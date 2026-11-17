"""OCR benchmark dashboard (local Gradio web app).

    python ocr_dashboard.py            # http://127.0.0.1:7860
    python ocr_dashboard.py --port 8000 --open

Tabs:
  Single image  - pick an image from data/ or upload one, optionally click 4 corners to warp,
                  run the selected engines, compare text / accuracy / time / detections.
  Batch         - label a folder of images (image, ground truth), run every engine on every image,
                  get per-engine accuracy / exact-match / latency summaries and a CSV export.
"""
import argparse
import os
import tempfile
import threading
import time

import cv2
import gradio as gr
import numpy as np
import pandas as pd

import benchmark_ocr as bo

DATA_DIR = os.path.join(bo.BASE_DIR, "data")
OUTPUT_DIR = os.path.join(bo.BASE_DIR, "output")
LABELS_FILENAME = "ocr_labels.csv"
IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
TMP_DIR = tempfile.mkdtemp(prefix="ocr_dashboard_")

YOLO_ENGINES = [f"yolo-ocr-{s}" for s in bo.YOLO_OCR_SIZES]
DEFAULT_ENGINES = [n for n in bo.ENGINE_NAMES if n != "glm-ocr"]  # glm-ocr currently fails to load

# One engine set per device, shared by all runs. Models are not thread-safe -> serialize runs.
_engine_cache = {"key": None, "engines": None}
_run_lock = threading.Lock()


def get_engines(device):
    key = device or "auto"
    if _engine_cache["key"] != key:
        _engine_cache["engines"] = None  # drop the old set before loading a new one
        _engine_cache["engines"] = bo.OCREngines(device_override=None if key == "auto" else key)
        _engine_cache["key"] = key
    return _engine_cache["engines"]


# ---------- helpers ----------

def list_images(folder, name_filter=""):
    if not folder or not os.path.isdir(folder):
        return []
    return sorted(
        f for f in os.listdir(folder)
        if f.lower().endswith(IMG_EXT) and (name_filter or "") in f
    )


def bgr_to_rgb(img):
    return None if img is None else cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def exact_match(gt, text):
    """Exact match ignoring whitespace (so 'TIS50-2565' == 'TIS 50-2565')."""
    if not gt:
        return None
    return "".join(gt.split()) == "".join((text or "").split())


def draw_engine_boxes(img, boxes, min_display=480):
    """Upscale small images for display, then draw one engine's boxes and texts."""
    scale = max(1.0, min_display / max(img.shape[:2]))
    view = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST) if scale > 1 else img.copy()
    for box in boxes:
        pts = bo._points_from_box(box)
        if pts is None or pts.size == 0:
            continue
        pts = (pts.astype(np.float32) * scale).round().astype(np.int32)
        cv2.polylines(view, [pts.reshape(-1, 1, 2)], True, (0, 200, 0), 2)
        x, y = int(pts[:, 0].min()), int(pts[:, 1].min())
        label = f"{box.get('text', '')} {box.get('score', 0):.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        y0 = max(y - 4, th + 4)
        cv2.rectangle(view, (x, y0 - th - 4), (x + tw + 4, y0 + 2), (0, 200, 0), -1)
        cv2.putText(view, label, (x + 2, y0 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return view


def results_frame(results, gt):
    rows = []
    for r in results:
        err = r.get("error")
        acc = r["acc"] if isinstance(r["acc"], float) else None
        rows.append({
            "engine": r["name"],
            "text": r["text"],
            "accuracy": None if acc is None else round(acc, 1),
            "exact": exact_match(gt, r["text"]) if not err else None,
            "time_ms": None if err else round(r["time_ms"], 1),
            "boxes": len(r.get("boxes") or []),
            "error": err or "",
        })
    df = pd.DataFrame(rows, columns=["engine", "text", "accuracy", "exact", "time_ms", "boxes", "error"])
    if gt:
        df = df.sort_values(["accuracy", "time_ms"], ascending=[False, True], na_position="last")
    return df


def bar_data(df, value_col):
    d = df[df["error"] == ""][["engine", value_col]].dropna()
    return d.rename(columns={value_col: "value"})


def clean_opt(s):
    s = (s or "").strip()
    return s or None


# ---------- single image ----------

def load_from_data(name):
    if not name:
        return None, None, [], None, None
    img = cv2.imread(os.path.join(DATA_DIR, name))
    if img is None:
        raise gr.Error(f"Cannot read {name}")
    return img, name, [], bgr_to_rgb(img), None


def load_from_upload(rgb):
    if rgb is None:
        return gr.skip(), gr.skip(), gr.skip(), gr.skip(), gr.skip()
    img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return img, "upload", [], rgb, None


def add_point(orig, points, mode, evt: gr.SelectData):
    if orig is None or mode != "pickpoints":
        return gr.skip(), gr.skip(), gr.skip()
    points = list(points or [])
    if len(points) >= 4:
        gr.Info("Already have 4 points - press 'Reset points' to start over.")
        return points, gr.skip(), gr.skip()
    x, y = evt.index
    points.append((int(x), int(y)))
    overlay = bgr_to_rgb(bo.draw_points_overlay(orig, points))
    warped = bgr_to_rgb(bo.warp_perspective(orig, points)) if len(points) == 4 else None
    return points, overlay, warped


def reset_points(orig):
    return [], bgr_to_rgb(orig), None


def run_single(orig, src_name, points, mode, gt, engine_names, whitelist, min_score, device,
               yolo_resize, warmup, progress=gr.Progress()):
    if orig is None:
        raise gr.Error("Pick an image from data/ or upload one first.")
    if not engine_names:
        raise gr.Error("Select at least one engine.")
    if mode == "pickpoints":
        if len(points or []) != 4:
            raise gr.Error("Pick-points mode: click 4 corners (TL, TR, BR, BL) on the image first.")
        processed = bo.warp_perspective(orig, points)
    else:
        processed = orig.copy()

    img_path = os.path.join(TMP_DIR, "single_input.png")  # lossless, for path-based engines
    cv2.imwrite(img_path, processed)
    gt, whitelist = clean_opt(gt), whitelist or None  # don't strip: a space can be a whitelisted char

    log_lines = []
    results = []
    with _run_lock:
        progress(0, desc=f"Loading engines on {device}...")
        engines = get_engines(device)
        for i, name in enumerate(engine_names):
            progress(i / len(engine_names), desc=f"{name} ({i + 1}/{len(engine_names)})")
            results += bo.run_benchmark(
                engines, processed, img_path, [name], min_score=min_score, gt=gt, whitelist=whitelist,
                yolo_ocr_resize=int(yolo_resize or 0), warmup=warmup, log=log_lines.append,
            )

    df = results_frame(results, gt)
    gallery = [
        (bgr_to_rgb(draw_engine_boxes(processed, r["boxes"])), f"{r['name']}: {r['text'] or '-'}")
        for r in results if r.get("boxes")
    ]

    # Same summary figure as the CLI, saved to output/
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    base = os.path.splitext(os.path.basename(src_name or "image"))[0]
    fig_path = os.path.join(OUTPUT_DIR, f"ocr_results_{base}.png")
    overlay = bo.draw_points_overlay(orig, points) if mode == "pickpoints" else orig
    bo.save_results_figure(orig, overlay, processed, results, fig_path, mode)

    acc_df = bar_data(df, "accuracy") if gt else pd.DataFrame(columns=["engine", "value"])
    return (
        df,
        gr.update(value=acc_df, visible=bool(gt)),
        bar_data(df, "time_ms"),
        gallery,
        fig_path,
        "\n".join(log_lines),
    )


# ---------- batch ----------

def labels_path(folder):
    return os.path.join(folder, LABELS_FILENAME)


def load_folder(folder, name_filter):
    files = list_images(folder, name_filter)
    if not files:
        raise gr.Error(f"No images in {folder!r} matching {name_filter!r}")
    known = {}
    path = labels_path(folder)
    if os.path.exists(path):
        saved = pd.read_csv(path, dtype=str, keep_default_na=False)
        known = dict(zip(saved["image"], saved["gt"]))
    df = pd.DataFrame({"image": files, "gt": [known.get(f, "") for f in files]})
    n_labeled = int((df["gt"] != "").sum())
    return df, f"{len(files)} images, {n_labeled} labeled (labels file: `{path}`)"


def save_labels(folder, table):
    if table is None or len(table) == 0:
        raise gr.Error("Nothing to save - load a folder first.")
    path = labels_path(folder)
    new = table[["image", "gt"]].fillna("").astype(str)
    if os.path.exists(path):  # keep labels for images hidden by the current filter
        old = pd.read_csv(path, dtype=str, keep_default_na=False)
        new = pd.concat([old[~old["image"].isin(new["image"])], new])
    new.sort_values("image").to_csv(path, index=False)
    gr.Info(f"Saved {int((new['gt'] != '').sum())} labels to {path}")
    return f"{len(table)} images, {int((table['gt'].fillna('') != '').sum())} labeled (labels file: `{path}`)"


def run_batch(folder, table, only_labeled, engine_names, whitelist, min_score, device, yolo_resize, warmup,
              progress=gr.Progress()):
    if table is None or len(table) == 0:
        raise gr.Error("Load a folder first.")
    if not engine_names:
        raise gr.Error("Select at least one engine.")
    rows = table.fillna("").astype(str)
    if only_labeled:
        rows = rows[rows["gt"].str.strip() != ""]
    if rows.empty:
        raise gr.Error("No labeled images - fill in the gt column (or untick 'only labeled').")
    whitelist = whitelist or None  # don't strip: a space can be a whitelisted char

    records = []
    total = len(rows) * len(engine_names)
    done = 0
    with _run_lock:
        progress(0, desc=f"Loading engines on {device}...")
        engines = get_engines(device)
        for name in engine_names:  # load + warm up each engine once, outside the timed loop
            engines.load(name)
        for _, row in rows.iterrows():
            path = os.path.join(folder, row["image"])
            img = cv2.imread(path)
            gt = clean_opt(row["gt"])
            for name in engine_names:
                progress(done / total, desc=f"{row['image']} - {name}")
                done += 1
                if img is None:
                    records.append({"image": row["image"], "gt": gt or "", "engine": name, "text": "",
                                    "accuracy": None, "exact": None, "time_ms": None, "error": "cannot read image"})
                    continue
                r = bo.run_benchmark(
                    engines, img, path, [name], min_score=min_score, gt=gt, whitelist=whitelist,
                    yolo_ocr_resize=int(yolo_resize or 0), warmup=warmup and done <= len(engine_names),
                    log=lambda *_: None,
                )[0]
                err = r.get("error", "")
                records.append({
                    "image": row["image"], "gt": gt or "", "engine": name, "text": r["text"],
                    "accuracy": r["acc"] if isinstance(r["acc"], float) else None,
                    "exact": exact_match(gt, r["text"]) if not err else None,
                    "time_ms": None if err else round(r["time_ms"], 1),
                    "error": err,
                })

    long_df = pd.DataFrame(records)
    summary = (
        long_df.groupby("engine", sort=False)
        .agg(
            images=("image", "count"),
            mean_accuracy=("accuracy", "mean"),
            exact_matches=("exact", lambda s: int(s.fillna(False).astype(bool).sum())),
            labeled=("exact", lambda s: int(s.notna().sum())),
            median_ms=("time_ms", "median"),
            mean_ms=("time_ms", "mean"),
            errors=("error", lambda s: int((s != "").sum())),
        )
        .reset_index()
    )
    summary["exact_rate_%"] = (100 * summary["exact_matches"] / summary["labeled"].where(summary["labeled"] > 0)).round(1)
    summary = summary.round({"mean_accuracy": 1, "median_ms": 1, "mean_ms": 1})
    summary = summary[["engine", "images", "mean_accuracy", "exact_matches", "exact_rate_%",
                       "median_ms", "mean_ms", "errors"]]
    summary = summary.sort_values(["mean_accuracy", "median_ms"], ascending=[False, True], na_position="last")

    texts_wide = long_df.pivot_table(index=["image", "gt"], columns="engine", values="text", aggfunc="first",
                                     sort=False).reset_index()
    texts_wide = texts_wide[["image", "gt", *[e for e in engine_names if e in texts_wide.columns]]]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUTPUT_DIR, f"batch_results_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    long_df.to_csv(csv_path, index=False)

    acc_chart = summary[["engine", "mean_accuracy"]].dropna().rename(columns={"mean_accuracy": "value"})
    time_chart = summary[["engine", "median_ms"]].dropna().rename(columns={"median_ms": "value"})
    return summary, acc_chart, time_chart, texts_wide, long_df, csv_path


# ---------- UI ----------

def build_app():
    data_files = list_images(DATA_DIR)

    with gr.Blocks(title="OCR Benchmark") as demo:
        gr.Markdown("# OCR Benchmark\nCompare OCR engines on your images: text, accuracy vs ground truth, latency.")

        with gr.Sidebar(open=True, width=340):
            gr.Markdown("### Engines")
            engines_cb = gr.CheckboxGroup(bo.ENGINE_NAMES, value=DEFAULT_ENGINES, show_label=False)
            with gr.Row():
                btn_all = gr.Button("All", size="sm")
                btn_yolo = gr.Button("YOLO sizes", size="sm")
                btn_none = gr.Button("None", size="sm")
            gr.Markdown("### Settings")
            whitelist_tb = gr.Textbox(label="Whitelist characters", placeholder="e.g. 0123456789TIS.- (empty = keep all)")
            min_score_sl = gr.Slider(0, 1, value=0.4, step=0.05, label="Min confidence")
            device_rd = gr.Radio(["auto", "cpu", "mps", "cuda"], value="auto", label="Device")
            yolo_resize_nb = gr.Number(value=0, precision=0, minimum=0, label="yolo-ocr resize (longest side px, 0 = off)",
                                       info="Tiny crops read better upscaled; ~160 worked on TIS labels")
            warmup_cb = gr.Checkbox(value=True, label="Warm up engines before timing",
                                    info="Excludes first-call overhead from latency")

        btn_all.click(lambda: bo.ENGINE_NAMES, outputs=engines_cb)
        btn_yolo.click(lambda: YOLO_ENGINES, outputs=engines_cb)
        btn_none.click(lambda: [], outputs=engines_cb)
        settings = [engines_cb, whitelist_tb, min_score_sl, device_rd, yolo_resize_nb, warmup_cb]

        with gr.Tab("Single image"):
            orig_state = gr.State(None)
            name_state = gr.State(None)
            points_state = gr.State([])
            with gr.Row():
                with gr.Column(scale=5):
                    with gr.Row():
                        data_dd = gr.Dropdown(data_files, label=f"Image from {os.path.relpath(DATA_DIR, bo.BASE_DIR)}/",
                                              value=None, scale=3)
                        upload_img = gr.Image(type="numpy", sources=["upload", "clipboard"], label="…or upload",
                                              height=110, scale=1)
                    mode_rd = gr.Radio(["full", "pickpoints"], value="full", label="Mode",
                                       info="pickpoints: click 4 corners on the image (TL, TR, BR, BL) to warp")
                    canvas = gr.Image(label="Image (click here to pick points)", interactive=False, height=420)
                    with gr.Row():
                        reset_btn = gr.Button("Reset points", size="sm")
                    warped_img = gr.Image(label="Warped (what the engines read)", interactive=False, height=200)
                with gr.Column(scale=7):
                    with gr.Row():
                        gt_tb = gr.Textbox(label="Ground truth", placeholder="e.g. TIS 50-2565 (optional)", scale=4)
                        run_btn = gr.Button("Run benchmark", variant="primary", scale=1)
                    results_df = gr.Dataframe(label="Results", interactive=False, wrap=True)
                    with gr.Row():
                        acc_plot = gr.BarPlot(x="engine", y="value", title="Accuracy vs ground truth (fuzz ratio)",
                                              y_title="accuracy", y_lim=[0, 100], sort="-y", x_label_angle=-35, height=280)
                        time_plot = gr.BarPlot(x="engine", y="value", title="Latency per image",
                                               y_title="ms", sort="y", x_label_angle=-35, height=280)
                    gallery = gr.Gallery(label="Detections per engine", columns=3, height="auto", object_fit="contain")
                    with gr.Accordion("Log & summary figure", open=False):
                        log_tb = gr.Textbox(lines=8, show_label=False)
                        fig_file = gr.File(label="Summary figure (also saved in output/)")

            load_outputs = [orig_state, name_state, points_state, canvas, warped_img]
            data_dd.change(load_from_data, data_dd, load_outputs)
            upload_img.upload(load_from_upload, upload_img, load_outputs)
            canvas.select(add_point, [orig_state, points_state, mode_rd], [points_state, canvas, warped_img])
            reset_btn.click(reset_points, orig_state, [points_state, canvas, warped_img])
            mode_rd.change(reset_points, orig_state, [points_state, canvas, warped_img])
            run_btn.click(
                run_single,
                [orig_state, name_state, points_state, mode_rd, gt_tb, *settings],
                [results_df, acc_plot, time_plot, gallery, fig_file, log_tb],
            )

        with gr.Tab("Batch"):
            gr.Markdown(
                "Label a folder once (image → ground truth), then run every selected engine on every image. "
                f"Labels are saved as `{LABELS_FILENAME}` inside the folder. Images are read as-is (no warping)."
            )
            with gr.Row():
                folder_tb = gr.Textbox(value=DATA_DIR, label="Folder", scale=4)
                filter_tb = gr.Textbox(value="_warped", label="Filename contains", scale=2)
                load_btn = gr.Button("Load folder", scale=1)
            status_md = gr.Markdown()
            labels_df = gr.Dataframe(headers=["image", "gt"], datatype=["str", "str"], interactive=True,
                                     label="Labels (edit the gt column)", max_height=320)
            with gr.Row():
                save_btn = gr.Button("Save labels")
                only_labeled_cb = gr.Checkbox(value=True, label="Only labeled images")
                batch_btn = gr.Button("Run batch", variant="primary")
            summary_df = gr.Dataframe(label="Per-engine summary", interactive=False)
            with gr.Row():
                b_acc_plot = gr.BarPlot(x="engine", y="value", title="Mean accuracy", y_title="accuracy",
                                        y_lim=[0, 100], sort="-y", x_label_angle=-35, height=280)
                b_time_plot = gr.BarPlot(x="engine", y="value", title="Median latency", y_title="ms",
                                         sort="y", x_label_angle=-35, height=280)
            texts_df = gr.Dataframe(label="Text per image × engine", interactive=False, wrap=True, max_height=400)
            with gr.Accordion("All results", open=False):
                long_df = gr.Dataframe(interactive=False, wrap=True, max_height=400)
                csv_file = gr.File(label="CSV (also saved in output/)")

            load_btn.click(load_folder, [folder_tb, filter_tb], [labels_df, status_md])
            save_btn.click(save_labels, [folder_tb, labels_df], status_md)
            batch_btn.click(
                run_batch,
                [folder_tb, labels_df, only_labeled_cb, *settings],
                [summary_df, b_acc_plot, b_time_plot, texts_df, long_df, csv_file],
            )

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR benchmark dashboard")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--host", default="127.0.0.1", help="Use 0.0.0.0 to reach it from other machines on the LAN")
    parser.add_argument("--open", action="store_true", help="Open the browser on start")
    args = parser.parse_args()
    build_app().queue(default_concurrency_limit=1).launch(
        server_name=args.host, server_port=args.port, inbrowser=args.open, allowed_paths=[OUTPUT_DIR],
    )

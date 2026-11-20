import cv2
import time
import datetime
import os
import threading
import logging
import base64
import json
import csv
import tkinter as tk
import numpy as np
from tkinter import Label, Frame, Canvas, Listbox, Scrollbar, Button
from PIL import Image, ImageTk
from collections import Counter
from enum import Enum, auto

# --- MISTRAL IMPORT ---
try:
    from mistralai.client import Mistral  # mistralai 2.x
except ImportError:
    from mistralai import Mistral  # mistralai 1.x compatibility

# ================= CONFIGURATION =================

API_KEY = os.getenv("MISTRAL_API_KEY", "").strip()
MODEL_ID = os.getenv("MISTRAL_MODEL_ID", "").strip() or "mistral-large-4-0"

# File Paths
LOG_FILE = "production_mistral_lib.log"
SAVE_DIR = "inspection_logs"
CSV_DIR = os.path.join(SAVE_DIR, "csv")
TXT_DIR = os.path.join(SAVE_DIR, "txt")
IMAGES_DIR = os.path.join(SAVE_DIR, "images")
CURRENT_TOTALS_FILE = os.path.join(SAVE_DIR, "current_totals.txt")
CAM_ID = 0

# CSV / TXT behavior
CSV_DELIMITER = ";"
CSV_ENCODING = "utf-8-sig"
TXT_ENCODING = "utf-8"
PASS_SAMPLE_INTERVAL = 20  # Save every Nth PASS image. FAIL images are always saved.

# --- TOKEN / COST REDUCTION FOR MODEL INPUT ---
# These affect only the image sent to the model, not the saved production evidence image.
MODEL_INPUT_MAX_DIM = 1200
MODEL_JPEG_QUALITY = 90
MODEL_MAX_TOKENS = 32

# Camera / UI
CAM_WIDTH = 1920
CAM_HEIGHT = 1060
UI_WIDTH = 640
UI_HEIGHT = 360

# Timing
COOLDOWN_SECONDS = 10
STABILITY_FRAMES = 8
EMPTY_CONFIRM_FRAMES = 2
PRODUCT_CONFIRM_FRAMES = 2

# ROI / preprocessing
MIN_ROI_SIZE = 20
BLUR_KERNEL = (7, 7)

# ======== GRAYSCALE DIFFERENCE THRESHOLDS ========
GRAY_PIXEL_DIFF_THRESHOLD = 70
GRAY_EMPTY_THRESHOLD = 4.0
GRAY_PRODUCT_THRESHOLD = 9.0

# Frame-to-frame stillness
GRAY_STILLNESS_PIXEL_THRESHOLD = 12
GRAY_STILLNESS_THRESHOLD = 1.5

# Optional edge confirmation
USE_EDGE_CONFIRMATION = False
EDGE_PRODUCT_THRESHOLD = 5.0

# Preview brightening only
PREVIEW_GAMMA = 1.8
PREVIEW_ALPHA = 1.15
PREVIEW_BETA = 18

CSV_FIELDNAMES = [
    "timestamp",
    "date",
    "inspection_id",
    "daily_sequence",
    "verdict",
    "reason_code",
    "reason_text",
    "image_saved",
    "image_file",
    "save_error",
    "gray_diff_empty",
    "edge_diff_empty",
    "gray_stillness",
    "state_before_result",
    "model_verdict_raw",
    "model_reason_raw",
]

MODEL_REASON_MAP = {
    "ok": ("ok", "OK"),
    "mp": ("missing_pinhole", "Missing required dark pinhole/mark"),
    "rd": ("rim_damage", "Major rim damage / chip"),
    "ri": ("rim_incomplete", "Incomplete outer rim / missing section"),
    "sd": ("structural_damage", "Crack / deformation / structural damage"),
    "co": ("color_not_ok", "Color outside allowed range"),
    "un": ("uncertain", "Uncertain - rejected"),
    "od": ("defect_other", "Other detected defect"),
}

logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

for d in [SAVE_DIR, CSV_DIR, TXT_DIR, IMAGES_DIR]:
    os.makedirs(d, exist_ok=True)


class VisionState(Enum):
    SET_ROI = auto()
    CALIBRATE = auto()
    ARMED = auto()
    TRANSITION = auto()
    PRODUCT_MOVING = auto()
    PRODUCT_STILL = auto()
    ANALYZING = auto()
    COOLDOWN = auto()
    WAIT_RESET = auto()


class CloudInspectorGrayscaleDebug:
    def __init__(self, window):
        self.window = window
        self.window.title(f"Pixtral Inspector Grayscale Debug - {MODEL_ID}")
        self.window.geometry("1400x860")
        self.window.configure(bg="#0f172a")

        # --- MISTRAL ---
        self.client = None
        try:
            if API_KEY.strip():
                self.client = Mistral(api_key=API_KEY)
            else:
                print("Warning: MISTRAL_API_KEY is not set.")
        except Exception as e:
            logging.exception("Client Init Error")
            print(f"Client Init Error: {e}")

        # --- STATE ---
        self.state = VisionState.SET_ROI
        self.is_analyzing = False
        self.last_scan_time = 0
        self.latest_display_frame = None

        # Baselines
        self.baseline_empty_gray = None
        self.baseline_empty_edges = None

        # Previous frame for stillness
        self.prev_gray = None

        # Counters
        self.stability_counter = 0
        self.empty_counter = 0
        self.product_counter = 0

        # ROI state
        self.roi_defined = False
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.roi_coords = None
        # Telemetry
        self.last_gray_diff_empty = None
        self.last_edge_diff_empty = None
        self.last_gray_stillness = None
        self.last_classification = "N/A"
        self.last_empty_bool = False
        self.last_product_bool = False
        self.last_uncertain_bool = False

        # Daily stats
        self.pass_count = 0
        self.fail_count = 0
        self.failure_reasons = Counter()
        self.current_log_date = None
        self.current_csv_path = None
        self.current_txt_path = None
        self.current_image_dir = None
        self.current_failed_dir = None
        self.daily_inspection_seq = 0

        # Canvas image handle
        self.canvas_image_id = None

        self._setup_ui()
        self._set_state(
            VisionState.SET_ROI,
            "SET ROI",
            "#94a3b8",
            "",
            "1. Draw ROI.\n2. Set EMPTY ARM baseline.\n3. Run cycle."
        )

        self._rollover_daily_context(datetime.datetime.now(), force_reload=True)
        self._refresh_stats_ui()
        self._write_current_totals_file(datetime.datetime.now())

        self._configure_camera()
        self.update_loop()

    # =========================================================
    # Camera
    # =========================================================

    def _configure_camera(self):
        self.cap = cv2.VideoCapture(CAM_ID, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 25)

        # Safer defaults: use auto exposure / auto white balance first.
        self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
        self.cap.set(cv2.CAP_PROP_AUTO_WB, 1)

        # Conservative values; some webcams ignore these, but they are sane.
        self.cap.set(cv2.CAP_PROP_BRIGHTNESS, 128)
        self.cap.set(cv2.CAP_PROP_CONTRAST, 32)
        self.cap.set(cv2.CAP_PROP_GAIN, 0)
        self.cap.set(cv2.CAP_PROP_GAMMA, 120)
        self.cap.set(cv2.CAP_PROP_BACKLIGHT, 0)

        if not self.cap.isOpened():
            logging.error("Could not open camera ID %s", CAM_ID)
            self.lbl_status.config(text="CAMERA ERROR", fg="#f87171")
            self.lbl_reason.config(text=f"Could not open camera ID {CAM_ID}")

    def _enhance_preview(self, frame):
        """
        Brighten preview only. Does not affect saved inspection evidence image.
        """
        if frame is None:
            return frame

        gamma = PREVIEW_GAMMA
        inv_gamma = 1.0 / gamma
        table = np.array(
            [((i / 255.0) ** inv_gamma) * 255 for i in np.arange(256)],
            dtype=np.uint8
        )
        bright = cv2.LUT(frame, table)
        bright = cv2.convertScaleAbs(bright, alpha=PREVIEW_ALPHA, beta=PREVIEW_BETA)
        return bright

    # =========================================================
    # UI
    # =========================================================

    def _setup_ui(self):
        self.left_frame = Frame(self.window, bg="black")
        self.left_frame.pack(side=tk.LEFT, padx=15, pady=15, anchor="n")

        Label(
            self.left_frame,
            text="DRAW ROI TO START",
            fg="#fbbf24",
            bg="black",
            font=("Arial", 12, "bold")
        ).pack()

        self.canvas = Canvas(
            self.left_frame,
            width=UI_WIDTH,
            height=UI_HEIGHT,
            bg="black",
            highlightthickness=0
        )
        self.canvas.pack()

        self.canvas.bind("<Button-1>", self.on_mouse_down)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_up)

        self.right_frame = Frame(self.window, bg="#0f172a")
        self.right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=20)

        self.lbl_status = Label(
            self.right_frame,
            text="SET ROI",
            font=("Helvetica", 28, "bold"),
            fg="#94a3b8",
            bg="#0f172a"
        )
        self.lbl_status.pack(pady=(20, 5))

        self.lbl_timer = Label(
            self.right_frame,
            text="",
            font=("Consolas", 14),
            fg="#fbbf24",
            bg="#0f172a"
        )
        self.lbl_timer.pack(pady=5)

        self.lbl_reason = Label(
            self.right_frame,
            text="Draw ROI and set empty-arm baseline.",
            font=("Consolas", 11),
            fg="#cbd5e1",
            bg="#1e293b",
            wraplength=430,
            justify="left",
            padx=10,
            pady=10,
            width=52
        )
        self.lbl_reason.pack(pady=10)

        self.calib_frame = Frame(self.right_frame, bg="#0f172a")
        self.calib_frame.pack(pady=5)

        self.btn_empty = Button(
            self.calib_frame,
            text="Set EMPTY ARM Baseline",
            font=("Arial", 10, "bold"),
            bg="#3b82f6",
            fg="white",
            command=self.set_baseline_empty
        )
        self.btn_empty.pack(side=tk.LEFT, padx=5)

        self.btn_rearm = Button(
            self.calib_frame,
            text="Force Rearm",
            font=("Arial", 10, "bold"),
            bg="#475569",
            fg="white",
            command=self.force_rearm
        )
        self.btn_rearm.pack(side=tk.LEFT, padx=5)

        self.stats_frame = Frame(self.right_frame, bg="#0f172a")
        self.stats_frame.pack(pady=10)

        self.lbl_pass = Label(
            self.stats_frame,
            text="PASS: 0",
            font=("Helvetica", 16, "bold"),
            fg="#4ade80",
            bg="#0f172a"
        )
        self.lbl_pass.pack(side="left", padx=15)

        self.lbl_fail = Label(
            self.stats_frame,
            text="FAIL: 0",
            font=("Helvetica", 16, "bold"),
            fg="#f87171",
            bg="#0f172a"
        )
        self.lbl_fail.pack(side="right", padx=15)

        self.debug_frame = Frame(self.right_frame, bg="#111827", bd=0, relief=tk.FLAT)
        self.debug_frame.pack(fill=tk.X, pady=(15, 10))

        Label(
            self.debug_frame,
            text="Live Telemetry",
            font=("Arial", 11, "bold"),
            fg="#93c5fd",
            bg="#111827"
        ).pack(anchor="w", padx=10, pady=(10, 6))

        self.lbl_debug = Label(
            self.debug_frame,
            text="No telemetry yet",
            font=("Consolas", 10),
            fg="#e5e7eb",
            bg="#111827",
            justify="left",
            anchor="w",
            padx=10,
            pady=10
        )
        self.lbl_debug.pack(fill=tk.X)

        Label(
            self.right_frame,
            text="Failure Analysis:",
            fg="#94a3b8",
            bg="#0f172a",
            font=("Arial", 10, "bold")
        ).pack(pady=(20, 0), anchor="w")

        self.list_frame = Frame(self.right_frame)
        self.list_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.fail_list = Listbox(
            self.list_frame,
            bg="#1e293b",
            fg="#f87171",
            font=("Consolas", 10),
            borderwidth=0,
            highlightthickness=0
        )
        self.fail_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        scrollbar = Scrollbar(self.list_frame, orient="vertical", command=self.fail_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill="y")
        self.fail_list.config(yscrollcommand=scrollbar.set)

    def _set_state(self, state, status_text=None, status_fg=None, timer_text=None, reason_text=None):
        self.state = state
        if status_text is not None:
            self.lbl_status.config(text=status_text)
        if status_fg is not None:
            self.lbl_status.config(fg=status_fg)
        if timer_text is not None:
            self.lbl_timer.config(text=timer_text)
        if reason_text is not None:
            self.lbl_reason.config(text=reason_text)

    def _update_debug_labels(self):
        ge = "n/a" if self.last_gray_diff_empty is None else f"{self.last_gray_diff_empty:.2f}%"
        ee = "n/a" if self.last_edge_diff_empty is None else f"{self.last_edge_diff_empty:.2f}%"
        st = "n/a" if self.last_gray_stillness is None else f"{self.last_gray_stillness:.2f}%"

        txt = (
            f"State:              {self.state.name}\n"
            f"Classification:     {self.last_classification}\n"
            f"\n"
            f"Gray diff empty:    {ge}\n"
            f"Edge diff empty:    {ee}\n"
            f"Stillness:          {st}\n"
            f"\n"
            f"Empty?              {self.last_empty_bool}\n"
            f"Product?            {self.last_product_bool}\n"
            f"Uncertain?          {self.last_uncertain_bool}\n"
            f"\n"
            f"Product counter:    {self.product_counter}/{PRODUCT_CONFIRM_FRAMES}\n"
            f"Empty counter:      {self.empty_counter}/{EMPTY_CONFIRM_FRAMES}\n"
            f"Stable counter:     {self.stability_counter}/{STABILITY_FRAMES}\n"
            f"\n"
            f"Thresh empty <      {GRAY_EMPTY_THRESHOLD:.2f}\n"
            f"Thresh product >    {GRAY_PRODUCT_THRESHOLD:.2f}\n"
            f"Thresh still <      {GRAY_STILLNESS_THRESHOLD:.2f}\n"
            f"Pixel diff thresh   {GRAY_PIXEL_DIFF_THRESHOLD}\n"
        )

        if USE_EDGE_CONFIRMATION:
            txt += f"Edge confirm >      {EDGE_PRODUCT_THRESHOLD:.2f}\n"

        self.lbl_debug.config(text=txt)

    def _refresh_stats_ui(self):
        self.lbl_pass.config(text=f"PASS: {self.pass_count}")
        self.lbl_fail.config(text=f"FAIL: {self.fail_count}")
        self._update_fail_list()

    # =========================================================
    # Daily persistence helpers
    # =========================================================

    def _rollover_daily_context(self, now, force_reload=False):
        date_str = now.strftime("%Y-%m-%d")
        if not force_reload and self.current_log_date == date_str:
            return

        self.current_log_date = date_str
        self.current_csv_path = os.path.join(CSV_DIR, f"inspection_events_{date_str}.csv")
        self.current_txt_path = os.path.join(TXT_DIR, f"inspection_events_{date_str}.txt")
        self.current_image_dir = os.path.join(IMAGES_DIR, date_str)
        self.current_failed_dir = os.path.join(self.current_image_dir, "FAILED")

        os.makedirs(self.current_image_dir, exist_ok=True)
        os.makedirs(self.current_failed_dir, exist_ok=True)

        self.pass_count = 0
        self.fail_count = 0
        self.failure_reasons = Counter()
        self.daily_inspection_seq = 0

        self._restore_daily_state_from_csv()

    def _restore_daily_state_from_csv(self):
        if not self.current_csv_path or not os.path.exists(self.current_csv_path):
            return

        try:
            with open(self.current_csv_path, "r", encoding=CSV_ENCODING, newline="") as f:
                reader = csv.DictReader(f, delimiter=CSV_DELIMITER)
                for row in reader:
                    seq = row.get("daily_sequence", "").strip()
                    if seq.isdigit():
                        self.daily_inspection_seq = max(self.daily_inspection_seq, int(seq))
                    else:
                        inspection_id = row.get("inspection_id", "")
                        seq_tail = inspection_id.split("-")[-1] if inspection_id else ""
                        if seq_tail.isdigit():
                            self.daily_inspection_seq = max(self.daily_inspection_seq, int(seq_tail))

                    verdict = row.get("verdict", "").strip().upper()
                    if verdict == "PASS":
                        self.pass_count += 1
                    elif verdict == "FAIL":
                        self.fail_count += 1
                        reason_code = row.get("reason_code", "").strip() or "defect_other"
                        self.failure_reasons[reason_code] += 1
        except Exception:
            logging.exception("Failed to restore daily state from CSV")

    def _next_inspection_id(self, now):
        self.daily_inspection_seq += 1
        return f"{now.strftime('%Y-%m-%d')}-{self.daily_inspection_seq:06d}"

    def _format_metric(self, value):
        if value is None:
            return ""
        try:
            return f"{float(value):.2f}"
        except Exception:
            return ""

    def _sanitize_reason_fragment(self, value, max_len=40):
        safe = "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in value.lower())
        while "__" in safe:
            safe = safe.replace("__", "_")
        return safe.strip("_")[:max_len] or "unknown"

    def _reason_code_to_display(self, code):
        display_map = {
            "ok": "OK",
            "missing_pinhole": "Missing pinhole",
            "rim_damage": "Rim damage",
            "rim_incomplete": "Rim incomplete",
            "structural_damage": "Structural damage",
            "color_not_ok": "Color not OK",
            "multiple_holes": "Multiple holes",
            "api_error": "API error",
            "invalid_model_output": "Invalid model output",
            "uncertain": "Uncertain",
            "defect_other": "Other defect",
        }
        return display_map.get(code, code.replace("_", " ").title())

    def _map_reason_code(self, verdict, reason_text, model_data=None):
        if verdict == "PASS":
            return "ok"

        lower_reason = (reason_text or "").strip().lower()

        if "api error" in lower_reason:
            return "api_error"
        if "invalid verdict" in lower_reason or "invalid model" in lower_reason or "json" in lower_reason:
            return "invalid_model_output"
        if "uncertain" in lower_reason:
            return "uncertain"

        if isinstance(model_data, dict):
            raw_rc = str(model_data.get("rc", "")).strip().lower()
            if raw_rc in MODEL_REASON_MAP:
                return MODEL_REASON_MAP[raw_rc][0]

            if model_data.get("pinhole_present") is False:
                return "missing_pinhole"
            if model_data.get("major_rim_damage") is True:
                return "rim_damage"
            if model_data.get("rim_complete") is False:
                return "rim_incomplete"
            if model_data.get("structural_damage") is True:
                return "structural_damage"
            if model_data.get("color_ok") is False:
                return "color_not_ok"

        keyword_map = [
            ("multiple holes", "multiple_holes"),
            ("pinhole", "missing_pinhole"),
            ("rim damage", "rim_damage"),
            ("chip", "rim_damage"),
            ("crack", "structural_damage"),
            ("deformation", "structural_damage"),
            ("structural", "structural_damage"),
            ("incomplete outer circle", "rim_incomplete"),
            ("incomplete", "rim_incomplete"),
            ("color", "color_not_ok"),
        ]
        for keyword, code in keyword_map:
            if keyword in lower_reason:
                return code

        return "defect_other"

    def _decode_model_reason(self, verdict, raw_rc):
        raw_rc = (raw_rc or "").strip().lower()

        if verdict == "PASS":
            if raw_rc == "" or raw_rc == "ok":
                return "ok", "OK"

        if raw_rc in MODEL_REASON_MAP:
            return MODEL_REASON_MAP[raw_rc]

        if verdict == "PASS":
            return "ok", "OK"

        return "defect_other", "Other detected defect"

    def _should_save_pass_image(self, next_pass_count):
        if PASS_SAMPLE_INTERVAL <= 0:
            return False
        return next_pass_count % PASS_SAMPLE_INTERVAL == 0

    def _build_image_filename(self, now, inspection_id, verdict, reason_code):
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        seq_part = inspection_id.split("-")[-1]
        safe_reason_code = self._sanitize_reason_fragment(reason_code, max_len=32)
        return f"{timestamp}_ID{seq_part}_{verdict}_{safe_reason_code}.jpg"

    def _fsync_file(self, file_obj):
        file_obj.flush()
        os.fsync(file_obj.fileno())

    def _append_txt_line(self, row):
        line = (
            f"[{row['timestamp']}] {row['inspection_id']} {row['verdict']} | "
            f"code={row['reason_code']} | reason={row['reason_text']}"
        )
        if row["image_saved"] == "1" and row["image_file"]:
            line += f" | image={row['image_file']}"
        else:
            line += " | image=NOT_SAVED"
        if row["save_error"]:
            line += f" | save_error={row['save_error']}"
        line += "\n"

        with open(self.current_txt_path, "a", encoding=TXT_ENCODING) as f:
            f.write(line)
            self._fsync_file(f)

    def _append_csv_row(self, row):
        needs_header = not os.path.exists(self.current_csv_path) or os.path.getsize(self.current_csv_path) == 0

        with open(self.current_csv_path, "a", encoding=CSV_ENCODING, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, delimiter=CSV_DELIMITER)
            if needs_header:
                writer.writeheader()
            writer.writerow(row)
            self._fsync_file(f)

    def _write_current_totals_file(self, now):
        total = self.pass_count + self.fail_count
        lines = [
            f"Last Update: {now.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Date: {self.current_log_date or now.strftime('%Y-%m-%d')}",
            f"Total Products: {total}",
            f"Passed: {self.pass_count}",
            f"Failed: {self.fail_count}",
            "",
            "Failure Reasons Summary:",
        ]

        if self.failure_reasons:
            for code, count in self.failure_reasons.most_common():
                lines.append(f"- {self._reason_code_to_display(code)}: {count}")
        else:
            lines.append("- none")

        content = "\n".join(lines) + "\n"
        temp_path = CURRENT_TOTALS_FILE + ".tmp"

        with open(temp_path, "w", encoding=TXT_ENCODING) as f:
            f.write(content)
            self._fsync_file(f)

        os.replace(temp_path, CURRENT_TOTALS_FILE)

    def _delete_temp_file(self, temp_path):
        try:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)
        except Exception:
            logging.exception("Could not delete temp image %s", temp_path)

    def _persist_completed_inspection(
        self,
        temp_path,
        verdict,
        reason_text,
        model_data,
        inspection_context,
        model_verdict_raw,
        model_reason_raw,
    ):
        now = datetime.datetime.now()
        self._rollover_daily_context(now)

        reason_code = self._map_reason_code(verdict, reason_text, model_data)
        inspection_id = self._next_inspection_id(now)

        next_pass_count = self.pass_count + 1 if verdict == "PASS" else self.pass_count
        image_should_save = verdict == "FAIL" or self._should_save_pass_image(next_pass_count)

        image_saved = False
        image_file = ""
        save_error = ""

        if image_should_save:
            final_name = self._build_image_filename(now, inspection_id, verdict, reason_code)
            target_dir = self.current_failed_dir if verdict == "FAIL" else self.current_image_dir
            target_path = os.path.join(target_dir, final_name)

            try:
                if os.path.exists(temp_path):
                    os.replace(temp_path, target_path)
                    image_saved = True
                    image_file = final_name
                else:
                    save_error = "temp image missing"
                    logging.error("Temp image missing before save for %s", inspection_id)
            except Exception as e:
                save_error = str(e)
                logging.exception("File save error for %s", inspection_id)
                self._delete_temp_file(temp_path)
        else:
            self._delete_temp_file(temp_path)

        row = {
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "date": self.current_log_date,
            "inspection_id": inspection_id,
            "daily_sequence": str(self.daily_inspection_seq),
            "verdict": verdict,
            "reason_code": reason_code,
            "reason_text": reason_text,
            "image_saved": "1" if image_saved else "0",
            "image_file": image_file,
            "save_error": save_error,
            "gray_diff_empty": self._format_metric(inspection_context.get("gray_diff_empty")),
            "edge_diff_empty": self._format_metric(inspection_context.get("edge_diff_empty")),
            "gray_stillness": self._format_metric(inspection_context.get("gray_stillness")),
            "state_before_result": inspection_context.get("state_before_result", ""),
            "model_verdict_raw": model_verdict_raw,
            "model_reason_raw": model_reason_raw,
        }

        self._append_txt_line(row)
        self._append_csv_row(row)

        if verdict == "PASS":
            self.pass_count += 1
        else:
            self.fail_count += 1
            self.failure_reasons[reason_code] += 1

        self._write_current_totals_file(now)
        return row

    def finalize_completed_inspection(
        self,
        temp_path,
        verdict,
        reason_text,
        model_data,
        inspection_context,
        model_verdict_raw,
        model_reason_raw,
    ):
        try:
            row = self._persist_completed_inspection(
                temp_path=temp_path,
                verdict=verdict,
                reason_text=reason_text,
                model_data=model_data,
                inspection_context=inspection_context,
                model_verdict_raw=model_verdict_raw,
                model_reason_raw=model_reason_raw,
            )
        except Exception as e:
            logging.exception("Persistence error after inspection")
            row = {
                "inspection_id": "PERSISTENCE_ERROR",
                "verdict": verdict,
                "reason_text": f"{reason_text} | Persistence Error: {e}",
                "reason_code": self._map_reason_code(verdict, reason_text, model_data),
                "image_saved": "0",
                "image_file": "",
                "save_error": str(e),
            }
            self._delete_temp_file(temp_path)

        self.ui_finish_scan(row)

    # =========================================================
    # ROI / helpers
    # =========================================================

    def get_normalized_roi(self):
        if not self.roi_coords:
            return None
        x, y, w, h = self.roi_coords
        if w < 0:
            x += w
            w = abs(w)
        if h < 0:
            y += h
            h = abs(h)
        return (x, y, w, h)

    def get_current_roi_view(self):
        if not self.roi_defined or self.latest_display_frame is None:
            return None

        roi = self.get_normalized_roi()
        if roi is None:
            return None

        x, y, w, h = roi
        x = max(0, min(UI_WIDTH - 1, x))
        y = max(0, min(UI_HEIGHT - 1, y))
        w = max(1, min(UI_WIDTH - x, w))
        h = max(1, min(UI_HEIGHT - y, h))

        roi_view = self.latest_display_frame[y:y + h, x:x + w]
        if roi_view.size == 0:
            return None
        return roi_view

    def get_current_roi_gray(self):
        roi_view = self.get_current_roi_view()
        if roi_view is None:
            return None
        gray = cv2.cvtColor(roi_view, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
        return gray

    def get_current_roi_edges(self):
        roi_view = self.get_current_roi_view()
        if roi_view is None:
            return None
        gray = cv2.cvtColor(roi_view, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, BLUR_KERNEL, 0)
        return cv2.Canny(gray, 50, 150)

    def get_gray_diff_percent(self, img1, img2, pixel_threshold=25):
        diff = cv2.absdiff(img1, img2)
        _, thresh = cv2.threshold(diff, pixel_threshold, 255, cv2.THRESH_BINARY)
        changed = cv2.countNonZero(thresh)
        total = thresh.shape[0] * thresh.shape[1]
        if total <= 0:
            return 0.0
        return (changed / total) * 100.0

    def get_edge_diff_percent(self, edge1, edge2):
        diff = cv2.absdiff(edge1, edge2)
        changed = cv2.countNonZero(diff)
        total = diff.shape[0] * diff.shape[1]
        if total <= 0:
            return 0.0
        return (changed / total) * 100.0

    def reset_motion_state(self):
        self.prev_gray = None
        self.stability_counter = 0
        self.empty_counter = 0
        self.product_counter = 0
        self.last_gray_stillness = None

    # =========================================================
    # Baseline
    # =========================================================

    def set_baseline_empty(self):
        gray = self.get_current_roi_gray()
        edges = self.get_current_roi_edges()

        if gray is None or edges is None:
            self._set_state(
                VisionState.CALIBRATE,
                "CALIBRATE",
                "#f59e0b",
                "ROI/frame unavailable",
                "Could not capture baseline. Check ROI and camera."
            )
            return

        self.baseline_empty_gray = gray
        self.baseline_empty_edges = edges
        self.reset_motion_state()

        self.btn_empty.config(bg="#10b981", text="EMPTY ARM - OK")

        self._set_state(
            VisionState.ARMED,
            "ARMED",
            "#38bdf8",
            "Waiting for product...",
            "Empty-arm baseline captured. System ready."
        )

    def force_rearm(self):
        if self.is_analyzing:
            self.lbl_reason.config(text="Analysis in progress. Wait for it to finish before rearming.")
            return

        if self.baseline_empty_gray is None:
            self._set_state(
                VisionState.CALIBRATE,
                "CALIBRATE",
                "#f59e0b",
                "Set empty baseline first",
                "Force rearm unavailable until empty-arm baseline exists."
            )
            return

        self.is_analyzing = False
        self.last_scan_time = 0
        self.reset_motion_state()
        self._set_state(
            VisionState.ARMED,
            "ARMED",
            "#38bdf8",
            "Waiting for product...",
            "Manually rearmed."
        )

    # =========================================================
    # Mouse
    # =========================================================

    def on_mouse_down(self, event):
        self.drawing = True
        self.ix, self.iy = event.x, event.y
        self.roi_coords = (self.ix, self.iy, 0, 0)

    def on_mouse_drag(self, event):
        if self.drawing:
            self.roi_coords = (self.ix, self.iy, event.x - self.ix, event.y - self.iy)

    def on_mouse_up(self, event):
        self.drawing = False

        w = abs(event.x - self.ix)
        h = abs(event.y - self.iy)
        x = min(self.ix, event.x)
        y = min(self.iy, event.y)

        if w >= MIN_ROI_SIZE and h >= MIN_ROI_SIZE:
            self.roi_coords = (x, y, w, h)
            self.roi_defined = True

            self.baseline_empty_gray = None
            self.baseline_empty_edges = None
            self.reset_motion_state()
            self.btn_empty.config(bg="#3b82f6", text="Set EMPTY ARM Baseline")

            self._set_state(
                VisionState.CALIBRATE,
                "CALIBRATE",
                "#f59e0b",
                "Capture empty-arm baseline",
                "ROI set. Put empty arm in normal waiting position, then click 'Set EMPTY ARM Baseline'."
            )
        else:
            self.roi_coords = None
            self.roi_defined = False
            self._set_state(
                VisionState.SET_ROI,
                "SET ROI",
                "#94a3b8",
                "",
                "ROI too small. Draw a larger box."
            )

    # =========================================================
    # Main loop
    # =========================================================

    def update_loop(self):
        if self.cap is None or not self.cap.isOpened():
            self.window.after(100, self.update_loop)
            return

        ret, raw_frame = self.cap.read()
        if ret:
            display_frame = cv2.resize(raw_frame, (UI_WIDTH, UI_HEIGHT))
            display_frame = self._enhance_preview(display_frame)
            self.latest_display_frame = display_frame.copy()

            if self.roi_defined and not self.is_analyzing:
                self._process_vision_logic(raw_frame)

            overlay_frame = display_frame.copy()

            if self.roi_coords:
                x, y, w, h = self.get_normalized_roi()
                cv2.rectangle(overlay_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            self._draw_overlay_telemetry(overlay_frame)
            self._display_on_canvas(overlay_frame)
            self._update_debug_labels()

        self.window.after(15, self.update_loop)

    def _process_vision_logic(self, raw_frame):
        current_time = time.time()
        time_since_scan = current_time - self.last_scan_time

        if time_since_scan < COOLDOWN_SECONDS:
            remaining = int(COOLDOWN_SECONDS - time_since_scan)
            self.last_classification = "COOLDOWN"
            # Keep WAIT_RESET active while only the visible status is in cooldown.
            # Otherwise a stationary product can be inspected again after cooldown.
            self.lbl_status.config(text=f"COOLDOWN ({remaining}s)", fg="#64748b")
            self.lbl_timer.config(text="Saving Image..." if remaining > 8 else "Resetting...")
            return

        if self.baseline_empty_gray is None:
            self.last_classification = "NO_BASELINE"
            self._set_state(
                VisionState.CALIBRATE,
                "CALIBRATE",
                "#f59e0b",
                "Capture empty-arm baseline",
                "Awaiting empty-arm baseline."
            )
            return

        current_gray = self.get_current_roi_gray()
        if current_gray is None:
            return

        current_edges = None
        if USE_EDGE_CONFIRMATION:
            current_edges = self.get_current_roi_edges()
            if current_edges is None:
                return

        gray_diff_empty = self.get_gray_diff_percent(
            current_gray,
            self.baseline_empty_gray,
            pixel_threshold=GRAY_PIXEL_DIFF_THRESHOLD
        )
        self.last_gray_diff_empty = gray_diff_empty

        edge_diff_empty = None
        if USE_EDGE_CONFIRMATION:
            edge_diff_empty = self.get_edge_diff_percent(current_edges, self.baseline_empty_edges)
            self.last_edge_diff_empty = edge_diff_empty
        else:
            self.last_edge_diff_empty = None

        if self.prev_gray is None:
            self.prev_gray = current_gray
            return

        gray_stillness = self.get_gray_diff_percent(
            current_gray,
            self.prev_gray,
            pixel_threshold=GRAY_STILLNESS_PIXEL_THRESHOLD
        )
        self.prev_gray = current_gray
        self.last_gray_stillness = gray_stillness

        definitely_empty = gray_diff_empty < GRAY_EMPTY_THRESHOLD
        definitely_product = gray_diff_empty > GRAY_PRODUCT_THRESHOLD

        if USE_EDGE_CONFIRMATION and definitely_product:
            definitely_product = edge_diff_empty is not None and edge_diff_empty > EDGE_PRODUCT_THRESHOLD

        uncertain = not definitely_empty and not definitely_product

        self.last_empty_bool = definitely_empty
        self.last_product_bool = definitely_product
        self.last_uncertain_bool = uncertain

        if definitely_empty:
            self.last_classification = "EMPTY"
        elif definitely_product:
            self.last_classification = "PRODUCT"
        else:
            self.last_classification = "TRANSITION"

        print(
            f"GrayEmpty: {gray_diff_empty:.2f}% | "
            f"EdgeEmpty: {self.last_edge_diff_empty if self.last_edge_diff_empty is not None else 'n/a'} | "
            f"Stillness: {gray_stillness:.2f}% | "
            f"Class: {self.last_classification} | "
            f"State: {self.state.name}"
        )

        if self.state == VisionState.WAIT_RESET:
            if definitely_empty:
                self.empty_counter += 1
            else:
                self.empty_counter = 0

            if self.empty_counter >= EMPTY_CONFIRM_FRAMES:
                self.reset_motion_state()
                self._set_state(
                    VisionState.ARMED,
                    "ARMED",
                    "#38bdf8",
                    "Waiting for product...",
                    "Reset complete. Empty arm detected."
                )
            else:
                self._set_state(
                    VisionState.WAIT_RESET,
                    "WAITING CLEAR",
                    "#fbbf24",
                    f"Reset confirm ({self.empty_counter}/{EMPTY_CONFIRM_FRAMES})",
                    self._format_telemetry()
                )
            return

        if definitely_empty:
            self.empty_counter += 1
            self.product_counter = 0
            self.stability_counter = 0

            if self.empty_counter >= EMPTY_CONFIRM_FRAMES:
                self._set_state(
                    VisionState.ARMED,
                    "ARMED",
                    "#38bdf8",
                    "Waiting for product...",
                    self._format_telemetry()
                )
            else:
                self._set_state(
                    VisionState.TRANSITION,
                    "EMPTY?",
                    "#38bdf8",
                    f"Confirming empty ({self.empty_counter}/{EMPTY_CONFIRM_FRAMES})",
                    self._format_telemetry()
                )
            return

        self.empty_counter = 0

        if uncertain:
            self.product_counter = 0
            self.stability_counter = 0
            self._set_state(
                VisionState.TRANSITION,
                "TRANSITION",
                "#fbbf24",
                "State uncertain...",
                self._format_telemetry()
            )
            return

        self.product_counter += 1
        if self.product_counter < PRODUCT_CONFIRM_FRAMES:
            self.stability_counter = 0
            self._set_state(
                VisionState.PRODUCT_MOVING,
                "PRODUCT?",
                "#38bdf8",
                f"Confirming product ({self.product_counter}/{PRODUCT_CONFIRM_FRAMES})",
                self._format_telemetry()
            )
            return

        if gray_stillness < GRAY_STILLNESS_THRESHOLD:
            self.stability_counter += 1
            self._set_state(
                VisionState.PRODUCT_STILL,
                f"PARKED ({self.stability_counter}/{STABILITY_FRAMES})",
                "#a78bfa",
                "Locking on...",
                self._format_telemetry()
            )

            if self.stability_counter >= STABILITY_FRAMES:
                self.trigger_inspection(raw_frame)
        else:
            self.stability_counter = 0
            self._set_state(
                VisionState.PRODUCT_MOVING,
                "MOTION",
                "#38bdf8",
                "Product moving...",
                self._format_telemetry()
            )

    def _format_telemetry(self):
        ge = "n/a" if self.last_gray_diff_empty is None else f"{self.last_gray_diff_empty:.2f}%"
        ee = "n/a" if self.last_edge_diff_empty is None else f"{self.last_edge_diff_empty:.2f}%"
        st = "n/a" if self.last_gray_stillness is None else f"{self.last_gray_stillness:.2f}%"

        return (
            f"Class: {self.last_classification}\n"
            f"Gray diff to empty: {ge}\n"
            f"Edge diff to empty: {ee}\n"
            f"Stillness: {st}\n"
            f"Product ctr: {self.product_counter}/{PRODUCT_CONFIRM_FRAMES}\n"
            f"Empty ctr: {self.empty_counter}/{EMPTY_CONFIRM_FRAMES}\n"
            f"Stable ctr: {self.stability_counter}/{STABILITY_FRAMES}"
        )

    def _draw_overlay_telemetry(self, frame):
        panel_w = 320
        panel_h = 140

        cv2.rectangle(frame, (5, 5), (5 + panel_w, 5 + panel_h), (20, 20, 20), -1)
        cv2.rectangle(frame, (5, 5), (5 + panel_w, 5 + panel_h), (120, 120, 120), 1)

        ge = "n/a" if self.last_gray_diff_empty is None else f"{self.last_gray_diff_empty:.2f}%"
        ee = "n/a" if self.last_edge_diff_empty is None else f"{self.last_edge_diff_empty:.2f}%"
        st = "n/a" if self.last_gray_stillness is None else f"{self.last_gray_stillness:.2f}%"

        lines = [
            f"State: {self.state.name}",
            f"Class: {self.last_classification}",
            f"GrayEmpty: {ge}",
            f"EdgeEmpty: {ee}",
            f"Stillness: {st}",
            f"P:{self.product_counter}/{PRODUCT_CONFIRM_FRAMES}  E:{self.empty_counter}/{EMPTY_CONFIRM_FRAMES}  S:{self.stability_counter}/{STABILITY_FRAMES}",
        ]

        y = 25
        for line in lines:
            cv2.putText(
                frame,
                line,
                (15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
            y += 22

    # =========================================================
    # Inspection
    # =========================================================

    def trigger_inspection(self, raw_frame):
        if self.is_analyzing:
            return

        inspection_context = {
            "state_before_result": self.state.name,
            "gray_diff_empty": self.last_gray_diff_empty,
            "edge_diff_empty": self.last_edge_diff_empty,
            "gray_stillness": self.last_gray_stillness,
        }

        self.is_analyzing = True
        self._set_state(
            VisionState.ANALYZING,
            "ANALYZING...",
            "#a78bfa",
            "Sending to model...",
            self._format_telemetry()
        )

        roi = self.get_normalized_roi()
        if roi is None:
            self.is_analyzing = False
            return

        x, y, w, h = roi

        h_img, w_img = raw_frame.shape[:2]
        scale_x = w_img / float(UI_WIDTH)
        scale_y = h_img / float(UI_HEIGHT)

        sx = int(x * scale_x)
        sy = int(y * scale_y)
        sw = int(w * scale_x)
        sh = int(h * scale_y)

        sx = max(0, min(w_img - 1, sx))
        sy = max(0, min(h_img - 1, sy))
        sw = max(1, min(sw, w_img - sx))
        sh = max(1, min(sh, h_img - sy))

        final_image = raw_frame[sy:sy + sh, sx:sx + sw].copy()
        threading.Thread(
            target=self.worker_logic,
            args=(final_image, inspection_context),
            daemon=True
        ).start()

    def encode_frame_for_model(self, frame):
        if frame is None or frame.size == 0:
            raise ValueError("Empty frame for model encoding")

        h, w = frame.shape[:2]
        max_dim = max(h, w)

        send_frame = frame
        if MODEL_INPUT_MAX_DIM > 0 and max_dim > MODEL_INPUT_MAX_DIM:
            scale = MODEL_INPUT_MAX_DIM / float(max_dim)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            send_frame = cv2.resize(send_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        ok, encoded = cv2.imencode(
            ".jpg",
            send_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), int(MODEL_JPEG_QUALITY)]
        )
        if not ok:
            raise RuntimeError("Could not JPEG-encode frame for model")

        return base64.b64encode(encoded.tobytes()).decode("utf-8")

    def worker_logic(self, frame, inspection_context):
        temp_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        temp_path = os.path.join(SAVE_DIR, f"temp_scan_{temp_stamp}.jpg")

        cv2.imwrite(temp_path, frame)

        verdict = "FAIL"
        reason_text = "Unknown Error"
        model_data = {}
        model_verdict_raw = ""
        model_reason_raw = ""

        try:
            if self.client is None:
                raise RuntimeError("Mistral client not initialized. Set MISTRAL_API_KEY.")

            base64_image = self.encode_frame_for_model(frame)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                "Inspect one projector remote. "
                                "Ignore the background. "
                                "Return FAIL if the product is incomplete or badly damaged, "
                                "if there is a crack, deformation, or structural damage, "
                                "or if uncertain. "
                                "Use rc=ok only with PASS. For FAIL use: "
                                "mp=missing pinhole/mark, rd=major rim damage/chip, "
                                "ri=incomplete rim, sd=structural damage, "
                                "co=color not acceptable, un=uncertain, od=other defect. "
                                "Return JSON only: "
                                "{\"v\":\"P|F\",\"rc\":\"ok|mp|rd|ri|sd|co|un|od\"}"
                            )
                        },
                        {
                            "type": "image_url",
                            "image_url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    ]
                }
            ]

            chat_response = self.client.chat.complete(
                model=MODEL_ID,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=MODEL_MAX_TOKENS
            )

            content = chat_response.choices[0].message.content
            model_data = json.loads(content)
            if not isinstance(model_data, dict):
                model_data = {}

            raw_v = str(model_data.get("v", "")).strip().upper()
            raw_rc = str(model_data.get("rc", "")).strip().lower()

            model_verdict_raw = raw_v
            model_reason_raw = raw_rc

            valid_fail_codes = set(MODEL_REASON_MAP) - {"ok"}

            if raw_v == "P" and raw_rc == "ok":
                verdict = "PASS"
                reason_text = "OK"
            elif raw_v == "F" and raw_rc in valid_fail_codes:
                verdict = "FAIL"
                _, reason_text = self._decode_model_reason(verdict, raw_rc)
            else:
                verdict = "FAIL"
                reason_text = "Invalid model output"

        except Exception:
            verdict = "FAIL"
            reason_text = "API Error; see local log"
            logging.exception("Inspection worker error")

        self.window.after(
            0,
            lambda: self.finalize_completed_inspection(
                temp_path=temp_path,
                verdict=verdict,
                reason_text=reason_text,
                model_data=model_data,
                inspection_context=inspection_context,
                model_verdict_raw=model_verdict_raw,
                model_reason_raw=model_reason_raw,
            )
        )

    def ui_finish_scan(self, row):
        verdict = row["verdict"]
        reason_text = row["reason_text"]
        inspection_id = row.get("inspection_id", "")
        image_saved = row.get("image_saved", "0") == "1"
        image_file = row.get("image_file", "")

        color = "#4ade80" if verdict == "PASS" else "#f87171"

        self._refresh_stats_ui()
        self.lbl_status.config(text=verdict, fg=color)

        reason_lines = [f"{inspection_id} | {reason_text}"]
        if verdict == "PASS" and not image_saved:
            reason_lines.append(f"PASS image not saved (sampling every {PASS_SAMPLE_INTERVAL})")
        elif image_saved and image_file:
            reason_lines.append(f"Image: {image_file}")
        elif row.get("save_error"):
            reason_lines.append(f"Save error: {row['save_error']}")

        self.lbl_reason.config(text="\n".join(reason_lines))
        self.lbl_timer.config(text="Inspection done")

        self.is_analyzing = False
        self.last_scan_time = time.time()
        self.reset_motion_state()
        self.state = VisionState.WAIT_RESET

    def _update_fail_list(self):
        self.fail_list.delete(0, tk.END)
        for reason_code, count in self.failure_reasons.most_common():
            self.fail_list.insert(tk.END, f"[{count}] {self._reason_code_to_display(reason_code)}")

    # =========================================================
    # Canvas
    # =========================================================

    def _display_on_canvas(self, cv_img):
        cv2image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.canvas.imgtk = imgtk

        if self.canvas_image_id is None:
            self.canvas_image_id = self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
        else:
            self.canvas.itemconfig(self.canvas_image_id, image=imgtk)

    # =========================================================
    # Shutdown
    # =========================================================

    def shutdown(self):
        try:
            if self.cap is not None and self.cap.isOpened():
                self.cap.release()
        except Exception:
            pass
        try:
            self.window.destroy()
        except Exception:
            pass


if __name__ == "__main__":
    if API_KEY.strip() == "":
        print("STOP: Set MISTRAL_API_KEY environment variable before running.")
    else:
        root = tk.Tk()
        app = CloudInspectorGrayscaleDebug(root)
        root.protocol("WM_DELETE_WINDOW", app.shutdown)
        root.mainloop()

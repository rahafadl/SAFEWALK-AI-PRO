"""
SafeWalk AI Pro — Smart Crosswalk Accessibility System
Stack: Python · Streamlit · OpenCV · Ultralytics YOLO · ByteTrack · Plotly · SQLite

Run:
    pip install -r requirements.txt
    streamlit run app.py

Expected model classes:
    0: pedestrian
    1: wheelchair

Important logic:
- Waiting Zone starts GREEN.
- Crosswalk Zone keeps GREEN while a person is still crossing.
- Wheelchair triggers accessibility extension.
"""

from __future__ import annotations

import os
import cv2
import time
import math
import sqlite3
import tempfile
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Set

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


# ============================================================
# CONFIG
# ============================================================
APP_TITLE = "SafeWalk AI Pro"
DB_PATH = "safewalk_events.db"
DEFAULT_MODEL_PATH = "best.pt"
TRACKER_CONFIG = "bytetrack.yaml"

CLASS_MAP = {
    0: "pedestrian",
    1: "wheelchair",
}

PEDESTRIAN_LABELS = {"pedestrian", "person", "people"}
WHEELCHAIR_LABELS = {"wheelchair", "wheelchair_user", "wheelchair user"}

BASE_RED_SECONDS = 8
BASE_GREEN_SECONDS = 10
WHEELCHAIR_GREEN_SECONDS = 18
YELLOW_SECONDS = 3
MAX_GREEN_SECONDS = 28

MIN_EVENT_COOLDOWN = 8
TRACK_STALE_SECONDS = 2.5

PROCESS_EVERY_N_FRAMES = 2
DEFAULT_CONF = 0.45


# ============================================================
# PAGE CONFIG + STYLE
# ============================================================
st.set_page_config(
    page_title="SafeWalk AI Pro | Smart Crosswalk",
    page_icon="♿",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;600;700;900&family=Rajdhani:wght@300;400;500;600;700&family=Cairo:wght@300;400;600;700&display=swap');
:root{
  --gold:#FFD700;--cyan:#00FFCA;--green:#00FF78;--red:#FF3C3C;--yellow:#FFD700;
  --bg:#020409;--card:rgba(10,14,26,.84);--border:rgba(255,215,0,.18);
  --text:#E8DEB3;--muted:#8A7F5C;
}
html, body, .stApp{
  background:var(--bg)!important;
  background-image:radial-gradient(ellipse 70% 40% at 50% -10%,rgba(255,215,0,.08),transparent 60%),
  repeating-linear-gradient(0deg,transparent,transparent 39px,rgba(255,215,0,.025) 40px),
  repeating-linear-gradient(90deg,transparent,transparent 39px,rgba(255,215,0,.025) 40px)!important;
  color:var(--text)!important;font-family:'Rajdhani','Cairo',sans-serif;
}
#MainMenu, footer, header, .stDeployButton{visibility:hidden;display:none;}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#05080F,#0A0E1A)!important;border-right:1px solid var(--border);}
[data-testid="metric-container"]{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:14px!important;padding:15px!important;box-shadow:0 0 22px rgba(255,215,0,.12)!important;}
[data-testid="stMetricValue"]{font-family:'Orbitron',monospace!important;color:var(--gold)!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;}
.stButton>button{background:linear-gradient(135deg,rgba(255,215,0,.14),rgba(255,215,0,.04))!important;border:1px solid rgba(255,215,0,.45)!important;color:var(--gold)!important;border-radius:10px!important;font-family:'Orbitron',monospace!important;letter-spacing:.08em!important;}
.stButton>button:hover{box-shadow:0 0 25px rgba(255,215,0,.35)!important;transform:translateY(-1px);}
.block-card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:18px;box-shadow:0 0 24px rgba(255,215,0,.10);}
.small-title{font-family:'Orbitron',monospace;color:var(--gold);font-size:.75rem;letter-spacing:.15em;text-transform:uppercase;margin-bottom:10px;}
.ar{font-family:'Cairo',sans-serif;direction:rtl;}
</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# DATA MODELS
# ============================================================
class SignalState(str, Enum):
    RED = "RED"
    GREEN = "GREEN"
    YELLOW = "YELLOW"


@dataclass
class Detection:
    track_id: int
    cls_id: int
    label: str
    conf: float
    box: Tuple[int, int, int, int]
    center: Tuple[int, int]
    in_waiting_zone: bool
    in_crosswalk: bool


@dataclass
class TrackMemory:
    track_id: int
    label: str
    first_seen: float
    last_seen: float
    frames_seen: int = 0
    entered_waiting_zone: bool = False
    entered_crosswalk: bool = False
    counted: bool = False
    first_waiting_time: Optional[float] = None
    last_center: Optional[Tuple[int, int]] = None
    confidence_values: List[float] = field(default_factory=list)


@dataclass
class EventItem:
    ts: str
    event_type: str
    message: str
    severity: str
    track_id: Optional[int] = None


# ============================================================
# DATABASE
# ============================================================
def init_db(path: str = DB_PATH):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            event_type TEXT NOT NULL,
            message TEXT NOT NULL,
            severity TEXT NOT NULL,
            track_id INTEGER,
            signal_state TEXT,
            pedestrian_count INTEGER,
            wheelchair_count INTEGER
        )
        """
    )
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            frame_idx INTEGER,
            track_id INTEGER,
            label TEXT,
            confidence REAL,
            cx INTEGER,
            cy INTEGER,
            in_waiting_zone INTEGER,
            in_crosswalk INTEGER
        )
        """
    )
    conn.commit()
    conn.close()


def save_event_to_db(event: EventItem, signal: str, ped: int, chair: int):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO events(ts,event_type,message,severity,track_id,signal_state,pedestrian_count,wheelchair_count)
        VALUES(?,?,?,?,?,?,?,?)
        """,
        (event.ts, event.event_type, event.message, event.severity, event.track_id, signal, ped, chair),
    )
    conn.commit()
    conn.close()


def save_detection_to_db(frame_idx: int, d: Detection):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO detections(ts,frame_idx,track_id,label,confidence,cx,cy,in_waiting_zone,in_crosswalk)
        VALUES(?,?,?,?,?,?,?,?,?)
        """,
        (
            datetime.now().isoformat(timespec="seconds"),
            frame_idx,
            d.track_id,
            d.label,
            d.conf,
            d.center[0],
            d.center[1],
            int(d.in_waiting_zone),
            int(d.in_crosswalk),
        ),
    )
    conn.commit()
    conn.close()


# ============================================================
# ROI HELPERS
# ============================================================
def default_rois(width: int, height: int, signal_state: Optional[SignalState] = None):

    waiting = np.array([
     [int(width*0.20), int(height*0.78)],
     [int(width*0.58), int(height*0.78)],
     [int(width*0.58), int(height*0.88)],
     [int(width*0.20), int(height*0.88)]
    ], dtype=np.int32)

    crosswalk = np.array([
     [int(width*0.16), int(height*0.42)],
     [int(width*0.60), int(height*0.42)],
     [int(width*0.60), int(height*0.77)],
     [int(width*0.16), int(height*0.77)]
    ], dtype=np.int32)

    return waiting, crosswalk

def point_in_polygon(point: Tuple[int, int], polygon: np.ndarray) -> bool:
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def draw_roi_overlay(
    frame: np.ndarray,
    waiting: np.ndarray,
    crosswalk: np.ndarray,
    state: Optional[SignalState] = None
) -> np.ndarray:

    overlay = frame.copy()

    # ========================================================
    # WAITING ZONE (YELLOW)
    # ========================================================
    cv2.fillPoly(overlay, [waiting], (0, 255, 255))
    frame = cv2.addWeighted(overlay, 0.35, frame, 0.65, 0)

    # ========================================================
    # CROSSWALK ZONE (GREEN)
    # ========================================================
    overlay2 = frame.copy()

    if state == SignalState.GREEN:
        cv2.fillPoly(overlay2, [crosswalk], (120, 255, 0))
        frame = cv2.addWeighted(overlay2, 0.35, frame, 0.65, 0)

        cv2.polylines(
            frame,
            [crosswalk],
            True,
            (120, 255, 0),
            4
        )

        cross_text = "CROSSWALK ACTIVE"
        cross_color = (120, 255, 0)

    else:
        cv2.fillPoly(overlay2, [crosswalk], (70, 120, 70))
        frame = cv2.addWeighted(overlay2, 0.20, frame, 0.80, 0)

        cv2.polylines(
            frame,
            [crosswalk],
            True,
            (70, 120, 70),
            2
        )

        cross_text = "CROSSWALK"
        cross_color = (70, 120, 70)

    # ========================================================
    # WAITING BORDER
    # ========================================================
    cv2.polylines(
        frame,
        [waiting],
        True,
        (0, 255, 255),
        4
    )

    # ========================================================
    # LABELS
    # ========================================================
    cv2.putText(
        frame,
        "WAITING ZONE",
        (waiting[0][0] + 10, waiting[0][1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 255, 255),
        2
    )

    cv2.putText(
        frame,
        cross_text,
        (crosswalk[0][0] + 10, crosswalk[0][1] - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        cross_color,
        2
    )

    return frame


# ============================================================
# YOLO + TRACKING
# ============================================================
@st.cache_resource(show_spinner=False)
def load_model(model_path: str):
    if not YOLO_AVAILABLE:
        return None
    if not os.path.exists(model_path):
        return None
    return YOLO(model_path)


def normalize_label(model: Any, cls_id: int) -> str:
    try:
        raw = str(model.names.get(cls_id, CLASS_MAP.get(cls_id, str(cls_id)))).lower().strip()
    except Exception:
        raw = CLASS_MAP.get(cls_id, str(cls_id))

    if raw in PEDESTRIAN_LABELS:
        return "pedestrian"
    if raw in WHEELCHAIR_LABELS:
        return "wheelchair"
    return CLASS_MAP.get(cls_id, raw)


def mock_track(frame: np.ndarray, conf: float, waiting: np.ndarray, crosswalk: np.ndarray, frame_idx: int) -> List[Detection]:
    h, w = frame.shape[:2]
    rng = np.random.default_rng(frame_idx // 10 + 4)
    detections: List[Detection] = []
    n = int(rng.integers(0, 4))

    for i in range(n):
        label = "wheelchair" if rng.random() < 0.25 else "pedestrian"
        cls_id = 1 if label == "wheelchair" else 0
        x1 = int(rng.integers(w * 0.05, w * 0.75))
        y1 = int(rng.integers(h * 0.40, h * 0.75))
        bw = int(rng.integers(45, 90))
        bh = int(rng.integers(75, 150)) if label == "pedestrian" else int(rng.integers(60, 110))
        x2, y2 = min(w - 1, x1 + bw), min(h - 1, y1 + bh)
        cx, cy = (x1 + x2) // 2, y2

        detections.append(Detection(
            track_id=i + 1,
            cls_id=cls_id,
            label=label,
            conf=float(rng.uniform(conf, 0.96)),
            box=(x1, y1, x2, y2),
            center=(cx, cy),
            in_waiting_zone=point_in_polygon((cx, cy), waiting),
            in_crosswalk=point_in_polygon((cx, cy), crosswalk),
        ))

    return detections


def run_tracking(frame: np.ndarray, model: Any, conf: float, waiting: np.ndarray, crosswalk: np.ndarray, frame_idx: int) -> List[Detection]:
    if model is None:
        return mock_track(frame, conf, waiting, crosswalk, frame_idx)

    results = model.track(
        frame,
        conf=conf,
        persist=True,
        tracker=TRACKER_CONFIG,
        verbose=False,
    )[0]

    detections: List[Detection] = []
    if results.boxes is None:
        return detections

    for box in results.boxes:
        if box.id is None:
            continue

        track_id = int(box.id[0])
        cls_id = int(box.cls[0])
        label = normalize_label(model, cls_id)

        if label not in {"pedestrian", "wheelchair"}:
            continue

        confidence = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

        # Bottom-center أفضل للـ ROI لأنه يمثل مكان القدم/الكرسي على الأرض
        cx = int((x1 + x2) / 2)
        cy = int(y2)
        center = (cx, cy)

        detections.append(Detection(
            track_id=track_id,
            cls_id=cls_id,
            label=label,
            conf=confidence,
            box=(x1, y1, x2, y2),
            center=center,
            in_waiting_zone=point_in_polygon(center, waiting),
            in_crosswalk=point_in_polygon(center, crosswalk),
        ))

    return detections


# ============================================================
# EVENT LOGGER
# ============================================================
class EventLogger:
    def __init__(self):
        self.last_event_time: Dict[str, float] = {}

    def should_log(self, key: str, cooldown: int = MIN_EVENT_COOLDOWN) -> bool:
        now = time.time()
        last = self.last_event_time.get(key, 0)
        if now - last >= cooldown:
            self.last_event_time[key] = now
            return True
        return False

    def log(self, event_type: str, message: str, severity: str = "info", track_id: Optional[int] = None):
        ev = EventItem(
            ts=datetime.now().strftime("%H:%M:%S"),
            event_type=event_type,
            message=message,
            severity=severity,
            track_id=track_id,
        )
        st.session_state.events.append(ev)
        save_event_to_db(
            ev,
            st.session_state.signal_state.value,
            st.session_state.current_pedestrians,
            st.session_state.current_wheelchairs,
        )

    def log_once_cooldown(self, key: str, event_type: str, message: str, severity: str = "info", track_id: Optional[int] = None):
        if self.should_log(key):
            self.log(event_type, message, severity, track_id)

    def log_state_change(self, old: SignalState, new: SignalState):
        if old != new:
            self.log("SIGNAL_CHANGE", f"Signal changed: {old.value} -> {new.value}", "success")


# ============================================================
# TRAFFIC SIGNAL STATE MACHINE
# ============================================================
class TrafficSignalController:
    def __init__(self):
        self.state = SignalState.RED
        self.state_started_at = time.time()
        self.duration = BASE_RED_SECONDS
        self.extension_active = False
        self.extension_count = 0
        self.max_green_seconds = MAX_GREEN_SECONDS

    def reset(self):
        self.state = SignalState.RED
        self.state_started_at = time.time()
        self.duration = BASE_RED_SECONDS
        self.extension_active = False
        self.extension_count = 0

    def elapsed(self) -> float:
        return time.time() - self.state_started_at

    def time_left(self) -> int:
        return max(0, int(math.ceil(self.duration - self.elapsed())))

    def transition(self, new_state: SignalState, duration: int, logger: Optional[EventLogger] = None):
        old = self.state
        self.state = new_state
        self.duration = duration
        self.state_started_at = time.time()

        if new_state != SignalState.GREEN:
            self.extension_active = False

        if logger:
            logger.log_state_change(old, new_state)

    def update(
        self,
        has_ped_waiting: bool,
        has_wheelchair_waiting: bool,
        has_person_crossing: bool,
        has_wheelchair_crossing: bool,
        logger: EventLogger,
    ):
        left = self.time_left()

        if self.state == SignalState.RED:
            if has_ped_waiting or has_wheelchair_waiting:
                duration = WHEELCHAIR_GREEN_SECONDS if has_wheelchair_waiting else BASE_GREEN_SECONDS
                self.extension_active = has_wheelchair_waiting
                self.transition(SignalState.GREEN, duration, logger)

                if has_wheelchair_waiting:
                    self.extension_count += 1
                    logger.log_once_cooldown(
                        "accessibility_green",
                        "ACCESSIBILITY_EXTENSION",
                        f"Wheelchair waiting: GREEN extended to {duration}s",
                        "alert",
                    )
                else:
                    logger.log_once_cooldown(
                        "normal_green",
                        "NORMAL_CROSSING",
                        f"Pedestrian waiting: GREEN for {duration}s",
                        "info",
                    )

        elif self.state == SignalState.GREEN:
            # لو كرسي متحرك ظهر أثناء الأخضر، مدد مرة واحدة
            if (has_wheelchair_waiting or has_wheelchair_crossing) and not self.extension_active:
                if self.elapsed() < self.max_green_seconds:
                    remaining = self.time_left()
                    desired_total = max(WHEELCHAIR_GREEN_SECONDS, remaining + 6)
                    self.duration = min(self.max_green_seconds, max(self.duration, desired_total))
                    self.extension_active = True
                    self.extension_count += 1
                    logger.log_once_cooldown(
                        "wheelchair_extension",
                        "ACCESSIBILITY_EXTENSION",
                        "Wheelchair detected: keeping GREEN longer",
                        "alert",
                    )

            # أهم منطق: لا تتحول أصفر إذا فيه شخص داخل منطقة العبور
            if left <= 0:
                if has_person_crossing and self.elapsed() < self.max_green_seconds:
                    self.duration = min(self.max_green_seconds, self.duration + 2)
                    logger.log_once_cooldown(
                        "crossing_hold_green",
                        "SAFETY_HOLD",
                        "Person still crossing: holding GREEN",
                        "warn",
                    )
                else:
                    self.transition(SignalState.YELLOW, YELLOW_SECONDS, logger)

        elif self.state == SignalState.YELLOW:
            if left <= 0:
                self.transition(SignalState.RED, BASE_RED_SECONDS, logger)


# ============================================================
# SESSION STATE
# ============================================================
def ensure_state():
    if "initialized" not in st.session_state:
        init_db()
        st.session_state.initialized = True
        st.session_state.running = False
        st.session_state.model = None
        st.session_state.model_loaded = False
        st.session_state.signal_controller = TrafficSignalController()
        st.session_state.event_logger = EventLogger()
        st.session_state.track_memory: Dict[int, TrackMemory] = {}
        st.session_state.seen_unique_ids: Set[int] = set()
        st.session_state.unique_pedestrians = 0
        st.session_state.unique_wheelchairs = 0
        st.session_state.current_pedestrians = 0
        st.session_state.current_wheelchairs = 0
        st.session_state.current_crossing = 0
        st.session_state.signal_state = SignalState.RED
        st.session_state.countdown = BASE_RED_SECONDS
        st.session_state.frame_idx = 0
        st.session_state.fps = 0.0
        st.session_state.events: List[EventItem] = []
        st.session_state.analytics = pd.DataFrame(columns=[
            "time", "frame", "active_pedestrians", "active_wheelchairs", "active_crossing",
            "unique_pedestrians", "unique_wheelchairs", "signal", "countdown",
            "avg_confidence", "accessibility_active", "avg_waiting_time"
        ])
        st.session_state.detection_points = pd.DataFrame(columns=[
            "time", "frame", "track_id", "label", "confidence", "cx", "cy",
            "in_waiting_zone", "in_crosswalk"
        ])
        st.session_state.last_frame = None


ensure_state()


# ============================================================
# ANALYTICS + TRACK MEMORY
# ============================================================
def update_track_memory(detections: List[Detection], logger: EventLogger):
    now = time.time()

    for d in detections:
        if d.track_id not in st.session_state.track_memory:
            st.session_state.track_memory[d.track_id] = TrackMemory(
                track_id=d.track_id,
                label=d.label,
                first_seen=now,
                last_seen=now,
                frames_seen=1,
                entered_waiting_zone=d.in_waiting_zone,
                entered_crosswalk=d.in_crosswalk,
                first_waiting_time=now if d.in_waiting_zone else None,
                last_center=d.center,
                confidence_values=[d.conf],
            )
        else:
            mem = st.session_state.track_memory[d.track_id]
            mem.last_seen = now
            mem.frames_seen += 1
            mem.entered_waiting_zone = mem.entered_waiting_zone or d.in_waiting_zone
            mem.entered_crosswalk = mem.entered_crosswalk or d.in_crosswalk
            mem.last_center = d.center
            mem.confidence_values.append(d.conf)

            if d.in_waiting_zone and mem.first_waiting_time is None:
                mem.first_waiting_time = now

        mem = st.session_state.track_memory[d.track_id]

        if d.in_waiting_zone and not mem.counted:
            mem.counted = True
            st.session_state.seen_unique_ids.add(d.track_id)

            if d.label == "wheelchair":
                st.session_state.unique_wheelchairs += 1
                logger.log_once_cooldown(
                    f"new_wheelchair_{d.track_id}",
                    "NEW_WHEELCHAIR",
                    f"New wheelchair user entered waiting zone | ID #{d.track_id}",
                    "alert",
                    d.track_id,
                )
            else:
                st.session_state.unique_pedestrians += 1
                logger.log_once_cooldown(
                    f"new_pedestrian_{d.track_id}",
                    "NEW_PEDESTRIAN",
                    f"New pedestrian entered waiting zone | ID #{d.track_id}",
                    "info",
                    d.track_id,
                )

    stale = [
        tid for tid, mem in st.session_state.track_memory.items()
        if now - mem.last_seen > TRACK_STALE_SECONDS
    ]
    for tid in stale:
        del st.session_state.track_memory[tid]


def compute_avg_waiting_time() -> float:
    now = time.time()
    waits = []
    for mem in st.session_state.track_memory.values():
        if mem.first_waiting_time is not None and mem.entered_waiting_zone:
            waits.append(now - mem.first_waiting_time)
    return float(np.mean(waits)) if waits else 0.0


def append_analytics(frame_idx: int, detections: List[Detection], signal: SignalState, countdown: int):
    active_waiting = [d for d in detections if d.in_waiting_zone]
    active_crossing = [d for d in detections if d.in_crosswalk]

    ped = sum(1 for d in active_waiting if d.label == "pedestrian")
    chair = sum(1 for d in active_waiting if d.label == "wheelchair")
    crossing_count = len(active_crossing)

    avg_conf = float(np.mean([d.conf for d in detections])) if detections else 0.0
    avg_wait = compute_avg_waiting_time()

    new_row = pd.DataFrame([{
        "time": datetime.now().strftime("%H:%M:%S"),
        "frame": frame_idx,
        "active_pedestrians": ped,
        "active_wheelchairs": chair,
        "active_crossing": crossing_count,
        "unique_pedestrians": st.session_state.unique_pedestrians,
        "unique_wheelchairs": st.session_state.unique_wheelchairs,
        "signal": signal.value,
        "countdown": countdown,
        "avg_confidence": avg_conf,
        "accessibility_active": chair > 0 or st.session_state.signal_controller.extension_active,
        "avg_waiting_time": avg_wait,
    }])

    st.session_state.analytics = pd.concat(
        [st.session_state.analytics, new_row],
        ignore_index=True
    ).tail(600)

    point_rows = []
    for d in detections:
        point_rows.append({
            "time": datetime.now().strftime("%H:%M:%S"),
            "frame": frame_idx,
            "track_id": d.track_id,
            "label": d.label,
            "confidence": d.conf,
            "cx": d.center[0],
            "cy": d.center[1],
            "in_waiting_zone": d.in_waiting_zone,
            "in_crosswalk": d.in_crosswalk,
        })
        if frame_idx % 10 == 0:
            save_detection_to_db(frame_idx, d)

    if point_rows:
        st.session_state.detection_points = pd.concat(
            [st.session_state.detection_points, pd.DataFrame(point_rows)],
            ignore_index=True
        ).tail(3000)

    st.session_state.current_pedestrians = ped
    st.session_state.current_wheelchairs = chair
    st.session_state.current_crossing = crossing_count


# ============================================================
# DRAWING
# ============================================================
def draw_detections(frame: np.ndarray, detections: List[Detection]) -> np.ndarray:
    out = frame.copy()

    for d in detections:
        x1, y1, x2, y2 = d.box

        if d.in_waiting_zone:
            color = (0, 215, 255)
            zone = "WAITING - COUNTED"
        elif d.in_crosswalk:
            color = (0, 255, 120)
            zone = "CROSSING - KEEP GREEN"
        else:
            color = (90, 90, 90)
            zone = "OUTSIDE ROI - IGNORED"

        if d.label == "wheelchair":
            color = (0, 255, 200)

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.circle(out, d.center, 5, color, -1)

        label = f"ID {d.track_id} | {d.label.upper()} | {d.conf:.0%} | {zone}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)

        cv2.rectangle(
            out,
            (x1, max(0, y1 - th - 10)),
            (min(out.shape[1] - 1, x1 + tw + 8), y1),
            color,
            -1,
        )
        cv2.putText(
            out,
            label,
            (x1 + 4, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return out


def draw_signal_hud(frame: np.ndarray, state: SignalState, countdown: int, ped: int, chair: int, crossing: int) -> np.ndarray:
    out = frame.copy()
    panel_w, panel_h = 430, 118
    x0, y0 = 15, 15

    cv2.rectangle(out, (x0, y0), (x0 + panel_w, y0 + panel_h), (5, 8, 15), -1)
    cv2.rectangle(out, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 215, 255), 1)

    color_map = {
        SignalState.RED: (40, 40, 255),
        SignalState.YELLOW: (0, 215, 255),
        SignalState.GREEN: (0, 255, 120),
    }
    color = color_map[state]

    cv2.circle(out, (x0 + 35, y0 + 34), 16, color, -1)
    cv2.putText(out, f"{state.value}  {countdown:02d}s", (x0 + 65, y0 + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(out, "Decision: WAITING starts | CROSSING holds green",
                (x0 + 18, y0 + 72), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 215, 255), 1)

    cv2.putText(out, f"Waiting: pedestrians={ped} wheelchair={chair} | Crossing={crossing}",
                (x0 + 18, y0 + 98), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (232, 222, 179), 1)

    return out


# ============================================================
# CHARTS
# ============================================================
def chart_counts(df: pd.DataFrame):
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(go.Scatter(x=df["time"], y=df["active_pedestrians"], name="Waiting Pedestrians", mode="lines", line=dict(width=2)))
        fig.add_trace(go.Scatter(x=df["time"], y=df["active_wheelchairs"], name="Waiting Wheelchairs", mode="lines", line=dict(width=2)))
        if "active_crossing" in df.columns:
            fig.add_trace(go.Scatter(x=df["time"], y=df["active_crossing"], name="Crossing Now", mode="lines", line=dict(width=2)))
    fig.update_layout(height=260, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#E8DEB3"), margin=dict(l=20,r=20,t=20,b=20))
    return fig


def chart_confidence(df: pd.DataFrame):
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(go.Scatter(x=df["time"], y=df["avg_confidence"], name="Avg Confidence", mode="lines"))
    fig.update_layout(height=230, yaxis=dict(range=[0,1]), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#E8DEB3"), margin=dict(l=20,r=20,t=20,b=20))
    return fig


def chart_signal(df: pd.DataFrame):
    if df.empty:
        return go.Figure()
    counts = df["signal"].value_counts().reset_index()
    counts.columns = ["signal", "count"]
    fig = go.Figure(go.Pie(labels=counts["signal"], values=counts["count"], hole=.55))
    fig.update_layout(height=230, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#E8DEB3"), margin=dict(l=20,r=20,t=20,b=20))
    return fig


def chart_heatmap(points: pd.DataFrame):
    fig = go.Figure()
    if not points.empty:
        fig = px.density_heatmap(points, x="cx", y="cy", nbinsx=24, nbinsy=16, title="Movement / Waiting Heatmap")
    fig.update_layout(height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#E8DEB3"), margin=dict(l=20,r=20,t=40,b=20))
    return fig


def render_events(events: List[EventItem]):
    if not events:
        st.info("No events yet.")
        return

    html = "<div class='block-card' style='max-height:280px;overflow:auto'>"
    for ev in list(reversed(events[-35:])):
        color = {"alert":"#FF3C3C", "success":"#00FF78", "warn":"#FFD700", "info":"#8A7F5C"}.get(ev.severity, "#8A7F5C")
        html += f"""
        <div style='border-bottom:1px solid rgba(255,215,0,.08);padding:7px 0;'>
          <span style='font-family:Orbitron;color:rgba(255,215,0,.45);font-size:.62rem'>{ev.ts}</span>
          <span style='color:{color};font-size:.86rem;margin-left:10px'>{ev.message}</span>
        </div>
        """
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


def traffic_light_html(state: SignalState, countdown: int):
    colors = {
        "RED": ("#FF3C3C", "#2A0707", "#2A2200", "#002A14"),
        "YELLOW": ("#2A0707", "#FFD700", "#002A14", "#FFD700"),
        "GREEN": ("#2A0707", "#2A2200", "#00FF78", "#00FF78"),
    }
    red, yellow, green, active = colors[state.value]

    st.markdown(f"""
    <div class='block-card' style='text-align:center'>
      <div class='small-title'>Traffic Signal</div>
      <div style='margin:auto;background:#080B14;border:1px solid rgba(255,215,0,.25);border-radius:42px;width:82px;padding:14px;display:flex;flex-direction:column;gap:12px'>
        <div style='width:50px;height:50px;border-radius:50%;background:{red};box-shadow:{'0 0 30px #FF3C3C' if state.value=='RED' else 'none'}'></div>
        <div style='width:50px;height:50px;border-radius:50%;background:{yellow};box-shadow:{'0 0 30px #FFD700' if state.value=='YELLOW' else 'none'}'></div>
        <div style='width:50px;height:50px;border-radius:50%;background:{green};box-shadow:{'0 0 30px #00FF78' if state.value=='GREEN' else 'none'}'></div>
      </div>
      <div style='font-family:Orbitron;color:{active};font-size:1.2rem;margin-top:12px'>{state.value}</div>
      <div style='font-family:Orbitron;color:#FFD700;font-size:2.3rem;margin-top:4px'>{countdown:02d}s</div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.markdown("<div style='text-align:center;font-family:Orbitron;color:#FFD700;font-size:1.05rem'>⬡ SAFEWALK AI PRO</div>", unsafe_allow_html=True)
    st.markdown("<div class='ar' style='text-align:center;color:#8A7F5C'>نظام عبور ذكي احترافي</div>", unsafe_allow_html=True)
    st.divider()

    input_mode = st.radio("Input Mode", ["Video Upload", "Webcam"], index=0)
    model_path = st.text_input("YOLO model path", DEFAULT_MODEL_PATH)
    conf = st.slider("Confidence", 0.10, 0.95, DEFAULT_CONF, 0.05)
    process_n = st.slider("Process every N frames", 1, 5, PROCESS_EVERY_N_FRAMES, 1)

    if st.button("LOAD MODEL", use_container_width=True):
        st.session_state.model = load_model(model_path)
        st.session_state.model_loaded = st.session_state.model is not None

        if st.session_state.model_loaded:
            st.success("Model loaded")
            try:
                st.write(st.session_state.model.names)
            except Exception:
                pass
        else:
            if not YOLO_AVAILABLE:
                st.warning("Ultralytics not installed. Demo mode is active.")
            elif not os.path.exists(model_path):
                st.warning("Model not found. Demo mode is active.")
            else:
                st.error("Failed to load model.")

    st.divider()

    if st.button("RESET SESSION", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()

    if not st.session_state.model_loaded:
        st.info("Demo mode will run if no best.pt is loaded.")


# ============================================================
# HEADER
# ============================================================
st.markdown(
    """
<div style='text-align:center;padding:18px 0 22px;border-bottom:1px solid rgba(255,215,0,.15);margin-bottom:20px'>
  <div style='font-family:Orbitron;font-size:2.5rem;font-weight:900;letter-spacing:.08em;background:linear-gradient(135deg,#FFD700,#FFF3A3,#B8960C);-webkit-background-clip:text;-webkit-text-fill-color:transparent'>SAFEWALK AI PRO</div>
  <div class='ar' style='color:#8A7F5C;font-size:1rem'>Smart Crosswalk Accessibility System —  عبور ذكي   </div>
</div>
""",
    unsafe_allow_html=True,
)

tab_live, tab_analytics, tab_arch = st.tabs(["Live System", "Analytics", "Architecture Notes"])


# ============================================================
# LIVE TAB
# ============================================================
with tab_live:
    left, right = st.columns([3, 1.35], gap="large")

    with left:
        st.markdown("<div class='small-title'>Live Detection + Tracking + ROI</div>", unsafe_allow_html=True)
        feed_ph = st.empty()
        control_cols = st.columns(3)

        uploaded_file = None
        if input_mode == "Video Upload":
            uploaded_file = st.file_uploader("Upload traffic / crosswalk video", type=["mp4", "avi", "mov", "mkv"])

        with control_cols[0]:
            start_btn = st.button("START ANALYSIS", use_container_width=True)
        with control_cols[1]:
            stop_btn = st.button("STOP", use_container_width=True)
        with control_cols[2]:
            st.download_button(
                "EXPORT CSV",
                data=st.session_state.analytics.to_csv(index=False).encode("utf-8"),
                file_name="safewalk_analytics.csv",
                mime="text/csv",
                use_container_width=True,
            )

    with right:
        signal_ph = st.empty()
        kpi_ph = st.empty()
        events_ph = st.empty()

    if stop_btn:
        st.session_state.running = False

    if start_btn:
        st.session_state.running = True

    with signal_ph.container():
        traffic_light_html(st.session_state.signal_controller.state, st.session_state.signal_controller.time_left())

    with kpi_ph.container():
        a, b = st.columns(2)
        a.metric("Waiting Ped.", st.session_state.current_pedestrians)
        b.metric("Waiting Chair", st.session_state.current_wheelchairs)
        c, d = st.columns(2)
        c.metric("Unique Ped.", st.session_state.unique_pedestrians)
        d.metric("Unique Chair", st.session_state.unique_wheelchairs)
        st.metric("Crossing Now", st.session_state.current_crossing)
        st.metric("FPS", f"{st.session_state.fps:.1f}")

    with events_ph.container():
        st.markdown("<div class='small-title'>Event Log</div>", unsafe_allow_html=True)
        render_events(st.session_state.events)

    if st.session_state.last_frame is not None:
        feed_ph.image(st.session_state.last_frame, use_container_width=True)
    else:
        feed_ph.markdown("""
        <div class='block-card' style='height:360px;display:flex;align-items:center;justify-content:center;flex-direction:column'>
          <div style='font-size:3rem'>🎥</div>
          <div style='font-family:Orbitron;color:#FFD700'>Waiting for input</div>
          <div class='ar' style='color:#8A7F5C'>ارفعي فيديو أو افتحي الكاميرا ثم اضغطي START ANALYSIS</div>
        </div>
        """, unsafe_allow_html=True)

    if st.session_state.running:
        model = st.session_state.model
        cap = None
        tmp_path = None

        if input_mode == "Video Upload":
            if uploaded_file is None:
                st.warning("ارفعي فيديو أولاً.")
                st.session_state.running = False
                st.stop()

            suffix = Path(uploaded_file.name).suffix or ".mp4"
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(uploaded_file.read())
            tmp.close()
            tmp_path = tmp.name
            cap = cv2.VideoCapture(tmp_path)
        else:
            cap = cv2.VideoCapture(0)

        if not cap or not cap.isOpened():
            st.error("Cannot open video/camera.")
            st.session_state.running = False
            st.stop()

        t0 = time.time()
        frame_idx = 0
        logger: EventLogger = st.session_state.event_logger
        controller: TrafficSignalController = st.session_state.signal_controller
        last_detections: List[Detection] = []

        while cap.isOpened() and st.session_state.running:
            ok, frame = cap.read()
            if not ok:
                break

            frame_idx += 1
            st.session_state.frame_idx = frame_idx

            max_w = 960
            h, w = frame.shape[:2]
            if w > max_w:
                scale = max_w / w
                frame = cv2.resize(frame, (max_w, int(h * scale)))

            h, w = frame.shape[:2]
            waiting_roi, crosswalk_roi = default_rois(w, h, controller.state)

            if frame_idx % process_n == 0:
                detections = run_tracking(frame, model, conf, waiting_roi, crosswalk_roi, frame_idx)
                last_detections = detections

                update_track_memory(detections, logger)

                # FINAL DECISION LOGIC:
                # Waiting Zone starts GREEN.
                # Crosswalk Zone keeps GREEN until crossing is clear.
                waiting_dets = [d for d in detections if d.in_waiting_zone]
                crossing_dets = [d for d in detections if d.in_crosswalk]

                has_ped_waiting = any(d.label == "pedestrian" for d in waiting_dets)
                has_wheelchair_waiting = any(d.label == "wheelchair" for d in waiting_dets)

                has_person_crossing = len(crossing_dets) > 0
                has_wheelchair_crossing = any(d.label == "wheelchair" for d in crossing_dets)

                controller.update(
                    has_ped_waiting,
                    has_wheelchair_waiting,
                    has_person_crossing,
                    has_wheelchair_crossing,
                    logger,
                )

                st.session_state.signal_state = controller.state
                st.session_state.countdown = controller.time_left()

                append_analytics(frame_idx, detections, controller.state, controller.time_left())

                draw = draw_roi_overlay(frame, waiting_roi, crosswalk_roi, controller.state)
                draw = draw_detections(draw, detections)
                draw = draw_signal_hud(
                    draw,
                    controller.state,
                    controller.time_left(),
                    st.session_state.current_pedestrians,
                    st.session_state.current_wheelchairs,
                    st.session_state.current_crossing,
                )

                rgb = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
                st.session_state.last_frame = rgb

                elapsed = time.time() - t0
                st.session_state.fps = frame_idx / elapsed if elapsed > 0 else 0.0

                feed_ph.image(
                    rgb,
                    use_container_width=True,
                    caption=f"Frame {frame_idx} | Tracking: ByteTrack | FPS {st.session_state.fps:.1f}",
                )

                with signal_ph.container():
                    traffic_light_html(controller.state, controller.time_left())

                with kpi_ph.container():
                    a, b = st.columns(2)
                    a.metric("Waiting Ped.", st.session_state.current_pedestrians)
                    b.metric("Waiting Chair", st.session_state.current_wheelchairs)
                    c, d = st.columns(2)
                    c.metric("Unique Ped.", st.session_state.unique_pedestrians)
                    d.metric("Unique Chair", st.session_state.unique_wheelchairs)
                    st.metric("Crossing Now", st.session_state.current_crossing)
                    st.metric("FPS", f"{st.session_state.fps:.1f}")
                    st.metric("Extensions", controller.extension_count)

                with events_ph.container():
                    st.markdown("<div class='small-title'>Event Log</div>", unsafe_allow_html=True)
                    render_events(st.session_state.events)

            if input_mode == "Video Upload":
                time.sleep(0.02)
            else:
                time.sleep(0.01)

        cap.release()

        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)

        st.session_state.running = False
        logger.log_once_cooldown("analysis_complete", "SYSTEM", "Analysis complete", "success")
        st.rerun()


# ============================================================
# ANALYTICS TAB
# ============================================================
with tab_analytics:
    df = st.session_state.analytics.copy()
    points = st.session_state.detection_points.copy()

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Unique Pedestrians", st.session_state.unique_pedestrians)
    k2.metric("Unique Wheelchairs", st.session_state.unique_wheelchairs)
    k3.metric("Accessibility Events", st.session_state.signal_controller.extension_count)
    k4.metric("Avg Waiting", f"{compute_avg_waiting_time():.1f}s")
    k5.metric("Avg FPS", f"{st.session_state.fps:.1f}")

    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("<div class='small-title'>Real-time Detection Timeline</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_counts(df), use_container_width=True)
    with c2:
        st.markdown("<div class='small-title'>Signal Distribution</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_signal(df), use_container_width=True)

    c3, c4 = st.columns([1, 1])
    with c3:
        st.markdown("<div class='small-title'>Detection Confidence Trend</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_confidence(df), use_container_width=True)
    with c4:
        st.markdown("<div class='small-title'>ROI Heatmap-ready Coordinates</div>", unsafe_allow_html=True)
        st.plotly_chart(chart_heatmap(points), use_container_width=True)

    st.markdown("<div class='small-title'>Raw Analytics</div>", unsafe_allow_html=True)
    st.dataframe(df.tail(150), use_container_width=True, height=260)

    st.markdown("<div class='small-title'>Detection Points for Heatmap / Audit</div>", unsafe_allow_html=True)
    st.dataframe(points.tail(250), use_container_width=True, height=260)


# ============================================================
# ARCHITECTURE TAB
# ============================================================
with tab_arch:
    st.markdown(
        """
<div class='block-card'>
<div class='small-title'>Production Architecture Idea</div>
<pre style='color:#E8DEB3;white-space:pre-wrap;font-size:.9rem'>
Camera / RTSP Stream
        ↓
Frame Ingestion Service
        ↓
YOLO Detection Service + ByteTrack
        ↓
ROI Filter: Waiting Zone + Crosswalk Zone
        ↓
Decision Engine
        ↓
Traffic Signal State Machine
        ↓
Event Queue / Database
        ↓
FastAPI Real-time API
        ↓
Streamlit / React Dashboard
</pre>
</div>

<br>

<div class='block-card ar'>
<b style='color:#FFD700'>ملاحظات مهمة:</b><br><br>
1. Waiting Zone تبدأ الإشارة الخضراء.<br>
2. Crosswalk Zone تمنع الإشارة من التحول للأصفر إذا الشخص لا يزال يعبر.<br>
3. الكرسي المتحرك يفعّل تمديد إضافي.<br>
4. يوجد حد أقصى للأخضر حتى لا يعلق النظام للأبد.<br>
5. العد يعتمد على Unique Track IDs وليس كل Frame.<br>
</div>
""",
        unsafe_allow_html=True,
    )

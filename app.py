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
BASE_GREEN_SECONDS = 14
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
[data-testid="metric-container"]{background:linear-gradient(145deg,rgba(16,20,34,.94),rgba(6,9,18,.88))!important;border:1px solid rgba(255,215,0,.24)!important;border-radius:18px!important;padding:18px!important;box-shadow:0 0 28px rgba(255,215,0,.13), inset 0 0 22px rgba(255,215,0,.035)!important;min-height:104px!important;}
[data-testid="stMetricValue"]{font-family:'Rajdhani','Orbitron',monospace!important;font-weight:700!important;color:var(--gold)!important;font-size:2.15rem!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:.86rem!important;letter-spacing:.03em!important;}
.stButton>button{background:linear-gradient(135deg,rgba(255,215,0,.14),rgba(255,215,0,.04))!important;border:1px solid rgba(255,215,0,.45)!important;color:var(--gold)!important;border-radius:10px!important;font-family:'Orbitron',monospace!important;letter-spacing:.08em!important;}
.stButton>button:hover{box-shadow:0 0 25px rgba(255,215,0,.35)!important;transform:translateY(-1px);}
.block-card{background:var(--card);border:1px solid var(--border);border-radius:16px;padding:18px;box-shadow:0 0 24px rgba(255,215,0,.10);}
.small-title{font-family:'Orbitron',monospace;color:var(--gold);font-size:.75rem;letter-spacing:.15em;text-transform:uppercase;margin-bottom:10px;}
.ar{font-family:'Cairo',sans-serif;direction:rtl;}
.kpi-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin-top:16px;}
.kpi-card{
  background:linear-gradient(145deg,rgba(18,22,36,.74),rgba(4,7,15,.72));
  border:1px solid rgba(255,215,0,.28);
  border-radius:18px;
  padding:15px 16px;
  min-height:65px;
  box-shadow:0 0 24px rgba(255,215,0,.12), inset 0 0 24px rgba(255,215,0,.035);
  backdrop-filter:blur(14px);
}
.kpi-card.wide{grid-column:1 / -1;}
.kpi-label{
  font-family:'Rajdhani','Cairo',sans-serif;
  color:rgba(232,222,179,.62);
  font-size:.76rem;
  letter-spacing:.08em;
  text-transform:uppercase;
  margin-bottom:8px;
}
.kpi-value{
  font-family:'Orbitron','Rajdhani',monospace;
  color:var(--gold);
  font-size:1.55rem;
  line-height:1;
  font-weight:800;
  text-shadow:0 0 18px rgba(255,215,0,.32);
}

.status-row{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid rgba(255,215,0,.08);padding:6px 0;color:rgba(232,222,179,.78);font-size:.86rem;}
.status-row:last-child{border-bottom:0;}
.status-dot{color:var(--green);font-family:Orbitron;text-shadow:0 0 12px rgba(0,255,120,.45);}
[data-testid="stSidebar"] .stExpander{border:1px solid rgba(255,215,0,.18)!important;border-radius:14px!important;background:rgba(255,215,0,.03)!important;}
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
     [int(width*0.20), int(height*0.77)],
     [int(width*0.64), int(height*0.77)],
     [int(width*0.64), int(height*0.88)],
     [int(width*0.20), int(height*0.88)]
   ], dtype=np.int32)

    crosswalk = np.array([
     [int(width*0.20), int(height*0.42)],
     [int(width*0.64), int(height*0.42)],
     [int(width*0.64), int(height*0.77)],
     [int(width*0.20), int(height*0.77)]
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

        cross_text = "CROSSWALK"
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
        "WAITING",
        (waiting[0][0] + 8, waiting[0][1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    cv2.putText(
        frame,
        cross_text,
        (crosswalk[0][0] + 8, crosswalk[0][1] - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        cross_color,
        1,
        cv2.LINE_AA,
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
    """Clean commercial-style detection overlay."""
    out = frame.copy()

    for d in detections:
        x1, y1, x2, y2 = d.box

        if d.label == "wheelchair":
            color = (0, 255, 200)
            label_name = "WHEELCHAIR"
        else:
            color = (0, 255, 120) if d.in_crosswalk else (0, 215, 255)
            label_name = "PEDESTRIAN"

        if d.in_waiting_zone:
            zone = "WAITING"
        elif d.in_crosswalk:
            zone = "CROSSING"
        else:
            zone = "TRACKED"

        # thinner, cleaner box
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1)
        cv2.circle(out, d.center, 4, color, -1)

        # short presentation label
        label = f"ID {d.track_id} | {label_name} | {zone}"
        font_scale = 0.36
        thickness = 1
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

        y_label = max(0, y1 - th - 8)
        cv2.rectangle(
            out,
            (x1, y_label),
            (min(out.shape[1] - 1, x1 + tw + 8), y1),
            color,
            -1,
        )
        cv2.putText(
            out,
            label,
            (x1 + 4, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            thickness,
            cv2.LINE_AA,
        )

    return out

def draw_signal_hud(frame: np.ndarray, state: SignalState, countdown: int, ped: int, chair: int, crossing: int) -> np.ndarray:
    """Compact signal HUD for presentation/demo view."""
    out = frame.copy()
    panel_w, panel_h = 210, 68
    x0, y0 = 14, 14

    panel = out.copy()
    cv2.rectangle(panel, (x0, y0), (x0 + panel_w, y0 + panel_h), (5, 8, 15), -1)
    out = cv2.addWeighted(panel, 0.58, out, 0.42, 0)
    cv2.rectangle(out, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 215, 255), 1)

    color_map = {
        SignalState.RED: (40, 40, 255),
        SignalState.YELLOW: (0, 215, 255),
        SignalState.GREEN: (0, 255, 120),
    }
    color = color_map[state]

    cv2.circle(out, (x0 + 26, y0 + 27), 11, color, -1)
    cv2.putText(
        out,
        f"{state.value}  {countdown:02d}s",
        (x0 + 48, y0 + 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        color,
        1,
        cv2.LINE_AA,
    )

   

    return out

# ============================================================
# CHARTS
# ============================================================
GOLD = "#FFD700"
GREEN = "#00FF78"
CYAN = "#00FFCA"
RED = "#FF3C3C"
DARK_RED = "#8B1E1E"
MUTED = "#8A7F5C"
TEXT = "#E8DEB3"
CARD_BG = "rgba(5,8,15,0)"
GRID_GOLD = "rgba(255,215,0,.10)"


def _premium_layout(fig: go.Figure, height: int = 330, showlegend: bool = True):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Rajdhani, Cairo, sans-serif"),
        margin=dict(l=22, r=22, t=24, b=36),
        legend=dict(
            orientation="h",
            y=-0.18,
            x=0,
            font=dict(color="rgba(232,222,179,.72)", size=11),
        ) if showlegend else dict(font=dict(color=TEXT)),
        hoverlabel=dict(bgcolor="#05080F", font_color=TEXT, bordercolor=GOLD),
    )
    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(255,215,0,.055)",
        zeroline=False,
        linecolor="rgba(255,215,0,.18)",
        tickfont=dict(color="rgba(232,222,179,.56)", size=10),
    )
    fig.update_yaxes(
        showgrid=True,
        gridcolor=GRID_GOLD,
        zeroline=False,
        linecolor="rgba(255,215,0,.18)",
        tickfont=dict(color="rgba(232,222,179,.56)", size=10),
    )
    return fig


def chart_counts(df: pd.DataFrame):
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(go.Scatter(
            x=df["time"],
            y=df["active_pedestrians"],
            name="Waiting Pedestrians",
            mode="lines",
            line=dict(width=3, color=GREEN),
            fill="tozeroy",
            fillcolor="rgba(0,255,120,.08)",
        ))
        fig.add_trace(go.Scatter(
            x=df["time"],
            y=df["active_wheelchairs"],
            name="Waiting Wheelchairs",
            mode="lines",
            line=dict(width=3, color=GOLD),
            fill="tozeroy",
            fillcolor="rgba(255,215,0,.09)",
        ))
        if "active_crossing" in df.columns:
            fig.add_trace(go.Scatter(
                x=df["time"],
                y=df["active_crossing"],
                name="Crossing Now",
                mode="lines",
                line=dict(width=3, color=CYAN),
                fill="tozeroy",
                fillcolor="rgba(0,255,202,.07)",
            ))
    fig = _premium_layout(fig, height=345, showlegend=True)
    fig.update_yaxes(rangemode="tozero")
    return fig


def chart_confidence(df: pd.DataFrame):
    fig = go.Figure()
    if not df.empty:
        fig.add_trace(go.Scatter(
            x=df["time"],
            y=df["avg_confidence"],
            name="Avg Confidence",
            mode="lines",
            line=dict(width=3, color=GOLD),
            fill="tozeroy",
            fillcolor="rgba(255,215,0,.10)",
        ))
    fig = _premium_layout(fig, height=250, showlegend=False)
    fig.update_yaxes(range=[0, 1])
    return fig


def chart_signal(df: pd.DataFrame):
    if df.empty:
        fig = go.Figure()
        return _premium_layout(fig, height=345, showlegend=False)

    counts = df["signal"].value_counts().reset_index()
    counts.columns = ["signal", "count"]

    color_map = {
        "GREEN": GREEN,
        "YELLOW": GOLD,
        "RED": RED,
    }

    fig = go.Figure(go.Pie(
        labels=counts["signal"],
        values=counts["count"],
        hole=.66,
        marker=dict(
            colors=[color_map.get(str(s), MUTED) for s in counts["signal"]],
            line=dict(color="rgba(255,215,0,.30)", width=2),
        ),
        textfont=dict(color=TEXT, size=12),
        hovertemplate="%{label}<br>%{percent}<extra></extra>",
    ))

    fig.update_layout(
        height=345,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Rajdhani, Cairo, sans-serif"),
        margin=dict(l=10, r=10, t=20, b=20),
        legend=dict(font=dict(color="rgba(232,222,179,.72)")),
        annotations=[dict(
            text="SIGNAL<br>STATES",
            x=.5,
            y=.5,
            font=dict(color=GOLD, size=13, family="Orbitron"),
            showarrow=False,
        )],
    )
    return fig


def chart_accessibility_activity(df: pd.DataFrame):
    fig = go.Figure()
    if not df.empty:
        access_df = (
            df.groupby("time")["accessibility_active"]
            .sum()
            .reset_index()
        )
        fig.add_trace(go.Bar(
            x=access_df["time"],
            y=access_df["accessibility_active"],
            name="Accessibility Active",
            marker=dict(
                color=GOLD,
                line=dict(color="rgba(255,255,255,.22)", width=1),
            ),
            hovertemplate="Accessibility Active: %{y}<extra></extra>",
        ))
    fig = _premium_layout(fig, height=315, showlegend=False)
    fig.update_yaxes(rangemode="tozero")
    return fig


def chart_ratio(pedestrians: int, wheelchairs: int):
    fig = go.Figure(go.Pie(
        labels=["Pedestrians", "Wheelchairs"],
        values=[pedestrians, wheelchairs],
        hole=.70,
        marker=dict(
            colors=[GREEN, GOLD],
            line=dict(color="rgba(255,215,0,.30)", width=2),
        ),
        textfont=dict(color=TEXT, size=12),
        hovertemplate="%{label}<br>%{percent}<extra></extra>",
    ))
    fig.update_layout(
        height=315,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT, family="Rajdhani, Cairo, sans-serif"),
        margin=dict(l=10, r=10, t=18, b=18),
        legend=dict(font=dict(color="rgba(232,222,179,.72)")),
        annotations=[dict(
            text="USER<br>RATIO",
            x=.5,
            y=.5,
            font=dict(color=GOLD, size=13, family="Orbitron"),
            showarrow=False,
        )],
    )
    return fig


def chart_scenarios():
    """Portfolio scenario comparison based on the four demo clips."""
    scenarios = [
        "Pedestrian Only",
        "Wheelchair Only",
        "Pedestrian + Wheelchair",
        "2 Pedestrians + Wheelchair",
    ]
    pedestrians = [1, 0, 1, 2]
    wheelchairs = [0, 1, 1, 1]

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=scenarios,
        y=pedestrians,
        name="Pedestrians",
        marker=dict(
            color=GREEN,
            line=dict(color="rgba(255,255,255,.18)", width=1),
        ),
        hovertemplate="Pedestrians: %{y}<extra></extra>",
    ))

    fig.add_trace(go.Bar(
        x=scenarios,
        y=wheelchairs,
        name="Wheelchairs",
        marker=dict(
            color=GOLD,
            line=dict(color="rgba(255,255,255,.18)", width=1),
        ),
        hovertemplate="Wheelchairs: %{y}<extra></extra>",
    ))

    fig.update_layout(
        barmode="stack",
        title=dict(
            text="Scenario Comparison",
            font=dict(color=GOLD, family="Orbitron, Rajdhani, sans-serif", size=18),
            x=0.02,
        ),
    )
    fig = _premium_layout(fig, height=360, showlegend=True)
    fig.update_yaxes(rangemode="tozero", dtick=1, title_text="Detected Users")
    fig.update_xaxes(tickangle=0)
    return fig


def chart_accessibility_impact():
    """Shows how wheelchair scenarios receive longer green time."""
    scenarios = [
        "Pedestrian Only",
        "Wheelchair Only",
        "Pedestrian + Wheelchair",
        "2 Pedestrians + Wheelchair",
    ]
    green_seconds = [BASE_GREEN_SECONDS, WHEELCHAIR_GREEN_SECONDS, WHEELCHAIR_GREEN_SECONDS, WHEELCHAIR_GREEN_SECONDS]

    fig = go.Figure(go.Bar(
        x=scenarios,
        y=green_seconds,
        name="Green Time",
        marker=dict(
            color=[GREEN, GOLD, GOLD, GOLD],
            line=dict(color="rgba(255,255,255,.20)", width=1),
        ),
        text=[f"{v}s" for v in green_seconds],
        textposition="outside",
        hovertemplate="Green Time: %{y}s<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="Accessibility Impact on Signal Timing",
            font=dict(color=GOLD, family="Orbitron, Rajdhani, sans-serif", size=18),
            x=0.02,
        ),
    )
    fig = _premium_layout(fig, height=360, showlegend=False)
    fig.update_yaxes(rangemode="tozero", title_text="Green Seconds")
    fig.update_xaxes(tickangle=0)
    return fig


def chart_heatmap(points: pd.DataFrame):
    fig = go.Figure()
    if not points.empty:
        fig = px.density_heatmap(
            points,
            x="cx",
            y="cy",
            nbinsx=24,
            nbinsy=16,
            title="Movement / Waiting Heatmap",
            color_continuous_scale=[[0, "#05080F"], [0.45, "#8A7F5C"], [1, "#FFD700"]],
        )
    fig.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT),
        margin=dict(l=20,r=20,t=40,b=20),
    )
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
    <div class='block-card' style='text-align:center;padding:28px 18px;box-shadow:0 0 44px rgba(255,215,0,.18), inset 0 0 34px rgba(255,215,0,.04)'>
      <div class='small-title' style='font-size:.86rem;margin-bottom:18px'>Traffic Signal</div>
      <div style='margin:auto;background:#080B14;border:1px solid rgba(255,215,0,.34);border-radius:62px;width:105px;padding:20px;display:flex;flex-direction:column;gap:16px;box-shadow:0 0 34px rgba(255,215,0,.12)'>
        <div style='width:65px;height:65px;border-radius:50%;background:{red};box-shadow:{'0 0 50px #FF3C3C, 0 0 76px rgba(255,60,60,.35)' if state.value=='RED' else 'inset 0 0 20px rgba(0,0,0,.62)'}'></div>
        <div style='width:65px;height:65px;border-radius:50%;background:{yellow};box-shadow:{'0 0 50px #FFD700, 0 0 76px rgba(255,215,0,.35)' if state.value=='YELLOW' else 'inset 0 0 20px rgba(0,0,0,.62)'}'></div>
        <div style='width:65px;height:65px;border-radius:50%;background:{green};box-shadow:{'0 0 50px #00FF78, 0 0 76px rgba(0,255,120,.35)' if state.value=='GREEN' else 'inset 0 0 20px rgba(0,0,0,.62)'}'></div>
      </div>
      <div style='font-family:Orbitron;color:{active};font-size:1.55rem;margin-top:20px;letter-spacing:.08em'>{state.value}</div>
      <div style='font-family:Orbitron;color:#FFD700;font-size:2.3rem;margin-top:4px;text-shadow:0 0 26px rgba(255,215,0,.38)'>{countdown:02d}s</div>
    </div>
    """, unsafe_allow_html=True)



def accessibility_alert_html():
    st.markdown("""
    <div style="
        margin-top:12px;
        margin-bottom:12px;
        background:linear-gradient(135deg,rgba(0,255,120,.14),rgba(0,20,10,.74));
        border:1px solid rgba(0,255,120,.44);
        border-radius:16px;
        padding:15px 20px;
        display:flex;
        align-items:center;
        gap:16px;
        box-shadow:
            0 0 25px rgba(0,255,120,.25),
            0 0 45px rgba(0,255,120,.15),
            inset 0 0 18px rgba(0,255,120,.08);
    ">
        <div style="
            font-size:42px;
            color:#00FF78;
            line-height:1;
            min-width:48px;
            text-align:center;
            text-shadow:0 0 18px rgba(0,255,120,.55);
        ">♿</div>
        <div>
            <div style="
                font-family:Orbitron;
                color:#00FF78;
                font-size:.90rem;
                font-weight:900;
                letter-spacing:.06em;
                text-transform:uppercase;
            ">
                ACCESSIBILITY PRIORITY ACTIVE
            </div>
            <div style="color:#D9E5DA;font-size:.86rem;margin-top:5px;">
                Green signal extended for wheelchair crossing
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

def render_kpi_cards():
    avg_wait = compute_avg_waiting_time()
    html = f"""
    <div class='kpi-grid'>
      <div class='kpi-card'>
        <div class='kpi-label'>Waiting Pedestrians</div>
        <div class='kpi-value'>{st.session_state.current_pedestrians}</div>
      </div>
      <div class='kpi-card'>
        <div class='kpi-label'>Waiting Wheelchairs</div>
        <div class='kpi-value'>{st.session_state.current_wheelchairs}</div>
      </div>
      <div class='kpi-card'>
        <div class='kpi-label'>Unique Pedestrians</div>
        <div class='kpi-value'>{st.session_state.unique_pedestrians}</div>
      </div>
      <div class='kpi-card'>
        <div class='kpi-label'>Unique Wheelchairs</div>
        <div class='kpi-value'>{st.session_state.unique_wheelchairs}</div>
      </div>
      <div class='kpi-card'>
        <div class='kpi-label'>Crossing Now</div>
        <div class='kpi-value'>{st.session_state.current_crossing}</div>
      </div>
      <div class='kpi-card'>
        <div class='kpi-label'>Accessibility Extensions</div>
        <div class='kpi-value'>{st.session_state.signal_controller.extension_count}</div>
      </div>
      <div class='kpi-card wide'>
        <div class='kpi-label'>Average Waiting Time</div>
        <div class='kpi-value'>{avg_wait:.1f}s</div>
      </div>
    </div>
   
    """
    st.markdown(html, unsafe_allow_html=True)


# ============================================================
# AUTO MODEL SETTINGS + SIDEBAR
# ============================================================
# Developer controls are kept as internal constants for a cleaner demo UI.
model_path = DEFAULT_MODEL_PATH
conf = DEFAULT_CONF
process_n = PROCESS_EVERY_N_FRAMES

# Load YOLO automatically once when the app starts. If best.pt is missing,
# model remains None and the existing demo/mock tracking mode can still run.
if not st.session_state.model_loaded:
    st.session_state.model = load_model(model_path)
    st.session_state.model_loaded = st.session_state.model is not None

with st.sidebar:
    st.markdown("<div style='text-align:center;font-family:Orbitron;color:#FFD700;font-size:1.05rem'>⬡ SAFEWALK AI PRO</div>", unsafe_allow_html=True)
    st.markdown("<div class='ar' style='text-align:center;color:#8A7F5C'>نظام عبور ذكي </div>", unsafe_allow_html=True)
    st.divider()

    with st.expander("⚙ System Controls", expanded=False):
        input_mode = st.radio("Input Mode", ["Video Upload", "Webcam"], index=0)

        st.divider()

        if st.button("RESET SESSION", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


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

tab_live, tab_analytics = st.tabs(["Live System", "Analytics"])


# ============================================================
# LIVE TAB
# ============================================================
with tab_live:
    left, right = st.columns([3, 1.35], gap="large")

    with left:
        st.markdown("<div class='small-title'>Live Detection + Tracking + ROI</div>", unsafe_allow_html=True)
        feed_ph = st.empty()
        control_cols = st.columns(2)

        with control_cols[0]:
            start_btn = st.button("START ANALYSIS", use_container_width=True)

        with control_cols[1]:
            stop_btn = st.button("STOP", use_container_width=True)

        # Alert appears directly under START / STOP and above the upload widget
        alert_ph = st.empty()

        uploaded_file = st.session_state.get("uploaded_file_ref", None)

        if input_mode == "Video Upload" and not st.session_state.running:
            uploaded_file = st.file_uploader(
                "Upload traffic / crosswalk video",
                type=["mp4", "avi", "mov", "mkv"]
            )
            if uploaded_file is not None:
                st.session_state.uploaded_file_ref = uploaded_file

    with right:
        signal_ph = st.empty()
        kpi_ph = st.empty()

    if stop_btn:
        st.session_state.running = False

    if start_btn:
        st.session_state.running = True

    with signal_ph.container():
        traffic_light_html(
            st.session_state.signal_controller.state,
            st.session_state.signal_controller.time_left()
        )

    with alert_ph.container():
        if (
            st.session_state.current_wheelchairs > 0
            or st.session_state.signal_controller.extension_active
        ):
            accessibility_alert_html()

    with kpi_ph.container():
        render_kpi_cards()

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
                )

                with signal_ph.container():
                    traffic_light_html(controller.state, controller.time_left())

                with alert_ph.container():
                    if (
                        st.session_state.current_wheelchairs > 0
                        or controller.extension_active
                    ):
                        accessibility_alert_html()
                    else:
                        st.empty()

                with kpi_ph.container():
                    render_kpi_cards()

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
# SIMPLE EXECUTIVE ANALYTICS TAB
# ============================================================
with tab_analytics:

    df = st.session_state.analytics.copy()

    total_crossings = (
        st.session_state.unique_pedestrians
        + st.session_state.unique_wheelchairs
    )

    extensions = st.session_state.signal_controller.extension_count

    st.markdown("""
    <style>
    .analytics-hero{
        text-align:center;
        margin:4px 0 30px;
        padding:14px 16px 16px;
        border:1px solid rgba(255,215,0,.18);
        border-radius:22px;
        background:
          radial-gradient(circle at 50% 0%, rgba(255,215,0,.16), transparent 45%),
          linear-gradient(135deg,rgba(18,22,36,.48),rgba(4,7,15,.38));
        box-shadow:0 0 40px rgba(255,215,0,.13), inset 0 0 36px rgba(255,215,0,.035);
    }
    .analytics-title{
        font-family:Orbitron,monospace;
        font-size:2.55rem;
        font-weight:900;
        letter-spacing:.09em;
        color:#FFD700;
        text-shadow:0 0 35px rgba(255,215,0,.42);
    }
    .analytics-subtitle{
        color:#8A7F5C;
        font-size:.95rem;
        margin-top:7px;
        letter-spacing:.04em;
    }
    .analytics-kpi-grid{
        display:grid;
        grid-template-columns:repeat(3,minmax(0,1fr));
        gap:18px;
        margin:12px 0 28px;
    }
    .analytics-kpi{
        min-height:125px;
        padding:20px 22px;
        border-radius:20px;
        border:1px solid rgba(255,215,0,.30);
        background:linear-gradient(145deg,rgba(18,22,36,.78),rgba(4,7,15,.76));
        box-shadow:0 0 30px rgba(255,215,0,.15), inset 0 0 25px rgba(255,215,0,.04);
        position:relative;
        overflow:hidden;
    }
    .analytics-kpi:before{
        content:"";
        position:absolute;
        inset:-40% -20% auto auto;
        width:120px;
        height:120px;
        background:radial-gradient(circle,rgba(255,215,0,.23),transparent 65%);
    }
    .analytics-kpi-label{
        font-family:Rajdhani,Cairo,sans-serif;
        color:rgba(232,222,179,.62);
        font-size:.82rem;
        letter-spacing:.11em;
        text-transform:uppercase;
    }
    .analytics-kpi-value{
        font-family:Orbitron,monospace;
        color:#FFD700;
        font-size:2.65rem;
        font-weight:900;
        margin-top:12px;
        line-height:1;
        text-shadow:0 0 25px rgba(255,215,0,.38);
    }
    .analytics-kpi-note{
        color:rgba(232,222,179,.48);
        margin-top:10px;
        font-size:.82rem;
    }

    .analytics-section-title{
        font-family:Orbitron,monospace;
        color:#FFD700;
        font-size:.92rem;
        letter-spacing:.13em;
        text-transform:uppercase;
        margin-bottom:6px;
    }
    .analytics-section-note{
        color:rgba(232,222,179,.54);
        font-size:.86rem;
        margin-bottom:12px;
    }
    .analytics-card{
        border:1px solid rgba(255,215,0,.22);
        border-radius:20px;
        padding:18px 18px 12px;
        background:linear-gradient(145deg,rgba(12,16,28,.72),rgba(3,6,12,.70));
        box-shadow:0 0 28px rgba(255,215,0,.10), inset 0 0 24px rgba(255,215,0,.03);
        margin-bottom:20px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='analytics-hero'>
        <div class='analytics-title'>ACCESSIBILITY ANALYTICS CENTER</div>
        <div class='analytics-subtitle'>Accessibility-Aware Smart Crosswalk Monitoring</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class='analytics-kpi-grid'>
      <div class='analytics-kpi'>
        <div class='analytics-kpi-label'>Total Crossings</div>
        <div class='analytics-kpi-value'>{total_crossings}</div>
        <div class='analytics-kpi-note'>Unique users detected in waiting zone</div>
      </div>
      <div class='analytics-kpi'>
        <div class='analytics-kpi-label'>Wheelchair Users</div>
        <div class='analytics-kpi-value'>{st.session_state.unique_wheelchairs}</div>
        <div class='analytics-kpi-note'>Accessibility users detected</div>
      </div>
      <div class='analytics-kpi'>
        <div class='analytics-kpi-label'>Accessibility Extensions</div>
        <div class='analytics-kpi-value'>{extensions}</div>
        <div class='analytics-kpi-note'>Green-time priority activations</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


    

    if not df.empty:
        col_a, col_b = st.columns([1.35, 1], gap="large")
        with col_a:
            st.markdown("<div class='analytics-card'><div class='analytics-section-title'>Real-Time Detection Timeline</div></div>", unsafe_allow_html=True)
            st.plotly_chart(chart_counts(df), use_container_width=True, key="analytics_detection_timeline")
        with col_b:
            st.markdown("<div class='analytics-card'><div class='analytics-section-title'>Signal Distribution</div></div>", unsafe_allow_html=True)
            st.plotly_chart(chart_signal(df), use_container_width=True, key="analytics_signal_distribution")

    else:
        st.info("Run an analysis first to populate the real-time charts.")



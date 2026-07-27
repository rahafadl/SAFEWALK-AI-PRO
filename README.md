# SafeWalk AI PRO

**AI-Powered Smart Crosswalk Accessibility System**

SafeWalk AI PRO is an AI-powered smart crosswalk system that improves pedestrian safety and accessibility using real-time computer vision. The system detects pedestrians and wheelchair users, tracks their movement, and dynamically adjusts traffic signal timing to provide safer road crossings, especially for wheelchair users.

## Key Features

* Real-time pedestrian and wheelchair detection using YOLOv8
* Multi-object tracking with ByteTrack
* Waiting Zone and Crosswalk ROI monitoring
* Adaptive RED / GREEN / YELLOW traffic signal state machine
* Automatic wheelchair priority signal extension
* Unique object ID counting
* Event cooldown and duplicate event prevention
* SQLite event logging
* Real-time analytics dashboard built with Streamlit
* Heatmap-ready event data for future analysis

## Technologies

* Python
* YOLOv8
* OpenCV
* ByteTrack
* Streamlit
* Plotly
* SQLite

## Installation

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

Place your `best.pt` model file in the same directory as `app.py`, then click **Load Model** from the Streamlit sidebar.

## Expected Model Classes

```
0: pedestrian
1: wheelchair
```

If your model uses different class names, update the `CLASS_MAP` variable in `app.py`.

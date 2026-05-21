# SafeWalk AI Pro

Smart Crosswalk Accessibility System using YOLO + ByteTrack + ROI + Traffic Signal State Machine.

## Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

Put your `best.pt` next to `app.py`, then click **LOAD MODEL** in the sidebar.

## Expected model classes

Default:

```python
0: pedestrian
1: wheelchair
```

If your model class names are different, edit `CLASS_MAP` in `app.py`.

## Features

- YOLO + ByteTrack tracking
- Unique ID counting
- Waiting Zone ROI
- Crosswalk ROI
- RED/GREEN/YELLOW signal state machine
- Wheelchair accessibility extension
- Event cooldown and deduplication
- SQLite event logging
- Analytics and heatmap-ready points

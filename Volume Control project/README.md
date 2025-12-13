# Hand Gesture Volume Control

Camera-based controller that maps hand gestures to system audio. Uses MediaPipe Hands for landmark detection, OpenCV for capture/rendering, and `pycaw` for Windows audio endpoints.

## Stack and Constraints
- Python 3.9+ (Windows focus because `pycaw` targets WASAPI)
- OpenCV for video IO and drawing overlays
- MediaPipe Hands for real-time landmark detection
- NumPy for interpolation; `pycaw` + `comtypes` for volume control
- Webcam required; default index `0`

## Repository Layout
```
.
├── hand_control/                  # Reusable detection utilities
│   ├── __init__.py                # Exports HandDetector
│   └── hand_tracking.py           # MediaPipe wrapper + helpers
├── scripts/                       # Runnable entry points
│   ├── hand_tracking_debug.py     # Live landmark viewer + FPS overlay
│   ├── volume_control_basic.py    # Direct thumb-index → volume mapping
│   ├── volume_control_advanced.py # Adds safety gates, smoothing, pinky confirm
│   └── volume_control_single_hand.py # Locks to first detected hand
├── examples/
│   └── hand_tracking_minimal.py   # Smallest detector demo
├── archive/
│   └── hand_detector_legacy.py    # Early prototype for reference only
└── README.md
```

## Quickstart (Windows-friendly)
```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts\volume_control_advanced.py   # recommended entry point
```
Notes: scripts now prepend the repo root to `PYTHONPATH`, so you can also run them via an absolute path from any working directory. Press `Esc` to quit; change camera with `cv2.VideoCapture(<index>)` if needed.

## How It Works
1. Capture a frame from the webcam.
2. Detect hands via `HandDetector.findHands` (mirrors by default for a selfie view).
3. Extract landmarks + bounding box with `findPosition`.
4. Measure thumb–index distance (`findDistance`).
5. Interpolate distance → volume target (linear mapping).
6. Apply safety checks (advanced mode) then call `pycaw` to set system volume.
7. Render HUD (volume bar, FPS, safety cues) and show the frame.

## Core Module: `hand_control.hand_tracking`
`HandDetector` (exported in `hand_control/__init__.py`):
- `__init__(mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5, mirror=True)`: Configure MediaPipe Hands; `mirror` flips frames horizontally for a natural user view.
- `findHands(img, draw=True) -> img`: Runs detection, stores results, and optionally draws hand skeletons.
- `findPosition(img, listOfHighlights=None, handNo=0, draw=False) -> (lmList, bBox)`: Returns landmark list (`[id, x, y]`) and bounding box for the chosen hand. Optionally highlight specific landmark IDs and draw the box.
- `fingersUp() -> List[int]`: Returns five booleans (thumb uses horizontal check, others vertical) indicating raised fingers.
- `findDistance(p1, p2, img, draw=True) -> (length, img, [x1, y1, x2, y2, cx, cy])`: Pixel distance between two landmarks plus annotated frame and coordinates.

Legacy note: `archive/hand_detector_legacy.py` is kept only for historical comparison.

## Script Behavior (runnable in `scripts/`)
- `volume_control_basic.py`
  - Minimal demo. Maps thumb–index distance `[25, 150]` pixels to device volume range via `SetMasterVolumeLevel`.
  - Draws line and center dot; red dot when distance < 25 px.
- `volume_control_advanced.py`
  - Adds bounding-box area guard (ignores frames outside 30–500 scaled area).
  - Smooths `vol_perc` to nearest 10% to reduce jitter.
  - Requires pinky down (`fingersUp()[4] == 0`) to commit volume; otherwise shows blue HUD and skips writes.
  - Displays current system volume (`Vol Set`) and FPS.
  - Uses `SetMasterVolumeLevelScalar` for percentage-based control.
  - CLI flags: `--camera`, `--smoothness`, `--min-distance/--max-distance`, `--min-area/--max-area`, `--no-pinky-guard`, `--[no-]mirror`.
- `volume_control_single_hand.py`
  - Locks to the first detected hand to avoid flicker when multiple hands enter the frame.
  - Uses direct distance → volume mapping (like basic mode) on the locked hand only.
- `hand_tracking_debug.py`
  - Landmark visualizer with thumb/index highlighting and FPS overlay; prints tip coordinates to console.
- `examples/hand_tracking_minimal.py`
  - Smallest usage sample: detect hands, print thumb/index positions, draw highlights.

## Gesture Mapping and Safety
- Distance thresholds: default linear map from 25 px (min) to 150 px (max).
- Advanced safety gates:
  - Bounding box area filter to reject noise (too near/far).
  - Pinky confirmation to prevent accidental changes.
  - Smoothing rounds volume percentage to nearest 10.
- Visual cues: green volume bar when applying changes; blue when gated; red center dot when near-mute distance is reached.

## Configuration Knobs
- Detection reliability: tweak `detectionCon` / `trackCon` in `HandDetector`.
- Camera: change index in `cv2.VideoCapture` and adjust `CAP_PROP_FRAME_WIDTH/HEIGHT`.
- Sensitivity: adjust distance bounds `[25, 150]`, smoothing step (`smoothness`), or box area limits in advanced mode.
- Mirroring: set `mirror=False` in `HandDetector` for a non-selfie view.

## Troubleshooting
- Webcam not opening: verify the index; ensure no other app is locking the camera.
- Volume not changing: confirm Windows + `pycaw` installed; try `volume_control_basic.py` to bypass safety gates.
- Jittery movement: increase `smoothness`, improve lighting, or stabilize hand position.
- Landmarks missing: raise `detectionCon`, improve lighting, or ensure hand is fully in frame.

## Validation Checklist
- Visual sanity: volume bar moves smoothly with thumb–index distance at close/mid/far positions.
- Safety: advanced mode only commits when pinky is down; HUD color flips accordingly.
- Performance: FPS above 20 in `hand_tracking_debug.py` on target hardware.
- Recovery: graceful exit on `Esc`; camera released and windows destroyed.

## Agile Snapshot (5-person allocation)
- Product (Aisha): Define gesture UX, acceptance criteria, distance thresholds, and demo scenarios; maintain backlog.
- CV/ML (Ben): Own `hand_control/hand_tracking.py` tuning, finger state logic, and robustness checks.
- Audio/Platform (Chloe): Own `pycaw` integration, endpoint setup helpers, and Windows compatibility testing.
- App/UX (Diego): Own runtime scripts, HUD overlays, camera defaults, and CLI ergonomics.
- QA/DevOps (Emery): Own setup docs, reproducible env (venv, pins), smoke/regression checklist, and release notes.

Current sprint goals:
- Ship stable advanced mode with pinky safety and smoothing (Ben + Chloe).
- Harden single-hand lock when two hands enter frame (Diego).
- Deliver debug/minimal scripts as developer tools (Ben + Diego).
- Finalize setup/troubleshooting documentation for Windows users (Emery).
- Capture acceptance demo video with FPS/latency notes (Aisha + QA).

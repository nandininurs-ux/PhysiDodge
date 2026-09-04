# PhysiDodge

A controller-free arcade dodging game that turns your webcam into a motion
sensor. MediaPipe Pose tracks your shoulders and hips to build a "hitbox"
around your torso, and you physically step, duck, or sway to dodge falling
red dodgeballs.

This game needs a real webcam and a display, so it must be run **on your own
computer** — it can't run in a cloud sandbox or browser.

## 1. Install Python

Python 3.9–3.11 is recommended (MediaPipe support for very new Python
versions can lag). Check your version:

```bash
python3 --version
```

## 2. Set up a virtual environment (recommended)

```bash
cd physidodge
python3 -m venv venv
source venv/bin/activate      # on Windows: venv\Scripts\activate
```

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

## 4. Run the game

```bash
python physidodge.py
```

A window will open showing your mirrored webcam feed with a green box
around your torso. Dodge the falling red balls by moving your whole body —
stepping sideways, ducking, or leaning out of the way.

### Controls

| Key   | Action                      |
|-------|------------------------------|
| Q     | Quit                          |
| R     | Restart after Game Over       |
| SPACE | Pause / Resume                 |

## How it works

- **Motion tracking**: MediaPipe Pose detects 33 body landmarks per frame.
- **Hitbox**: the game takes your left/right shoulder and left/right hip
  landmarks, converts them from normalized (0–1) coordinates to pixel
  coordinates, and draws a padded bounding box around them.
- **Dodgeballs**: spawn at random x-positions at the top of the frame and
  fall at a speed that increases the longer you survive.
- **Collision**: each frame, the game does a circle-vs-rectangle collision
  check between each ball and your torso hitbox. A hit costs a life and
  triggers a red screen flash; letting a ball fall past you scores a point.
- **Mirroring**: the raw camera frame is flipped horizontally before display
  and before pose detection, so moving left on screen matches moving left
  in real life.

## Troubleshooting

- **"Could not open webcam"**: try changing `CAM_INDEX = 0` to `1` or `2` in
  `physidodge.py` (common on laptops with multiple cameras, or external
  USB webcams).
- **Low frame rate**: lower `FRAME_WIDTH` / `FRAME_HEIGHT` in the config
  section, or set `model_complexity=0` in `PoseTracker.__init__`.
- **Hitbox flickers or disappears**: make sure your shoulders and hips are
  both visible in frame — step back from the camera a bit, and make sure
  the room is reasonably well lit.
- **`ModuleNotFoundError: mediapipe`**: confirm your virtual environment is
  activated and `pip install -r requirements.txt` completed without errors.
  MediaPipe wheels aren't always available for the very latest Python
  version — if install fails, try Python 3.10 or 3.11.

## Tuning the difficulty

All the knobs live at the top of `physidodge.py`:

- `BASE_SPAWN_INTERVAL` / `MIN_SPAWN_INTERVAL` / `SPAWN_RAMP` — how often
  balls spawn, and how quickly that ramps up.
- `BASE_BALL_SPEED` / `MAX_BALL_SPEED` / `SPEED_RAMP` — fall speed and its
  ramp-up over time.
- `STARTING_LIVES` — how many hits you can take before Game Over.
- `BALL_RADIUS` / `TORSO_PADDING` — sizes of the ball and the forgiveness
  margin around your torso hitbox.

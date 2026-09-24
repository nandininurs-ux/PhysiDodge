"""
Fruit Ninja mode for PhysiDodge.

Your index fingers (tracked by MediaPipe's hand model, filtered for low jitter
and low lag) are the blades. The webcam is used ONLY for tracking - the camera feed is never shown;
the game draws its own wooden dojo scene instead.

Slice fruit, avoid bombs, don't let 3 fruits fall. Slice 3+ in one swipe for
a combo bonus.

Controls:
    Q / ESC  - Quit
    R        - Restart
    SPACE    - Pause / Resume
    F        - Toggle fullscreen
    Mouse    - Hold left button and drag to slice (also works with --no-cam)

Run (keep this file next to physidodge.py):
    python fruitninja.py
    python fruitninja.py --no-cam        # mouse only
    python fruitninja.py --cam 1         # different camera
"""

import os
import json
import math
import time
import random
import argparse
import threading
import urllib.request
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from physidodge import (CAM_INDEX, FRAME_WIDTH, FRAME_HEIGHT, MODEL_PATH,
                        ensure_model_downloaded, play_beeps,
                        draw_rounded_panel, FloatingText)

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
W, H = 1280, 720                 # game canvas size (independent of camera size)
TITLE = "Fruit Ninja"
GRAVITY = 1300                   # px/s^2
MAX_STRIKES = 3                  # fruits you may miss
MIN_SLICE_SPEED = 300            # px/s the blade must move to cut
MIN_SLICE_LEN = 6                # px per tracking sample
TRAIL_LIFE = 0.25                # seconds a blade trail lingers
X_RANGE = (0.10, 0.90)           # part of the camera view (left-right) that maps to the full screen
Y_RANGE = (0.10, 0.85)           # ...and top-bottom, so you can reach the top without leaving frame
CAM_W, CAM_H = 640, 480          # small + MJPG = higher fps on laptop webcams
HOLD = 0.20                      # seconds the blade coasts through a tracking dropout
MAX_LAT = 0.30                   # cap on latency compensation (s)
BLADE_WIDTH = 16                 # extra reach around the blade path (forgiving hits)
GAP_MAX = 0.15                   # a tracking gap longer than this breaks the swipe
SPEED_RAMP = 0.012               # game speed gained per second survived
MAX_SPEED = 2.2                  # cap on the game-speed multiplier
APEX_RANGE = (70, 300)           # y where fruit peaks - the top of the screen
HAND_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")
HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/latest/hand_landmarker.task")
COMBO_WINDOW = 0.35              # seconds - slices this close count as a combo
SAVE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "fruitninja_highscore.json")

FONT = cv2.FONT_HERSHEY_TRIPLEX
WHITE = (255, 255, 255)
GOLD = (40, 200, 255)
ORANGE = (30, 150, 255)
RED = (50, 50, 235)

# BGR colours. r = radius in px.
FRUITS = {
    "watermelon": dict(r=50, rind=(45, 125, 35), pith=(190, 235, 190), flesh=(75, 70, 235), juice=(75, 70, 235)),
    "orange":     dict(r=38, rind=(0, 130, 250), pith=(170, 220, 255), flesh=(30, 165, 255), juice=(30, 165, 255)),
    "apple":      dict(r=37, rind=(45, 45, 205), pith=(200, 235, 245), flesh=(215, 240, 250), juice=(200, 235, 245)),
    "lemon":      dict(r=34, rind=(30, 225, 250), pith=(170, 245, 255), flesh=(120, 235, 255), juice=(120, 235, 255)),
    "lime":       dict(r=32, rind=(50, 175, 60), pith=(170, 235, 180), flesh=(110, 225, 140), juice=(110, 225, 140)),
}
BOMB_R = 36
SEEDS = {"watermelon": [(-.45, .35), (-.15, .5), (.2, .4), (.45, .3), (.05, .2)],
         "apple": [(-.15, .22), (.15, .22)]}
CITRUS = ("orange", "lemon", "lime")


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def darken(c, f):
    return tuple(int(v * f) for v in c)


def lighten(c, f=0.45):
    return tuple(int(v + (255 - v) * f) for v in c)


def rot(x, y, a):
    return x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)


def outlined_text(img, text, org, scale=1.0, color=WHITE, thick=2, outline=(20, 25, 45)):
    cv2.putText(img, text, org, FONT, scale, outline, thick + 5, cv2.LINE_AA)
    cv2.putText(img, text, org, FONT, scale, color, thick, cv2.LINE_AA)


def centered_text(img, text, cx, cy, scale=1.0, color=WHITE, thick=2):
    (tw, th), _ = cv2.getTextSize(text, FONT, scale, thick)
    outlined_text(img, text, (int(cx - tw / 2), int(cy + th / 2)), scale, color, thick)


def seg_dist(p, q, c):
    """Distance from point c to the line segment p-q."""
    dx, dy = q[0] - p[0], q[1] - p[1]
    l2 = dx * dx + dy * dy
    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((c[0] - p[0]) * dx + (c[1] - p[1]) * dy) / l2))
    return math.hypot(c[0] - (p[0] + t * dx), c[1] - (p[1] + t * dy))


def make_background():
    """Warm wooden dojo wall: vertical planks, grain, faint red sun, vignette."""
    rng = np.random.default_rng(7)
    img = np.zeros((H, W, 3), np.float32)
    plank = 120
    base = np.array([70, 120, 175], np.float32)
    for x in range(0, W, plank):
        img[:, x:x + plank] = base * rng.uniform(0.82, 1.08)
    grain = rng.normal(0, 1, (H, W)).astype(np.float32)
    grain = cv2.GaussianBlur(grain, (1, 41), 1, sigmaY=12)
    grain = grain / (grain.std() + 1e-6) * 7
    img += grain[..., None]
    for x in range(0, W, plank):
        img[:, x:x + 3] *= 0.45
        img[:, x + 3:x + 5] *= 0.8
    sun = img.copy()
    cv2.circle(sun, (W // 2, H // 2 - 20), 260, (40, 50, 190), -1, cv2.LINE_AA)
    img = cv2.addWeighted(sun, 0.16, img, 0.84, 0)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    img *= (1 - 0.5 * np.clip(d / 1.25, 0, 1) ** 2)[..., None]
    return np.clip(img, 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Game objects
# --------------------------------------------------------------------------
class Item:
    """A whole fruit or bomb flying through the air."""
    def __init__(self, kind, x, y, vx, vy, r, spin, static=False):
        self.kind, self.x, self.y, self.vx, self.vy = kind, x, y, vx, vy
        self.r, self.spin, self.static = r, spin, static
        self.angle = random.uniform(0, 6.28)
        self.base_y = y
        self.g = GRAVITY
        self.sliced = False


class Half:
    """One half of a sliced fruit."""
    def __init__(self, x, y, vx, vy, r, angle, kind, side, spin):
        self.x, self.y, self.vx, self.vy, self.r = x, y, vx, vy, r
        self.angle, self.kind, self.side, self.spin = angle, kind, side, spin


class Particle:
    def __init__(self, x, y, vx, vy, r, color, life):
        self.x, self.y, self.vx, self.vy, self.r = x, y, vx, vy, r
        self.color, self.life, self.max_life = color, life, life

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 900 * dt
        self.life -= dt

    def draw(self, img):
        rad = max(1, int(self.r * self.life / self.max_life))
        cv2.circle(img, (int(self.x), int(self.y)), rad, self.color, -1, cv2.LINE_AA)


class Blade:
    """A tracked hand (or mouse) and its fading trail."""
    def __init__(self):
        self.pts = deque()
        self.pos = None
        self.last_xy = None
        self.t = 0.0          # time of latest point (real or coasted)
        self.real_t = 0.0     # time of latest REAL detection
        self.vel = (0.0, 0.0)

    def push(self, x, y, t, real=True):
        prev = None
        if self.pos is not None and t - self.t < GAP_MAX:
            prev = (self.pos, self.t)
            if real:
                dt = max(1e-3, t - self.t)
                vx, vy = (x - self.pos[0]) / dt, (y - self.pos[1]) / dt
                self.vel = (0.5 * self.vel[0] + 0.5 * vx, 0.5 * self.vel[1] + 0.5 * vy)
        elif real:
            self.vel = (0.0, 0.0)
        self.pos, self.t, self.last_xy = (x, y), t, (x, y)
        if real:
            self.real_t = t
        self.pts.append((x, y, t))
        return prev

    def update(self, now):
        if self.pos is not None and now - self.t > 0.25:
            self.pos = None
        while self.pts and now - self.pts[0][2] > TRAIL_LIFE:
            self.pts.popleft()


class OneEuro:
    """One Euro filter: heavy smoothing when the hand is slow (kills jitter),
    almost none when it's fast (no lag on swipes)."""
    def __init__(self, min_cutoff=1.0, beta=0.03, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = None

    @staticmethod
    def _alpha(cutoff, dt):
        return 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / dt)

    def reset(self):
        self.x = None

    def __call__(self, x, t):
        if self.x is None:
            self.x, self.dx, self.t = x, 0.0, t
            return x
        dt = max(1e-3, t - self.t)
        self.t = t
        a = self._alpha(self.d_cutoff, dt)
        self.dx = a * (x - self.x) / dt + (1 - a) * self.dx
        a = self._alpha(self.min_cutoff + self.beta * abs(self.dx), dt)
        self.x = a * x + (1 - a) * self.x
        return self.x


# --------------------------------------------------------------------------
# Drawing
# --------------------------------------------------------------------------
def draw_fruit(img, f):
    s = FRUITS[f.kind]
    r, a = f.r, f.angle
    c = (int(f.x), int(f.y))
    cv2.circle(img, c, r, s["rind"], -1, cv2.LINE_AA)
    if f.kind == "watermelon":
        for k in (0.35, 0.75):
            cv2.ellipse(img, c, (int(r * k), r - 2), math.degrees(a), 0, 360,
                        darken(s["rind"], 0.65), 3, cv2.LINE_AA)
    cv2.circle(img, c, r, darken(s["rind"], 0.55), 2, cv2.LINE_AA)
    cv2.ellipse(img, (int(f.x - r * 0.35), int(f.y - r * 0.38)), (max(4, r // 4), max(2, r // 8)),
                -40, 0, 360, lighten(s["rind"]), -1, cv2.LINE_AA)
    if f.kind != "watermelon":
        sx, sy = rot(0, -r * 0.88, a)
        cv2.ellipse(img, (int(f.x + sx), int(f.y + sy)), (10, 4), math.degrees(a) - 35,
                    0, 360, (40, 160, 50), -1, cv2.LINE_AA)


def draw_half(img, h):
    s = FRUITS[h.kind]
    r, deg = h.r, math.degrees(h.angle)
    c = (int(h.x), int(h.y))
    st, en = (0, 180) if h.side == 0 else (180, 360)
    sgn = 1 if h.side == 0 else -1
    cv2.ellipse(img, c, (r, r), deg, st, en, s["rind"], -1, cv2.LINE_AA)
    cv2.ellipse(img, c, (r - 4, r - 4), deg, st, en, s["pith"], -1, cv2.LINE_AA)
    cv2.ellipse(img, c, (r - 8, r - 8), deg, st, en, s["flesh"], -1, cv2.LINE_AA)
    if h.kind in CITRUS:
        for k in (1, 2, 3):
            ang = math.radians(st + k * 45) + h.angle
            cv2.line(img, c, (int(h.x + (r - 9) * math.cos(ang)), int(h.y + (r - 9) * math.sin(ang))),
                     s["pith"], 1, cv2.LINE_AA)
    for u, v in SEEDS.get(h.kind, []):
        sx, sy = rot(u * r, sgn * v * r, h.angle)
        cv2.circle(img, (int(h.x + sx), int(h.y + sy)), max(2, r // 14),
                   (25, 25, 25) if h.kind == "watermelon" else (30, 50, 80), -1, cv2.LINE_AA)


def draw_bomb(img, b, now):
    c = (int(b.x), int(b.y))
    r = b.r
    pulse = 0.5 + 0.5 * math.sin(now * 12)
    cv2.circle(img, c, r + 5, (0, 0, int(110 + 120 * pulse)), -1, cv2.LINE_AA)
    cv2.circle(img, c, r, (42, 42, 42), -1, cv2.LINE_AA)
    cv2.circle(img, c, r, (12, 12, 12), 3, cv2.LINE_AA)
    cv2.ellipse(img, (int(b.x - r * .35), int(b.y - r * .4)), (r // 4, r // 8), -40, 0, 360,
                (110, 110, 110), -1, cv2.LINE_AA)
    cv2.circle(img, c, int(r * .42), (225, 225, 225), -1, cv2.LINE_AA)          # skull
    for ex in (-.16, .16):
        cv2.circle(img, (int(b.x + ex * r * 2), int(b.y - r * .05)), max(2, r // 8), (20, 20, 20), -1)
    cv2.line(img, (int(b.x), int(b.y + r * .18)), (int(b.x), int(b.y + r * .3)), (20, 20, 20), 2)
    p0 = rot(0, -r, b.angle * 0.15)
    p0 = (int(b.x + p0[0]), int(b.y + p0[1]))
    p1 = (p0[0] + 12, p0[1] - 16)
    cv2.line(img, p0, p1, (90, 150, 190), 4, cv2.LINE_AA)
    cv2.circle(img, p1, random.randint(4, 8), (0, 220, 255), -1, cv2.LINE_AA)   # flickering spark
    cv2.circle(img, p1, 3, WHITE, -1, cv2.LINE_AA)


def draw_x(img, cx, cy, color, s=16):
    for col, th in (((15, 15, 25), 13), (color, 7)):
        cv2.line(img, (cx - s, cy - s), (cx + s, cy + s), col, th, cv2.LINE_AA)
        cv2.line(img, (cx - s, cy + s), (cx + s, cy - s), col, th, cv2.LINE_AA)


# --------------------------------------------------------------------------
# Hand tracking - runs on its own thread so the game never waits on the camera
# --------------------------------------------------------------------------
def to_screen(nx, ny):
    """Camera-normalised point -> game pixels (mirrored, active area stretched)."""
    sx = ((1 - nx) - X_RANGE[0]) / (X_RANGE[1] - X_RANGE[0])
    sy = (ny - Y_RANGE[0]) / (Y_RANGE[1] - Y_RANGE[0])
    return min(max(sx, 0.0), 1.0) * W, min(max(sy, 0.0), 1.0) * H


class HandBackend:
    """MediaPipe HandLandmarker - tracks the index fingertip directly."""
    name = "hand model"

    def __init__(self):
        if not os.path.exists(HAND_MODEL_PATH):
            print("First run: downloading hand landmarker model (~8 MB)...")
            urllib.request.urlretrieve(HAND_MODEL_URL, HAND_MODEL_PATH)
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO, num_hands=2,
            min_hand_detection_confidence=0.3, min_hand_presence_confidence=0.3,
            min_tracking_confidence=0.3)
        self.lm = mp_vision.HandLandmarker.create_from_options(opts)

    def detect(self, mp_img, ts):
        res = self.lm.detect_for_video(mp_img, ts)
        # 8 = fingertip, 7 = joint just behind it. The joint is steadier, so blend them.
        return [to_screen(0.7 * h[8].x + 0.3 * h[7].x, 0.7 * h[8].y + 0.3 * h[7].y)
                for h in res.hand_landmarks]

    def close(self):
        self.lm.close()


class PoseBackend:
    """Fallback if the hand model can't be downloaded: fingertips via body pose."""
    name = "pose fallback"

    def __init__(self):
        ensure_model_downloaded()
        opts = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=mp_vision.RunningMode.VIDEO, num_poses=1,
            min_pose_detection_confidence=0.4, min_pose_presence_confidence=0.4,
            min_tracking_confidence=0.4)
        self.lm = mp_vision.PoseLandmarker.create_from_options(opts)

    def detect(self, mp_img, ts):
        res = self.lm.detect_for_video(mp_img, ts)
        out = []
        if res.pose_landmarks:
            lm = res.pose_landmarks[0]
            for tip, wrist in ((19, 15), (20, 16)):
                for idx in (tip, wrist):
                    if getattr(lm[idx], "visibility", 1.0) > 0.3:
                        out.append(to_screen(lm[idx].x, lm[idx].y))
                        break
        return out

    def close(self):
        self.lm.close()


class TrackerThread(threading.Thread):
    """Grabs frames and detects hands in the background. The game loop just
    asks for the newest result, so rendering stays smooth regardless of
    tracking speed. Frames are used for tracking only and never displayed."""

    def __init__(self, cam_index):
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(cam_index)
        # MJPG matters: many laptop webcams drop to ~10 fps in their default raw format
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAM_W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAM_H)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)     # always the freshest frame = less lag
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        if not self.cap.isOpened():
            raise RuntimeError("Could not open webcam. Try --cam 1, or run with --no-cam for mouse play.")
        try:
            self.backend = HandBackend()
        except Exception as exc:
            print(f"Hand model unavailable ({exc}); falling back to pose tracking.")
            self.backend = PoseBackend()
        self.lock = threading.Lock()
        self.seq, self.pts, self.t = 0, [], 0.0
        self.fps, self.lat, self.brightness = 30.0, 0.0, 128.0
        self.failed, self.running = False, True
        self.t0, self.last_ts = time.time(), -1
        self.start()

    def run(self):
        prev = time.time()
        while self.running:
            ok, frame = self.cap.read()
            if not ok:
                self.failed = True
                break
            t = time.time()
            k = 640.0 / frame.shape[1]
            small = cv2.resize(frame, None, fx=k, fy=k)
            self.brightness = 0.9 * self.brightness + 0.1 * float(small[::8, ::8].mean())
            img = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(small, cv2.COLOR_BGR2RGB))
            ts = max(int((t - self.t0) * 1000), self.last_ts + 1)
            self.last_ts = ts
            try:
                pts = self.backend.detect(img, ts)
            except Exception:
                pts = []
            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 / max(1e-3, now - prev)
            self.lat = 0.9 * self.lat + 0.1 * (now - t)
            prev = now
            with self.lock:
                self.pts, self.t, self.seq = pts, t, self.seq + 1

    def get(self):
        with self.lock:
            return self.seq, list(self.pts), self.t

    def close(self):
        self.running = False
        self.join(timeout=1.5)
        self.cap.release()
        self.backend.close()


# --------------------------------------------------------------------------
# Game
# --------------------------------------------------------------------------
class FruitNinja:
    def __init__(self, use_cam=True, cam_index=CAM_INDEX, debug=False):
        self.tracker = TrackerThread(cam_index) if use_cam else None
        self.last_seq = -1
        self.debug, self.raw_dbg = debug, []
        self.filters = {k: (OneEuro(), OneEuro()) for k in "AB"}
        self.bg = make_background()
        self.splat = np.zeros((H // 2, W // 2, 3), np.uint8)
        self.splat_a = np.zeros((H // 2, W // 2), np.uint8)
        self.fade = 0.0
        self.blades = {k: Blade() for k in "ABM"}
        self.segments = []
        self.mouse_down, self.mouse_pending = False, None
        self.best = self._load_best()
        self.paused = self.fullscreen = False
        self._to_menu(time.time())

    # ---------- persistence ----------
    @staticmethod
    def _load_best():
        try:
            with open(SAVE_PATH) as f:
                return int(json.load(f).get("high_score", 0))
        except (OSError, ValueError):
            return 0

    def _save_best(self):
        try:
            with open(SAVE_PATH, "w") as f:
                json.dump({"high_score": self.best}, f)
        except OSError:
            pass

    # ---------- state ----------
    def _clear(self):
        self.items, self.halves, self.particles, self.texts, self.slashes = [], [], [], [], []
        self.score = self.strikes = 0
        self.play_time = 0.0
        self.next_wave = 0.6
        self.ending, self.end_at = False, 0.0
        self.flash_until = self.shake_until = 0.0
        self.recent, self.combo_text = [], None
        self.menu_fruit, self.start_at = None, None
        self.new_best = False

    def _make_menu_fruit(self, label):
        self.menu_fruit = Item("watermelon", W / 2, H * 0.64, 0, 0, 80, 0.4, static=True)
        self.menu_label = label

    def _to_menu(self, now):
        self._clear()
        self.state = "menu"
        self._make_menu_fruit("SLICE TO START")

    def _start_game(self, now):
        self._clear()
        self.state = "playing"

    def _end_game(self, now):
        self.state = "over"
        self.over_at = now
        self.new_best = self.score > self.best
        if self.new_best:
            self.best = self.score
            self._save_best()
        self.items = []
        self.menu_fruit = None

    # ---------- input ----------
    def _on_mouse(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_down = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_down = False
        if self.mouse_down and event in (cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONDOWN):
            self.mouse_pending = (x, y)

    def speed(self):
        """Game-speed multiplier: fruit flies faster the longer you survive."""
        return min(MAX_SPEED, 1.0 + SPEED_RAMP * self.play_time)

    def _assign(self, pts):
        """Give each detected hand a stable identity (A/B) by continuity."""
        if not pts:
            return {}
        if len(pts) == 1:
            known = [k for k in "AB" if self.blades[k].last_xy]
            key = min(known, key=lambda k: math.dist(pts[0], self.blades[k].last_xy)) if known else "A"
            return {key: pts[0]}
        a, b = sorted(pts[:2])
        la, lb = self.blades["A"].last_xy, self.blades["B"].last_xy
        if la and lb and math.dist(a, la) + math.dist(b, lb) > math.dist(a, lb) + math.dist(b, la):
            a, b = b, a
        return {"A": a, "B": b}

    def _push(self, key, x, y, t, filtered=True, real=True):
        b = self.blades[key]
        if filtered:
            fx, fy = self.filters[key]
            if b.pos is None:            # hand was lost - restart the filter, don't glide in
                fx.reset()
                fy.reset()
            x, y = fx(x, t), fy(y, t)
        prev = b.push(x, y, t, real)
        if prev is not None and t > prev[1]:
            length = math.dist(prev[0], b.pos)
            if length >= MIN_SLICE_LEN and length / (t - prev[1]) >= MIN_SLICE_SPEED:
                self.segments.append((prev[0], b.pos, t))

    def _feed_blades(self, now):
        self.segments = []
        if self.tracker:
            seq, pts, t = self.tracker.get()
            if seq != self.last_seq:           # only act on genuinely new tracking samples
                self.last_seq = seq
                self.raw_dbg = list(pts)
                assigned = self._assign(pts)
                for key in "AB":
                    b = self.blades[key]
                    if key in assigned:
                        self._push(key, *assigned[key], t)
                    elif b.pos is not None and t - b.real_t < HOLD and t > b.t \
                            and abs(b.vel[0]) + abs(b.vel[1]) > 150:
                        # dropout mid-swipe: keep the blade gliding along its last velocity
                        dt = t - b.t
                        x = min(max(b.pos[0] + b.vel[0] * dt, 0), W)
                        y = min(max(b.pos[1] + b.vel[1] * dt, 0), H)
                        self._push(key, x, y, t, filtered=False, real=False)
                        b.vel = (b.vel[0] * 0.85, b.vel[1] * 0.85)
        if self.mouse_pending:
            self._push("M", *self.mouse_pending, now, filtered=False)
            self.mouse_pending = None
        for b in self.blades.values():
            b.update(now)

    # ---------- spawning ----------
    def _launch(self, kind):
        """Throw from below the screen up to the top. The apex height is fixed by
        the throw; speed only compresses the flight time, so fruit still
        reaches the top but hangs there for less time as the game speeds up."""
        r = BOMB_R if kind == "bomb" else FRUITS[kind]["r"]
        sp = self.speed()
        g = GRAVITY * sp * sp
        y0 = H + r
        vy = -math.sqrt(2 * g * (y0 - random.uniform(*APEX_RANGE)))
        x = random.uniform(0.12 * W, 0.88 * W)
        vx = ((W / 2 - x) * random.uniform(0.12, 0.4) + random.uniform(-60, 60)) * sp
        it = Item(kind, x, y0, vx, vy, r, random.uniform(-5, 5) * sp)
        it.g = g
        self.items.append(it)

    def _spawn(self):
        if self.play_time < self.next_wave:
            return
        t = self.play_time
        self.next_wave = t + max(0.6, 1.5 - 0.01 * t) / self.speed() * random.uniform(0.85, 1.15)
        bomb_p = 0 if t < 6 else min(0.22, 0.08 + 0.0015 * t)
        for _ in range(random.randint(1, min(5, 2 + int(t / 25)))):
            self._launch("bomb" if random.random() < bomb_p else random.choice(list(FRUITS)))

    # ---------- slicing ----------
    def _check_slices(self, items, now):
        for p, q, t in self.segments:
            ang = math.atan2(q[1] - p[1], q[0] - p[0])
            back = min(max(now - t, 0.0), MAX_LAT)     # camera + detection delay
            for it in items:
                if it.sliced:
                    continue
                reach = it.r + (0 if it.kind == "bomb" else BLADE_WIDTH)
                # where the item was when the hand actually made this movement
                ox = it.x - it.vx * back
                oy = it.y - it.vy * back + 0.5 * it.g * back * back
                if seg_dist(p, q, (ox, oy)) <= reach or seg_dist(p, q, (it.x, it.y)) <= reach:
                    self._slice(it, ang, now)

    def _add_splat(self, x, y, color):
        sx, sy = int(x / 2), int(y / 2)
        cv2.circle(self.splat, (sx, sy), random.randint(16, 26), color, -1, cv2.LINE_AA)
        cv2.circle(self.splat_a, (sx, sy), random.randint(16, 26), 200, -1, cv2.LINE_AA)
        for _ in range(random.randint(4, 7)):
            ox, oy, rr = int(random.gauss(0, 22)), int(random.gauss(0, 22)), random.randint(3, 10)
            cv2.circle(self.splat, (sx + ox, sy + oy), rr, color, -1, cv2.LINE_AA)
            cv2.circle(self.splat_a, (sx + ox, sy + oy), rr, 200, -1, cv2.LINE_AA)

    def _slice(self, it, ang, now):
        it.sliced = True
        if it.kind == "bomb":
            self._explode(it, now)
            return
        s = FRUITS[it.kind]
        nx, ny = -math.sin(ang), math.cos(ang)
        for side, sgn in ((0, 1), (1, -1)):
            self.halves.append(Half(it.x, it.y, it.vx + sgn * nx * 170, it.vy + sgn * ny * 170 - 80,
                                    it.r, ang, it.kind, side, sgn * random.uniform(1.5, 4)))
        dx, dy = math.cos(ang) * it.r * 1.6, math.sin(ang) * it.r * 1.6
        self.slashes.append([(it.x - dx, it.y - dy), (it.x + dx, it.y + dy), 0.18])
        for _ in range(18):
            a = ang + math.pi / 2 * random.choice((-1, 1)) + random.uniform(-0.9, 0.9)
            sp = random.uniform(150, 520)
            self.particles.append(Particle(it.x, it.y, math.cos(a) * sp, math.sin(a) * sp - 120,
                                           random.uniform(2.5, 6), s["juice"], random.uniform(0.35, 0.8)))
        self._add_splat(it.x, it.y, s["juice"])
        if it is self.menu_fruit:
            self.start_at = now + 0.7
            play_beeps([(900, 60), (1200, 80)])
            return
        self.score += 1
        self.texts.append(FloatingText(it.x - 12, it.y - it.r, "+1", WHITE, scale=0.9, life=0.6))
        self.recent = [t for t in self.recent if now - t < COMBO_WINDOW] + [now]
        n = len(self.recent)
        if n >= 3:
            self.score += n
            if self.combo_text:
                self.combo_text.life = 0
            self.combo_text = FloatingText(min(max(it.x - 170, 20), W - 420), max(it.y - 60, 140),
                                           f"{n} FRUIT COMBO! +{n}", GOLD, scale=1.2, life=1.0, vy=-40)
            self.texts.append(self.combo_text)
        play_beeps([(800, 50), (1100, 60)])

    def _explode(self, it, now):
        self.flash_until, self.shake_until = now + 0.5, now + 0.45
        for _ in range(45):
            a, sp = random.uniform(0, 6.283), random.uniform(200, 900)
            self.particles.append(Particle(it.x, it.y, math.cos(a) * sp, math.sin(a) * sp,
                                           random.uniform(4, 11),
                                           random.choice([(0, 220, 255), (0, 140, 255), (60, 60, 70), (40, 40, 255)]),
                                           random.uniform(0.4, 1.0)))
        cv2.circle(self.splat, (int(it.x / 2), int(it.y / 2)), 42, (20, 20, 20), -1, cv2.LINE_AA)
        cv2.circle(self.splat_a, (int(it.x / 2), int(it.y / 2)), 42, 210, -1, cv2.LINE_AA)
        self.texts.append(FloatingText(it.x - 90, it.y, "BOOM!", ORANGE, scale=1.8, life=0.9, vy=-30))
        self.ending, self.end_at = True, now + 1.1
        play_beeps([(250, 120), (120, 300)])

    def _miss(self, it, now):
        self.strikes += 1
        self.texts.append(FloatingText(min(max(it.x - 40, 20), W - 140), H - 70, "MISS", RED,
                                       scale=1.1, life=0.8))
        play_beeps([(200, 120)])
        if self.strikes >= MAX_STRIKES:
            self.ending, self.end_at = True, now + 0.8

    # ---------- update ----------
    def _update(self, dt, now):
        for h in self.halves:
            h.x += h.vx * dt
            h.y += h.vy * dt
            h.vy += GRAVITY * self.speed() ** 2 * dt
            h.angle += h.spin * dt
        self.halves = [h for h in self.halves if h.y - h.r < H + 60]
        for p in self.particles:
            p.update(dt)
        self.particles = [p for p in self.particles if p.life > 0]
        for t in self.texts:
            t.update(dt)
        self.texts = [t for t in self.texts if t.alive()]
        for s in self.slashes:
            s[2] -= dt
        self.slashes = [s for s in self.slashes if s[2] > 0]
        self.fade += dt * 45
        k = int(self.fade)
        if k:
            self.splat_a = np.maximum(self.splat_a, k) - k
            self.fade -= k

        if self.state == "playing":
            self.play_time += dt
            if not self.ending:
                self._spawn()
            keep = []
            for it in self.items:
                it.x += it.vx * dt
                it.y += it.vy * dt
                it.vy += it.g * dt
                it.angle += it.spin * dt
                if it.y - it.r > H and it.vy > 0:
                    if it.kind != "bomb" and not it.sliced and not self.ending:
                        self._miss(it, now)
                    continue
                keep.append(it)
            self.items = keep
            if not self.ending:
                self._check_slices(self.items, now)
            self.items = [i for i in self.items if not i.sliced]
            if self.ending and now >= self.end_at:
                self._end_game(now)
        else:
            if self.state == "over" and self.menu_fruit is None and now - self.over_at > 1.0:
                self._make_menu_fruit("SLICE TO PLAY AGAIN")
            mf = self.menu_fruit
            if mf is not None:
                mf.angle += mf.spin * dt
                mf.y = H * 0.64 + 8 * math.sin(now * 2)
                if not mf.sliced:
                    self._check_slices([mf], now)
                elif self.start_at is not None and now >= self.start_at:
                    self._start_game(now)

    # ---------- render ----------
    def _draw_trails(self, img, now):
        overlay = img.copy()
        lines = []
        for b in self.blades.values():
            pts = list(b.pts)
            for i in range(1, len(pts)):
                t = max(0.0, 1 - (now - pts[i][2]) / TRAIL_LIFE)
                lines.append(((int(pts[i - 1][0]), int(pts[i - 1][1])), (int(pts[i][0]), int(pts[i][1])), t))
        for p, q, t in lines:
            cv2.line(overlay, p, q, (255, 190, 70), int(6 + 16 * t), cv2.LINE_AA)
        if lines:
            cv2.addWeighted(overlay, 0.45, img, 0.55, 0, img)
        for p, q, t in lines:
            cv2.line(img, p, q, WHITE, int(1 + 6 * t), cv2.LINE_AA)
        for k in "AB":
            pos = self.blades[k].pos
            if pos:
                cv2.circle(img, (int(pos[0]), int(pos[1])), 12, WHITE, 2, cv2.LINE_AA)

    def _draw_hud(self, img):
        outlined_text(img, str(self.score), (40, 92), 2.3, WHITE, 4)
        outlined_text(img, f"BEST {self.best}", (44, 134), 0.8, GOLD, 2)
        outlined_text(img, f"SPEED x{self.speed():.1f}", (44, 172), 0.7, (255, 215, 120), 2)
        for i in range(MAX_STRIKES):
            draw_x(img, W - 190 + i * 62, 62, RED if i < self.strikes else (105, 110, 120))

    def _render(self, now):
        img = self.bg.copy()
        if self.splat_a.any():
            a = cv2.resize(self.splat_a, (W, H))
            a3 = cv2.merge([a, a, a])
            sp = cv2.resize(self.splat, (W, H))
            img = cv2.add(cv2.multiply(img, 255 - a3, scale=1 / 255.0), cv2.multiply(sp, a3, scale=1 / 255.0))

        for h in self.halves:
            draw_half(img, h)
        for it in self.items:
            draw_bomb(img, it, now) if it.kind == "bomb" else draw_fruit(img, it)
        mf = self.menu_fruit
        if mf is not None and not mf.sliced:
            cv2.circle(img, (int(mf.x), int(mf.y)), mf.r + 14 + int(4 * math.sin(now * 4)), GOLD, 3, cv2.LINE_AA)
            draw_fruit(img, mf)
            centered_text(img, self.menu_label, W / 2, mf.y + mf.r + 55, 1.0, WHITE, 2)
        for p in self.particles:
            p.draw(img)
        for a, b, life in self.slashes:
            cv2.line(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), WHITE,
                     max(1, int(6 * life / 0.18)), cv2.LINE_AA)

        self._draw_trails(img, now)
        for t in self.texts:
            t.draw(img)

        if self.state == "menu":
            centered_text(img, "FRUIT NINJA", W / 2, 130, 3.2, ORANGE, 6)
            centered_text(img, "Slice fruit with your hands  -  avoid the bombs!", W / 2, 215, 0.85, WHITE, 2)
            centered_text(img, f"Best score: {self.best}", W / 2, 258, 0.8, GOLD, 2)
        else:
            self._draw_hud(img)
        if self.state == "over":
            draw_rounded_panel(img, (W // 2 - 300, 90), (W // 2 + 300, 340), (20, 20, 30), radius=24, alpha=0.7)
            centered_text(img, "GAME OVER", W / 2, 165, 2.0, RED, 4)
            centered_text(img, f"Score {self.score}", W / 2, 240, 1.2, WHITE, 2)
            centered_text(img, "NEW BEST!" if self.new_best else f"Best {self.best}", W / 2, 295, 0.9, GOLD, 2)
        if self.state in ("menu", "over") and self.tracker:
            seen = any(self.blades[k].pos for k in "AB")
            msg, col = (("Hands detected - swipe to slice!", (120, 255, 140)) if seen else
                        ("Show your hand to the camera (or drag with the mouse)", ORANGE))
            centered_text(img, msg, W / 2, H - 32, 0.7, col, 2)
        if self.tracker:
            tr, warn = self.tracker, []
            if tr.fps < 18:
                warn.append("low fps: add light / close other camera apps")
            if tr.brightness < 70:
                warn.append("too dark: face a light")
            txt = f"tracking: {tr.backend.name}  {tr.fps:.0f} fps  {tr.lat * 1000:.0f} ms"
            cv2.putText(img, txt + ("   !! " + "; ".join(warn) if warn else ""), (16, H - 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 200, 255) if warn else (235, 235, 235), 1, cv2.LINE_AA)
        if self.debug:                                  # raw detections (red) vs filtered blade (green)
            for x, y in self.raw_dbg:
                cv2.circle(img, (int(x), int(y)), 7, (0, 0, 255), -1, cv2.LINE_AA)
            for k in "AB":
                b = self.blades[k]
                if b.pos:
                    coast = b.real_t < b.t
                    cv2.circle(img, (int(b.pos[0]), int(b.pos[1])), 16, (0, 165, 255) if coast else (0, 230, 0), 2, cv2.LINE_AA)
        if self.paused:
            centered_text(img, "PAUSED", W / 2, H / 2, 2.0, WHITE, 4)

        if now < self.flash_until:
            a = 0.8 * (self.flash_until - now) / 0.5
            img = cv2.addWeighted(img, 1 - a, np.full_like(img, 255), a, 0)
        if now < self.shake_until:
            s = 18 * (self.shake_until - now) / 0.45
            m = np.float32([[1, 0, random.uniform(-s, s)], [0, 1, random.uniform(-s, s)]])
            img = cv2.warpAffine(img, m, (W, H), borderMode=cv2.BORDER_REFLECT)
        return img

    # ---------- main loop ----------
    def run(self):
        cv2.namedWindow(TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(TITLE, W, H)
        cv2.setMouseCallback(TITLE, self._on_mouse)
        last = time.time()
        try:
            while True:
                now = time.time()
                dt = min(0.05, now - last)
                last = now
                if self.tracker and self.tracker.failed:
                    print("Camera stopped delivering frames.")
                    break
                if not self.paused:
                    self._feed_blades(now)
                    self._update(dt, now)
                cv2.imshow(TITLE, self._render(now))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif key == ord('r'):
                    self._start_game(now)
                elif key == ord(' '):
                    self.paused = not self.paused
                elif key == ord('f'):
                    self.fullscreen = not self.fullscreen
                    cv2.setWindowProperty(TITLE, cv2.WND_PROP_FULLSCREEN,
                                          cv2.WINDOW_FULLSCREEN if self.fullscreen else cv2.WINDOW_NORMAL)
        finally:
            if self.tracker:
                self.tracker.close()
            cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description="Fruit Ninja - slice fruit with your hands")
    ap.add_argument("--no-cam", action="store_true", help="mouse-only mode (no webcam)")
    ap.add_argument("--cam", type=int, default=CAM_INDEX, help="camera index")
    ap.add_argument("--debug", action="store_true",
                    help="show raw tracking (red) vs filtered blade (green; orange = coasting through a dropout)")
    args = ap.parse_args()
    FruitNinja(use_cam=not args.no_cam, cam_index=args.cam, debug=args.debug).run()


if __name__ == "__main__":
    main()
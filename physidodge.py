"""
PhysiDodge - A controller-free, full-body arcade dodging game.

Uses a webcam + MediaPipe Pose to track your torso in real time. The game
waits until your torso is detected, then runs a 3-second "get ready"
countdown before dodgeballs start falling - so you don't get hit while
still stepping into frame.

Controls:
    Q       - Quit
    R       - Restart after Game Over
    SPACE   - Pause / Resume

Run:
    python physidodge.py
"""

import os
import json
import math
import time
import random
import threading
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

try:
    import winsound  # Windows-only stdlib module
    HAS_SOUND = True
except ImportError:
    HAS_SOUND = False


def play_beeps(sequence):
    """Plays a list of (frequency_hz, duration_ms) tones back to back on a
    background thread. Silently does nothing on non-Windows systems, since
    winsound.Beep is Windows-only and there's no cross-platform stdlib
    equivalent - the game is fully playable without it."""
    if not HAS_SOUND:
        return

    def _run():
        for freq, dur in sequence:
            try:
                winsound.Beep(int(freq), int(dur))
            except (RuntimeError, ValueError):
                return

    threading.Thread(target=_run, daemon=True).start()


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

CAM_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

STARTING_LIVES = 3
TORSO_PADDING = 20          # extra pixels around the shoulder/hip box
BALL_RADIUS = 22            # baseline radius; actual spawns vary around this
MIN_BALL_RADIUS = 14
MAX_BALL_RADIUS = 34

# Difficulty scaling (per second of survival) - balls spawn faster and
# faster, and get progressively more unpredictable, the longer you survive.
BASE_SPAWN_INTERVAL = 1.4   # seconds between spawns at game start
MIN_SPAWN_INTERVAL = 0.22
SPAWN_RAMP = 0.022          # spawn interval shrinks by this much per second

BASE_BALL_SPEED = 300       # pixels/sec at game start
MAX_BALL_SPEED = 1100
SPEED_RAMP = 9              # px/sec added per second survived
SPEED_JITTER = (0.85, 1.3)  # per-ball random multiplier on top of the ramp

# Randomness that grows with survival time: how much a ball's path can
# curve sideways, and how likely a second ball is to spawn alongside it.
BASE_DRIFT_RANGE = 30       # px/sec of sideways drift at game start
MAX_DRIFT_RANGE = 180
DRIFT_RAMP = 4              # px/sec of drift range added per second survived

BASE_DOUBLE_SPAWN_CHANCE = 0.04
MAX_DOUBLE_SPAWN_CHANCE = 0.5
DOUBLE_SPAWN_RAMP = 0.008   # chance added per second survived

HIT_FLASH_DURATION = 0.35   # seconds the red flash stays on screen

# "Get ready" countdown shown before balls start falling (also replayed on restart)
READY_COUNTDOWN_SECONDS = 3.0
GO_FLASH_DURATION = 0.6

# Combo scoring: consecutive dodges without a hit build a streak; every
# COMBO_MILESTONE dodges in a row adds a permanent point-per-dodge bonus.
COMBO_MILESTONE = 5

# Screen shake feedback on taking a hit
SHAKE_DURATION = 0.25
SHAKE_MAGNITUDE = 16

# Difficulty/"intensity" progress bar - roughly when the ramps below all
# reach their caps, so the bar visually fills up as the game gets harder.
INTENSITY_PLATEAU_SECONDS = 75

# Cinematic vignette darkening on the video feed (0 = none, 1 = extreme)
VIGNETTE_STRENGTH = 0.30

# High score persistence (saved next to the script, survives restarts)
HIGH_SCORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "physidodge_highscore.json")

# Sparkle celebration, triggered the moment you beat your saved high score
SPARKLE_BURST_COUNT = 60        # total sparkles per burst (split across both edges)
SPARKLE_MIN_LIFE = 1.0
SPARKLE_MAX_LIFE = 1.9
SPARKLE_MIN_SPEED = 260
SPARKLE_MAX_SPEED = 560
HIGH_SCORE_BANNER_DURATION = 2.2

# MediaPipe Pose Landmarker model (downloaded once, cached locally)
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(MODEL_DIR, "pose_landmarker_lite.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)

# Colors (BGR)
COLOR_BALL = (0, 0, 255)
COLOR_BALL_OUTLINE = (0, 0, 150)
COLOR_HITBOX = (80, 240, 120)
COLOR_HITBOX_MISSING = (0, 165, 255)
COLOR_TEXT = (240, 240, 240)
COLOR_FLASH = (0, 0, 255)
COLOR_HEART_FULL = (60, 40, 235)
COLOR_HEART_EMPTY = (95, 95, 95)
COLOR_PANEL_BG = (35, 25, 20)
COLOR_GOLD = (10, 200, 255)
COLOR_ACCENT = (200, 140, 40)
SPARKLE_COLORS = [(255, 255, 255), (10, 220, 255), (60, 200, 255), (150, 255, 255)]


def draw_heart(frame, center, size, color, filled=True, thickness=2):
    """Draws a simple heart icon (two circle lobes + a triangle point)."""
    x, y = center
    r = max(3, size // 2)
    lobe_offset_x = int(r * 0.55)
    lobe_offset_y = int(r * 0.25)
    lobe_r = int(r * 0.62)

    pts = np.array(
        [
            [x - r, y - lobe_offset_y],
            [x + r, y - lobe_offset_y],
            [x, y + r],
        ],
        np.int32,
    )

    if filled:
        cv2.circle(frame, (x - lobe_offset_x, y - lobe_offset_y), lobe_r, color, -1)
        cv2.circle(frame, (x + lobe_offset_x, y - lobe_offset_y), lobe_r, color, -1)
        cv2.fillConvexPoly(frame, pts, color)
    else:
        cv2.circle(frame, (x - lobe_offset_x, y - lobe_offset_y), lobe_r, color, thickness)
        cv2.circle(frame, (x + lobe_offset_x, y - lobe_offset_y), lobe_r, color, thickness)
        cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=thickness)


def draw_rounded_panel(frame, pt1, pt2, color, radius=18, alpha=1.0):
    """Draws a filled rounded rectangle, optionally alpha-blended onto frame."""
    x1, y1 = pt1
    x2, y2 = pt2
    radius = min(radius, (x2 - x1) // 2, (y2 - y1) // 2)

    target = frame if alpha >= 1.0 else frame.copy()

    cv2.rectangle(target, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(target, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in [(x1 + radius, y1 + radius), (x2 - radius, y1 + radius),
                   (x1 + radius, y2 - radius), (x2 - radius, y2 - radius)]:
        cv2.circle(target, (cx, cy), radius, color, -1)

    if alpha < 1.0:
        cv2.addWeighted(target, alpha, frame, 1 - alpha, 0, frame)


def draw_text_shadow(frame, text, org, scale=1.0, color=COLOR_TEXT, thickness=2,
                      font=cv2.FONT_HERSHEY_DUPLEX, shadow_color=(0, 0, 0)):
    """Draws text with a small drop shadow so it stays readable over video."""
    x, y = org
    cv2.putText(frame, text, (x + 2, y + 2), font, scale, shadow_color, thickness + 1, cv2.LINE_AA)
    cv2.putText(frame, text, (x, y), font, scale, color, thickness, cv2.LINE_AA)


def draw_bracket_box(frame, bbox, color, thickness=3, corner_len=26):
    """Draws targeting-style corner brackets instead of a full rectangle -
    a lighter-weight, more 'AR tracking' look than a solid box outline."""
    x1, y1, x2, y2 = bbox
    corner_len = min(corner_len, (x2 - x1) // 2, (y2 - y1) // 2)
    corners = [
        ((x1, y1), (1, 0), (0, 1)),
        ((x2, y1), (-1, 0), (0, 1)),
        ((x1, y2), (1, 0), (0, -1)),
        ((x2, y2), (-1, 0), (0, -1)),
    ]
    for (cx, cy), (dx, dy), (ex, ey) in corners:
        cv2.line(frame, (cx, cy), (cx + dx * corner_len, cy + dy * corner_len), color, thickness, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx + ex * corner_len, cy + ey * corner_len), color, thickness, cv2.LINE_AA)
    # faint fill so the tracked region still reads clearly at a glance
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
    cv2.addWeighted(overlay, 0.08, frame, 0.92, 0, frame)


class Sparkle:
    """A single star-shaped particle used in the high-score celebration burst."""

    def __init__(self, x, y, vx, vy, size, color, life):
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.size = size
        self.color = color
        self.life = life
        self.max_life = life

    def update(self, dt):
        self.x += self.vx * dt
        self.y += self.vy * dt
        self.vy += 25 * dt  # gentle gravity so paths arc slightly
        self.life -= dt

    def alive(self):
        return self.life > 0

    def draw(self, frame):
        t = max(0.0, self.life / self.max_life)
        twinkle = 0.55 + 0.45 * math.sin(self.life * 22 + self.x)
        size = max(2, int(self.size * t * twinkle))
        glow_color = tuple(int(c * 0.5 * t) for c in self.color)
        center = (int(self.x), int(self.y))
        cv2.circle(frame, center, size + 4, glow_color, -1, cv2.LINE_AA)
        cv2.drawMarker(frame, center, self.color, markerType=cv2.MARKER_STAR,
                        markerSize=max(4, size * 2), thickness=2, line_type=cv2.LINE_AA)


class FloatingText:
    """A short-lived label that drifts upward and fades - used for dodge
    points, combo milestones, and hit feedback so the game gives instant
    visual response to what just happened."""

    def __init__(self, x, y, text, color, scale=0.8, life=0.7, vy=-70):
        self.x = x
        self.y = y
        self.text = text
        self.color = color
        self.scale = scale
        self.life = life
        self.max_life = life
        self.vy = vy

    def update(self, dt):
        self.y += self.vy * dt
        self.life -= dt

    def alive(self):
        return self.life > 0

    def draw(self, frame):
        t = max(0.0, self.life / self.max_life)
        color = tuple(int(c * t) for c in self.color)
        draw_text_shadow(frame, self.text, (int(self.x), int(self.y)),
                          scale=self.scale, color=color, thickness=2)


# --------------------------------------------------------------------------
# Pose tracking wrapper
# --------------------------------------------------------------------------

def ensure_model_downloaded():
    """Downloads the MediaPipe Pose Landmarker model on first run."""
    if os.path.exists(MODEL_PATH):
        return
    print("First run: downloading pose landmarker model (~5-30 MB)...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Model downloaded to", MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(
            f"Could not download the pose model from {MODEL_URL}.\n"
            f"Check your internet connection, or manually download it and "
            f"place it at: {MODEL_PATH}\n"
            f"Original error: {exc}"
        )


class PoseTracker:
    """Wraps MediaPipe's PoseLandmarker (Tasks API) and extracts a torso bbox."""

    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24

    def __init__(self):
        ensure_model_downloaded()

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._clock_start = time.time()
        self._last_timestamp_ms = -1

    def process(self, frame_bgr):
        """Returns (result, torso_bbox_or_None). bbox = (x1, y1, x2, y2)."""
        h, w = frame_bgr.shape[:2]
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        # VIDEO mode requires strictly increasing timestamps.
        timestamp_ms = int((time.time() - self._clock_start) * 1000)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        bbox = None
        if result.pose_landmarks:
            lm = result.pose_landmarks[0]  # first (only) detected pose

            pts = [
                lm[self.LEFT_SHOULDER],
                lm[self.RIGHT_SHOULDER],
                lm[self.LEFT_HIP],
                lm[self.RIGHT_HIP],
            ]

            visible_pts = [p for p in pts if getattr(p, "visibility", 1.0) > 0.4]
            if len(visible_pts) >= 3:
                xs = [p.x * w for p in visible_pts]
                ys = [p.y * h for p in visible_pts]
                x1 = max(0, int(min(xs)) - TORSO_PADDING)
                x2 = min(w, int(max(xs)) + TORSO_PADDING)
                y1 = max(0, int(min(ys)) - TORSO_PADDING)
                y2 = min(h, int(max(ys)) + TORSO_PADDING)
                bbox = (x1, y1, x2, y2)

        return result, bbox

    def close(self):
        self.landmarker.close()


# --------------------------------------------------------------------------
# Dodgeball
# --------------------------------------------------------------------------

class Dodgeball:
    TRAIL_LENGTH = 7

    def __init__(self, x, speed, radius=BALL_RADIUS, vx=0.0):
        self.x = x
        self.y = -radius
        self.speed = speed
        self.radius = radius
        self.vx = vx            # sideways drift, px/sec (adds unpredictability)
        self.resolved = False   # True once it's been counted as dodge or hit
        self.trail = []         # recent (x, y) positions, oldest first

    def update(self, dt, width=None):
        self.trail.append((self.x, self.y))
        if len(self.trail) > self.TRAIL_LENGTH:
            self.trail.pop(0)

        self.y += self.speed * dt
        self.x += self.vx * dt

        # Bounce off the left/right edges instead of drifting off-screen,
        # so drift makes paths curvy and surprising rather than removing
        # balls from play early.
        if width is not None:
            if self.x - self.radius < 0:
                self.x = self.radius
                self.vx *= -1
            elif self.x + self.radius > width:
                self.x = width - self.radius
                self.vx *= -1

    def is_off_screen(self, height):
        return self.y - self.radius > height

    def draw(self, frame):
        # Comet trail: fading, shrinking circles behind the ball's current position
        n = len(self.trail)
        for i, (tx, ty) in enumerate(self.trail):
            t = (i + 1) / (n + 1)  # 0 (oldest/faintest) -> 1 (newest)
            trail_radius = max(1, int(self.radius * t * 0.8))
            trail_color = tuple(int(c * t * 0.6) for c in COLOR_BALL)
            cv2.circle(frame, (int(tx), int(ty)), trail_radius, trail_color, -1, cv2.LINE_AA)

        center = (int(self.x), int(self.y))
        cv2.circle(frame, center, self.radius, COLOR_BALL, -1)
        cv2.circle(frame, center, self.radius, COLOR_BALL_OUTLINE, 3)
        # simple highlight for a "3D" look
        hl = (int(self.x - self.radius * 0.35), int(self.y - self.radius * 0.35))
        cv2.circle(frame, hl, max(2, self.radius // 4), (120, 120, 255), -1)

    def collides_with_bbox(self, bbox):
        """Circle-vs-rectangle collision test."""
        x1, y1, x2, y2 = bbox
        closest_x = min(max(self.x, x1), x2)
        closest_y = min(max(self.y, y1), y2)
        dist_x = self.x - closest_x
        dist_y = self.y - closest_y
        return (dist_x * dist_x + dist_y * dist_y) <= (self.radius * self.radius)


# --------------------------------------------------------------------------
# Game
# --------------------------------------------------------------------------

class PhysiDodgeGame:
    def __init__(self):
        self.cap = cv2.VideoCapture(CAM_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

        if not self.cap.isOpened():
            raise RuntimeError(
                "Could not open webcam. Check CAM_INDEX or camera permissions."
            )

        self.tracker = PoseTracker()
        self.high_score = self._load_high_score()
        self.reset()

    # ---------------- high score persistence ----------------

    @staticmethod
    def _load_high_score():
        try:
            with open(HIGH_SCORE_PATH, "r") as f:
                data = json.load(f)
                return int(data.get("high_score", 0))
        except (FileNotFoundError, json.JSONDecodeError, ValueError, OSError):
            return 0

    def _save_high_score(self):
        try:
            with open(HIGH_SCORE_PATH, "w") as f:
                json.dump({"high_score": self.high_score}, f)
        except OSError:
            pass  # non-critical - just means the record won't persist

    def _trigger_sparkle_burst(self, w, h, now):
        """Spawns a burst of star sparkles flying in from both edges of the
        screen to celebrate a new high score."""
        per_side = SPARKLE_BURST_COUNT // 2
        for side in (-1, 1):  # -1 = enters from left edge, 1 = from right edge
            for _ in range(per_side):
                y = random.uniform(0.1 * h, 0.9 * h)
                x = -20 if side == -1 else w + 20
                speed = random.uniform(SPARKLE_MIN_SPEED, SPARKLE_MAX_SPEED)
                vx = speed if side == -1 else -speed
                vy = random.uniform(-80, 80)
                size = random.randint(5, 11)
                color = random.choice(SPARKLE_COLORS)
                life = random.uniform(SPARKLE_MIN_LIFE, SPARKLE_MAX_LIFE)
                self.sparkles.append(Sparkle(x, y, vx, vy, size, color, life))
        self.high_score_banner_until = now + HIGH_SCORE_BANNER_DURATION

    def _update_sparkles(self, dt):
        for s in self.sparkles:
            s.update(dt)
        self.sparkles = [s for s in self.sparkles if s.alive()]

    def _update_floating_texts(self, dt):
        for ft in self.floating_texts:
            ft.update(dt)
        self.floating_texts = [ft for ft in self.floating_texts if ft.alive()]

    def _apply_vignette(self, frame, w, h):
        """Darkens the frame edges slightly for a more cinematic look."""
        cache_key = (w, h)
        if getattr(self, "_vignette_cache_key", None) != cache_key:
            yy, xx = np.indices((h, w))
            cx, cy = w / 2.0, h / 2.0
            dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
            max_dist = math.sqrt(cx ** 2 + cy ** 2)
            norm = np.clip(dist / max_dist, 0, 1)
            mask = 1.0 - VIGNETTE_STRENGTH * (norm ** 2)
            self._vignette_mask = np.dstack([mask] * 3).astype(np.float32)
            self._vignette_cache_key = cache_key
        return (frame.astype(np.float32) * self._vignette_mask).astype(np.uint8)

    def _apply_shake(self, frame, w, h, now):
        remaining = self.shake_until - now
        strength = self.shake_magnitude * (remaining / SHAKE_DURATION)
        dx = random.uniform(-strength, strength)
        dy = random.uniform(-strength, strength)
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        return cv2.warpAffine(frame, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    # ---------------- state management ----------------

    def reset(self):
        self.balls = []
        self.score = 0
        self.lives = STARTING_LIVES
        self.game_over = False
        self.paused = False

        now = time.time()
        # The countdown doesn't begin until a torso is actually detected -
        # see run(). Until then these stay None/False.
        self.waiting_for_player = True
        self.start_time = None
        self.go_flash_until = None
        self.last_frame_time = now
        self.last_spawn_time = None

        self.flash_until = 0.0
        self.last_bbox = None
        self.sparkles = []
        self.new_high_triggered = False
        self.high_score_banner_until = 0.0

        self.combo = 0
        self.floating_texts = []
        self.shake_until = 0.0
        self.shake_magnitude = 0.0

    def survival_time(self):
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def current_spawn_interval(self):
        t = self.survival_time()
        interval = BASE_SPAWN_INTERVAL - SPAWN_RAMP * t
        return max(MIN_SPAWN_INTERVAL, interval)

    def current_ball_speed(self):
        t = self.survival_time()
        speed = BASE_BALL_SPEED + SPEED_RAMP * t
        return min(MAX_BALL_SPEED, speed)

    def current_drift_range(self):
        t = self.survival_time()
        drift = BASE_DRIFT_RANGE + DRIFT_RAMP * t
        return min(MAX_DRIFT_RANGE, drift)

    def current_double_spawn_chance(self):
        t = self.survival_time()
        chance = BASE_DOUBLE_SPAWN_CHANCE + DOUBLE_SPAWN_RAMP * t
        return min(MAX_DOUBLE_SPAWN_CHANCE, chance)

    # ---------------- main loop ----------------

    def run(self):
        try:
            while True:
                ok, frame = self.cap.read()
                if not ok:
                    break

                frame = cv2.flip(frame, 1)  # mirror mode
                h, w = frame.shape[:2]
                frame = self._apply_vignette(frame, w, h)

                now = time.time()
                dt = now - self.last_frame_time
                self.last_frame_time = now

                results, bbox = self.tracker.process(frame)
                if bbox is not None:
                    self.last_bbox = bbox

                # The countdown only begins once a torso is actually seen -
                # this way nobody gets hit while still stepping into frame,
                # and it works regardless of how long that takes.
                if self.waiting_for_player:
                    if bbox is not None:
                        self.waiting_for_player = False
                        self.start_time = now + READY_COUNTDOWN_SECONDS
                        self.go_flash_until = self.start_time + GO_FLASH_DURATION
                        self.last_spawn_time = self.start_time

                in_countdown = (not self.waiting_for_player) and now < self.start_time

                if not self.paused and not self.game_over and not self.waiting_for_player and not in_countdown:
                    self._update_game(dt, now, bbox, w, h)

                if not self.paused:
                    self._update_sparkles(dt)
                    self._update_floating_texts(dt)

                self._draw(frame, bbox, w, h, now, self.waiting_for_player, in_countdown)

                if now < self.shake_until:
                    frame = self._apply_shake(frame, w, h, now)

                cv2.imshow("PhysiDodge", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('r') and self.game_over:
                    self.reset()
                elif key == ord(' '):
                    self.paused = not self.paused
        finally:
            self.cap.release()
            self.tracker.close()
            cv2.destroyAllWindows()

    # ---------------- update ----------------

    def _update_game(self, dt, now, bbox, w, h):
        # Spawn new balls - faster and faster, with a growing chance of a
        # second ball landing at the same time as difficulty rises.
        if now - self.last_spawn_time >= self.current_spawn_interval():
            self.last_spawn_time = now
            self._spawn_ball(w)
            if random.random() < self.current_double_spawn_chance():
                self._spawn_ball(w)

        # Update balls, check collisions
        surviving_balls = []
        for ball in self.balls:
            ball.update(dt, width=w)

            # Only allow a hit once the ball has actually entered the visible
            # frame - otherwise a torso box that touches the top edge can
            # "hit" balls the instant they spawn, before the player ever
            # sees them.
            ball_fully_visible = (ball.y - ball.radius) >= 0
            if ball_fully_visible and not ball.resolved and bbox is not None and ball.collides_with_bbox(bbox):
                ball.resolved = True
                self._register_hit(now)
                continue  # remove ball on hit

            if ball.is_off_screen(h):
                if not ball.resolved:
                    self._register_dodge(ball, w, h, now)
                continue  # remove ball once off screen

            surviving_balls.append(ball)

        self.balls = surviving_balls

    def _register_dodge(self, ball, w, h, now):
        self.combo += 1
        bonus = self.combo // COMBO_MILESTONE
        points = 1 + bonus
        self.score += points

        popup_color = (120, 255, 120) if bonus == 0 else COLOR_GOLD
        self.floating_texts.append(
            FloatingText(ball.x, h - 60, f"+{points}", popup_color, scale=0.75, life=0.6)
        )

        if self.combo > 0 and self.combo % COMBO_MILESTONE == 0:
            self.floating_texts.append(
                FloatingText(w / 2 - 90, h / 2, f"COMBO x{self.combo}!", COLOR_GOLD,
                             scale=1.3, life=1.0, vy=-40)
            )
            play_beeps([(700, 70), (900, 90)])

        if self.score > self.high_score:
            self.high_score = self.score
            self._save_high_score()
            if not self.new_high_triggered:
                self.new_high_triggered = True
                self._trigger_sparkle_burst(w, h, now)
                play_beeps([(1000, 100), (1300, 140)])

    def _spawn_ball(self, w):
        """Creates one dodgeball with randomized speed, size, and drift -
        the ranges widen the longer the player has survived."""
        radius = int(BALL_RADIUS * random.uniform(0.75, 1.35))
        radius = max(MIN_BALL_RADIUS, min(MAX_BALL_RADIUS, radius))

        x = random.randint(radius + 10, max(radius + 11, w - radius - 10))

        speed = self.current_ball_speed() * random.uniform(*SPEED_JITTER)

        drift_range = self.current_drift_range()
        vx = random.uniform(-drift_range, drift_range)

        self.balls.append(Dodgeball(x, speed, radius=radius, vx=vx))

    def _register_hit(self, now):
        self.lives -= 1
        self.flash_until = now + HIT_FLASH_DURATION
        self.shake_until = now + SHAKE_DURATION
        self.shake_magnitude = SHAKE_MAGNITUDE
        self.combo = 0

        if self.last_bbox is not None:
            cx = (self.last_bbox[0] + self.last_bbox[2]) / 2
            cy = (self.last_bbox[1] + self.last_bbox[3]) / 2
            self.floating_texts.append(
                FloatingText(cx - 40, cy, "OUCH!", (60, 60, 255), scale=1.1, life=0.6, vy=-90)
            )

        if self.lives <= 0:
            self.game_over = True
            play_beeps([(300, 150), (150, 300)])
        else:
            play_beeps([(200, 130)])

    # ---------------- drawing ----------------

    def _draw(self, frame, bbox, w, h, now, waiting, in_countdown):
        # Hit flash overlay
        if now < self.flash_until:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), COLOR_FLASH, -1)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # Torso hitbox - targeting-style corner brackets rather than a full box
        draw_bbox = bbox if bbox is not None else self.last_bbox
        if draw_bbox is not None:
            color = COLOR_HITBOX if bbox is not None else COLOR_HITBOX_MISSING
            draw_bracket_box(frame, draw_bbox, color)

        if bbox is None:
            draw_text_shadow(
                frame, "Step into frame - full torso needed",
                (30, h - 30), scale=0.75, color=(0, 190, 255), thickness=2
            )

        # Balls
        for ball in self.balls:
            ball.draw(frame)

        # Sparkles and floating text (drawn above balls for clear feedback)
        for s in self.sparkles:
            s.draw(frame)
        for ft in self.floating_texts:
            ft.draw(frame)

        self._draw_hud(frame, w, h, now)

        if not waiting:
            self._draw_intensity_bar(frame, w, h)

        if self.high_score_banner_until and now < self.high_score_banner_until:
            self._draw_high_score_banner(frame, w)

        if waiting:
            self._draw_waiting_prompt(frame, w, h, now)
        elif in_countdown:
            self._draw_countdown(frame, w, h, now)
        elif now < self.go_flash_until:
            self._draw_go_flash(frame, w, h, now)

        if self.paused and not self.game_over:
            self._draw_center_panel(frame, w, h, [("PAUSED", 1.6, COLOR_TEXT)])

        if self.game_over:
            is_new_best = self.score >= self.high_score and self.score > 0
            lines = [
                ("GAME OVER", 1.8, (70, 70, 255)),
                (f"Score: {self.score}", 1.0, COLOR_TEXT),
                (f"High Score: {self.high_score}" + ("  \u2605 NEW BEST!" if is_new_best else ""),
                 0.8, COLOR_GOLD if is_new_best else COLOR_TEXT),
                ("Press R to restart, Q to quit", 0.7, (180, 180, 180)),
            ]
            self._draw_center_panel(frame, w, h, lines)

    def _draw_waiting_prompt(self, frame, w, h, now):
        pulse = 0.65 + 0.35 * math.sin(now * 4)
        text = "STEP INTO FRAME TO BEGIN"
        scale = 1.3
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 3)
        x = (w - text_w) // 2
        y = h // 2

        pad = 24
        draw_rounded_panel(frame, (x - pad, y - text_h - pad),
                            (x + text_w + pad, y + pad), (20, 20, 20), radius=16, alpha=0.55)

        color = tuple(int(c * pulse) for c in (80, 220, 255))
        draw_text_shadow(frame, text, (x, y), scale=scale, color=color, thickness=3)
        draw_text_shadow(frame, "Make sure your shoulders and hips are both visible",
                          (w // 2 - 260, y + 45), scale=0.65, color=COLOR_TEXT, thickness=2)

    def _draw_countdown(self, frame, w, h, now):
        remaining = self.start_time - now
        n = max(1, int(math.ceil(remaining)))
        text = str(n)
        pulse = 1.0 + 0.3 * (remaining - int(remaining))
        scale = 3.2 * pulse
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 6)
        x = (w - text_w) // 2
        y = h // 2 + text_h // 2
        draw_text_shadow(frame, text, (x, y), scale=scale, color=COLOR_GOLD, thickness=6)
        draw_text_shadow(frame, "Get ready to dodge...", (w // 2 - 150, h // 2 + 90),
                          scale=0.8, color=COLOR_TEXT, thickness=2)

    def _draw_go_flash(self, frame, w, h, now):
        remaining = self.go_flash_until - now
        alpha = max(0.0, remaining / GO_FLASH_DURATION)
        text = "GO!"
        scale = 3.0
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 6)
        x = (w - text_w) // 2
        y = h // 2
        color = tuple(int(c * alpha) for c in (80, 255, 120))
        draw_text_shadow(frame, text, (x, y), scale=scale, color=color, thickness=6)

    def _draw_intensity_bar(self, frame, w, h):
        """A thin bar showing how close the difficulty ramp is to maxing out."""
        progress = min(1.0, max(0.0, self.survival_time()) / INTENSITY_PLATEAU_SECONDS)
        bar_w, bar_h = 220, 14
        x1, y1 = w - bar_w - 26, 26
        x2, y2 = x1 + bar_w, y1 + bar_h

        draw_rounded_panel(frame, (x1 - 8, y1 - 26), (x2 + 8, y2 + 8), COLOR_PANEL_BG,
                            radius=10, alpha=0.55)
        draw_text_shadow(frame, "INTENSITY", (x1, y1 - 8), scale=0.55, thickness=1,
                          color=(200, 200, 200))

        cv2.rectangle(frame, (x1, y1), (x2, y2), (70, 70, 70), -1)
        fill_w = int(bar_w * progress)
        # green -> gold -> red as it fills
        bar_color = (
            int(60 + 140 * progress),
            int(220 - 60 * progress),
            int(60 + 195 * progress),
        )
        if fill_w > 0:
            cv2.rectangle(frame, (x1, y1), (x1 + fill_w, y2), bar_color, -1)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (200, 200, 200), 1, cv2.LINE_AA)

    def _draw_hud(self, frame, w, h, now):
        """Top-left status panel: score, high score, lives, combo, time."""
        panel_w, panel_h = 340, 200
        draw_rounded_panel(frame, (14, 14), (14 + panel_w, 14 + panel_h),
                            COLOR_PANEL_BG, radius=16, alpha=0.55)
        cv2.rectangle(frame, (14, 14), (14 + panel_w, 14 + panel_h), COLOR_ACCENT, 1, cv2.LINE_AA)

        pad_x = 30
        draw_text_shadow(frame, f"Score  {self.score}", (pad_x, 52), scale=0.95, thickness=2)
        draw_text_shadow(frame, f"Best   {self.high_score}", (pad_x, 82), scale=0.7,
                          color=COLOR_GOLD, thickness=2)
        draw_text_shadow(frame, f"Best   {self.high_score}", (pad_x, 82), scale=0.7,
                          color=COLOR_GOLD, thickness=2)

        lives_label = "Lives"
        draw_text_shadow(frame, lives_label, (pad_x, 122), scale=0.8, thickness=2)
        (label_w, _), _ = cv2.getTextSize(lives_label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)

        heart_size = 26
        heart_spacing = 34
        heart_y = 112
        heart_start_x = pad_x + label_w + 18
        for i in range(STARTING_LIVES):
            cx = heart_start_x + i * heart_spacing
            filled = i < self.lives
            color = COLOR_HEART_FULL if filled else COLOR_HEART_EMPTY
            draw_heart(frame, (cx, heart_y), heart_size, color, filled=filled)

        combo_color = COLOR_GOLD if self.combo >= COMBO_MILESTONE else (200, 200, 200)
        draw_text_shadow(frame, f"Combo  x{self.combo}", (pad_x, 158), scale=0.7,
                          color=combo_color, thickness=2)

        display_time = max(0.0, self.survival_time())
        draw_text_shadow(frame, f"Time  {display_time:.1f}s", (pad_x, 188),
                          scale=0.65, color=(200, 200, 200), thickness=1)

    def _draw_high_score_banner(self, frame, w):
        remaining = self.high_score_banner_until - time.time()
        pulse = 0.7 + 0.3 * math.sin(time.time() * 10)
        text = "NEW HIGH SCORE!"
        scale = 1.3
        thickness = 3
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, thickness)
        x = (w - text_w) // 2
        y = 60

        pad = 18
        alpha = min(1.0, remaining / 0.4) if remaining < 0.4 else 1.0
        draw_rounded_panel(frame, (x - pad, y - text_h - pad),
                            (x + text_w + pad, y + pad), (20, 20, 20),
                            radius=14, alpha=0.5 * alpha)

        glow_color = tuple(int(c * pulse) for c in COLOR_GOLD)
        draw_text_shadow(frame, text, (x, y), scale=scale, color=glow_color,
                          thickness=thickness, shadow_color=(0, 0, 0))

    def _draw_center_panel(self, frame, w, h, lines):
        """Draws a dark rounded panel centered on screen with stacked lines
        of (text, scale, color) tuples."""
        line_specs = []
        total_h = 0
        for text, scale, color in lines:
            (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, scale, 2)
            line_specs.append((text, scale, color, text_w, text_h))
            total_h += text_h + 26

        panel_w = max(tw for _, _, _, tw, _ in line_specs) + 100
        panel_h = total_h + 50
        cx, cy = w // 2, h // 2
        x1, y1 = cx - panel_w // 2, cy - panel_h // 2
        x2, y2 = cx + panel_w // 2, cy + panel_h // 2

        draw_rounded_panel(frame, (x1, y1), (x2, y2), (15, 15, 15), radius=24, alpha=0.72)
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ACCENT, 2, cv2.LINE_AA)

        y = y1 + 45
        for text, scale, color, text_w, text_h in line_specs:
            x = cx - text_w // 2
            draw_text_shadow(frame, text, (x, y), scale=scale, color=color, thickness=2)
            y += text_h + 26


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    game = PhysiDodgeGame()
    game.run()


if __name__ == "__main__":
    main()

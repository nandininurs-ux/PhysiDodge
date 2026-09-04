"""
PhysiDodge - A controller-free, full-body arcade dodging game.

Uses a webcam + MediaPipe Pose to track your torso in real time, then
challenges you to physically dodge falling "dodgeballs" by stepping,
ducking, or swaying out of the way.

Controls:
    Q       - Quit
    R       - Restart after Game Over
    SPACE   - Pause / Resume

Run:
    python physidodge.py
"""

import os
import time
import random
import urllib.request

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision


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
COLOR_HITBOX = (0, 255, 0)
COLOR_HITBOX_MISSING = (0, 165, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_FLASH = (0, 0, 255)
COLOR_HEART_FULL = (60, 40, 235)
COLOR_HEART_EMPTY = (90, 90, 90)


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
    def __init__(self, x, speed, radius=BALL_RADIUS, vx=0.0):
        self.x = x
        self.y = -radius
        self.speed = speed
        self.radius = radius
        self.vx = vx            # sideways drift, px/sec (adds unpredictability)
        self.resolved = False   # True once it's been counted as dodge or hit

    def update(self, dt, width=None):
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
        self.reset()

    # ---------------- state management ----------------

    def reset(self):
        self.balls = []
        self.score = 0
        self.lives = STARTING_LIVES
        self.game_over = False
        self.paused = False
        self.start_time = time.time()
        self.last_frame_time = self.start_time
        self.last_spawn_time = self.start_time
        self.flash_until = 0.0
        self.last_bbox = None

    def survival_time(self):
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
                now = time.time()
                dt = now - self.last_frame_time
                self.last_frame_time = now

                results, bbox = self.tracker.process(frame)
                if bbox is not None:
                    self.last_bbox = bbox

                if not self.paused and not self.game_over:
                    self._update_game(dt, now, bbox, w, h)

                self._draw(frame, bbox, w, h)

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
            # frame - otherwise a torso box that touches the top edge (e.g.
            # when standing very close to the camera) can "hit" balls the
            # instant they spawn, before the player ever sees them.
            ball_fully_visible = (ball.y - ball.radius) >= 0
            if (
                not ball.resolved
                and bbox is not None
                and ball_fully_visible
                and ball.collides_with_bbox(bbox)
            ):
                ball.resolved = True
                self._register_hit(now)
                continue  # remove ball on hit

            if ball.is_off_screen(h):
                if not ball.resolved:
                    self.score += 1  # successful dodge
                continue  # remove ball once off screen

            surviving_balls.append(ball)

        self.balls = surviving_balls

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
        if self.lives <= 0:
            self.game_over = True

    # ---------------- drawing ----------------

    def _draw(self, frame, bbox, w, h):
        # Hit flash overlay
        if time.time() < self.flash_until:
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (w, h), COLOR_FLASH, -1)
            cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

        # Torso hitbox
        draw_bbox = bbox if bbox is not None else self.last_bbox
        if draw_bbox is not None:
            color = COLOR_HITBOX if bbox is not None else COLOR_HITBOX_MISSING
            x1, y1, x2, y2 = draw_bbox
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)

        if bbox is None:
            cv2.putText(
                frame, "Step into frame - full torso needed",
                (30, h - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2
            )

        # Balls
        for ball in self.balls:
            ball.draw(frame)

        # HUD
        cv2.putText(frame, f"Score: {self.score}", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, COLOR_TEXT, 2)

        lives_label = "Lives:"
        (label_w, _), _ = cv2.getTextSize(lives_label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
        cv2.putText(frame, lives_label, (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.1, COLOR_TEXT, 2)

        heart_size = 28
        heart_spacing = 36
        heart_y = 78
        heart_start_x = 30 + label_w + 20
        for i in range(STARTING_LIVES):
            cx = heart_start_x + i * heart_spacing
            filled = i < self.lives
            color = COLOR_HEART_FULL if filled else COLOR_HEART_EMPTY
            draw_heart(frame, (cx, heart_y), heart_size, color, filled=filled)

        cv2.putText(frame, f"Time: {self.survival_time():.1f}s", (30, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_TEXT, 2)

        if self.paused and not self.game_over:
            self._center_text(frame, "PAUSED", w, h, scale=2.0)

        if self.game_over:
            self._center_text(frame, "GAME OVER", w, h // 2 - 40, scale=2.2)
            self._center_text(frame, f"Final Score: {self.score}", w, h // 2 + 20, scale=1.2)
            self._center_text(frame, "Press R to restart, Q to quit", w, h // 2 + 70, scale=0.9)

    @staticmethod
    def _center_text(frame, text, w, y, scale=1.0, color=COLOR_TEXT, thickness=3):
        (text_w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        x = (w - text_w) // 2
        cv2.putText(frame, text, (x, int(y) if isinstance(y, (int, float)) else y),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def main():
    game = PhysiDodgeGame()
    game.run()


if __name__ == "__main__":
    main()
"""Comprehensive on-screen visual overlay (HUD) for camera test, recording, and replay."""

from __future__ import annotations

from typing import List
import cv2
import numpy as np

from motion_mentor.storage.models import HandLandmarkData
from motion_mentor.tracking.hand_tracker import draw_hand_skeleton


class HUDOverlay:
    """Renders capture zones, telemetry metrics, countdowns, and recording banners."""

    def __init__(
        self,
        show_skeleton: bool = True,
        show_telemetry: bool = True,
        show_capture_zone: bool = True,
        capture_zone_padding: float = 0.12,
    ) -> None:
        self.show_skeleton = show_skeleton
        self.show_telemetry = show_telemetry
        self.show_capture_zone = show_capture_zone
        self.padding = capture_zone_padding

    def render(
        self,
        frame: np.ndarray,
        hands: List[HandLandmarkData],
        fps: float = 0.0,
        latency_ms: float = 0.0,
        dropped_frames: int = 0,
        status: str = "preview",  # "preview", "countdown", "recording", "replaying"
        countdown_sec: int = 0,
        recording_time_sec: float = 0.0,
        activity_title: str = "",
        role: str = "",
    ) -> np.ndarray:
        """Render complete HUD layer onto the frame."""
        h, w = frame.shape[:2]

        # 1. Capture zone
        if self.show_capture_zone:
            self._draw_capture_zone(frame, w, h)

        # 2. Hand skeletons
        if self.show_skeleton and hands:
            draw_hand_skeleton(frame, hands, show_labels=True)

        # 3. Telemetry card (top-left)
        if self.show_telemetry:
            self._draw_telemetry_card(frame, fps, latency_ms, len(hands), dropped_frames)

        # 4. Status badge / banner
        if status == "recording":
            self._draw_recording_banner(frame, recording_time_sec, w, activity_title, role)
        elif status == "countdown":
            self._draw_countdown_overlay(frame, countdown_sec, w, h)
        elif status == "replaying":
            self._draw_replay_banner(frame, recording_time_sec, w)

        return frame

    def _draw_capture_zone(self, frame: np.ndarray, w: int, h: int) -> None:
        x1 = int(w * self.padding)
        y1 = int(h * self.padding)
        x2 = int(w * (1.0 - self.padding))
        y2 = int(h * (1.0 - self.padding))

        # Thin bounding rectangle with dashed appearance or corner brackets
        corner_len = 30
        bracket_color = (0, 200, 255)
        thickness = 2

        # Draw semi-transparent box outline
        cv2.rectangle(frame, (x1, y1), (x2, y2), (60, 60, 60), 1, cv2.LINE_AA)

        # Top-left corner
        cv2.line(frame, (x1, y1), (x1 + corner_len, y1), bracket_color, thickness)
        cv2.line(frame, (x1, y1), (x1, y1 + corner_len), bracket_color, thickness)
        # Top-right corner
        cv2.line(frame, (x2, y1), (x2 - corner_len, y1), bracket_color, thickness)
        cv2.line(frame, (x2, y1), (x2, y1 + corner_len), bracket_color, thickness)
        # Bottom-left corner
        cv2.line(frame, (x1, y2), (x1 + corner_len, y2), bracket_color, thickness)
        cv2.line(frame, (x1, y2), (x1, y2 - corner_len), bracket_color, thickness)
        # Bottom-right corner
        cv2.line(frame, (x2, y2), (x2 - corner_len, y2), bracket_color, thickness)
        cv2.line(frame, (x2, y2), (x2, y2 - corner_len), bracket_color, thickness)

        # Label at bottom of zone
        label = "CAPTURE ZONE"
        cv2.putText(
            frame,
            label,
            (x1 + 10, y2 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )

    def _draw_telemetry_card(
        self,
        frame: np.ndarray,
        fps: float,
        latency_ms: float,
        hand_count: int,
        dropped: int,
    ) -> None:
        # Dark glass panel top-left
        px, py, pw, ph = 20, 20, 260, 115
        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + pw, py + ph), (18, 22, 28), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (px, py), (px + pw, py + ph), (60, 70, 85), 1, cv2.LINE_AA)

        # Title
        cv2.putText(
            frame,
            "MOTIONMENTOR TELEMETRY",
            (px + 12, py + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 210, 225),
            1,
            cv2.LINE_AA,
        )

        # FPS indicator
        fps_color = (80, 240, 120) if fps >= 24.0 else (80, 120, 240)
        cv2.putText(
            frame,
            f"FPS: {fps:4.1f}",
            (px + 12, py + 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            fps_color,
            1,
            cv2.LINE_AA,
        )

        # Latency indicator
        lat_color = (80, 240, 120) if latency_ms < 35.0 else (80, 160, 240)
        cv2.putText(
            frame,
            f"Latency: {latency_ms:4.1f} ms",
            (px + 12, py + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            lat_color,
            1,
            cv2.LINE_AA,
        )

        # Hands & drops
        hand_color = (80, 240, 120) if hand_count > 0 else (120, 120, 140)
        cv2.putText(
            frame,
            f"Hands: {hand_count}  |  Drops: {dropped}",
            (px + 12, py + 94),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            hand_color,
            1,
            cv2.LINE_AA,
        )

    def _draw_recording_banner(
        self,
        frame: np.ndarray,
        time_sec: float,
        width: int,
        activity_title: str,
        role: str,
    ) -> None:
        # Centered recording pill at top
        pill_w = 340
        pill_h = 42
        px = (width - pill_w) // 2
        py = 20

        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + pill_w, py + pill_h), (20, 15, 30), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (px, py), (px + pill_w, py + pill_h), (0, 0, 240), 2, cv2.LINE_AA)

        # Blinking red dot
        blink = int(time_sec * 2) % 2 == 0
        dot_color = (0, 0, 255) if blink else (80, 60, 100)
        cv2.circle(frame, (px + 22, py + 21), 8, dot_color, -1, cv2.LINE_AA)

        # Timer format MM:SS.S
        mins = int(time_sec // 60)
        secs = time_sec % 60
        timer_str = f"REC {mins:02d}:{secs:04.1f}"
        cv2.putText(
            frame,
            timer_str,
            (px + 40, py + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        role_str = f"[{role.upper()}]" if role else ""
        cv2.putText(
            frame,
            role_str,
            (px + 230, py + 27),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 220, 255),
            1,
            cv2.LINE_AA,
        )

    def _draw_countdown_overlay(
        self,
        frame: np.ndarray,
        countdown_sec: int,
        width: int,
        height: int,
    ) -> None:
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

        num_str = str(countdown_sec) if countdown_sec > 0 else "START!"
        # Large glowing number
        cx = width // 2
        cy = height // 2

        cv2.putText(
            frame,
            "GET READY",
            (cx - 110, cy - 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.1,
            (200, 220, 255),
            2,
            cv2.LINE_AA,
        )

        font_scale = 3.5 if countdown_sec > 0 else 2.2
        text_offset_x = 40 if countdown_sec > 0 else 120
        cv2.putText(
            frame,
            num_str,
            (cx - text_offset_x, cy + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 0, 0),
            12,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            num_str,
            (cx - text_offset_x, cy + 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 240, 255),
            5,
            cv2.LINE_AA,
        )

    def _draw_replay_banner(
        self,
        frame: np.ndarray,
        time_sec: float,
        width: int,
    ) -> None:
        pill_w = 260
        pill_h = 38
        px = (width - pill_w) // 2
        py = 20

        overlay = frame.copy()
        cv2.rectangle(overlay, (px, py), (px + pill_w, py + pill_h), (25, 25, 30), -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
        cv2.rectangle(frame, (px, py), (px + pill_w, py + pill_h), (0, 220, 200), 2, cv2.LINE_AA)

        mins = int(time_sec // 60)
        secs = time_sec % 60
        timer_str = f"REPLAY  {mins:02d}:{secs:04.1f}"
        cv2.putText(
            frame,
            timer_str,
            (px + 20, py + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 240, 220),
            2,
            cv2.LINE_AA,
        )

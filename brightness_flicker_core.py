import math
import os
import re
import shutil
import subprocess
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence

import cv2
import numpy as np


DEFAULT_SEQUENCE_FPS = 24.0
ProgressCallback = Optional[Callable[[int, int], None]]


@dataclass(frozen=True)
class GlobalFlickerSettings:
    threshold: float = 185.0
    softness: float = 35.0
    amount: float = 0.28
    glow: float = 0.22
    pulse_duration: int = 1
    seed: int = 7


@dataclass(frozen=True)
class IndependentFlickerSettings:
    threshold: float = 185.0
    softness: float = 35.0
    amount: float = 0.30
    glow: float = 0.22
    pulse_duration: int = 1
    min_area: int = 100
    max_distance: float = 60.0
    max_missed: int = 8
    seed: int = 13


@dataclass
class Track:
    track_id: int
    centroid: tuple[float, float]
    missed_frames: int
    amplitude_scale: float


def smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def build_bright_mask(frame_bgr: np.ndarray, threshold: float, softness: float) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    start = threshold
    end = threshold + max(softness, 1.0)
    mask = smoothstep((gray - start) / (end - start))
    mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=5.0, sigmaY=5.0)
    return np.clip(mask, 0.0, 1.0)


def clamp_pulse_duration(pulse_duration: int) -> int:
    return max(1, int(pulse_duration))


def loop_phase_array(frame_count: int) -> np.ndarray:
    if frame_count <= 0:
        return np.zeros(0, dtype=np.float32)
    return (np.arange(frame_count, dtype=np.float32) + 1.0) / float(frame_count)


def loop_phase_at(frame_index: int, frame_count: int) -> float:
    if frame_count <= 0:
        return 1.0
    return ((frame_index % frame_count) + 1.0) / float(frame_count)


def make_pulse_envelope(phases: np.ndarray, pulse_duration: int) -> np.ndarray:
    duration = clamp_pulse_duration(pulse_duration)
    return (-np.cos(2.0 * math.pi * duration * phases)).astype(np.float32)


def pulse_value_at_phase(loop_phase: float, pulse_duration: int) -> float:
    duration = clamp_pulse_duration(pulse_duration)
    return float(-math.cos(2.0 * math.pi * duration * loop_phase))


def make_flicker_curve(frame_count: int, amount: float, pulse_duration: int) -> np.ndarray:
    if frame_count <= 0:
        return np.zeros(0, dtype=np.float32)

    phases = loop_phase_array(frame_count)
    curve = make_pulse_envelope(phases, pulse_duration)
    return (curve * amount).astype(np.float32)


def apply_global_flicker(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    flicker_value: float,
    glow_strength: float,
) -> np.ndarray:
    frame = frame_bgr.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]

    darken = 1.0 + min(flicker_value, 0.0)
    brighten = max(flicker_value, 0.0)
    value *= 1.0 + mask * (darken - 1.0)
    value += mask * brighten * (0.8 + 0.2 * value)
    hsv[:, :, 2] = np.clip(value, 0.0, 1.0)

    flickered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    if brighten > 0.0 and glow_strength > 0.0:
        blur = cv2.GaussianBlur(flickered, (0, 0), sigmaX=10.0, sigmaY=10.0)
        glow = blur * (mask[:, :, None] * brighten * glow_strength)
        flickered = np.clip(flickered + glow, 0.0, 1.0)

    return (flickered * 255.0).astype(np.uint8)


class GlobalFlickerProcessor:
    def __init__(self, frame_count: int, settings: GlobalFlickerSettings):
        self.settings = settings
        self.flicker_curve = make_flicker_curve(
            frame_count,
            settings.amount,
            settings.pulse_duration,
        )

    def process_frame(self, frame_bgr: np.ndarray, frame_index: int) -> np.ndarray:
        mask = build_bright_mask(frame_bgr, self.settings.threshold, self.settings.softness)
        flicker_value = float(self.flicker_curve[frame_index]) if frame_index < len(self.flicker_curve) else 0.0
        return apply_global_flicker(frame_bgr, mask, flicker_value, self.settings.glow)


def build_brightness_mask(frame_bgr: np.ndarray, threshold: float, softness: float) -> np.ndarray:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    start = threshold
    end = threshold + max(softness, 1.0)
    mask = smoothstep((gray - start) / (end - start))
    return cv2.GaussianBlur(mask, (0, 0), sigmaX=4.0, sigmaY=4.0)


def detect_light_regions(mask: np.ndarray, min_area: int) -> list[dict]:
    binary = (mask > 0.35).astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    binary = cv2.dilate(binary, np.ones((3, 3), np.uint8), iterations=1)

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    regions: list[dict] = []
    for label in range(1, count):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue

        component_mask = (labels == label).astype(np.float32)
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        w = int(stats[label, cv2.CC_STAT_WIDTH])
        h = int(stats[label, cv2.CC_STAT_HEIGHT])
        region_mask = component_mask[y : y + h, x : x + w] * mask[y : y + h, x : x + w]
        region_mask = cv2.GaussianBlur(region_mask, (0, 0), sigmaX=3.0, sigmaY=3.0)
        regions.append(
            {
                "centroid": (float(centroids[label][0]), float(centroids[label][1])),
                "bbox": (x, y, w, h),
                "mask": np.clip(region_mask, 0.0, 1.0),
            }
        )
    return regions


def create_track(track_id: int, centroid: tuple[float, float], rng: np.random.Generator) -> Track:
    return Track(
        track_id=track_id,
        centroid=centroid,
        missed_frames=0,
        amplitude_scale=float(rng.uniform(0.9, 1.1)),
    )


def update_tracks(
    tracks: dict[int, Track],
    regions: list[dict],
    next_track_id: int,
    max_distance: float,
    max_missed: int,
    rng: np.random.Generator,
) -> tuple[list[tuple[Track, dict]], int]:
    pairs: list[tuple[Track, dict]] = []
    unmatched_tracks = set(tracks.keys())
    unmatched_regions = set(range(len(regions)))

    distances: list[tuple[float, int, int]] = []
    for track_id, track in tracks.items():
        for region_index, region in enumerate(regions):
            dx = track.centroid[0] - region["centroid"][0]
            dy = track.centroid[1] - region["centroid"][1]
            distance = math.hypot(dx, dy)
            if distance <= max_distance:
                distances.append((distance, track_id, region_index))

    for _, track_id, region_index in sorted(distances, key=lambda item: item[0]):
        if track_id not in unmatched_tracks or region_index not in unmatched_regions:
            continue
        track = tracks[track_id]
        region = regions[region_index]
        track.centroid = region["centroid"]
        track.missed_frames = 0
        pairs.append((track, region))
        unmatched_tracks.remove(track_id)
        unmatched_regions.remove(region_index)

    for track_id in list(unmatched_tracks):
        track = tracks[track_id]
        track.missed_frames += 1
        if track.missed_frames > max_missed:
            del tracks[track_id]

    for region_index in unmatched_regions:
        region = regions[region_index]
        track = create_track(next_track_id, region["centroid"], rng)
        tracks[next_track_id] = track
        pairs.append((track, region))
        next_track_id += 1

    return pairs, next_track_id


def sample_flicker(track: Track, loop_phase: float, amount: float, pulse_duration: int) -> float:
    pulse_value = pulse_value_at_phase(loop_phase, pulse_duration)
    flicker = pulse_value * amount * track.amplitude_scale
    return float(np.clip(flicker, -0.55, 0.55))


def apply_region_flicker(
    frame_bgr: np.ndarray,
    pairs: list[tuple[Track, dict]],
    loop_phase: float,
    amount: float,
    glow_strength: float,
    pulse_duration: int,
) -> np.ndarray:
    frame = frame_bgr.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    glow_accum = np.zeros_like(frame)

    for track, region in pairs:
        flicker_value = sample_flicker(track, loop_phase, amount, pulse_duration)
        x, y, w, h = region["bbox"]
        mask = region["mask"]

        roi_value = value[y : y + h, x : x + w]
        darken = 1.0 + min(flicker_value, 0.0)
        brighten = max(flicker_value, 0.0)
        roi_value *= 1.0 + mask * (darken - 1.0)
        roi_value += mask * brighten * (0.85 + 0.15 * roi_value)
        value[y : y + h, x : x + w] = np.clip(roi_value, 0.0, 1.0)

        if brighten > 0.0 and glow_strength > 0.0:
            glow_accum[y : y + h, x : x + w] += mask[:, :, None] * brighten * glow_strength

    hsv[:, :, 2] = value
    flickered = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    if glow_strength > 0.0:
        blur = cv2.GaussianBlur(flickered, (0, 0), sigmaX=10.0, sigmaY=10.0)
        flickered = np.clip(flickered + blur * glow_accum, 0.0, 1.0)

    return (flickered * 255.0).astype(np.uint8)


class IndependentFlickerProcessor:
    def __init__(self, frame_count: int, settings: IndependentFlickerSettings):
        self.frame_count = max(frame_count, 1)
        self.settings = settings
        self.rng = np.random.default_rng(settings.seed)
        self.tracks: dict[int, Track] = {}
        self.next_track_id = 1

    def process_frame(self, frame_bgr: np.ndarray, frame_index: int) -> np.ndarray:
        brightness_mask = build_brightness_mask(
            frame_bgr,
            self.settings.threshold,
            self.settings.softness,
        )
        regions = detect_light_regions(brightness_mask, self.settings.min_area)
        pairs, self.next_track_id = update_tracks(
            self.tracks,
            regions,
            self.next_track_id,
            self.settings.max_distance,
            self.settings.max_missed,
            self.rng,
        )
        loop_phase = loop_phase_at(frame_index, self.frame_count)
        return apply_region_flicker(
            frame_bgr,
            pairs,
            loop_phase,
            self.settings.amount,
            self.settings.glow,
            self.settings.pulse_duration,
        )


def _call_progress(progress_callback: ProgressCallback, processed: int, total: int) -> None:
    if progress_callback is not None:
        progress_callback(processed, total)


def process_global_image_sequence(
    frames_bgr: Sequence[np.ndarray],
    settings: GlobalFlickerSettings,
    fps: float = DEFAULT_SEQUENCE_FPS,
    progress_callback: ProgressCallback = None,
) -> list[np.ndarray]:
    processor = GlobalFlickerProcessor(len(frames_bgr), settings)
    total = len(frames_bgr)
    output_frames: list[np.ndarray] = []
    for index, frame_bgr in enumerate(frames_bgr):
        output_frames.append(processor.process_frame(frame_bgr, index))
        _call_progress(progress_callback, index + 1, total)
    return output_frames


def process_independent_image_sequence(
    frames_bgr: Sequence[np.ndarray],
    settings: IndependentFlickerSettings,
    fps: float = DEFAULT_SEQUENCE_FPS,
    progress_callback: ProgressCallback = None,
) -> list[np.ndarray]:
    processor = IndependentFlickerProcessor(len(frames_bgr), settings)
    total = len(frames_bgr)
    output_frames: list[np.ndarray] = []
    for index, frame_bgr in enumerate(frames_bgr):
        output_frames.append(processor.process_frame(frame_bgr, index))
        _call_progress(progress_callback, index + 1, total)
    return output_frames


def resolve_output_video_path(input_path: Path, suffix: str) -> Path:
    cleaned_suffix = suffix.strip()
    if not cleaned_suffix:
        raise ValueError("Suffix must not be empty.")
    output_path = input_path.with_name(f"{input_path.stem}{cleaned_suffix}{input_path.suffix}")
    if output_path.resolve() == input_path.resolve():
        raise ValueError("Output path would overwrite the input file. Use a different suffix.")
    return output_path


def choose_fourcc(output_path: Path) -> int:
    extension = output_path.suffix.lower()
    codec = {
        ".avi": "XVID",
        ".mov": "mp4v",
        ".mkv": "mp4v",
        ".mp4": "mp4v",
        ".m4v": "mp4v",
    }.get(extension, "mp4v")
    return cv2.VideoWriter_fourcc(*codec)


def create_temp_video_path(output_path: Path, label: str) -> Path:
    unique = uuid.uuid4().hex
    return output_path.with_name(f"{output_path.stem}.{label}.{unique}{output_path.suffix}")


def detect_audio_stream(input_path: Path) -> Optional[bool]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        cmd = [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "csv=p=0",
            str(input_path),
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed while inspecting audio streams: {result.stderr.strip()}")
        return bool(result.stdout.strip())

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None

    result = subprocess.run([ffmpeg, "-i", str(input_path)], capture_output=True, text=True, check=False)
    return re.search(r"Stream #.*Audio:", result.stderr) is not None


def mux_audio_into_video(silent_video_path: Path, source_video_path: Path, output_path: Path) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found on PATH. It is required to preserve audio for video processing.")

    cmd = [
        ffmpeg,
        "-y",
        "-i",
        str(silent_video_path),
        "-i",
        str(source_video_path),
        "-map",
        "0:v:0",
        "-map",
        "1:a?",
        "-c:v",
        "copy",
        "-c:a",
        "copy",
        "-shortest",
        str(output_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg failed while muxing audio: {result.stderr.strip()}")


def _finalize_video_output(
    input_path: Path,
    output_path: Path,
    silent_temp_path: Path,
    delete_original: bool,
) -> Path:
    muxed_temp_path = create_temp_video_path(output_path, "muxed")
    final_source_path = silent_temp_path

    try:
        audio_present = detect_audio_stream(input_path)
        if audio_present is None:
            raise RuntimeError(
                "Could not determine whether the input video has audio because ffmpeg/ffprobe is unavailable. "
                "Install ffmpeg and ensure it is on PATH."
            )
        if audio_present:
            mux_audio_into_video(silent_temp_path, input_path, muxed_temp_path)
            final_source_path = muxed_temp_path

        if output_path.exists():
            output_path.unlink()
        os.replace(final_source_path, output_path)
        final_source_path = output_path

        if delete_original:
            input_path.unlink()
        return output_path
    finally:
        if silent_temp_path.exists():
            silent_temp_path.unlink(missing_ok=True)
        if muxed_temp_path.exists() and muxed_temp_path != final_source_path:
            muxed_temp_path.unlink(missing_ok=True)


def process_video_file(
    input_path: Path,
    output_path: Path,
    processor_factory: Callable[[int, float], object],
    delete_original: bool = False,
    progress_callback: ProgressCallback = None,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if output_path.resolve() == input_path.resolve():
        raise ValueError("Output path must be different from the input path.")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or DEFAULT_SEQUENCE_FPS
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    silent_temp_path = create_temp_video_path(output_path, "silent")
    writer = cv2.VideoWriter(
        str(silent_temp_path),
        choose_fourcc(output_path),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {silent_temp_path}")

    processor = processor_factory(frame_count, fps)
    frame_index = 0

    try:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break
            processed_frame = processor.process_frame(frame_bgr, frame_index)
            writer.write(processed_frame)
            frame_index += 1
            _call_progress(progress_callback, frame_index, frame_count)
    finally:
        capture.release()
        writer.release()

    return _finalize_video_output(input_path, output_path, silent_temp_path, delete_original)


def process_global_video(
    input_path: Path,
    output_path: Path,
    settings: GlobalFlickerSettings,
    delete_original: bool = False,
    progress_callback: ProgressCallback = None,
) -> Path:
    return process_video_file(
        input_path,
        output_path,
        lambda frame_count, _fps: GlobalFlickerProcessor(frame_count, settings),
        delete_original=delete_original,
        progress_callback=progress_callback,
    )


def process_independent_video(
    input_path: Path,
    output_path: Path,
    settings: IndependentFlickerSettings,
    delete_original: bool = False,
    progress_callback: ProgressCallback = None,
) -> Path:
    return process_video_file(
        input_path,
        output_path,
        lambda frame_count, _fps: IndependentFlickerProcessor(frame_count, settings),
        delete_original=delete_original,
        progress_callback=progress_callback,
    )

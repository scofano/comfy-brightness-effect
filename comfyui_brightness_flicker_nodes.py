from pathlib import Path

import numpy as np
import torch

try:
    from .brightness_flicker_core import (
        DEFAULT_SEQUENCE_FPS,
        GlobalFlickerSettings,
        IndependentFlickerSettings,
        process_global_image_sequence,
        process_global_video,
        process_independent_image_sequence,
        process_independent_video,
        resolve_output_video_path,
    )
except ImportError:  # pragma: no cover - fallback for direct local imports
    from brightness_flicker_core import (
        DEFAULT_SEQUENCE_FPS,
        GlobalFlickerSettings,
        IndependentFlickerSettings,
        process_global_image_sequence,
        process_global_video,
        process_independent_image_sequence,
        process_independent_video,
        resolve_output_video_path,
    )

try:
    from comfy.utils import ProgressBar
except ImportError:  # pragma: no cover - local fallback outside ComfyUI
    class ProgressBar:  # type: ignore[override]
        def __init__(self, total: int):
            self.total = total
            self.value = 0

        def update(self, amount: int) -> None:
            self.value += amount

        def update_absolute(self, amount: int) -> None:
            self.value = amount


CATEGORY = "Brightness Flicker"


def _update_progress(progress_bar: ProgressBar, processed: int, total: int) -> None:
    if total <= 0:
        return
    if hasattr(progress_bar, "update_absolute"):
        progress_bar.update_absolute(processed)
    else:
        progress_bar.update(1)


def _tensor_to_bgr_frames(images: torch.Tensor) -> list[np.ndarray]:
    if images.ndim != 4:
        raise ValueError("IMAGE input must be a 4D tensor in BHWC format.")

    images_cpu = images.detach().cpu().clamp(0.0, 1.0).numpy()
    frames_bgr: list[np.ndarray] = []
    for image in images_cpu:
        rgb = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
        frames_bgr.append(rgb[:, :, ::-1].copy())
    return frames_bgr


def _bgr_frames_to_tensor(frames_bgr: list[np.ndarray], reference: torch.Tensor) -> torch.Tensor:
    frames_rgb = [frame[:, :, ::-1].astype(np.float32) / 255.0 for frame in frames_bgr]
    output = torch.from_numpy(np.stack(frames_rgb, axis=0))
    return output.to(device=reference.device, dtype=reference.dtype)


class BrightAreaFlickerVideoPath:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "process"
    CATEGORY = CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("STRING", {"default": "", "multiline": False}),
                "suffix": ("STRING", {"default": "_flicker", "multiline": False}),
                "enabled": ("BOOLEAN", {"default": False}),
                "delete_original": ("BOOLEAN", {"default": False}),
                "threshold": ("FLOAT", {"default": 185.0, "min": 0.0, "max": 255.0, "step": 1.0}),
                "softness": ("FLOAT", {"default": 35.0, "min": 1.0, "max": 255.0, "step": 1.0}),
                "amount": ("FLOAT", {"default": 0.28, "min": 0.0, "max": 1.0, "step": 0.01}),
                "glow": ("FLOAT", {"default": 0.22, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pulse_duration": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "seed": ("INT", {"default": 7, "min": 0, "max": 2147483647}),
            }
        }

    def process(self, input_path, suffix, enabled, delete_original, threshold, softness, amount, glow, pulse_duration, seed):
        raw_input_path = input_path.strip()
        if not enabled:
            return (str(Path(raw_input_path)),)
        if not raw_input_path:
            raise ValueError("input_path must be a full file path.")

        source_path = Path(raw_input_path)
        output_path = resolve_output_video_path(source_path, suffix)
        settings = GlobalFlickerSettings(
            threshold=threshold,
            softness=softness,
            amount=amount,
            glow=glow,
            pulse_duration=pulse_duration,
            seed=seed,
        )
        progress_state = {"bar": ProgressBar(1), "total": 1}

        def progress(processed: int, total: int) -> None:
            if total > 0 and progress_state["total"] != total:
                progress_state["bar"] = ProgressBar(total)
                progress_state["total"] = total
            _update_progress(progress_state["bar"], processed, total)

        final_path = process_global_video(
            source_path,
            output_path,
            settings,
            delete_original=delete_original,
            progress_callback=progress,
        )
        return (str(final_path),)


class BrightAreaFlickerImages:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": False}),
                "images": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 185.0, "min": 0.0, "max": 255.0, "step": 1.0}),
                "softness": ("FLOAT", {"default": 35.0, "min": 1.0, "max": 255.0, "step": 1.0}),
                "amount": ("FLOAT", {"default": 0.28, "min": 0.0, "max": 1.0, "step": 0.01}),
                "glow": ("FLOAT", {"default": 0.22, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pulse_duration": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "seed": ("INT", {"default": 7, "min": 0, "max": 2147483647}),
            }
        }

    def process(self, enabled, images, threshold, softness, amount, glow, pulse_duration, seed):
        if not enabled:
            return (images,)

        frames_bgr = _tensor_to_bgr_frames(images)
        progress_bar = ProgressBar(max(len(frames_bgr), 1))
        settings = GlobalFlickerSettings(
            threshold=threshold,
            softness=softness,
            amount=amount,
            glow=glow,
            pulse_duration=pulse_duration,
            seed=seed,
        )
        output_frames = process_global_image_sequence(
            frames_bgr,
            settings,
            fps=DEFAULT_SEQUENCE_FPS,
            progress_callback=lambda processed, total: _update_progress(progress_bar, processed, total),
        )
        return (_bgr_frames_to_tensor(output_frames, images),)


class BrightAreaFlickerIndependentVideoPath:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("output_path",)
    FUNCTION = "process"
    CATEGORY = CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_path": ("STRING", {"default": "", "multiline": False}),
                "suffix": ("STRING", {"default": "_independent_flicker", "multiline": False}),
                "enabled": ("BOOLEAN", {"default": False}),
                "delete_original": ("BOOLEAN", {"default": False}),
                "threshold": ("FLOAT", {"default": 185.0, "min": 0.0, "max": 255.0, "step": 1.0}),
                "softness": ("FLOAT", {"default": 35.0, "min": 1.0, "max": 255.0, "step": 1.0}),
                "amount": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
                "glow": ("FLOAT", {"default": 0.22, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_area": ("INT", {"default": 100, "min": 1, "max": 1000000}),
                "max_distance": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 10000.0, "step": 1.0}),
                "max_missed": ("INT", {"default": 8, "min": 0, "max": 1000}),
                "pulse_duration": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "seed": ("INT", {"default": 13, "min": 0, "max": 2147483647}),
            }
        }

    def process(
        self,
        input_path,
        suffix,
        enabled,
        delete_original,
        threshold,
        softness,
        amount,
        glow,
        pulse_duration,
        min_area,
        max_distance,
        max_missed,
        seed,
    ):
        raw_input_path = input_path.strip()
        if not enabled:
            return (str(Path(raw_input_path)),)
        if not raw_input_path:
            raise ValueError("input_path must be a full file path.")

        source_path = Path(raw_input_path)
        output_path = resolve_output_video_path(source_path, suffix)
        settings = IndependentFlickerSettings(
            threshold=threshold,
            softness=softness,
            amount=amount,
            glow=glow,
            pulse_duration=pulse_duration,
            min_area=min_area,
            max_distance=max_distance,
            max_missed=max_missed,
            seed=seed,
        )
        progress_state = {"bar": ProgressBar(1), "total": 1}

        def progress(processed: int, total: int) -> None:
            if total > 0 and progress_state["total"] != total:
                progress_state["bar"] = ProgressBar(total)
                progress_state["total"] = total
            _update_progress(progress_state["bar"], processed, total)

        final_path = process_independent_video(
            source_path,
            output_path,
            settings,
            delete_original=delete_original,
            progress_callback=progress,
        )
        return (str(final_path),)


class BrightAreaFlickerIndependentImages:
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = CATEGORY

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": False}),
                "images": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 185.0, "min": 0.0, "max": 255.0, "step": 1.0}),
                "softness": ("FLOAT", {"default": 35.0, "min": 1.0, "max": 255.0, "step": 1.0}),
                "amount": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
                "glow": ("FLOAT", {"default": 0.22, "min": 0.0, "max": 1.0, "step": 0.01}),
                "min_area": ("INT", {"default": 100, "min": 1, "max": 1000000}),
                "max_distance": ("FLOAT", {"default": 60.0, "min": 1.0, "max": 10000.0, "step": 1.0}),
                "max_missed": ("INT", {"default": 8, "min": 0, "max": 1000}),
                "pulse_duration": ("INT", {"default": 1, "min": 1, "max": 1000}),
                "seed": ("INT", {"default": 13, "min": 0, "max": 2147483647}),
            }
        }

    def process(self, enabled, images, threshold, softness, amount, glow, pulse_duration, min_area, max_distance, max_missed, seed):
        if not enabled:
            return (images,)

        frames_bgr = _tensor_to_bgr_frames(images)
        progress_bar = ProgressBar(max(len(frames_bgr), 1))
        settings = IndependentFlickerSettings(
            threshold=threshold,
            softness=softness,
            amount=amount,
            glow=glow,
            pulse_duration=pulse_duration,
            min_area=min_area,
            max_distance=max_distance,
            max_missed=max_missed,
            seed=seed,
        )
        output_frames = process_independent_image_sequence(
            frames_bgr,
            settings,
            fps=DEFAULT_SEQUENCE_FPS,
            progress_callback=lambda processed, total: _update_progress(progress_bar, processed, total),
        )
        return (_bgr_frames_to_tensor(output_frames, images),)


NODE_CLASS_MAPPINGS = {
    "BrightAreaFlickerVideoPath": BrightAreaFlickerVideoPath,
    "BrightAreaFlickerImages": BrightAreaFlickerImages,
    "BrightAreaFlickerIndependentVideoPath": BrightAreaFlickerIndependentVideoPath,
    "BrightAreaFlickerIndependentImages": BrightAreaFlickerIndependentImages,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BrightAreaFlickerVideoPath": "Bright Area Flicker (Video Path)",
    "BrightAreaFlickerImages": "Bright Area Flicker (Images)",
    "BrightAreaFlickerIndependentVideoPath": "Bright Area Flicker Independent (Video Path)",
    "BrightAreaFlickerIndependentImages": "Bright Area Flicker Independent (Images)",
}

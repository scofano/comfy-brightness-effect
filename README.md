# ComfyUI Brightness Flicker Effect

A custom node suite for **ComfyUI** that applies a realistic, lamp-like flicker (dimming and glowing bloom) to the bright regions of a video or image sequence. It features both **global flickering** (all lights flicker in unison) and **independent flickering** (individual light sources are detected, tracked, and flickered independently with randomized variations).

It also includes standalone **CLI tools** to run the effects directly on video files from the command line, completely independent of the ComfyUI interface.

---

## Features

- **Global Flicker**: A single coherent flicker wave is applied across all detected bright areas. Ideal for simulating camera exposure changes, overall power fluctuations, or environment-wide light changes.
- **Independent Flicker**: Automatically detects individual bright regions (e.g., streetlights, candles, neon signs) using connected-components analysis, tracks them across frames, and applies unique, randomized flicker behaviors to each.
- **Video Path Nodes**: Directly processes video files on disk. By streaming frames and writing them back to disk, these nodes avoid loading huge high-resolution video sequences into system RAM.
- **Audio Preservation**: If `ffmpeg` and `ffprobe` are installed on your system PATH, the Video Path nodes will automatically extract and remux the original audio track back into the processed video.
- **Standard Image Batch Nodes**: Process standard ComfyUI `IMAGE` tensors (BHWC format) natively, allowing seamless integration with other ComfyUI nodes (e.g., Load Video, VAE Decode, Save Image).

---

## Installation

1. **Clone the Repository**:
   Clone this repository into your ComfyUI `custom_nodes` directory:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/comfyanonymous/comfy-brightness-effect.git
   ```

2. **Install Python Dependencies**:
   The nodes rely on standard python packages. Run:
   ```bash
   pip install opencv-python numpy torch
   ```
   *(Note: `numpy` and `torch` are usually pre-installed with ComfyUI.)*

3. **Audio Preservation Support (Optional)**:
   To preserve audio when using the Video Path nodes, make sure `ffmpeg` and `ffprobe` are installed and available on your system's PATH.

---

## ComfyUI Nodes

All nodes can be found in the **Brightness Flicker** category in the ComfyUI context menu.

### 1. Bright Area Flicker (Images)
Applies global flicker to a batch of images/frames directly within your workflow.
* **Inputs**:
  - `images` (`IMAGE`): The input image tensor sequence (BHWC).
  - `enabled` (`BOOLEAN`): Toggle the effect on/off.
  - `threshold` (`FLOAT`, default: `185.0`): The brightness threshold (0-255) above which pixels are considered "bright".
  - `softness` (`FLOAT`, default: `35.0`): The range above the threshold used to smoothly transition the mask from dark to bright, avoiding harsh edges.
  - `amount` (`FLOAT`, default: `0.28`): The maximum intensity/amplitude of the flicker.
  - `glow` (`FLOAT`, default: `0.22`): Bloom intensity mixed back into bright regions during positive flicker peaks.
  - `pulse_duration` (`INT`, default: `1`): The number of full cosine pulse cycles across the entire sequence duration. Higher values create faster, more frequent pulses.
  - `seed` (`INT`, default: `7`): Random seed for reproducible flicker curves.
* **Outputs**:
  - `images` (`IMAGE`): The processed image batch.

### 2. Bright Area Flicker (Video Path)
Processes a video file on disk using global flicker.
* **Inputs**:
  - `input_path` (`STRING`): Absolute file path to the input video.
  - `suffix` (`STRING`, default: `_flicker`): Suffix appended to the output file name.
  - `delete_original` (`BOOLEAN`, default: `false`): Delete the input video after successful processing.
  - *Plus all common global parameters (`enabled`, `threshold`, `softness`, `amount`, `glow`, `pulse_duration`, `seed`).*
* **Outputs**:
  - `output_path` (`STRING`): The absolute path to the processed output video.

### 3. Bright Area Flicker Independent (Images)
Detects, tracks, and applies independent flicker variations to individual light sources in an image batch.
* **Inputs**:
  - *All common parameters from the global images node, plus:*
  - `min_area` (`INT`, default: `100`): The minimum pixel area of a bright region to be tracked as a distinct light source.
  - `max_distance` (`FLOAT`, default: `60.0`): The maximum distance in pixels a light's centroid can move between consecutive frames to still be matched as the same light source.
  - `max_missed` (`INT`, default: `8`): How many consecutive frames a tracked light can disappear (e.g., due to temporary occlusion) before its track is deleted.
* **Outputs**:
  - `images` (`IMAGE`): The processed image batch.

### 4. Bright Area Flicker Independent (Video Path)
Processes a video file on disk, detecting and tracking individual bright regions to flicker them independently.
* **Inputs**:
  - `input_path` (`STRING`): Absolute file path to the input video.
  - `suffix` (`STRING`, default: `_independent_flicker`): Suffix appended to the output file name.
  - `delete_original` (`BOOLEAN`, default: `false`): Delete the input video after successful processing.
  - *Plus all independent tracking parameters (`min_area`, `max_distance`, `max_missed`) and common parameters.*
* **Outputs**:
  - `output_path` (`STRING`): The absolute path to the processed output video.

---

## Standalone CLI Usage

You can also run these effects outside of ComfyUI as standalone python scripts.

### Global Flicker CLI
```bash
python flicker_bright_areas.py input.mp4 output.mp4 --threshold 185.0 --softness 35.0 --amount 0.28 --glow 0.22 --seed 7
```

### Independent Flicker CLI
```bash
python flicker_bright_areas_independent.py input.mp4 output.mp4 --threshold 185.0 --softness 35.0 --amount 0.30 --glow 0.22 --min-area 100 --max-distance 60.0 --max-missed 8 --seed 13
```

---

## How it Works Under the Hood

1. **Mask Generation**: The video frames are converted to grayscale. A smooth threshold function (using a smoothstep curve) isolates pixels above the `threshold` value. The mask is blurred using a Gaussian filter to ensure soft, realistic transitions around the edges of lights.
2. **Coherent Cosine Waves**: A cosine wave is calculated over the sequence length based on the `pulse_duration` parameter. This wave oscillates between positive (glow/bloom peaks) and negative (dimming/decay valleys) values.
3. **HSV Manipulation**: The node processes frames in HSV color space, adjusting the **V**alue (brightness) channel dynamically:
   - During **decay phases**, the brightness of the masked regions is reduced.
   - During **peak phases**, the brightness of the masked regions is boosted, and a blurred version of the brightened areas is added back as a bloom layer using the `glow` factor.
4. **Centroid-Based Tracking (Independent Mode)**: Connected components analyze the bright mask to identify separate blobs. Centroids are matched frame-to-frame by finding the minimal Euclidean distance within the `max_distance` radius. Each active track is assigned a randomized amplitude multiplier so they do not flicker in perfect synchronization.

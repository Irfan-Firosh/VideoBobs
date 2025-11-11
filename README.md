# VideoBobs - Mathematical Foundations

A Python system for generating animated videos with talking circular avatars (bobs) that respond to audio energy through mathematical transformations.

## Mathematical Concepts

### 1. RMS Energy Extraction

Root Mean Square (RMS) energy is computed from audio signals to quantify signal amplitude:

```
RMS = √(1/N * Σ(x_i²))
```

Where:
- `N` is the number of samples in the analysis window
- `x_i` are the audio sample values
- The frame length is 2048 samples with a hop length of 512 samples

RMS energy provides a robust measure of audio intensity that correlates with perceived loudness and is less sensitive to outliers than peak amplitude.

### 2. Time-to-Frame Conversion

Audio time is mapped to video frames using linear interpolation:

```
frame_index = floor(time * fps)
```

Where:
- `time` is the audio timestamp in seconds
- `fps` is the video framerate (default: 30 fps)
- `floor()` ensures frame indices are integers

The total number of frames is calculated as:
```
total_frames = ceil(total_duration * fps)
```

### 3. Linear Interpolation

Energy values from audio analysis (at irregular time intervals) are interpolated to match video frame times using linear interpolation:

```
E(t) = E(t₁) + (E(t₂) - E(t₁)) * (t - t₁) / (t₂ - t₁)
```

Where:
- `E(t)` is the interpolated energy at frame time `t`
- `t₁` and `t₂` are adjacent RMS analysis times
- `E(t₁)` and `E(t₂)` are the corresponding RMS values

This ensures smooth energy transitions between audio analysis points.

### 4. Exponential Moving Average Smoothing

To reduce jitter and create smoother animations, an exponential moving average (EMA) is applied:

```
smoothed[i] = α * energy[i] + (1 - α) * smoothed[i-1]
```

Where:
- `α` (alpha) is the smoothing factor (default: 0.2)
- `energy[i]` is the raw energy at frame `i`
- `smoothed[i-1]` is the previous smoothed value

Lower α values (closer to 0) create more smoothing but slower response. Higher values (closer to 1) create less smoothing but faster response.

### 5. Energy Normalization

Energy values are normalized to a 0-1 range with a minimum threshold:

```
normalized = smoothed / max(smoothed)
normalized = max(normalized, 0.1)
```

This ensures:
- All values are in the range [0.1, 1.0]
- Maximum energy maps to 1.0
- Minimum visual presence is maintained at 0.1

### 6. Circular Position Calculation

Speakers are arranged in a semicircle using polar coordinates:

```
θ = π * (1 - i / (n - 1))
x = center_x + radius * cos(θ)
y = center_y + radius * sin(θ)
```

Where:
- `i` is the speaker index (0 to n-1)
- `n` is the number of speakers
- `radius = min(width, height) / 4`
- Angles range from π (left) to 0 (right)

For a single speaker, the position is simply the center of the frame.

### 7. HSV Color Space

Colors are generated using evenly spaced hues in HSV color space:

```
hue = 180 * i / max(n, 1)
```

Where:
- `i` is the speaker index
- `n` is the number of speakers
- Saturation = 200, Value = 255 (fixed)

HSV colors are then converted to BGR for OpenCV rendering.

### 8. Parametric Circle with Sine Wave Modulation

The fluid outline is generated using a parametric circle with multiple sine wave modulations:

```
angles = linspace(0, 2π, num_points)
phase = frame_idx * 0.1

wobble = sin(angles * 3 + phase) * 0.1 +
         sin(angles * 5 + phase * 1.3) * 0.05 +
         sin(angles * 7 + phase * 0.7) * 0.03
```

The multiple sine waves with different frequencies (3, 5, 7) and phase offsets create complex, organic-looking motion. The amplitudes (0.1, 0.05, 0.03) decrease with frequency to emphasize lower-frequency motion.

### 9. Energy-Based Scaling

The radius of each bob scales with audio energy:

```
energy_scale = 1.0 + (energy - 0.1) * (max_scale - 1.0) / 0.9
radius = base_radius * energy_scale + wobble * base_radius * energy
```

Where:
- `energy` is the normalized energy value [0.1, 1.0]
- `max_scale` is the maximum scale factor (default: 1.5)
- `base_radius` is the base size (default: 80 pixels)

The scaling formula ensures:
- Minimum energy (0.1) maps to scale 1.0
- Maximum energy (1.0) maps to `max_scale`
- Wobble amplitude also scales with energy

### 10. Inner Circle and Glow Effects

Additional visual elements use scaled radii:

```
inner_radius = base_radius * 0.7 * (1.0 + (energy - 0.1) * 0.5)
glow_radius = base_radius * 1.2 * (1.0 + (energy - 0.1) * 0.5)  [if energy > 0.3]
```

The inner circle provides depth, and the glow effect (only when energy > 0.3) adds emphasis for active speakers.

## Dependencies

- `numpy` - Numerical operations and array manipulation
- `librosa` - Audio feature extraction (RMS energy)
- `scipy` - Interpolation functions
- `opencv-python` - Video rendering and image processing
- `moviepy` - Video/audio combination
- `pydub` - Audio file manipulation

## Usage

```python
from main import TalkingBobsPipeline

conversation = [
    (0, "Hello everyone, welcome to the discussion."),
    (1, "Thanks for having me!"),
    (2, "I'd like to add something here."),
]

pipeline = TalkingBobsPipeline(output_file="talking_bobs.mp4")
output_path = pipeline.process_conversation(conversation)
```

## Parameters

### ChunkedAudioProcessor
- `sample_rate` (int, default=44100): Audio sample rate in Hz
- `video_fps` (int, default=30): Target video framerate
- `smoothing_alpha` (float, default=0.2): EMA smoothing factor (0-1)

### VideoGenerator
- `video_fps` (int, default=30): Video framerate
- `width` (int, default=1920): Video width in pixels
- `height` (int, default=1080): Video height in pixels
- `base_radius` (int, default=80): Base radius of bobs in pixels
- `max_scale` (float, default=1.5): Maximum scale factor for energy-based size increase

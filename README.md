# 4D Hypersphere Visualization Reference Guide

This document provides a comprehensive overview of the 4D Hypersphere Visualization system, including key parameters, recent changes, and customization options.

## Key Components

The visualization system consists of two main scripts:
1. **`generate_4d_sphere_v5_audio_opt.py`** - Main rendering script for creating visualization frames
2. **Encoder script** - Combines frames with audio into a high-quality video

## Configuration Options

The system is controlled via a JSON configuration file (`hypersphere_config.json`) with the following parameters:

| Parameter | Description | Default Value | Notes |
|-----------|-------------|---------------|-------|
| `resolution` | Output resolution [width, height] | [1280, 720] | Higher resolutions require more memory |
| `framerate` | Frames per second | 60 | Higher framerates create smoother animations |
| `point_count` | Number of points in hypersphere | 25000 | More points = more detail but slower rendering |
| `duration` | Animation duration in seconds | 6 | Overridden by audio length when used |
| `color_depth` | Bit depth for colors | 8 | Can be 8 or 10 |
| `color_gradient` | Color range for visualization | {"low": [0, 0, 255], "high": [255, 0, 0]} | Maps 4D coordinates to colors |
| `batch_size` | Number of frames per batch | 120 | Adjusts automatically for memory optimization |

## Recent Changes

- **Memory Optimization**: Implemented batch processing with automatic sizing based on system memory
- **Beat Detection**: Added audio beat analysis to drive color changes
- **Performance Improvements**: Added shared memory management and multi-threading
- **Error Recovery**: Added detection and re-rendering of missing frames
- **Rotation Changes**: Modified pitch rotation for more interesting visual effects
- **Frame Tracking**: Added progress tracking across worker processes
- **Temporary File Management**: Improved file I/O with temporary files to prevent corruption

## Customization Guide

### Visual Style Adjustments

#### 1. Color Gradient

To modify the color scheme, adjust the `color_gradient` parameter in the config file:

```json
"color_gradient": {
  "low": [0, 0, 255],  // Blue (RGB)
  "high": [255, 0, 0]  // Red (RGB)
}
```

Alternative color schemes:
- Purple to Yellow: `"low": [128, 0, 128], "high": [255, 255, 0]`
- Green to Cyan: `"low": [0, 128, 0], "high": [0, 255, 255]`
- Orange to Blue: `"low": [255, 165, 0], "high": [0, 0, 255]`

#### 2. Projection Parameters

To modify the 3D projection, adjust these lines in the `render_frame_static` method:

```python
# OpenGL perspective
gluPerspective(45, (width / height), 0.1, 50.0)

# Camera position
glTranslatef(0, 0, -1)  # Adjust z-distance to zoom in/out

# Rotation parameters
pitch_angle = w_angle  # Adjust multiplier to change rotation speed
glRotatef(np.degrees(pitch_angle), 0.5, 0.5, 0.0)  # Change rotation axes
```

#### 3. Beat Reactivity

To adjust how beats affect the visualization, modify the `get_zoom_envelope` method:

```python
# Amplify beat effect (increase for stronger effects)
decay_factor = np.exp(-decay_pos) * 3.0  # Try values between 1.0-5.0

# Smoothness of beat transitions (lower = smoother)
smooth_factor = 0.15  # Try values between 0.05-0.5
```

### Rendering Performance

#### 1. Memory Usage

The most critical parameters for memory usage are:

```python
# Estimated memory per thread
mem_per_thread = 100 * 1024 * 1024  # Adjust based on your system

# Batch size for frame processing
self.batch_size = 120  # Smaller batches use less memory
```

#### 2. CPU Usage

To control CPU usage:

```python
# Thread count
self.thread_count = max(1, multiprocessing.cpu_count() - 1)  # Leave 1 core free

# Or specify via command line:
python generate_4d_sphere_v5_audio_opt.py --threads 4
```

#### 3. Point Count vs Quality

More points create better detail but require more processing power:

```json
"point_count": 25000
```

Recommended ranges:
- Low-end systems: 10,000 - 15,000 points
- Mid-range systems: 25,000 - 50,000 points 
- High-end systems: 75,000 - 100,000 points

### Advanced Projection Techniques

#### 1. Stereographic Projection Factor

In `project_4d_to_3d_optimized`, the projection factor controls how 4D points map to 3D:

```python
# Static projection factor
factor = 1  # Try values between 0.5-2.0
```

#### 2. 4D Rotation

The 4D rotation can be adjusted for different effects:

```python
# 4D rotation (vectorized)
cos_w, sin_w = np.cos(w_angle), np.sin(w_angle)
rotated_w4 = w4 * cos_w - x4 * sin_w
rotated_x4 = w4 * sin_w + x4 * cos_w

# Try different rotation planes by replacing x4 with y4 or z4
```

#### 3. Point Rendering

Point appearance can be adjusted in the OpenGL setup:

```python
# Point size (larger = more visible points)
glPointSize(2.0)  # Try values between 1.0-5.0

# Point smoothing quality
glHint(GL_POINT_SMOOTH_HINT, GL_NICEST)  # Options: GL_FASTEST, GL_NICEST
```

## Command Line Usage

```bash
python generate_4d_sphere_v5_audio_opt.py --audio music.wav --config hypersphere_config.json --output /path/to/frames --batch-size 120 --threads 8
```

| Argument | Description |
|----------|-------------|
| `--audio` / `-a` | Path to audio file for beat detection |
| `--config` / `-c` | Path to configuration file |
| `--output` / `-o` | Output directory for rendered frames |
| `--batch-size` / `-b` | Override batch size for processing |
| `--threads` / `-t` | Override thread count |
| `--verbose` / `-v` | Enable verbose logging |

## Video Encoding

After rendering frames, encode them with the audio file:

```bash
python encode_video.py --input /path/to/frames --output final_video.mkv --audio music.wav --config hypersphere_config.json --quality 20 --preset medium
```

## Troubleshooting

1. **Memory Issues**: Reduce batch size, point count, or thread count
2. **Missing Frames**: Check for errors in logs; the system will attempt to re-render missing frames
3. **OpenGL Errors**: Ensure your system has proper graphics drivers installed
4. **Performance Issues**: Monitor CPU and memory usage, adjust parameters accordingly
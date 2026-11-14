# VisionScale AI - Google Colab Setup Guide

This guide is for running VisionScale AI in Google Colab with a free or paid GPU runtime.

Use the commands below in Colab cells. Colab is a notebook environment, so commands must be written as shell commands with `!` or as normal Python code inside a cell.

---

## 1) Open a Colab notebook

1. Go to Google Colab.
2. Create a new notebook.
3. In the runtime menu, choose:
   - `Runtime` -> `Change runtime type`
   - Set `Hardware accelerator` to `T4 GPU` or another available GPU
4. Start the runtime.

Check GPU status:

```python
!nvidia-smi
```

---

## 2) Clone the repository

```python
!git clone https://github.com/webdevaditya1/VisionScale-AI.git
%cd VisionScale-AI
!ls -la
```

This clones the repo and moves into the project directory.

---

## 3) Install FFmpeg

```python
!apt-get update
!apt-get install -y ffmpeg
!ffmpeg -version
```

---

## 4) Install Python dependencies

```python
!pip install -r requirements-gpu.txt
```

Colab usually provides CUDA-enabled PyTorch. Avoid reinstalling PyTorch unless the hardware check below shows CUDA is unavailable.

---

## 5) Verify the project works

```python
!python -m pytest tests/ -v
```

This should run the test suite and confirm the setup is valid.

---

## 6) Check hardware detection

```python
from visionscale.core.hardware import detect_hardware
print(detect_hardware().summary())
```

You should see CUDA enabled and the GPU model in the output.

---

## 7) Upload or locate your input video

You can upload from your computer or use a file already in Colab storage.

Check your files:

```python
import os
for root, dirs, files in os.walk('.'):
    for f in files:
        if f.lower().endswith(('.mp4', '.mov', '.mkv', '.avi', '.webm', '.m4v')):
            print(os.path.join(root, f))
```

Or use a specific path:

```python
VIDEO_PATH = '/content/your_video.mp4'
print(VIDEO_PATH)
```

---

## 8) Enhance the video and export an MP4

```python
from pathlib import Path
from visionscale.config import VisionScaleConfig, ResolutionTarget, ContentCategory
from visionscale.pipeline import enhance_video

assert Path(VIDEO_PATH).is_file(), f"Video not found: {VIDEO_PATH}"

config = VisionScaleConfig()
config.processing.target_resolution = ResolutionTarget.P4K
config.processing.content_category = ContentCategory.AUTO

output_path = enhance_video(VIDEO_PATH, config)
print(f"Enhanced video saved to: {output_path}")
```

The first run downloads model weights, so enable Internet access. The output is stored under `/content/visionscale_work/outputs`.

## 9) Optional: run the command-line script

```python
!python kaggle_t4_run.py --input "/content/your_video.mp4" --target-res 4k --category auto
```

---

## 10) Memory cleanup

If you hit memory issues:

```python
import gc, torch
gc.collect()
torch.cuda.empty_cache()
print('GPU memory cleared.')
```

---

## 11) Quick copy-paste setup block

Use this if you want a fast full setup in one cell:

```python
!git clone https://github.com/webdevaditya1/VisionScale-AI.git
%cd VisionScale-AI
!apt-get update
!apt-get install -y ffmpeg
!pip install -r requirements-gpu.txt
!python -m pytest tests/ -v
```

Then run:

```python
from visionscale.core.hardware import detect_hardware
print(detect_hardware().summary())
```

---

## 12) Best practice

- Use shorter clips for your first test
- Start with a short clip before processing long videos
- `P4K` retains or targets 4K; `P1080` creates 1080p output
- Keep your output in `/content` or `/content/visionscale_work`
- Clear GPU memory if you run multiple jobs in a session

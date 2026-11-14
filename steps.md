# VisionScale AI - Kaggle T4 GPU Setup Steps

This is the step-by-step guide to run this project on Kaggle with a free NVIDIA T4 GPU.

Use this in a Kaggle notebook and follow the steps in order.

---

## 1) Open a Kaggle notebook and enable GPU

1. Go to Kaggle.
2. Create a new notebook.
3. In the notebook settings:
   - Set Accelerator to `GPU`
   - Choose `T4 x 1` (or the available GPU)
4. Keep the notebook type as `Python`.

Check the GPU is active:

```python
!nvidia-smi
```

You should see an NVIDIA T4 device in the output.

---

## 2) Confirm environment and Python version

```python
!python --version
!pip --version
```

Expected:
- Python 3.10+
- pip is available

---

## 3) Clone the repository

```bash
!git clone https://github.com/webdevaditya1/VisionScale-AI.git
%cd VisionScale-AI
```

Check files:

```bash
ls -la
```

You should see the project files such as `README.md`, `requirements.txt`, `requirements-gpu.txt`, and the `visionscale/` folder.

---

## 4) Install FFmpeg system dependency

Kaggle usually already has some packages, but install FFmpeg to be safe:

```bash
!apt-get update
!apt-get install -y ffmpeg
```

Verify installation:

```bash
!ffmpeg -version
```

---

## 5) Use Kaggle's existing Python environment

Skip virtual-environment creation and activation on Kaggle. The notebook already runs in an isolated environment, and shell activation in a `!` cell does not persist to later cells.

---

## 6) Install the required GPU dependencies

Install the project's GPU requirements (this file includes the core requirements):

```bash
!python -m pip install -r requirements-gpu.txt
!python -m pip check
```

Kaggle normally already provides a CUDA-enabled PyTorch build; verify it with `torch.cuda.is_available()`.

---

## 7) Verify the installation

Run the test suite:

```bash
!python -m pytest tests/ -v
```

If everything is installed correctly, you should see tests pass.

If a test fails, inspect the complete traceback before changing packages. Avoid reinstalling PyTorch unless the hardware check confirms CUDA is unavailable.

---

## 8) Check hardware detection and GPU status

Run the project’s hardware diagnostic:

```python
from visionscale.core.hardware import detect_hardware
print(detect_hardware().summary())
```

This will tell you whether the project detected:
- CUDA available
- GPU device count
- VRAM profile
- recommended FP16 usage

Example expected output includes:
- NVIDIA Tesla T4
- CUDA: True
- FP16: True

---

## 9) Prepare your input video

Upload a video to Kaggle or place it in the working directory.

For example:

```bash
ls -la
```

If using Kaggle upload UI, the file will usually appear in:

```python
'/kaggle/working/'
```

Make sure you know the exact file name:

```python
video_path = '/kaggle/working/input_video.mp4'
print(video_path)
```

---

## 10) Enhance the video and export an MP4

The pipeline downloads the selected Real-ESRGAN weights on first use, runs tiled frame-by-frame inference, and writes the result under the Kaggle working directory. Enable Internet in Kaggle notebook settings for the first run. For a 4K source, choose `P4K` to retain 4K output; choose `P1080` to produce a 1080p output.

```python
from pathlib import Path
from visionscale.config import VisionScaleConfig, ResolutionTarget, ContentCategory
from visionscale.pipeline import enhance_video

video_path = "/kaggle/input/your-dataset/input_video.mp4"
assert Path(video_path).is_file(), f"Video not found: {video_path}"

config = VisionScaleConfig()
config.processing.target_resolution = ResolutionTarget.P4K
config.processing.content_category = ContentCategory.AUTO

def show_progress(done, total):
    if done == total or done % 25 == 0:
        print(f"Processed {done}/{total} frames")

output_path = enhance_video(video_path, config, progress_callback=show_progress)
print(f"Enhanced video saved to: {output_path}")
```

---

## 11) Save output files

The enhanced MP4 is saved under `/kaggle/working/visionscale_work/outputs/`. Use Kaggle's output/file browser to download it. Start with a short clip before processing a long video.

---

## 12) Best practices for T4 GPU on Kaggle

- Use the smallest reasonable output resolution first.
- Prefer `1080p` before `4K` if you are trying to save time and memory.
- Keep the input short while testing.
- Clean GPU memory when needed:

```python
import gc, torch
gc.collect()
torch.cuda.empty_cache()
```

- Run on `FP16` when supported; the project is designed for that on T4.
- Use `/tmp` for temporary frame work if the project needs scratch storage.

---

## 13) If the run fails with OOM or memory issues

Try these fixes:

1. Reduce output resolution
2. Reduce processing tile size
3. Use a shorter input clip
4. Clear GPU cache
5. Restart the notebook if memory is stuck

Example:

```python
import gc, torch
gc.collect()
torch.cuda.empty_cache()
```

If the memory manager triggers tiling, the project is designed to handle it automatically for T4-class GPUs.

---

## 14) Final checklist before training or processing

Before starting a long job, confirm:

- [ ] GPU is enabled
- [ ] `nvidia-smi` works
- [ ] `ffmpeg` is installed
- [ ] `requirements-gpu.txt` is installed
- [ ] `pytest` tests pass
- [ ] Input video file exists
- [ ] Output folder exists
- [ ] GPU memory is available

---

## 15) Quick Kaggle command block

Use this all-in-one notebook cell if you want a fast setup:

```bash
!nvidia-smi
!git clone https://github.com/webdevaditya1/VisionScale-AI.git
%cd VisionScale-AI
!apt-get update
!apt-get install -y ffmpeg
!pip install -r requirements-gpu.txt
!python -m pytest tests/ -v
```

Then verify hardware:

```python
from visionscale.core.hardware import detect_hardware
print(detect_hardware().summary())
```

---

## 16) Recommended first run

For a first test, use a short clip and target 1080p.

This makes debugging faster and avoids wasting GPU time.

After the first run succeeds, you can try higher-quality settings and longer videos.

---

For a complete Kaggle notebook workflow, use `VisionScaleAI_Kaggle_T4.ipynb`.

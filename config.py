"""
Configuration file for Image-to-Video Generator
Edit these settings to customize your pipeline
"""

import os
from pathlib import Path

# ===== PROJECT PATHS =====
PROJECT_ROOT = Path(__file__).parent
ASSETS_DIR = PROJECT_ROOT / "assets"
INPUT_DIR = ASSETS_DIR / "input"
OUTPUT_DIR = ASSETS_DIR / "output"
MODELS_DIR = PROJECT_ROOT / "models"

# Create directories if they don't exist
for directory in [INPUT_DIR, OUTPUT_DIR, MODELS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ===== MODEL CONFIGURATION =====
# Stable Diffusion model to use
STABLE_DIFFUSION_MODEL = "runwayml/stable-diffusion-v1-5"

# ControlNet model (for more control over generation)
CONTROLNET_MODEL = "lllyasviel/control_canny"

# Real-ESRGAN model (for upscaling)
UPSCALE_MODEL = "RealESRGAN_x4plus"

# Device to use (cuda for GPU, cpu for CPU)
DEVICE = "cuda"  # Change to "cpu" if you don't have a GPU

# ===== VIDEO GENERATION SETTINGS =====
VIDEO_FPS = 30  # Frames per second for output video
VIDEO_DURATION = 5  # Duration in seconds
VIDEO_WIDTH = 512
VIDEO_HEIGHT = 512
VIDEO_FORMAT = "mp4"

# ===== STYLE TRANSFER SETTINGS =====
STYLE_TRANSFER_STRENGTH = 0.5  # 0.0 to 1.0 (higher = more style applied)

# ===== COLORIZATION SETTINGS =====
COLORIZATION_MODEL = "user/instance_colorization"  # Hugging Face model

# ===== UPSCALING SETTINGS =====
UPSCALE_FACTOR = 4  # Upscale by 4x

# ===== API SETTINGS (if running web server) =====
API_HOST = "0.0.0.0"
API_PORT = 8000
API_DEBUG = True

# ===== LOGGING SETTINGS =====
LOG_LEVEL = "INFO"
SAVE_INTERMEDIATE_STEPS = True  # Save intermediate results for debugging

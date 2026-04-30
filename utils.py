"""
Utility functions for Image-to-Video Generator
Helper functions for loading, processing, and saving images/videos
"""

import os
import cv2
import numpy as np
from PIL import Image
import imageio
from pathlib import Path
from tqdm import tqdm
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== IMAGE LOADING & PROCESSING =====

def load_image(image_path, resize_to=None):
    """
    Load an image from a file path.
    
    Args:
        image_path (str): Path to the image file
        resize_to (tuple): Optional. Resize to (width, height)
    
    Returns:
        PIL.Image: Loaded image
    """
    try:
        image = Image.open(image_path).convert('RGB')
        if resize_to:
            image = image.resize(resize_to, Image.Resampling.LANCZOS)
        logger.info(f"✓ Loaded image: {image_path}")
        return image
    except Exception as e:
        logger.error(f"✗ Failed to load image: {e}")
        return None


def save_image(image, output_path):
    """
    Save a PIL Image to a file.
    
    Args:
        image (PIL.Image): Image to save
        output_path (str): Path to save the image
    """
    try:
        image.save(output_path)
        logger.info(f"✓ Saved image: {output_path}")
    except Exception as e:
        logger.error(f"✗ Failed to save image: {e}")


def image_to_tensor(image, device='cuda'):
    """
    Convert PIL Image to PyTorch tensor.
    
    Args:
        image (PIL.Image): Image to convert
        device (str): Device to move tensor to ('cuda' or 'cpu')
    
    Returns:
        torch.Tensor: Normalized tensor [1, 3, H, W]
    """
    import torch
    
    # Convert to numpy array
    image_np = np.array(image).astype(np.float32) / 255.0
    
    # Convert to tensor and normalize (ImageNet normalization)
    tensor = torch.from_numpy(image_np).permute(2, 0, 1)
    tensor = tensor.unsqueeze(0)  # Add batch dimension
    
    # Normalize
    tensor = (tensor - 0.5) / 0.5  # Normalize to [-1, 1]
    
    return tensor.to(device)


def tensor_to_image(tensor):
    """
    Convert PyTorch tensor back to PIL Image.
    
    Args:
        tensor (torch.Tensor): Tensor to convert [1, 3, H, W] or [3, H, W]
    
    Returns:
        PIL.Image: Converted image
    """
    import torch
    
    # Remove batch dimension if present
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    # Move to CPU and detach
    tensor = tensor.cpu().detach()
    
    # Denormalize from [-1, 1] to [0, 1]
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to numpy
    image_np = tensor.permute(1, 2, 0).numpy() * 255.0
    image_np = image_np.astype(np.uint8)
    
    return Image.fromarray(image_np)


# ===== VIDEO PROCESSING =====

def create_video_from_frames(frames, output_path, fps=30, codec='mp4v'):
    """
    Create a video from a list of frames.
    
    Args:
        frames (list): List of PIL Images or numpy arrays
        output_path (str): Path to save the video
        fps (int): Frames per second
        codec (str): Video codec ('mp4v' for mp4)
    """
    try:
        # Convert frames to numpy arrays if needed
        frame_array = []
        for frame in tqdm(frames, desc="Converting frames"):
            if isinstance(frame, Image.Image):
                frame = np.array(frame)
            frame_array.append(frame)
        
        # Create video writer
        writer = imageio.get_writer(output_path, fps=fps)
        
        for frame in tqdm(frame_array, desc="Writing video"):
            writer.append_data(frame)
        
        writer.close()
        logger.info(f"✓ Video saved: {output_path}")
    except Exception as e:
        logger.error(f"✗ Failed to create video: {e}")


def interpolate_frames(frame1, frame2, num_frames=10):
    """
    Create intermediate frames between two images for smooth animation.
    
    Args:
        frame1 (PIL.Image): First frame
        frame2 (PIL.Image): Last frame
        num_frames (int): Number of intermediate frames to create
    
    Returns:
        list: List of interpolated PIL Images
    """
    frames = [frame1]
    
    # Convert to numpy arrays
    img1 = np.array(frame1, dtype=np.float32)
    img2 = np.array(frame2, dtype=np.float32)
    
    # Linear interpolation
    for i in range(1, num_frames - 1):
        alpha = i / (num_frames - 1)
        interpolated = (1 - alpha) * img1 + alpha * img2
        frames.append(Image.fromarray(interpolated.astype(np.uint8)))
    
    frames.append(frame2)
    return frames


# ===== BATCH PROCESSING =====

def process_images_in_directory(input_dir, process_fn, output_dir=None):
    """
    Process all images in a directory with a given function.
    
    Args:
        input_dir (str): Directory containing images
        process_fn (function): Function to apply to each image
        output_dir (str): Optional output directory
    
    Returns:
        list: List of processed images
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.webp']
    image_paths = []
    
    # Find all images
    for ext in image_extensions:
        image_paths.extend(Path(input_dir).glob(f'*{ext}'))
        image_paths.extend(Path(input_dir).glob(f'*{ext.upper()}'))
    
    processed_images = []
    for image_path in tqdm(image_paths, desc="Processing images"):
        image = load_image(str(image_path))
        if image:
            processed = process_fn(image)
            processed_images.append(processed)
            
            if output_dir:
                output_path = Path(output_dir) / image_path.name
                save_image(processed, str(output_path))
    
    return processed_images


# ===== UTILITY FUNCTIONS =====

def get_image_dimensions(image_path):
    """
    Get the dimensions of an image without loading it fully.
    
    Args:
        image_path (str): Path to image
    
    Returns:
        tuple: (width, height)
    """
    try:
        image = Image.open(image_path)
        return image.size
    except Exception as e:
        logger.error(f"✗ Failed to get image dimensions: {e}")
        return None


def ensure_dir_exists(directory):
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory (str): Path to directory
    """
    Path(directory).mkdir(parents=True, exist_ok=True)


def list_files(directory, extension=None):
    """
    List all files in a directory, optionally filtered by extension.
    
    Args:
        directory (str): Path to directory
        extension (str): Optional file extension filter (e.g., '.jpg')
    
    Returns:
        list: List of file paths
    """
    files = list(Path(directory).iterdir())
    if extension:
        files = [f for f in files if f.suffix.lower() == extension.lower()]
    return sorted(files)

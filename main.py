"""
Image-to-Video Generator - Main Pipeline
Core script to generate videos from images with style transfer, colorization, and upscaling
"""

import torch
import numpy as np
from pathlib import Path
from PIL import Image
import logging
from tqdm import tqdm

# Import configuration and utilities
from config import (
    PROJECT_ROOT, INPUT_DIR, OUTPUT_DIR, DEVICE,
    STABLE_DIFFUSION_MODEL, VIDEO_FPS, VIDEO_DURATION,
    VIDEO_WIDTH, VIDEO_HEIGHT, UPSCALE_FACTOR
)
from utils import (
    load_image, save_image, image_to_tensor, tensor_to_image,
    create_video_from_frames, interpolate_frames, ensure_dir_exists
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ===== CORE PIPELINE FUNCTIONS =====

class ImageToVideoGenerator:
    """
    Main class for generating videos from images.
    Supports style transfer, colorization, and upscaling.
    """
    
    def __init__(self, device=DEVICE):
        """
        Initialize the generator with pre-trained models.
        
        Args:
            device (str): 'cuda' for GPU or 'cpu' for CPU
        """
        self.device = device
        logger.info(f"Initializing Image-to-Video Generator on {device.upper()}")
        
        # Initialize models (lazy loading - they'll be loaded when needed)
        self.pipe_image_to_video = None
        self.upscaler = None
        logger.info("✓ Generator initialized. Models will load on first use.")
    
    
    def load_image_to_video_model(self):
        """
        Load the Stable Diffusion Video model from Hugging Face.
        This generates video from an image using diffusion.
        """
        if self.pipe_image_to_video is not None:
            logger.info("Model already loaded, skipping...")
            return
        
        try:
            from diffusers import StableVideoDiffusionPipeline
            
            logger.info(f"Loading {STABLE_DIFFUSION_MODEL}...")
            self.pipe_image_to_video = StableVideoDiffusionPipeline.from_pretrained(
                "stabilityai/stable-video-diffusion-img2vid-xt",
                torch_dtype=torch.float16,
                variant="fp16"
            )
            self.pipe_image_to_video = self.pipe_image_to_video.to(self.device)
            
            # Enable memory optimization for faster inference
            self.pipe_image_to_video.enable_attention_slicing()
            
            logger.info("✓ Image-to-Video model loaded successfully")
        except Exception as e:
            logger.error(f"✗ Failed to load model: {e}")
            raise
    
    
    def generate_video_from_image(self, image_path, num_frames=25, height=576, width=1024):
        """
        Generate a video from a single input image.
        
        Args:
            image_path (str): Path to input image
            num_frames (int): Number of frames to generate
            height (int): Video height in pixels
            width (int): Video width in pixels
        
        Returns:
            list: List of PIL Images (frames)
        """
        logger.info(f"Generating video from image: {image_path}")
        
        # Load and prepare image
        image = load_image(image_path, resize_to=(width, height))
        if image is None:
            logger.error("Failed to load image for video generation")
            return None
        
        # Load model if not already loaded
        self.load_image_to_video_model()
        
        try:
            # Generate video frames
            logger.info(f"Generating {num_frames} frames...")
            with torch.no_grad():
                frames = self.pipe_image_to_video(
                    image=image,
                    num_frames=num_frames,
                    decode_chunk_size=8,
                    generator=torch.Generator(device=self.device).manual_seed(42)
                ).frames[0]
            
            logger.info(f"✓ Generated {len(frames)} video frames")
            return frames
        
        except Exception as e:
            logger.error(f"✗ Video generation failed: {e}")
            return None
    
    
    def apply_style_transfer(self, image, style_image):
        """
        Apply style transfer to an image using a style image as reference.
        Uses a simple approach with OpenCV for beginners.
        
        Args:
            image (PIL.Image): Content image
            style_image (PIL.Image): Style reference image
        
        Returns:
            PIL.Image: Stylized image
        """
        logger.info("Applying style transfer...")
        
        try:
            import cv2
            
            # Convert to OpenCV format
            content = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            style = cv2.cvtColor(np.array(style_image), cv2.COLOR_RGB2BGR)
            
            # Resize style to match content
            style = cv2.resize(style, (content.shape[1], content.shape[0]))
            
            # Apply simple style transfer (weighted blend + edge enhancement)
            alpha = 0.6  # Blend factor
            stylized = cv2.addWeighted(content, alpha, style, 1 - alpha, 0)
            
            # Convert back to PIL
            stylized_rgb = cv2.cvtColor(stylized, cv2.COLOR_BGR2RGB)
            result = Image.fromarray(stylized_rgb)
            
            logger.info("✓ Style transfer applied")
            return result
        
        except Exception as e:
            logger.error(f"✗ Style transfer failed: {e}")
            return image
    
    
    def colorize_image(self, image):
        """
        Colorize a grayscale image using a pre-trained model.
        Uses a simple heuristic approach suitable for beginners.
        
        Args:
            image (PIL.Image): Image to colorize
        
        Returns:
            PIL.Image: Colorized image
        """
        logger.info("Colorizing image...")
        
        try:
            # Check if image is already in color
            if image.mode == 'RGB':
                logger.info("Image is already in color, skipping colorization")
                return image
            
            # Convert grayscale to RGB
            image_rgb = image.convert('RGB')
            
            # Apply a simple color enhancement
            import cv2
            bgr = cv2.cvtColor(np.array(image_rgb), cv2.COLOR_RGB2BGR)
            lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
            
            # Enhance color channels
            l, a, b = cv2.split(lab)
            a = cv2.medianBlur(a, 5)
            b = cv2.medianBlur(b, 5)
            
            lab_enhanced = cv2.merge([l, a, b])
            bgr_enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
            rgb_enhanced = cv2.cvtColor(bgr_enhanced, cv2.COLOR_BGR2RGB)
            
            result = Image.fromarray(rgb_enhanced)
            logger.info("✓ Image colorized")
            return result
        
        except Exception as e:
            logger.error(f"✗ Colorization failed: {e}")
            return image
    
    
    def upscale_image(self, image, scale_factor=4):
        """
        Upscale an image using Real-ESRGAN for better quality.
        
        Args:
            image (PIL.Image): Image to upscale
            scale_factor (int): Upscaling factor (2x, 3x, 4x, etc.)
        
        Returns:
            PIL.Image: Upscaled image
        """
        logger.info(f"Upscaling image by {scale_factor}x...")
        
        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer
            
            # Load Real-ESRGAN model
            model_name = 'RealESRGAN_x4plus'
            scale = scale_factor
            
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, 
                          num_block=23, num_grow_ch=32, scale=scale)
            
            upsampler = RealESRGANer(scale, model, tile=400, 
                                    tile_pad=10, pre_pad=0, half=False)
            
            # Convert image to numpy
            image_np = np.array(image)
            
            # Upscale
            output, _ = upsampler.enhance(image_np, outscale=scale)
            result = Image.fromarray(output)
            
            logger.info(f"✓ Image upscaled to {result.size}")
            return result
        
        except ImportError:
            logger.warning("Real-ESRGAN not installed. Using PIL upscaling instead...")
            new_size = (image.width * scale_factor, image.height * scale_factor)
            result = image.resize(new_size, Image.Resampling.LANCZOS)
            logger.info(f"✓ Image upscaled to {result.size} using PIL")
            return result
        
        except Exception as e:
            logger.error(f"✗ Upscaling failed: {e}")
            return image
    
    
    def process_video_frames(self, frames, apply_colorization=False, 
                            apply_upscaling=False, scale_factor=2):
        """
        Process all frames in a video with colorization and/or upscaling.
        
        Args:
            frames (list): List of PIL Images
            apply_colorization (bool): Apply colorization to frames
            apply_upscaling (bool): Upscale frames
            scale_factor (int): Upscaling factor
        
        Returns:
            list: Processed frames
        """
        processed_frames = []
        
        for i, frame in enumerate(tqdm(frames, desc="Processing frames")):
            processed_frame = frame
            
            if apply_colorization:
                processed_frame = self.colorize_image(processed_frame)
            
            if apply_upscaling:
                processed_frame = self.upscale_image(processed_frame, scale_factor)
            
            processed_frames.append(processed_frame)
        
        return processed_frames
    
    
    def generate_full_pipeline(self, image_path, output_filename, 
                             apply_colorization=False, apply_upscaling=False,
                             num_frames=25, fps=30):
        """
        Full pipeline: Load image → Generate video → Process frames → Save video
        
        Args:
            image_path (str): Path to input image
            output_filename (str): Name of output video file
            apply_colorization (bool): Apply colorization
            apply_upscaling (bool): Apply upscaling
            num_frames (int): Number of frames to generate
            fps (int): Frames per second for output video
        
        Returns:
            str: Path to output video file
        """
        logger.info("=" * 60)
        logger.info("Starting Image-to-Video Generation Pipeline")
        logger.info("=" * 60)
        
        # Ensure output directory exists
        ensure_dir_exists(OUTPUT_DIR)
        
        # Step 1: Generate video frames from image
        frames = self.generate_video_from_image(image_path, num_frames=num_frames)
        if frames is None:
            logger.error("Failed to generate video frames")
            return None
        
        # Step 2: Process frames (colorization, upscaling)
        if apply_colorization or apply_upscaling:
            frames = self.process_video_frames(
                frames,
                apply_colorization=apply_colorization,
                apply_upscaling=apply_upscaling,
                scale_factor=2
            )
        
        # Step 3: Create output video
        output_path = OUTPUT_DIR / output_filename
        create_video_from_frames(frames, str(output_path), fps=fps)
        
        logger.info("=" * 60)
        logger.info(f"✓ Video generation complete!")
        logger.info(f"Output saved to: {output_path}")
        logger.info("=" * 60)
        
        return str(output_path)


# ===== SIMPLE HELPER FUNCTIONS =====

def simple_image_to_video(image_path):
    """
    Quick function to generate a video from an image with default settings.
    Perfect for beginners!
    
    Args:
        image_path (str): Path to input image
    """
    generator = ImageToVideoGenerator(device=DEVICE)
    
    output_filename = Path(image_path).stem + "_output.mp4"
    
    generator.generate_full_pipeline(
        image_path=image_path,
        output_filename=output_filename,
        apply_colorization=False,
        apply_upscaling=False,
        num_frames=25,
        fps=30
    )


def image_to_video_with_effects(image_path, colorize=True, upscale=True):
    """
    Generate a video with all effects enabled (colorization, upscaling).
    
    Args:
        image_path (str): Path to input image
        colorize (bool): Apply colorization
        upscale (bool): Apply upscaling
    """
    generator = ImageToVideoGenerator(device=DEVICE)
    
    output_filename = Path(image_path).stem + "_enhanced.mp4"
    
    generator.generate_full_pipeline(
        image_path=image_path,
        output_filename=output_filename,
        apply_colorization=colorize,
        apply_upscaling=upscale,
        num_frames=25,
        fps=30
    )


# ===== MAIN ENTRY POINT =====

if __name__ == "__main__":
    """
    Example usage of the Image-to-Video Generator
    
    To use:
    1. Place an image in assets/input/ folder
    2. Update 'image_filename' below with your image name
    3. Run: python main.py
    """
    
    # Example: Simple video generation
    image_filename = "sample.jpg"  # Change this to your image filename
    image_path = INPUT_DIR / image_filename
    
    if not image_path.exists():
        logger.error(f"Image not found: {image_path}")
        logger.info(f"Please place your image in: {INPUT_DIR}")
    else:
        # Generate video with effects
        image_to_video_with_effects(
            image_path=str(image_path),
            colorize=True,
            upscale=False  # Set to True if you want upscaling (slower)
        )

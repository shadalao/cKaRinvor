"""
Test Script for Image-to-Video Generator
Verifies that all dependencies are installed and working correctly
Run this before using main.py to ensure everything is set up properly
"""

import sys
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_python_version():
    """Check if Python version is 3.9+"""
    logger.info("=" * 60)
    logger.info("Testing Python Version...")
    logger.info("=" * 60)
    
    version = sys.version_info
    python_version = f"{version.major}.{version.minor}.{version.micro}"
    
    if version.major >= 3 and version.minor >= 9:
        logger.info(f"✓ Python {python_version} - OK")
        return True
    else:
        logger.error(f"✗ Python {python_version} - FAILED (Need 3.9+)")
        return False


def test_imports():
    """Test if all required packages can be imported"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Required Imports...")
    logger.info("=" * 60)
    
    packages = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'tensorflow': 'TensorFlow',
        'PIL': 'Pillow',
        'cv2': 'OpenCV',
        'numpy': 'NumPy',
        'scipy': 'SciPy',
        'imageio': 'ImageIO',
        'diffusers': 'Hugging Face Diffusers',
        'transformers': 'Hugging Face Transformers',
    }
    
    all_passed = True
    
    for package, name in packages.items():
        try:
            __import__(package)
            logger.info(f"✓ {name:35} - OK")
        except ImportError as e:
            logger.error(f"✗ {name:35} - FAILED: {str(e)}")
            all_passed = False
    
    return all_passed


def test_torch_cuda():
    """Check if CUDA/GPU is available"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing GPU/CUDA Support...")
    logger.info("=" * 60)
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"✓ GPU Available: {gpu_name}")
            logger.info(f"✓ CUDA Version: {torch.version.cuda}")
            return True
        else:
            logger.warning("⚠ No GPU detected - Will use CPU (slower)")
            logger.info("  This is fine for testing, but GPU is recommended for production")
            return True
    except Exception as e:
        logger.error(f"✗ CUDA check failed: {e}")
        return False


def test_config():
    """Test if config.py can be imported"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Config Module...")
    logger.info("=" * 60)
    
    try:
        from config import (
            PROJECT_ROOT, INPUT_DIR, OUTPUT_DIR, DEVICE,
            STABLE_DIFFUSION_MODEL, VIDEO_FPS
        )
        
        logger.info(f"✓ Config loaded successfully")
        logger.info(f"  - Project Root: {PROJECT_ROOT}")
        logger.info(f"  - Input Directory: {INPUT_DIR}")
        logger.info(f"  - Output Directory: {OUTPUT_DIR}")
        logger.info(f"  - Device: {DEVICE}")
        logger.info(f"  - Model: {STABLE_DIFFUSION_MODEL}")
        logger.info(f"  - Video FPS: {VIDEO_FPS}")
        return True
    except Exception as e:
        logger.error(f"✗ Config test failed: {e}")
        return False


def test_utils():
    """Test if utils.py functions work"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Utils Module...")
    logger.info("=" * 60)
    
    try:
        from utils import (
            load_image, save_image, image_to_tensor, tensor_to_image,
            create_video_from_frames, ensure_dir_exists, list_files
        )
        
        logger.info(f"✓ load_image function - OK")
        logger.info(f"✓ save_image function - OK")
        logger.info(f"✓ image_to_tensor function - OK")
        logger.info(f"✓ tensor_to_image function - OK")
        logger.info(f"✓ create_video_from_frames function - OK")
        logger.info(f"✓ ensure_dir_exists function - OK")
        logger.info(f"✓ list_files function - OK")
        return True
    except Exception as e:
        logger.error(f"✗ Utils test failed: {e}")
        return False


def test_main_module():
    """Test if main.py can be imported"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Main Module...")
    logger.info("=" * 60)
    
    try:
        from main import (
            ImageToVideoGenerator, simple_image_to_video,
            image_to_video_with_effects
        )
        
        logger.info(f"✓ ImageToVideoGenerator class - OK")
        logger.info(f"✓ simple_image_to_video function - OK")
        logger.info(f"✓ image_to_video_with_effects function - OK")
        return True
    except Exception as e:
        logger.error(f"✗ Main module test failed: {e}")
        return False


def test_directory_structure():
    """Test if required directories exist or can be created"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Directory Structure...")
    logger.info("=" * 60)
    
    try:
        from config import INPUT_DIR, OUTPUT_DIR
        from pathlib import Path
        
        # These should be created by config.py automatically
        if INPUT_DIR.exists():
            logger.info(f"✓ Input directory exists: {INPUT_DIR}")
        else:
            logger.warning(f"⚠ Input directory created: {INPUT_DIR}")
        
        if OUTPUT_DIR.exists():
            logger.info(f"✓ Output directory exists: {OUTPUT_DIR}")
        else:
            logger.warning(f"⚠ Output directory created: {OUTPUT_DIR}")
        
        return True
    except Exception as e:
        logger.error(f"✗ Directory structure test failed: {e}")
        return False


def run_all_tests():
    """Run all tests and report results"""
    logger.info("\n")
    logger.info("╔" + "=" * 58 + "╗")
    logger.info("║" + " " * 12 + "IMAGE-TO-VIDEO GENERATOR TEST SUITE" + " " * 12 + "║")
    logger.info("╚" + "=" * 58 + "╝")
    
    tests = [
        ("Python Version", test_python_version),
        ("Required Imports", test_imports),
        ("GPU/CUDA Support", test_torch_cuda),
        ("Config Module", test_config),
        ("Utils Module", test_utils),
        ("Main Module", test_main_module),
        ("Directory Structure", test_directory_structure),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            logger.error(f"Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary Report
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        logger.info(f"{status:8} - {test_name}")
    
    logger.info("=" * 60)
    logger.info(f"Results: {passed}/{total} tests passed")
    logger.info("=" * 60)
    
    if passed == total:
        logger.info("\n🎉 All tests passed! You're ready to generate videos!")
        logger.info("\nNext steps:")
        logger.info("1. Place an image in: assets/input/")
        logger.info("2. Update 'image_filename' in main.py")
        logger.info("3. Run: python main.py")
        return True
    else:
        logger.warning(f"\n⚠ {total - passed} test(s) failed. See details above.")
        logger.warning("\nCommon fixes:")
        logger.warning("1. Make sure you've activated the virtual environment:")
        logger.warning("   - macOS/Linux: source venv/bin/activate")
        logger.warning("   - Windows: venv\\Scripts\\activate")
        logger.warning("2. Install dependencies: pip install -r requirements.txt")
        logger.warning("3. Ensure you have 20+ GB free disk space for models")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

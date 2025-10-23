import os
import sys
import json
import torch
import logging
from pathlib import Path

def setup_environment():
    """Set up the environment for Codette with modern language models."""
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create necessary directories
    config_dir = Path(__file__).parent.parent / 'config'
    config_dir.mkdir(exist_ok=True)
    
    # Check system requirements
    logger.info("Checking system requirements...")
    
    # Check Python version
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        logger.error("Python 3.8 or higher is required")
        return False
        
    # Check CUDA availability
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        gpu_count = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0) if gpu_count > 0 else "Unknown"
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3 if gpu_count > 0 else 0
        
        logger.info(f"CUDA is available with {gpu_count} device(s)")
        logger.info(f"GPU: {gpu_name} with {gpu_memory:.1f}GB memory")
    else:
        logger.warning("CUDA is not available - running in CPU mode will be very slow")
        
    # Create or update environment configuration
    env_config = {
        'cuda_available': cuda_available,
        'gpu_count': gpu_count if cuda_available else 0,
        'gpu_memory': gpu_memory if cuda_available else 0,
        'python_version': f"{python_version.major}.{python_version.minor}.{python_version.micro}",
        'torch_version': torch.__version__
    }
    
    with open(config_dir / 'environment.json', 'w') as f:
        json.dump(env_config, f, indent=2)
        
    # Set environment variables
    os.environ['TRANSFORMERS_CACHE'] = str(Path.home() / '.cache' / 'huggingface')
    os.environ['TORCH_HOME'] = str(Path.home() / '.cache' / 'torch')
    
    # Optimize for inference
    if cuda_available:
        torch.backends.cudnn.benchmark = True
        
    logger.info("Environment setup complete")
    return True

if __name__ == '__main__':
    setup_environment()
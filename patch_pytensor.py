import os
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def patch_environment():
    """Set environment variables to force CPU-only mode"""
    os.environ['PYTENSOR_FLAGS'] = 'cxx='
    os.environ['PYTENSOR_DEFAULT_MODE'] = 'FAST_COMPILE'
    logger.info("PyTensor environment patched for CPU-only mode")

if __name__ == '__main__':
    patch_environment()

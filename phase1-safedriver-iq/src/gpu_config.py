"""
GPU Configuration Module

Centralized GPU detection and configuration for all ML frameworks.
This module automatically detects available GPUs and configures
all supported frameworks (XGBoost, LightGBM, PyTorch, TensorFlow).
"""

import os
import subprocess
import logging
from typing import Dict, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GPUConfig:
    """Centralized GPU configuration for all ML frameworks."""
    
    def __init__(self):
        """Initialize GPU configuration."""
        self.gpu_available = self._check_gpu_available()
        self.gpu_info = self._get_gpu_info() if self.gpu_available else {}
        self.cuda_version = self._get_cuda_version() if self.gpu_available else None
        
        if self.gpu_available:
            self._configure_frameworks()
            self._log_gpu_info()
    
    def _check_gpu_available(self) -> bool:
        """Check if NVIDIA GPU is available."""
        try:
            result = subprocess.run(
                ['nvidia-smi'], 
                capture_output=True, 
                timeout=5
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return False
    
    def _get_gpu_info(self) -> Dict[str, str]:
        """Get GPU information."""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv,noheader'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                name, memory, driver = result.stdout.strip().split(', ')
                return {
                    'name': name,
                    'memory': memory,
                    'driver_version': driver
                }
        except:
            pass
        return {}
    
    def _get_cuda_version(self) -> Optional[str]:
        """Get CUDA version."""
        try:
            # Try PyTorch first
            import torch
            if torch.cuda.is_available():
                return torch.version.cuda
        except:
            pass
        
        try:
            # Try nvidia-smi
            result = subprocess.run(
                ['nvidia-smi'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if 'CUDA Version' in line:
                        return line.split('CUDA Version:')[1].strip().split()[0]
        except:
            pass
        
        return None
    
    def _configure_frameworks(self):
        """Configure all ML frameworks for GPU."""
        # Configure PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                # Set default device
                torch.set_default_device('cuda')
                # Enable TF32 for better performance on Ampere+ GPUs
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                logger.info("✅ PyTorch GPU acceleration enabled")
        except ImportError:
            pass
        
        # Configure TensorFlow
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                try:
                    # Enable memory growth to prevent OOM errors
                    for gpu in gpus:
                        tf.config.experimental.set_memory_growth(gpu, True)
                    logger.info("✅ TensorFlow GPU acceleration enabled")
                except RuntimeError as e:
                    logger.warning(f"TensorFlow GPU config failed: {e}")
        except ImportError:
            pass
        
        # Set environment variables for better GPU performance
        os.environ.setdefault('CUDA_DEVICE_ORDER', 'PCI_BUS_ID')
        os.environ.setdefault('TF_FORCE_GPU_ALLOW_GROWTH', 'true')
    
    def _log_gpu_info(self):
        """Log GPU configuration information."""
        logger.info("="*60)
        logger.info("🚀 GPU CONFIGURATION")
        logger.info("="*60)
        logger.info(f"GPU Available: YES")
        if self.gpu_info:
            logger.info(f"GPU Name: {self.gpu_info.get('name', 'Unknown')}")
            logger.info(f"GPU Memory: {self.gpu_info.get('memory', 'Unknown')}")
            logger.info(f"Driver Version: {self.gpu_info.get('driver_version', 'Unknown')}")
        if self.cuda_version:
            logger.info(f"CUDA Version: {self.cuda_version}")
        logger.info("="*60)
    
    def get_xgboost_params(self) -> Dict:
        """Get XGBoost GPU parameters."""
        if not self.gpu_available:
            return {'n_jobs': -1}
        
        return {
            'tree_method': 'hist',
            'device': 'cuda'
        }
    
    def get_lightgbm_params(self) -> Dict:
        """Get LightGBM GPU parameters."""
        if not self.gpu_available:
            return {'n_jobs': -1}
        
        return {
            'device': 'gpu',
            'gpu_platform_id': 0,
            'gpu_device_id': 0
        }
    
    def get_pytorch_device(self) -> str:
        """Get PyTorch device string."""
        try:
            import torch
            if torch.cuda.is_available():
                return 'cuda'
        except ImportError:
            pass
        return 'cpu'
    
    def get_tensorflow_device(self) -> str:
        """Get TensorFlow device string."""
        try:
            import tensorflow as tf
            if tf.config.list_physical_devices('GPU'):
                return '/GPU:0'
        except ImportError:
            pass
        return '/CPU:0'
    
    def print_status(self):
        """Print detailed GPU status for all frameworks."""
        print("\n" + "="*70)
        print("🖥️  GPU ACCELERATION STATUS")
        print("="*70)
        
        if not self.gpu_available:
            print("⚠️  No GPU detected - all frameworks will use CPU")
            print("="*70)
            return
        
        print(f"\n✅ GPU Detected: {self.gpu_info.get('name', 'Unknown')}")
        print(f"   Memory: {self.gpu_info.get('memory', 'Unknown')}")
        print(f"   CUDA Version: {self.cuda_version or 'Unknown'}")
        print(f"   Driver Version: {self.gpu_info.get('driver_version', 'Unknown')}")
        
        print("\n📦 Framework GPU Support:")
        print("-" * 70)
        
        # Check PyTorch
        try:
            import torch
            if torch.cuda.is_available():
                print(f"   ✅ PyTorch {torch.__version__}")
                print(f"      Device: {torch.cuda.get_device_name(0)}")
                print(f"      Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
            else:
                print(f"   ⚠️  PyTorch {torch.__version__} - CUDA not available")
        except ImportError:
            print("   ❌ PyTorch not installed")
        
        # Check TensorFlow
        try:
            import tensorflow as tf
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                print(f"   ✅ TensorFlow {tf.__version__}")
                print(f"      Devices: {len(gpus)} GPU(s) detected")
            else:
                print(f"   ⚠️  TensorFlow {tf.__version__} - No GPU devices")
        except ImportError:
            print("   ❌ TensorFlow not installed")
        
        # Check XGBoost
        try:
            import xgboost as xgb
            print(f"   ✅ XGBoost {xgb.__version__}")
            print(f"      GPU Config: tree_method='hist', device='cuda'")
        except ImportError:
            print("   ❌ XGBoost not installed")
        
        # Check LightGBM
        try:
            import lightgbm as lgb
            print(f"   ✅ LightGBM {lgb.__version__}")
            print(f"      GPU Config: device='gpu' (requires GPU build)")
        except ImportError:
            print("   ❌ LightGBM not installed")
        
        print("\n" + "="*70)
        print("💡 All GPU-capable frameworks will automatically use GPU acceleration")
        print("="*70 + "\n")


# Global GPU configuration instance
_gpu_config = None


def get_gpu_config() -> GPUConfig:
    """Get or create global GPU configuration instance."""
    global _gpu_config
    if _gpu_config is None:
        _gpu_config = GPUConfig()
    return _gpu_config


def is_gpu_available() -> bool:
    """Check if GPU is available."""
    return get_gpu_config().gpu_available


def print_gpu_status():
    """Print GPU status for all frameworks."""
    get_gpu_config().print_status()


# Auto-configure on import
if __name__ != "__main__":
    _gpu_config = GPUConfig()

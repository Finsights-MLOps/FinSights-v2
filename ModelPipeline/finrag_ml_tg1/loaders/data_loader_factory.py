"""
Data Loader Factory - Auto-detect environment and create appropriate loader

Usage:
    config = MLConfig()
    loader = create_data_loader(config)
    
    # Automatically returns LocalCacheLoader or S3StreamingLoader
"""

import logging
from .ml_config_loader import MLConfig
from .data_loader_strategy import DataLoaderStrategy, LocalCacheLoader, S3StreamingLoader

logger = logging.getLogger(__name__)


def create_data_loader(config: MLConfig) -> DataLoaderStrategy:
    """
    Factory: Auto-detect environment and return appropriate data loader.
    
    Detection Logic:
        1. Check config.data_loading_mode ('LOCAL_CACHE' or 'S3_STREAMING')
        2. If Lambda env detected → S3StreamingLoader
        3. If local ModelPipeline exists → LocalCacheLoader
        4. Fallback → S3StreamingLoader
    
    Args:
        config: Initialized MLConfig instance
    
    Returns:
        DataLoaderStrategy implementation (LocalCacheLoader or S3StreamingLoader)
    """
    
    mode = config.data_loading_mode
    logger.info(f"Creating data loader: mode={mode}")
    
    if mode == 'S3_STREAMING':
        logger.info("  → S3StreamingLoader (Lambda or cloud environment)")
        return S3StreamingLoader(config)
    
    elif mode == 'LOCAL_CACHE':
        logger.info(f"  → LocalCacheLoader (local dev, model_root={config.model_root})")
        return LocalCacheLoader(config.model_root, config)
    
    else:
        # Should never happen, but safe fallback
        logger.warning(f"Unknown data_loading_mode: {mode}, defaulting to S3StreamingLoader")
        return S3StreamingLoader(config)
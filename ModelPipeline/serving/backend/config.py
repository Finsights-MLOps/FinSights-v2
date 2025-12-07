"""
Configuration management for FinRAG backend. C1.

Loads settings from:
1. Environment variables (.env file)
2. Default values
3. Auto-detected paths

Usage:
    from backend.config import get_config
    
    config = get_config()
    print(config.model_root)  # Path to ML code
    print(config.backend_port)  # 8000
"""

from pathlib import Path
from typing import Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from loguru import logger
import os


class BackendConfig(BaseSettings):
    """
    Backend configuration with validation.
    
    Fields are loaded from environment variables or defaults.
    Pydantic validates types and required fields on startup.
    """
    
    # ========================================================================
    # PATHS
    # ========================================================================
    
    model_pipeline_root: Path = Field(
        default_factory=lambda: _detect_model_pipeline_root(),
        description="Path to ModelPipeline root directory (parent of finrag_ml_tg1)"
    )
    
    # ========================================================================
    # SERVER SETTINGS
    # ========================================================================
    
    backend_host: str = Field(
        default="0.0.0.0",
        description="Host to bind backend server"
    )
    
    backend_port: int = Field(
        default=8000,
        description="Port for backend API"
    )
    
    backend_reload: bool = Field(
        default=True,
        description="Auto-reload on code changes (dev only)"
    )
    
    # ========================================================================
    # LOGGING
    # ========================================================================
    
    log_level: str = Field(
        default="INFO",
        description="Logging level: DEBUG, INFO, WARNING, ERROR"
    )
    
    log_format: str = Field(
        default="text",
        description="Log format: 'text' or 'json'"
    )
    

    #: Export control for debugging
    enable_exports: bool = Field(
        default=True,  # True in dev, False in production
        description="Export context and response files for debugging (backend dev feature)"
    )


    # ========================================================================
    # CACHING
    # ========================================================================
    
    enable_cache: bool = Field(
        default=True,
        description="Enable query result caching"
    )
    
    cache_ttl_seconds: int = Field(
        default=300,
        description="Cache time-to-live (5 minutes default)"
    )
    
    # ========================================================================
    # PYDANTIC SETTINGS CONFIG
    # ========================================================================
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        protected_namespaces=()  # Allow model_* field names
    )
    
    

    def validate_config(self) -> None:
        """Validate configuration after loading."""
        # Check ModelPipeline root exists
        if not self.model_pipeline_root.exists():
            raise ValueError(
                f"ModelPipeline root does not exist: {self.model_pipeline_root}"
            )
        
        # Check finrag_ml_tg1 exists inside it
        finrag_ml_dir = self.model_pipeline_root / "finrag_ml_tg1"
        if not finrag_ml_dir.exists():
            raise ValueError(
                f"finrag_ml_tg1 not found in ModelPipeline: {self.model_pipeline_root}"
            )
        
        # Check orchestrator exists
        orchestrator_path = (
            finrag_ml_dir / 
            "rag_modules_src" / 
            "synthesis_pipeline" / 
            "orchestrator.py"
        )
        if not orchestrator_path.exists():
            raise ValueError(
                f"Orchestrator not found at: {orchestrator_path}"
            )
        
        # Check AWS credentials
        if not (os.getenv("AWS_PROFILE") or os.getenv("AWS_ACCESS_KEY_ID")):
            logger.warning(
                "AWS credentials not detected in environment. "
                "Ensure AWS CLI is configured or credentials are set."
            )
        
        logger.info(f"✅ Configuration validated: ModelPipeline root={self.model_pipeline_root}")




def _detect_model_pipeline_root() -> Path:
    """
    Auto-detect ModelPipeline root directory by walking up from current file.
    
    Walks up parent directories until it finds a directory named 'ModelPipeline'.
    This matches the pattern used in notebooks and ensures consistent path resolution.
    
    Returns:
        Path to ModelPipeline directory (parent of finrag_ml_tg1)
        
    Raises:
        RuntimeError: If ModelPipeline directory cannot be found
        
    Example:
        /path/to/ModelPipeline/serving/backend/config.py
        → Returns: /path/to/ModelPipeline
    """
    # Start from this file's location (backend/config.py)
    current = Path(__file__).resolve().parent  # backend/
    
    # Walk up to find ModelPipeline directory (same as notebook)
    for parent in [current] + list(current.parents):
        if parent.name == "ModelPipeline":
            logger.info(f"✅ Auto-detected ModelPipeline root: {parent}")
            return parent
    
    # If not found, raise error
    raise RuntimeError(
        "Cannot find 'ModelPipeline' directory in path tree. "
        f"Started from: {current}"
    )


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_config_instance: Optional[BackendConfig] = None


def get_config() -> BackendConfig:
    """
    Get singleton configuration instance.
    
    Loads configuration once on first call, reuses thereafter.
    
    Returns:
        BackendConfig instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = BackendConfig()
        _config_instance.validate_config()
    
    return _config_instance


def reload_config() -> BackendConfig:
    """
    Force reload configuration (useful for testing).
    
    Returns:
        New BackendConfig instance
    """
    global _config_instance
    _config_instance = None
    return get_config()
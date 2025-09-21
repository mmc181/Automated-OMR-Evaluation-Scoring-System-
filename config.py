"""
Configuration Management for OMR Evaluation System
Centralized configuration handling with environment variable support
"""
import os
from pathlib import Path
from typing import Dict, Any, Optional
import json

class Config:
    """Main configuration class"""
    
    # Base directories
    BASE_DIR = Path(__file__).parent.absolute()
    UPLOAD_DIR = BASE_DIR / "uploads"
    PROCESSED_DIR = BASE_DIR / "processed"
    EXPORTS_DIR = BASE_DIR / "exports"
    LOGS_DIR = BASE_DIR / "logs"
    TEMP_DIR = BASE_DIR / "temp"
    
    # Database configuration
    DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/omr_system.db")
    DATABASE_ECHO = os.getenv("DATABASE_ECHO", "false").lower() == "true"
    
    # API configuration
    API_HOST = os.getenv("API_HOST", "127.0.0.1")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    API_RELOAD = os.getenv("API_RELOAD", "true").lower() == "true"
    API_WORKERS = int(os.getenv("API_WORKERS", "1"))
    
    # Frontend configuration
    FRONTEND_HOST = os.getenv("FRONTEND_HOST", "127.0.0.1")
    FRONTEND_PORT = int(os.getenv("FRONTEND_PORT", "8501"))
    
    # File upload configuration
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    ALLOWED_EXTENSIONS = {
        "images": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
        "documents": [".pdf"]
    }
    
    # OMR Processing configuration
    OMR_CONFIG = {
        "min_confidence_threshold": float(os.getenv("OMR_MIN_CONFIDENCE", "0.7")),
        "max_processing_retries": int(os.getenv("OMR_MAX_RETRIES", "3")),
        "enable_quality_checks": os.getenv("OMR_QUALITY_CHECKS", "true").lower() == "true",
        "enable_auto_rotation": os.getenv("OMR_AUTO_ROTATION", "true").lower() == "true",
        "enable_perspective_correction": os.getenv("OMR_PERSPECTIVE_CORRECTION", "true").lower() == "true",
        "enable_noise_reduction": os.getenv("OMR_NOISE_REDUCTION", "true").lower() == "true",
        "bubble_fill_threshold": float(os.getenv("OMR_BUBBLE_THRESHOLD", "0.6")),
        "image_quality_threshold": float(os.getenv("OMR_QUALITY_THRESHOLD", "0.8")),
        "max_skew_angle": float(os.getenv("OMR_MAX_SKEW", "5.0")),
        "min_resolution": tuple(map(int, os.getenv("OMR_MIN_RESOLUTION", "600,800").split(","))),
        "processing_timeout": int(os.getenv("OMR_TIMEOUT", "300"))
    }
    
    # Security configuration
    SECRET_KEY = os.getenv("SECRET_KEY", "your-secret-key-change-in-production")
    ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "30"))
    
    # Logging configuration
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE_MAX_SIZE = int(os.getenv("LOG_FILE_MAX_SIZE", "10485760"))  # 10MB
    LOG_FILE_BACKUP_COUNT = int(os.getenv("LOG_FILE_BACKUP_COUNT", "5"))
    
    # Email configuration (for notifications)
    SMTP_SERVER = os.getenv("SMTP_SERVER", "")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USERNAME = os.getenv("SMTP_USERNAME", "")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
    SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
    
    # System configuration
    ENABLE_BACKGROUND_PROCESSING = os.getenv("ENABLE_BACKGROUND_PROCESSING", "true").lower() == "true"
    MAX_CONCURRENT_PROCESSING = int(os.getenv("MAX_CONCURRENT_PROCESSING", "3"))
    ENABLE_AUTO_BACKUP = os.getenv("ENABLE_AUTO_BACKUP", "true").lower() == "true"
    BACKUP_RETENTION_DAYS = int(os.getenv("BACKUP_RETENTION_DAYS", "30"))
    
    @classmethod
    def create_directories(cls):
        """Create necessary directories if they don't exist"""
        directories = [
            cls.UPLOAD_DIR,
            cls.PROCESSED_DIR,
            cls.EXPORTS_DIR,
            cls.LOGS_DIR,
            cls.TEMP_DIR
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_database_url(cls) -> str:
        """Get database URL with proper formatting"""
        return cls.DATABASE_URL
    
    @classmethod
    def get_upload_path(cls, filename: str) -> Path:
        """Get full path for uploaded file"""
        return cls.UPLOAD_DIR / filename
    
    @classmethod
    def get_processed_path(cls, filename: str) -> Path:
        """Get full path for processed file"""
        return cls.PROCESSED_DIR / filename
    
    @classmethod
    def get_export_path(cls, filename: str) -> Path:
        """Get full path for export file"""
        return cls.EXPORTS_DIR / filename
    
    @classmethod
    def is_allowed_file(cls, filename: str) -> bool:
        """Check if file extension is allowed"""
        file_ext = Path(filename).suffix.lower()
        all_extensions = []
        for ext_list in cls.ALLOWED_EXTENSIONS.values():
            all_extensions.extend(ext_list)
        return file_ext in all_extensions
    
    @classmethod
    def get_file_type(cls, filename: str) -> Optional[str]:
        """Get file type category"""
        file_ext = Path(filename).suffix.lower()
        for file_type, extensions in cls.ALLOWED_EXTENSIONS.items():
            if file_ext in extensions:
                return file_type
        return None
    
    @classmethod
    def load_from_file(cls, config_file: str = "config.json"):
        """Load configuration from JSON file"""
        config_path = cls.BASE_DIR / config_file
        if config_path.exists():
            try:
                with open(config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Update class attributes with loaded config
                for key, value in config_data.items():
                    if hasattr(cls, key):
                        setattr(cls, key, value)
                
                print(f"✅ Configuration loaded from {config_file}")
            except Exception as e:
                print(f"❌ Error loading configuration from {config_file}: {e}")
    
    @classmethod
    def save_to_file(cls, config_file: str = "config.json"):
        """Save current configuration to JSON file"""
        config_path = cls.BASE_DIR / config_file
        
        # Collect configuration data
        config_data = {
            "DATABASE_URL": cls.DATABASE_URL,
            "API_HOST": cls.API_HOST,
            "API_PORT": cls.API_PORT,
            "FRONTEND_HOST": cls.FRONTEND_HOST,
            "FRONTEND_PORT": cls.FRONTEND_PORT,
            "MAX_FILE_SIZE_MB": cls.MAX_FILE_SIZE_MB,
            "OMR_CONFIG": cls.OMR_CONFIG,
            "LOG_LEVEL": cls.LOG_LEVEL,
            "ENABLE_BACKGROUND_PROCESSING": cls.ENABLE_BACKGROUND_PROCESSING,
            "MAX_CONCURRENT_PROCESSING": cls.MAX_CONCURRENT_PROCESSING
        }
        
        try:
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            print(f"✅ Configuration saved to {config_file}")
        except Exception as e:
            print(f"❌ Error saving configuration to {config_file}: {e}")
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate configuration and return status"""
        validation_results = {
            "valid": True,
            "errors": [],
            "warnings": []
        }
        
        # Check required directories
        try:
            cls.create_directories()
        except Exception as e:
            validation_results["errors"].append(f"Cannot create directories: {e}")
            validation_results["valid"] = False
        
        # Check database URL
        if not cls.DATABASE_URL:
            validation_results["errors"].append("DATABASE_URL is not set")
            validation_results["valid"] = False
        
        # Check OMR configuration
        if cls.OMR_CONFIG["min_confidence_threshold"] < 0 or cls.OMR_CONFIG["min_confidence_threshold"] > 1:
            validation_results["errors"].append("OMR confidence threshold must be between 0 and 1")
            validation_results["valid"] = False
        
        if cls.OMR_CONFIG["bubble_fill_threshold"] < 0 or cls.OMR_CONFIG["bubble_fill_threshold"] > 1:
            validation_results["errors"].append("Bubble fill threshold must be between 0 and 1")
            validation_results["valid"] = False
        
        # Check ports
        if cls.API_PORT == cls.FRONTEND_PORT:
            validation_results["warnings"].append("API and Frontend ports are the same")
        
        # Check file size limits
        if cls.MAX_FILE_SIZE_MB > 100:
            validation_results["warnings"].append("Large file size limit may cause memory issues")
        
        return validation_results

class DevelopmentConfig(Config):
    """Development environment configuration"""
    DATABASE_ECHO = True
    API_RELOAD = True
    LOG_LEVEL = "DEBUG"

class ProductionConfig(Config):
    """Production environment configuration"""
    DATABASE_ECHO = False
    API_RELOAD = False
    LOG_LEVEL = "WARNING"
    API_WORKERS = 4
    MAX_CONCURRENT_PROCESSING = 6

class TestingConfig(Config):
    """Testing environment configuration"""
    DATABASE_URL = "sqlite:///:memory:"
    UPLOAD_DIR = Config.BASE_DIR / "test_uploads"
    PROCESSED_DIR = Config.BASE_DIR / "test_processed"
    EXPORTS_DIR = Config.BASE_DIR / "test_exports"
    LOG_LEVEL = "DEBUG"

def get_config() -> Config:
    """Get configuration based on environment"""
    env = os.getenv("ENVIRONMENT", "development").lower()
    
    if env == "production":
        return ProductionConfig()
    elif env == "testing":
        return TestingConfig()
    else:
        return DevelopmentConfig()

# Global configuration instance
config = get_config()

# Ensure directories exist
config.create_directories()
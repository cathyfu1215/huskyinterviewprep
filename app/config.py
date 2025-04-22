import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Base configuration settings"""
    # Secret key for session management
    SECRET_KEY = os.environ.get('SECRET_KEY') or os.urandom(24)
    
    # Together API configuration
    TOGETHER_API_KEY = os.environ.get('TOGETHER_API_KEY')
    TOGETHER_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"
    
    # Speech recognition settings
    DEFAULT_VOICE_OPTION = "US English"
    
    # Application settings
    DEBUG = False
    TESTING = False


class DevelopmentConfig(Config):
    """Development configuration settings"""
    DEBUG = True


class TestingConfig(Config):
    """Testing configuration settings"""
    TESTING = True
    DEBUG = True


class ProductionConfig(Config):
    """Production configuration settings"""
    # Production specific settings
    pass


# Configuration dictionary to easily select environment
config_by_name = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}


def get_config():
    """
    Get configuration based on environment variable.
    Defaults to development if not specified.
    """
    env = os.environ.get('FLASK_ENV', 'development')
    return config_by_name.get(env, config_by_name['default']) 
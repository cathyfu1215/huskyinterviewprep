from flask import Flask
import os
from dotenv import load_dotenv
import together

# Load environment variables
load_dotenv()

# Initialize Together client
together.api_key = os.getenv("TOGETHER_API_KEY")


def create_app():
    """Create and configure the Flask application"""
    # Create app with static and template folders inside the app directory
    app = Flask(__name__)
    
    # Load configuration
    from app.config import get_config
    app.config.from_object(get_config())
    
    # Initialize services
    from app.services.interview_manager import InterviewAgentManager
    app.interview_manager = InterviewAgentManager()
    
    # Register blueprints
    from app.routes.main import main_bp
    from app.routes.analysis import analysis_bp
    from app.routes.questions import questions_bp
    from app.routes.answers import answers_bp
    from app.routes.export import export_bp
    from app.routes.speech import speech_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(analysis_bp)
    app.register_blueprint(questions_bp)
    app.register_blueprint(answers_bp)
    app.register_blueprint(export_bp)
    app.register_blueprint(speech_bp)
    
    return app 
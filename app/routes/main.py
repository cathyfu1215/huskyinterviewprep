from flask import Blueprint, render_template, request, redirect, url_for

main_bp = Blueprint('main', __name__)


@main_bp.route('/')
def index():
    """Render the home page"""
    return render_template('index.html')


# Adapter routes for backward compatibility with the frontend
@main_bp.route('/analyze-info', methods=['POST'])
def analyze_info_adapter():
    """Adapter to redirect to the analysis blueprint"""
    from app.routes.analysis import analyze_info_endpoint
    return analyze_info_endpoint()


@main_bp.route('/generate-questions', methods=['POST'])
def generate_questions_adapter():
    """Adapter to redirect to the questions blueprint"""
    from app.routes.questions import generate_questions_endpoint
    return generate_questions_endpoint()


@main_bp.route('/speech-to-text', methods=['POST'])
def speech_to_text_adapter():
    """Adapter to redirect to the speech blueprint"""
    from app.routes.speech import speech_to_text_endpoint
    return speech_to_text_endpoint()


@main_bp.route('/analyze-answer', methods=['POST'])
def analyze_answer_adapter():
    """Adapter to redirect to the answers blueprint"""
    from app.routes.answers import analyze_answer_endpoint
    return analyze_answer_endpoint()


@main_bp.route('/generate-model-answer', methods=['POST'])
def generate_model_answer_adapter():
    """Adapter to redirect to the answers blueprint"""
    from app.routes.answers import generate_model_answer_endpoint
    return generate_model_answer_endpoint()


@main_bp.route('/text-to-speech', methods=['POST'])
def text_to_speech_adapter():
    """Adapter to redirect to the speech blueprint"""
    from app.routes.speech import text_to_speech_endpoint
    return text_to_speech_endpoint()


@main_bp.route('/generate-follow-up-questions', methods=['POST'])
def generate_follow_up_questions_adapter():
    """Adapter to redirect to the questions blueprint"""
    from app.routes.questions import generate_follow_up_questions_endpoint
    return generate_follow_up_questions_endpoint()


@main_bp.route('/save-to-html', methods=['POST'])
def save_to_html_adapter():
    """Adapter to redirect to the export blueprint"""
    from app.routes.export import save_to_html_endpoint
    return save_to_html_endpoint()


@main_bp.route('/download-html/<file_id>', methods=['GET'])
def download_html_adapter(file_id):
    """Adapter to redirect to the export blueprint"""
    from app.routes.export import download_html
    return download_html(file_id) 
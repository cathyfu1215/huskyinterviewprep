"""
Speech blueprint - Handles routes for speech-to-text and text-to-speech functionality
"""
from flask import Blueprint, request, jsonify

# Create a blueprint for speech-related routes
speech_bp = Blueprint('speech', __name__, url_prefix='/api/speech')


@speech_bp.route('/speech-to-text', methods=['POST'])
def speech_to_text_endpoint():
    """
    Convert audio data to text.
    
    This endpoint accepts base64 encoded audio data and returns 
    the transcribed text.
    
    Returns:
        JSON response with transcribed text
    """
    data = request.get_json()
    audio_data = data.get('audio', '')
    
    if not audio_data:
        return jsonify({'error': 'No audio data provided'}), 400
    
    # Import the speech-to-text function
    from app.services.speech import speech_to_text
    
    text = speech_to_text(audio_data)
    return jsonify({'text': text})


@speech_bp.route('/text-to-speech', methods=['POST'])
def text_to_speech_endpoint():
    """
    Convert text to speech.
    
    This endpoint accepts text and voice option and returns
    base64 encoded audio data.
    
    Returns:
        JSON response with audio data
    """
    data = request.get_json()
    text = data.get('text', '')
    voice_option = data.get('voice_option', 'US English')
    
    # Import the text-to-speech function
    from app.services.speech import text_to_speech
    
    audio_base64 = text_to_speech(text, voice_option)
    
    return jsonify({'audio': audio_base64})


@speech_bp.route('/voice-options', methods=['GET'])
def get_voice_options_endpoint():
    """
    Get available voice options for text-to-speech.
    
    Returns:
        JSON response with voice options
    """
    # Import the get_voice_options function
    from app.services.speech import get_voice_options
    
    voices = get_voice_options()
    return jsonify({'voices': voices}) 
"""
Speech Service - Handles speech-to-text and text-to-speech functionality
"""
import os
import tempfile
import base64
import speech_recognition as sr
from gtts import gTTS


def get_voice_options():
    """
    Returns a dictionary of available voice options with their language parameters.
    
    Returns:
        dict: Voice options with language and TLD parameters
    """
    return {
        "US English": {"lang": "en", "tld": "com"},
        "UK English": {"lang": "en", "tld": "co.uk"},
        "Australian English": {"lang": "en", "tld": "com.au"},
        "Indian English": {"lang": "en", "tld": "co.in"},
        "French": {"lang": "fr", "tld": "fr"},
        "German": {"lang": "de", "tld": "de"},
        "Spanish": {"lang": "es", "tld": "es"},
        "Italian": {"lang": "it", "tld": "it"},
        "Japanese": {"lang": "ja", "tld": "co.jp"},
        "Korean": {"lang": "ko", "tld": "co.kr"}
    }


def text_to_speech(text, voice_option="US English"):
    """
    Convert text to speech and return as base64 encoded audio data.
    
    Args:
        text (str): The text to convert to speech
        voice_option (str): The voice/accent option to use
        
    Returns:
        str: Base64 encoded audio data or None if conversion fails
    """
    voice_options = get_voice_options()
    selected_voice = voice_options.get(voice_option, {"lang": "en", "tld": "com"})
    
    try:
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_filename = fp.name
        
        # Generate the speech audio file with the selected voice
        tts = gTTS(
            text=text, 
            lang=selected_voice["lang"], 
            tld=selected_voice["tld"]
        )
        tts.save(temp_filename)
        
        # Read the file and convert to base64
        with open(temp_filename, 'rb') as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
        
        # Clean up the temporary file
        os.unlink(temp_filename)
        
        return f"data:audio/mp3;base64,{audio_data}"
    except Exception as e:
        print(f"TTS Error: {e}")
        return None


def speech_to_text(audio_data):
    """
    Convert speech to text using SpeechRecognition.
    
    Args:
        audio_data (str): Base64 encoded audio data
        
    Returns:
        str: Transcribed text or error message
    """
    recognizer = sr.Recognizer()
    
    try:
        # Save base64 audio data to a temporary file
        audio_bytes = base64.b64decode(audio_data.split(',')[1])
        
        # Save as webm file first
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_audio_file:
            temp_audio_file.write(audio_bytes)
            temp_webm_path = temp_audio_file.name
        
        # Convert to WAV using FFmpeg if available, otherwise use a direct approach
        try:
            import subprocess
            wav_path = temp_webm_path.replace('.webm', '.wav')
            subprocess.call([
                'ffmpeg', 
                '-i', temp_webm_path, 
                '-ar', '16000', 
                '-ac', '1', 
                wav_path
            ])
            os.unlink(temp_webm_path)  # Delete the webm file
            
            with sr.AudioFile(wav_path) as source:
                audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            os.unlink(wav_path)  # Delete temp WAV file
            return text
        except (ImportError, FileNotFoundError):
            # If FFmpeg is not available, try direct approach with the webm file
            with sr.AudioFile(temp_webm_path) as source:
                audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            os.unlink(temp_webm_path)  # Delete temp file
            return text
    except Exception as e:
        # If an error occurs, try to delete any temporary files
        try:
            if 'temp_webm_path' in locals():
                os.unlink(temp_webm_path)
            if 'wav_path' in locals():
                os.unlink(wav_path)
        except:
            pass
        
        # Use a different approach as fallback - send directly to Google's API
        try:
            audio_bytes = base64.b64decode(audio_data.split(',')[1])
            audio_data_obj = sr.AudioData(audio_bytes, 16000, 2)  # Default values
            text = recognizer.recognize_google(audio_data_obj)
            return text
        except Exception as inner_e:
            return f"Speech recognition failed: {str(e)}. Second attempt: {str(inner_e)}" 
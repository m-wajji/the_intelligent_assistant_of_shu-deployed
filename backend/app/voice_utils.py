import speech_recognition as sr
import langid
from deep_translator import GoogleTranslator
from pydub import AudioSegment
import tempfile
import os
import logging
import openai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Setup OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

def validate_audio_file(audio_path):
    """Validate if the audio file is properly formatted"""
    try:
        file_size = os.path.getsize(audio_path)
        logger.info(f"Audio file size: {file_size} bytes")
        
        if file_size < 100:
            logger.error("Audio file is too small")
            return False
            
        with open(audio_path, 'rb') as f:
            header = f.read(32)
            
        if len(header) < 8:
            logger.error("Audio file header is too short")
            return False
            
        # Check for common audio file signatures
        audio_signatures = [
            (b'RIFF', "WAV"),
            (b'ID3', "MP3"),
            (b'\xff\xfb', "MP3"),
            (b'\xff\xf3', "MP3"),
            (b'OggS', "OGG"),
            (b'\x1a\x45\xdf\xa3', "WebM/Matroska"),
            (b'fLaC', "FLAC")
        ]
        
        for signature, format_name in audio_signatures:
            if header.startswith(signature):
                logger.info(f"{format_name} file detected")
                return True
        
        # Check for M4A/MP4
        if header[4:8] == b'ftyp':
            logger.info("M4A/MP4 file detected")
            return True
            
        # Be permissive for unknown formats
        logger.warning(f"Unknown file format, but will attempt processing. Header: {header[:8].hex()}")
        return True
            
    except Exception as e:
        logger.error(f"Error validating audio file: {e}")
        return False

def convert_audio_to_wav(audio_path):
    """Convert any audio format to WAV format suitable for speech recognition"""
    try:
        if not validate_audio_file(audio_path):
            logger.error("Audio file validation failed")
            return None
            
        file_ext = os.path.splitext(audio_path)[1].lower()
        logger.info(f"Processing {file_ext} file")
        
        sound = None
        
        # Try format-specific loading first
        format_loaders = {
            '.mp3': lambda: AudioSegment.from_mp3(audio_path),
            '.wav': lambda: AudioSegment.from_wav(audio_path),
            '.m4a': lambda: AudioSegment.from_file(audio_path, format="m4a"),
            '.ogg': lambda: AudioSegment.from_ogg(audio_path),
            '.webm': lambda: AudioSegment.from_file(audio_path, format="webm"),
            '.flac': lambda: AudioSegment.from_file(audio_path, format="flac")
        }
        
        try:
            if file_ext in format_loaders:
                sound = format_loaders[file_ext]()
            else:
                sound = AudioSegment.from_file(audio_path)
                
        except Exception as load_error:
            logger.error(f"Failed to load with specific format: {load_error}")
            
            # Fallback methods
            try:
                sound = AudioSegment.from_file(audio_path)
            except Exception as generic_error:
                logger.error(f"Generic loading failed: {generic_error}")
                
                # Try header-based detection
                try:
                    with open(audio_path, 'rb') as f:
                        header = f.read(32)
                    
                    if header.startswith(b'\x1a\x45\xdf\xa3'):
                        sound = AudioSegment.from_file(audio_path, format="webm")
                    elif header.startswith(b'OggS'):
                        sound = AudioSegment.from_file(audio_path, format="ogg")
                    else:
                        return None
                        
                except Exception as final_error:
                    logger.error(f"All loading methods failed: {final_error}")
                    return None
        
        if sound is None or len(sound) == 0:
            logger.error("Could not load audio file or file contains no data")
            return None
            
        # Normalize for speech recognition
        sound = sound.set_frame_rate(16000).set_channels(1)
        
        logger.info(f"Audio duration: {len(sound)} ms")
        logger.info(f"Audio properties: {sound.frame_rate}Hz, {sound.channels} channels, {sound.sample_width*8}bit")
        
        # Create temporary WAV file
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        temp_path = temp_file.name
        temp_file.close()
        
        # Export to WAV
        sound.export(temp_path, format='wav')
        logger.info(f"Converted audio to temporary WAV: {temp_path}")
        
        # Validate the converted file
        if os.path.exists(temp_path) and os.path.getsize(temp_path) > 44:
            logger.info(f"Successfully created WAV file: {os.path.getsize(temp_path)} bytes")
            return temp_path
        else:
            logger.error("Converted file is invalid")
            return None
        
    except Exception as e:
        logger.error(f"Error converting audio: {str(e)}")
        return None

def transcribe_with_openai_whisper(audio_path, language=None):
    """Transcribe using OpenAI Whisper API - Primary method"""
    try:
        if not openai.api_key:
            logger.error("OpenAI API key not found")
            return ""
            
        logger.info("Starting Whisper transcription...")
        
        with open(audio_path, "rb") as audio_file:
            # Use the new OpenAI client
            client = openai.OpenAI()
            
            if language:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file,
                    language=language
                )
            else:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                
        text = transcript.text.strip()
        logger.info(f"Whisper transcription successful: {text[:100]}...")
        return text
        
    except Exception as e:
        logger.error(f"Whisper transcription error: {e}")
        return ""

def transcribe_with_speech_recognition(wav_path, language='en-US'):
    """Transcribe using SpeechRecognition library - Fallback method"""
    recognizer = sr.Recognizer()
    try:
        with sr.AudioFile(wav_path) as source:
            recognizer.adjust_for_ambient_noise(source, duration=0.5)
            audio_data = recognizer.record(source)
            
            text = recognizer.recognize_google(audio_data, language=language)
            logger.info(f"SpeechRecognition transcription ({language}): {text}")
            return text
            
    except sr.UnknownValueError:
        logger.warning(f"SpeechRecognition: Could not understand audio using {language}")
        return ""
    except sr.RequestError as e:
        logger.error(f"SpeechRecognition Google API error: {e}")
        return ""
    except Exception as e:
        logger.error(f"SpeechRecognition error: {e}")
        return ""

def detect_language(text):
    """Detect language of the given text"""
    try:
        if not text or len(text.strip()) < 3:
            return 'unknown'
        lang, confidence = langid.classify(text)
        logger.info(f"Detected language: {lang} (confidence: {confidence:.2f})")
        return lang
    except Exception as e:
        logger.error(f"Language detection error: {e}")
        return 'unknown'

def is_supported_language(lang_code):
    """Check if the detected language is supported"""
    supported_languages = ['en', 'ur']  # English and Urdu
    return lang_code in supported_languages

def translate_to_english(text, source_lang='ur'):
    """Translate text to English"""
    try:
        if source_lang == 'en':
            return text  # Already in English
            
        translator = GoogleTranslator(source=source_lang, target='en')
        translated = translator.translate(text)
        logger.info(f"Translated from {source_lang} to English: {translated}")
        return translated
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return text

def process_voice(audio_path):
    """
    Process voice file and return English transcription
    Main processing function with proper language handling
    """
    logger.info(f"Processing voice file: {audio_path}")
    
    # Convert audio to WAV format
    wav_path = convert_audio_to_wav(audio_path)
    if not wav_path:
        logger.error("Failed to convert audio file")
        return None

    final_result = None
    detected_language = None

    try:
        # Primary method: Use OpenAI Whisper (more accurate)
        logger.info("Attempting transcription with OpenAI Whisper...")
        transcription = transcribe_with_openai_whisper(wav_path)
        
        if transcription:
            # Detect language of transcription
            detected_language = detect_language(transcription)
            logger.info(f"Transcription: {transcription}")
            logger.info(f"Detected language: {detected_language}")
            
            # Check if language is supported
            if not is_supported_language(detected_language):
                logger.warning(f"Unsupported language detected: {detected_language}")
                return {
                    'error': 'unsupported_language',
                    'message': 'Currently I handle Urdu and English only.',
                    'detected_language': detected_language
                }
            
            # Translate to English if needed
            if detected_language == 'ur':
                final_result = translate_to_english(transcription, 'ur')
            else:
                final_result = transcription
                
        else:
            # Fallback: Try SpeechRecognition
            logger.info("Whisper failed, trying SpeechRecognition...")
            
            # Try English first
            english_text = transcribe_with_speech_recognition(wav_path, language='en-US')
            if english_text:
                detected_language = detect_language(english_text)
                if is_supported_language(detected_language):
                    final_result = english_text
                else:
                    return {
                        'error': 'unsupported_language',
                        'message': 'Currently I handle Urdu and English only.',
                        'detected_language': detected_language
                    }
            else:
                # Try Urdu
                urdu_text = transcribe_with_speech_recognition(wav_path, language='ur-PK')
                if urdu_text:
                    detected_language = 'ur'
                    final_result = translate_to_english(urdu_text, 'ur')

    except Exception as e:
        logger.error(f"Error in process_voice: {str(e)}")
        final_result = None
        
    finally:
        # Clean up temporary WAV file
        if wav_path and os.path.exists(wav_path):
            try:
                os.remove(wav_path)
                logger.info(f"Cleaned up temporary file: {wav_path}")
            except Exception as e:
                logger.warning(f"Could not remove temporary file: {e}")

    if final_result:
        logger.info(f"Final transcription result: {final_result}")
        return {
            'transcription': final_result,
            'detected_language': detected_language,
            'success': True
        }
    else:
        logger.error("All transcription methods failed")
        return {
            'error': 'transcription_failed',
            'message': 'Could not transcribe the audio. Please try again with clearer audio.',
            'success': False
        }

# Example usage
if __name__ == "__main__":
    audio_input = "uploads/recording.wav"
    result = process_voice(audio_input)
    if result:
        if result.get('success'):
            print(f"Transcription: {result['transcription']}")
            print(f"Language: {result['detected_language']}")
        else:
            print(f"Error: {result.get('message', 'Unknown error')}")
    else:
        print("Processing failed")
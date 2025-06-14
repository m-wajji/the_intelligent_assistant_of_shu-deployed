import os
import openai
from dotenv import load_dotenv
from gtts import gTTS
from pydub import AudioSegment
import sys
from langdetect import detect
from deep_translator import GoogleTranslator
from langchain_core.messages import HumanMessage, AIMessage
import logging
import traceback
load_dotenv()

# Add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "agent")))

from app.agent.app.core.graph import graph

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import voice_utils with error handling
try:
    from app.voice_utils import process_voice
    logger.info("Successfully imported process_voice")
except ImportError as e:
    logger.error(f"Failed to import voice_utils: {e}")
    logger.error("Make sure voice_utils.py exists and process_voice function is defined")
    process_voice = None

# --- OpenAI API Key ---
openai.api_key = os.getenv("OPENAI_API_KEY")
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY not found in .env file!")

uploads_dir = "uploads"
os.makedirs(uploads_dir, exist_ok=True)

# # --- Setup for ffmpeg ---
# AudioSegment.converter = "ffmpeg.exe"
# AudioSegment.ffprobe = "ffprobe.exe"

# --- Initializing persistent chat state ---
state = {"messages": []}

def is_supported_language(lang_code):
    """Check if the language is supported"""
    supported_languages = ['en', 'ur', 'urdu']
    return lang_code.lower() in supported_languages

def translate_text(text, target_lang):
    """Translate text to target language"""
    try:
        if target_lang.lower() in ['urdu', 'ur']:
            target_lang = 'ur'
        elif target_lang.lower() == 'en':
            target_lang = 'en'
        else:
            logger.warning(f"Unsupported target language: {target_lang}, defaulting to English")
            target_lang = 'en'
            
        translator = GoogleTranslator(source="auto", target=target_lang)
        translated = translator.translate(text)
        logger.info(f"Translated text to {target_lang}: {translated[:100]}...")
        return translated
    except Exception as e:
        logger.error(f"Translation error: {str(e)}")
        return text  # Return original text if translation fails

def text_to_speech_openai(text, voice="fable"):
    """Generate speech using OpenAI TTS API - Always in English"""
    try:
        os.makedirs("audio_responses", exist_ok=True)
        audio_file = os.path.join("audio_responses", "response.mp3")
        
        client = openai.OpenAI()
        
        response = client.audio.speech.create(
            model="tts-1",
            voice=voice,  # alloy, echo, fable, onyx, nova, shimmer
            input=text
        )
        
        response.stream_to_file(audio_file)
        logger.info(f"OpenAI TTS audio saved to: {audio_file}")
        return audio_file
        
    except Exception as e:
        logger.error(f"Error in OpenAI TTS: {str(e)}")
        return None

def text_to_speech_gtts(text, lang="en"):
    """Generate speech using Google TTS (fallback)"""
    try:
        os.makedirs("audio_responses", exist_ok=True)
        audio_file = os.path.join("audio_responses", "response.mp3")
        
        # Always use English for TTS since responses are always in English
        lang = 'en'
            
        tts = gTTS(text=text, lang=lang)
        tts.save(audio_file)
        logger.info(f"gTTS audio saved to: {audio_file}")
        return audio_file
    except Exception as e:
        logger.error(f"Error in gTTS: {str(e)}")
        return None

def text_to_speech(text, lang="en", use_openai=True):
    """Convert text to speech with OpenAI TTS as primary and gTTS as fallback - Always English"""
    # Always use English for TTS since responses are always in English
    if use_openai and openai.api_key:
        audio_file = text_to_speech_openai(text)
        if audio_file:
            return audio_file
        else:
            logger.info("OpenAI TTS failed, falling back to gTTS")
    
    return text_to_speech_gtts(text, "en") 

def process_agent_query(q: str) -> str:
    """Process query through the agent"""
    try:
        # Record the user's turn
        state["messages"].append(HumanMessage(content=q, name="user"))
        state["input"] = q

        # Run the graph
        new_state = graph.invoke(state)

        # Pull out the last AIMessage
        ai_msgs = [m for m in new_state["messages"] if isinstance(m, AIMessage)]
        reply = ai_msgs[-1].content if ai_msgs else "No response generated."

        # Persist full state for the next turn
        state.clear()
        state.update(new_state)
        return reply
    except Exception as e:
        logger.error(f"Error in process_agent_query: {str(e)}")
        logger.error(traceback.format_exc())
        return f"I apologize, but I encountered an error processing your query. Please try again."

def process_text_query(text, detected_lang=None):
    """Process text query with language detection and handling - Always respond in English"""
    try:
        # Detect language if not provided
        if not detected_lang:
            try:
                detected_lang = detect(text)
                logger.info(f"Detected language for text query: {detected_lang}")
            except Exception as e:
                logger.warning(f"Language detection failed: {e}, defaulting to English")
                detected_lang = "en"
        
        # Check if language is supported
        if not is_supported_language(detected_lang):
            return {
                "error": "unsupported_language",
                "message": "I currently handle English and Urdu only.",
                "detected_language": detected_lang,
                "success": False
            }
        
        # Translate to English if needed for agent processing
        english_text = text
        if detected_lang == 'ur':
            english_text = translate_text(text, 'en')
            logger.info(f"Translated query to English: {english_text}")
        
        # Get agent response
        agent_response = process_agent_query(english_text)
        
        # ALWAYS keep response in English (removed translation back to Urdu)
        final_response = agent_response
        logger.info("Keeping response in English regardless of input language")
        
        return {
            "answer": final_response,
            "original_text": text,
            "detected_language": detected_lang,
            "response_language": "en",  # Always English
            "success": True
        }
        
    except Exception as e:
        logger.error(f"Error processing text query: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "error": "processing_failed",
            "message": "Error processing your query. Please try again."
        }

def process_audio_file(audio_filename):
    """Process audio file with comprehensive language handling - Always respond in English"""
    try:
        audio_path = os.path.join(uploads_dir, audio_filename)
        logger.info(f"Processing audio file: {audio_path}")
        
        if not os.path.exists(audio_path):
            error_msg = f"File not found: {audio_path}"
            logger.error(error_msg)
            return {"error": error_msg}

        # Check if process_voice was imported successfully
        if process_voice is None:
            error_msg = "Voice processing module not available"
            logger.error(error_msg)
            return {"error": error_msg}

        # Transcribe audio
        logger.info("Starting voice transcription...")
        transcription_result = process_voice(audio_path)
        
        if not transcription_result or not transcription_result.get('success'):
            if transcription_result and transcription_result.get('error') == 'unsupported_language':
                return {
                    "error": "unsupported_language", 
                    "message": "I currently handle English and Urdu only.",
                    "detected_language": transcription_result.get('detected_language'),
                    "success": False
                }
            else:
                error_msg = transcription_result.get('message', 'Could not transcribe audio.')
                logger.error(error_msg)
                return {"error": error_msg}

        transcription = transcription_result['transcription']
        detected_lang = transcription_result['detected_language']
        
        logger.info(f"Transcription successful: {transcription[:100]}...")
        logger.info(f"Detected language: {detected_lang}")

        # Get agent response (transcription is already in English)
        logger.info("Getting agent response...")
        agent_response = process_agent_query(transcription)

        # ALWAYS keep response in English (removed translation back to original language)
        final_response = agent_response
        response_lang = 'en'
        
        logger.info("Keeping response in English regardless of input language")

        # Generate audio response using OpenAI TTS (primary) or gTTS (fallback) - Always in English
        logger.info("Generating audio response in English...")
        audio_file = text_to_speech(final_response, response_lang, use_openai=True)

        result = {
            "transcription": transcription,
            "original_language": detected_lang,
            "answer": final_response,
            "response_language": "en", 
            "audio_response_url": (
                f"/download-audio/{os.path.basename(audio_file)}"
                if audio_file
                else None
            ),
            "success": True
        }
        
        if not audio_file:
            result["audio_error"] = "Failed to generate audio response"
        
        logger.info("Audio processing completed successfully")
        return result

    except Exception as e:
        error_msg = f"Error processing audio file: {str(e)}"
        logger.error(error_msg)
        logger.error(traceback.format_exc())
        return {"error": error_msg}

def process_query(query_text=None, audio_filename=None):
    """Unified query processing function"""
    try:
        if audio_filename:
            return process_audio_file(audio_filename)
        elif query_text:
            return process_text_query(query_text)
        else:
            return {"error": "No query text or audio file provided"}
    except Exception as e:
        logger.error(f"Error in process_query: {str(e)}")
        return {"error": f"Processing error: {str(e)}"}

if __name__ == "__main__":
    print("SHU Assistant - Supports English and Urdu input, Always responds in English")
    print("Type 'quit' or 'exit' to end the session")
    print("-" * 50)
    
    while True:
        try:
            q = input("\nAsk SHU Assistant: ").strip()
            if not q or q.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            result = process_text_query(q)
            
            if result.get('success'):
                print(f"\nResponse: {result['answer']}")
                if result.get('detected_language'):
                    print(f"Input Language: {result['detected_language']}")
                print(f"Response Language: English")
            elif result.get('error') == 'unsupported_language':
                print(f"\nError: {result.get('message')}")
                print(f"Detected Language: {result.get('detected_language', 'Unknown')}")
            else:
                print(f"\nError: {result.get('message', 'Unknown error')}")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")
            logger.error(f"CLI error: {str(e)}")
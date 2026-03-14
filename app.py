import os
import tempfile
import logging
from flask import Flask, request, jsonify, render_template
import replicate

# ==================== CONFIGURATION ====================
# Set up logging (Render captures stdout)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Read API token from environment (must be set on Render)
REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')
if not REPLICATE_API_TOKEN:
    logger.error("REPLICATE_API_TOKEN environment variable not set")
    raise ValueError("REPLICATE_API_TOKEN environment variable not set")

# Initialize Replicate client
client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# ==================== MODEL VERSION MANAGEMENT ====================
# Hardcoded known good versions (as of March 2026)
# These can be overridden by environment variables for flexibility.
DEFAULT_TTS_VERSION = "9c3e0d6"                     # suno/bark
DEFAULT_SADTALKER_VERSION = "aad6e5d"               # lucataco/sadtalker (example – verify on replicate.com)
DEFAULT_ANIMATE_DIFF_VERSION = "a3f4b7c"             # lucataco/animate-diff (example – verify on replicate.com)

# Try to fetch latest versions dynamically, but fall back to hardcoded if it fails.
def get_model_version(model_name, default_version):
    """
    Attempt to fetch the latest version of a model from Replicate.
    If the API call fails (invalid token, network error, or model not found),
    return the default version and log a warning.
    """
    try:
        logger.info(f"Fetching latest version for {model_name}...")
        model = client.models.get(model_name)
        # versions.list() returns a list, newest first (index 0)
        latest = model.versions.list()[0].id
        logger.info(f"Latest version for {model_name} is {latest}")
        return latest
    except Exception as e:
        logger.warning(f"Could not fetch latest version for {model_name}: {e}")
        logger.warning(f"Using default version {default_version}")
        return default_version

# Determine which versions to use (environment variable > dynamic fetch > hardcoded default)
def get_version_from_env_or_fetch(env_var, model_name, default_version):
    """
    Priority:
      1. Environment variable (e.g., BARK_VERSION)
      2. Dynamically fetched version (if fetch succeeds)
      3. Hardcoded default version
    """
    env_version = os.environ.get(env_var)
    if env_version:
        logger.info(f"Using {env_var}={env_version} from environment")
        return env_version
    # Fetch dynamically (but may fail, in which case fallback to default)
    return get_model_version(model_name, default_version)

# Assemble full model identifiers
TTS_VERSION = get_version_from_env_or_fetch("BARK_VERSION", "suno/bark", DEFAULT_TTS_VERSION)
SADTALKER_VERSION = get_version_from_env_or_fetch("SADTALKER_VERSION", "lucataco/sadtalker", DEFAULT_SADTALKER_VERSION)
ANIMATE_DIFF_VERSION = get_version_from_env_or_fetch("ANIMATE_DIFF_VERSION", "lucataco/animate-diff", DEFAULT_ANIMATE_DIFF_VERSION)

TTS_MODEL = f"suno/bark:{TTS_VERSION}"
SADTALKER_MODEL = f"lucataco/sadtalker:{SADTALKER_VERSION}"
ANIMATE_DIFF_MODEL = f"lucataco/animate-diff:{ANIMATE_DIFF_VERSION}"

logger.info(f"Using TTS model: {TTS_MODEL}")
logger.info(f"Using SadTalker model: {SADTALKER_MODEL}")
logger.info(f"Using AnimateDiff model: {ANIMATE_DIFF_MODEL}")

# ==================== CONSTANTS ====================
MAX_SCRIPT_LENGTH = 1000  # roughly 1 minute of speech
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_url(output):
    """
    Replicate may return a FileOutput object (with .url) or a direct string URL.
    This function normalises both to a string URL.
    """
    if hasattr(output, 'url'):
        return output.url
    elif isinstance(output, str):
        return output
    # If it's a list, take the first element (common for some models)
    if isinstance(output, list) and len(output) > 0:
        first = output[0]
        if hasattr(first, 'url'):
            return first.url
        elif isinstance(first, str):
            return first
    return str(output)  # last resort

# ==================== ROUTES ====================
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    """
    Main endpoint that handles both Photo-to-Video and Text-to-Video.
    Expects form data with 'tab' = 'photo' or 'text'.
    """
    tab = request.form.get('tab')
    
    if tab == 'photo':
        return handle_photo_to_video()
    elif tab == 'text':
        return handle_text_to_video()
    else:
        return jsonify({'error': 'Invalid tab'}), 400

def handle_photo_to_video():
    """Photo + script → audio (Bark) → talking head video (SadTalker)"""
    image_file = request.files.get('image')
    script = request.form.get('script', '').strip()

    # Validate inputs
    if not image_file:
        return jsonify({'error': 'No image uploaded'}), 400
    if not script:
        return jsonify({'error': 'Script is required'}), 400
    if not allowed_file(image_file.filename):
        return jsonify({'error': 'Unsupported image type. Use jpg, png, etc.'}), 400

    # Trim long scripts
    if len(script) > MAX_SCRIPT_LENGTH:
        script = script[:MAX_SCRIPT_LENGTH] + "..."

    # Save uploaded image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        image_file.save(tmp.name)
        image_path = tmp.name
        logger.info(f"Saved uploaded image to {image_path}")

    try:
        # Step 1: Generate audio from script using Bark
        logger.info("Calling Bark TTS...")
        tts_input = {
            "prompt": script,
            "text_temp": 0.7,
            "waveform_temp": 0.7
        }
        tts_output = client.run(TTS_MODEL, input=tts_input)
        audio_url = extract_url(tts_output)
        logger.info(f"Audio generated: {audio_url}")

        # Step 2: Generate talking head video with SadTalker
        logger.info("Calling SadTalker...")
        sadtalker_input = {
            "source_image": open(image_path, 'rb'),
            "driven_audio": audio_url,
            "preprocess": "crop",
            "still": True,
            "enhancer": True
        }
        sadtalker_output = client.run(SADTALKER_MODEL, input=sadtalker_input)
        video_url = extract_url(sadtalker_output)
        logger.info(f"Video generated: {video_url}")

        return jsonify({'video_url': video_url})

    except replicate.exceptions.ReplicateError as e:
        logger.error(f"Replicate API error: {e}")
        return jsonify({'error': f'Replicate API error: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500
    finally:
        # Clean up temporary file
        if os.path.exists(image_path):
            os.unlink(image_path)
            logger.info(f"Deleted temporary file {image_path}")

def handle_text_to_video():
    """Text prompt → AnimateDiff video"""
    prompt = request.form.get('prompt', '').strip()
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400

    try:
        logger.info("Calling AnimateDiff...")
        animate_input = {
            "prompt": prompt,
            "num_frames": 16,
            "guidance_scale": 7.5,
            "num_inference_steps": 25
        }
        output = client.run(ANIMATE_DIFF_MODEL, input=animate_input)
        video_url = extract_url(output)
        logger.info(f"Video generated: {video_url}")
        return jsonify({'video_url': video_url})

    except replicate.exceptions.ReplicateError as e:
        logger.error(f"Replicate API error: {e}")
        return jsonify({'error': f'Replicate API error: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        return jsonify({'error': f'Server error: {str(e)}'}), 500

# ==================== HEALTH CHECK (optional) ====================
@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200

if __name__ == '__main__':
    # For local development only
    app.run(debug=True, host='0.0.0.0', port=5000)

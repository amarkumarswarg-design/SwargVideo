import os
import tempfile
from flask import Flask, request, jsonify, render_template
import replicate

app = Flask(__name__)

REPLICATE_API_TOKEN = os.environ.get('REPLICATE_API_TOKEN')
if not REPLICATE_API_TOKEN:
    raise ValueError("REPLICATE_API_TOKEN environment variable not set")

client = replicate.Client(api_token=REPLICATE_API_TOKEN)

# Model identifiers
TTS_MODEL = "suno/bark"
SADTALKER_MODEL = "lucataco/sadtalker"
ANIMATE_DIFF_MODEL = "lucataco/animate-diff"

# Limit script length to approx. 1 minute of audio (~1000 chars)
MAX_SCRIPT_LENGTH = 1000

def extract_url(output):
    """Extract URL from replicate output (FileOutput or string)."""
    if hasattr(output, 'url'):
        return output.url
    elif isinstance(output, str):
        return output
    return str(output)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    tab = request.form.get('tab')
    
    if tab == 'photo':
        # Photo to video: image + script
        image_file = request.files.get('image')
        script = request.form.get('script', '').strip()
        
        if not image_file or not script:
            return jsonify({'error': 'Image and script are required'}), 400
        
        # Truncate long scripts
        if len(script) > MAX_SCRIPT_LENGTH:
            script = script[:MAX_SCRIPT_LENGTH] + "..."
        
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            image_file.save(tmp.name)
            image_path = tmp.name
        
        try:
            # Step 1: Generate audio from script using Bark
            tts_input = {
                "prompt": script,
                "text_temp": 0.7,
                "waveform_temp": 0.7
            }
            tts_output = replicate.run(TTS_MODEL, input=tts_input)
            audio_url = extract_url(tts_output)
            
            # Step 2: Generate talking head video with SadTalker
            sadtalker_input = {
                "source_image": open(image_path, 'rb'),
                "driven_audio": audio_url,
                "preprocess": "crop",
                "still": True,
                "enhancer": True
            }
            sadtalker_output = replicate.run(SADTALKER_MODEL, input=sadtalker_input)
            video_url = extract_url(sadtalker_output)
            
            return jsonify({'video_url': video_url})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            os.unlink(image_path)
    
    elif tab == 'text':
        # Text to video
        prompt = request.form.get('prompt', '').strip()
        if not prompt:
            return jsonify({'error': 'Prompt is required'}), 400
        
        try:
            animate_input = {
                "prompt": prompt,
                "num_frames": 16,
                "guidance_scale": 7.5,
                "num_inference_steps": 25
            }
            output = replicate.run(ANIMATE_DIFF_MODEL, input=animate_input)
            video_url = extract_url(output)
            return jsonify({'video_url': video_url})
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    else:
        return jsonify({'error': 'Invalid tab'}), 400

if __name__ == '__main__':
    app.run(debug=True)

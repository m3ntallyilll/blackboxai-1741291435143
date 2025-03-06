from flask import Flask, request, jsonify, send_from_directory, render_template, url_for
import os
from werkzeug.utils import secure_filename
import logging

# Try to import Wav2Lip dependencies
WAV2LIP_AVAILABLE = False
try:
    from core_inference import run_inference, InferenceError
    WAV2LIP_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Wav2Lip dependencies not available: {str(e)}")
    logging.warning("Running in demo mode - lip sync functionality will be disabled")

app = Flask(__name__, 
            static_url_path='',
            static_folder='static',
            template_folder='templates')

app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['SECRET_KEY'] = os.urandom(24)

# Ensure upload and results directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('results', exist_ok=True)

ALLOWED_EXTENSIONS = {
    'video': {'mp4', 'avi', 'mov'},
    'audio': {'mp3', 'wav'}
}

def allowed_file(filename, file_type):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS[file_type]

@app.route('/test')
def test():
    return render_template('test.html', wav2lip_available=WAV2LIP_AVAILABLE)

@app.route('/')
def index():
    return render_template('index.html', wav2lip_available=WAV2LIP_AVAILABLE)

@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)

@app.route('/check-static')
def check_static():
    """Test endpoint to verify static files are being served correctly"""
    return '''
    <html>
        <head>
            <link rel="stylesheet" href="/static/css/style.css">
            <script src="/static/js/main.js"></script>
        </head>
        <body>
            <h1>Static Files Test</h1>
            <div class="drag-drop-zone">
                Test drag-drop zone styling
            </div>
            <script>
                // Verify main.js is loaded
                document.addEventListener('DOMContentLoaded', function() {
                    console.log('main.js loaded successfully');
                });
            </script>
        </body>
    </html>
    '''

@app.route('/process', methods=['POST'])
def process_video():
    if not WAV2LIP_AVAILABLE:
        return jsonify({
            'error': 'Lip sync functionality is not available in demo mode',
            'type': 'demo_mode'
        }), 503
        
    if 'face' not in request.files or 'audio' not in request.files:
        return jsonify({'error': 'Missing video or audio file'}), 400
        
    face_file = request.files['face']
    audio_file = request.files['audio']
    
    if face_file.filename == '' or audio_file.filename == '':
        return jsonify({'error': 'No selected files'}), 400
        
    if not (allowed_file(face_file.filename, 'video') and 
            allowed_file(audio_file.filename, 'audio')):
        return jsonify({'error': 'Invalid file type'}), 400
    
    try:
        # Save uploaded files
        face_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                secure_filename(face_file.filename))
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                 secure_filename(audio_file.filename))
        
        face_file.save(face_path)
        audio_file.save(audio_path)
        
        # Get parameters from form
        params = {
            'checkpoint_path': 'checkpoints/wav2lip.pth',
            'static': request.form.get('static', 'false') == 'true',
            'fps': float(request.form.get('fps', 25.0)),
            'pads': [int(x) for x in request.form.get('pads', '0,10,0,0').split(',')],
            'face_det_batch_size': int(request.form.get('face_det_batch_size', 16)),
            'wav2lip_batch_size': int(request.form.get('wav2lip_batch_size', 128)),
            'resize_factor': int(request.form.get('resize_factor', 1)),
            'crop': [int(x) for x in request.form.get('crop', '0,-1,0,-1').split(',')],
            'rotate': request.form.get('rotate', 'false') == 'true',
            'nosmooth': request.form.get('nosmooth', 'false') == 'true',
            'max_retries': int(request.form.get('max_retries', 3))
        }
        
        # Generate unique output filename
        output_path = os.path.join('results', 
                                  f"result_{os.path.splitext(face_file.filename)[0]}.mp4")
        
        # Run inference
        result_path = run_inference(face_path, audio_path, output_path, **params)
        
        # Clean up uploaded files
        os.remove(face_path)
        os.remove(audio_path)
        
        return jsonify({
            'success': True,
            'result_path': result_path
        })
        
    except InferenceError as e:
        return jsonify({
            'error': str(e),
            'type': 'inference_error'
        }), 500
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        return jsonify({
            'error': 'An unexpected error occurred',
            'details': str(e)
        }), 500

@app.route('/results/<path:filename>')
def download_file(filename):
    return send_from_directory('results', filename)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

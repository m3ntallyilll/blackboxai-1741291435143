# Enhanced Wav2Lip with Web Interface

This is an enhanced version of the Wav2Lip project that adds robust error handling, retry mechanisms, and a modern web interface for easier interaction.

## New Features

- **Web Interface**: Modern, responsive UI built with Tailwind CSS
- **Drag & Drop**: Easy file uploads with drag and drop support
- **Progress Tracking**: Real-time progress updates during processing
- **Error Handling**: Robust error handling with clear user feedback
- **Retry Mechanism**: Automatic retry for recoverable errors
- **Advanced Settings**: User-friendly controls for all parameters

## Quick Start

1. Install additional requirements:
```bash
pip install flask werkzeug
```

2. Start the web server:
```bash
python server.py
```

3. Open your browser and navigate to:
```
http://localhost:5000
```

## Components

### 1. Core Inference Module (`core_inference.py`)
- Enhanced error handling and logging
- Retry mechanism for recoverable errors
- Progress tracking
- Modular design for better maintainability

### 2. Web Server (`server.py`)
- Flask-based web server
- File upload handling
- Process management
- Result delivery

### 3. Web Interface (`templates/index.html`)
- Modern UI with Tailwind CSS
- Drag & drop file uploads
- Real-time progress updates
- Advanced settings controls
- Error feedback
- Result preview and download

## Usage

### Via Web Interface

1. Open the web interface in your browser
2. Upload a video file containing faces
3. Upload an audio file
4. Adjust advanced settings if needed
5. Click "Generate Lip-Sync Video"
6. Wait for processing to complete
7. Preview and download the result

### Via Command Line

The original command-line interface is still available:

```bash
python inference.py --checkpoint_path checkpoints/wav2lip.pth --face <video.mp4> --audio <audio.wav>
```

## Advanced Settings

The web interface provides easy access to all parameters:

- **Padding**: Adjust face detection padding (top, bottom, left, right)
- **Resize Factor**: Control output resolution
- **Smoothing**: Toggle temporal smoothing
- **Rotation**: Rotate input video if needed

## Error Handling

The enhanced version includes comprehensive error handling for:

- File upload issues
- Face detection failures
- Audio processing errors
- GPU memory issues
- General processing errors

Each error includes:
- Clear error message
- Suggested solution
- Automatic retry for recoverable errors

## Logging

Detailed logs are available in `wav2lip.log`, including:
- Processing steps
- Error details
- Retry attempts
- Performance metrics

## Requirements

Additional requirements for the enhanced version:
- Flask
- Werkzeug
- All original Wav2Lip requirements

## License

Same as the original Wav2Lip project - for research/academic purposes only (non-commercial).

## Acknowledgements

- Original Wav2Lip project and authors
- Tailwind CSS for UI components
- Flask for web framework

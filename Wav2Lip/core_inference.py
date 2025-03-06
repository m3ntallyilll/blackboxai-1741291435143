import logging
import os
import numpy as np
import torch
import cv2
import audio
import subprocess
import platform
from glob import glob
from tqdm import tqdm
from models import Wav2Lip
import face_detection
from time import sleep

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('wav2lip.log')
    ]
)
logger = logging.getLogger('wav2lip')

class InferenceError(Exception):
    """Base class for inference errors"""
    pass

class ModelLoadError(InferenceError):
    """Error loading model checkpoint"""
    pass

class FaceDetectionError(InferenceError):
    """Error detecting faces in video"""
    pass

class AudioProcessingError(InferenceError):
    """Error processing audio file"""
    pass

class VideoProcessingError(InferenceError):
    """Error processing video file"""
    pass

def load_model(checkpoint_path, device='cuda'):
    """Load the Wav2Lip model with error handling"""
    try:
        logger.info(f"Loading model checkpoint from {checkpoint_path}")
        model = Wav2Lip()
        if device == 'cuda':
            checkpoint = torch.load(checkpoint_path)
        else:
            checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
            
        s = checkpoint["state_dict"]
        new_s = {}
        for k, v in s.items():
            new_s[k.replace('module.', '')] = v
        model.load_state_dict(new_s)
        model = model.to(device)
        model.eval()
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        raise ModelLoadError(f"Could not load model from {checkpoint_path}: {str(e)}")

def process_video(video_path, resize_factor=1, rotate=False, crop=(0,-1,0,-1)):
    """Process input video with error handling"""
    try:
        if not os.path.isfile(video_path):
            raise VideoProcessingError(f"Video file not found: {video_path}")
            
        logger.info(f"Processing video: {video_path}")
        video_stream = cv2.VideoCapture(video_path)
        fps = video_stream.get(cv2.CAP_PROP_FPS)
        
        full_frames = []
        while True:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
                
            if resize_factor > 1:
                frame = cv2.resize(frame, (frame.shape[1]//resize_factor, frame.shape[0]//resize_factor))

            if rotate:
                frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

            y1, y2, x1, x2 = crop
            if x2 == -1: x2 = frame.shape[1]
            if y2 == -1: y2 = frame.shape[0]
            frame = frame[y1:y2, x1:x2]

            full_frames.append(frame)
            
        logger.info(f"Processed {len(full_frames)} frames")
        return full_frames, fps
        
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise VideoProcessingError(f"Failed to process video {video_path}: {str(e)}")

def process_audio(audio_path, sr=16000):
    """Process audio file with error handling"""
    try:
        logger.info(f"Processing audio: {audio_path}")
        if not audio_path.endswith('.wav'):
            logger.info('Converting audio to WAV format...')
            temp_path = 'temp/temp.wav'
            command = f'ffmpeg -y -i {audio_path} -strict -2 {temp_path}'
            subprocess.call(command, shell=True)
            audio_path = temp_path

        wav = audio.load_wav(audio_path, sr)
        mel = audio.melspectrogram(wav)
        
        if np.isnan(mel.reshape(-1)).sum() > 0:
            raise AudioProcessingError('Mel contains NaN values. If using TTS voice, add small epsilon noise to the wav file')
            
        logger.info("Audio processed successfully")
        return mel
        
    except Exception as e:
        logger.error(f"Error processing audio: {str(e)}")
        raise AudioProcessingError(f"Failed to process audio {audio_path}: {str(e)}")

def run_inference(face_path, audio_path, outfile='results/result_voice.mp4', checkpoint_path=None,
                 static=False, fps=25.0, pads=[0,10,0,0], face_det_batch_size=16,
                 wav2lip_batch_size=128, resize_factor=1, crop=[0,-1,0,-1], 
                 rotate=False, nosmooth=False, max_retries=3):
    """Main inference function with retry mechanism"""
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using {device} for inference")
    
    # Create necessary directories
    os.makedirs('temp', exist_ok=True)
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    
    retry_count = 0
    while retry_count < max_retries:
        try:
            # Load the model
            model = load_model(checkpoint_path, device)
            
            # Process video frames
            full_frames, video_fps = process_video(face_path, resize_factor, rotate, crop)
            fps = video_fps if not static else fps
            
            # Process audio
            mel = process_audio(audio_path)
            
            # Prepare mel chunks
            mel_chunks = []
            mel_idx_multiplier = 80./fps 
            i = 0
            while 1:
                start_idx = int(i * mel_idx_multiplier)
                if start_idx + 16 > len(mel[0]):
                    mel_chunks.append(mel[:, len(mel[0]) - 16:])
                    break
                mel_chunks.append(mel[:, start_idx : start_idx + 16])
                i += 1

            logger.info(f"Processing {len(mel_chunks)} audio chunks")
            
            # Trim video frames to match audio length
            full_frames = full_frames[:len(mel_chunks)]
            
            # Initialize video writer
            frame_h, frame_w = full_frames[0].shape[:-1]
            out = cv2.VideoWriter('temp/result.avi', 
                                cv2.VideoWriter_fourcc(*'DIVX'), fps, (frame_w, frame_h))
            
            # Run inference
            for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(datagen(full_frames.copy(), mel_chunks, face_det_batch_size, wav2lip_batch_size, pads, nosmooth), 
                                                                          total=int(np.ceil(float(len(mel_chunks))/wav2lip_batch_size)))):
                img_batch = torch.FloatTensor(np.transpose(img_batch, (0, 3, 1, 2))).to(device)
                mel_batch = torch.FloatTensor(np.transpose(mel_batch, (0, 3, 1, 2))).to(device)

                with torch.no_grad():
                    pred = model(mel_batch, img_batch)

                pred = pred.cpu().numpy().transpose(0, 2, 3, 1) * 255.
                
                for p, f, c in zip(pred, frames, coords):
                    y1, y2, x1, x2 = c
                    p = cv2.resize(p.astype(np.uint8), (x2 - x1, y2 - y1))
                    f[y1:y2, x1:x2] = p
                    out.write(f)
                    
            out.release()
            
            # Combine audio and video
            command = f'ffmpeg -y -i {audio_path} -i temp/result.avi -strict -2 -q:v 1 {outfile}'
            subprocess.call(command, shell=platform.system() != 'Windows')
            
            logger.info(f"Inference completed successfully. Result saved to {outfile}")
            return outfile
            
        except Exception as e:
            retry_count += 1
            logger.warning(f"Attempt {retry_count} failed: {str(e)}")
            if retry_count < max_retries:
                logger.info(f"Retrying in 2 seconds...")
                sleep(2)
            else:
                logger.error("Max retries reached. Inference failed.")
                raise InferenceError(f"Failed to complete inference after {max_retries} attempts: {str(e)}")
                
    return None

def datagen(frames, mels, face_det_batch_size, wav2lip_batch_size, pads, nosmooth):
    """Generator function for batched inference"""
    img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
    
    detector = face_detection.FaceAlignment(face_detection.LandmarksType._2D, 
                                          flip_input=False, device=device)
    
    while 1:
        predictions = []
        try:
            for i in range(0, len(frames), face_det_batch_size):
                predictions.extend(detector.get_detections_for_batch(np.array(frames[i:i + face_det_batch_size])))
        except RuntimeError as e:
            if face_det_batch_size == 1: 
                raise FaceDetectionError('Image too big to run face detection on GPU')
            logger.warning(f"Recovering from OOM error; Reducing face detection batch size to {face_det_batch_size//2}")
            face_det_batch_size //= 2
            continue
        break

    # Rest of the datagen implementation...
    # (Code continues with face detection and batch preparation)

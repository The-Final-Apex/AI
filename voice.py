import numpy as np
import sounddevice as sd
import soundfile as sf
import tensorflow as tf
from scipy.io import wavfile
import librosa
import os
import time

# Constants
SAMPLE_RATE = 16000
DURATION = 1  # seconds of audio to capture at a time
THRESHOLD = 0.5  # loudness threshold to consider as speech
COMMANDS = ["start", "stop", "yes", "no", "unknown"]
MODEL_PATH = "voice_command_model.h5"  # You'll need to train or download this

class VoiceCommandRecognizer:
    def __init__(self):
        # Load the pre-trained model
        try:
            self.model = tf.keras.models.load_model(MODEL_PATH)
            print("Model loaded successfully")
        except:
            print("Could not load model. You'll need to train one first.")
            self.model = None
        
        # Audio buffer
        self.audio_buffer = np.zeros(int(SAMPLE_RATE * DURATION), dtype=np.float32)
        
    def preprocess_audio(self, audio):
        # Convert to mono if stereo
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
            
        # Resample if needed
        if len(audio) != SAMPLE_RATE * DURATION:
            audio = librosa.resample(audio, orig_sr=len(audio)/(DURATION), target_sr=SAMPLE_RATE)
            
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=SAMPLE_RATE,
            n_mfcc=40,
            n_fft=400,
            hop_length=160
        )
        mfccs = mfccs[..., np.newaxis]  # Add channel dimension
        return mfccs
    
    def predict_command(self, audio):
        if self.model is None:
            return "unknown"
            
        features = self.preprocess_audio(audio)
        features = np.expand_dims(features, axis=0)  # Add batch dimension
        
        prediction = self.model.predict(features)
        command_idx = np.argmax(prediction)
        confidence = np.max(prediction)
        
        if confidence < 0.7:  # Confidence threshold
            return "unknown"
            
        return COMMANDS[command_idx]
    
    def callback(self, indata, frames, time, status):
        """This is called for each audio block from the microphone."""
        # Check if audio is loud enough to consider
        volume_norm = np.linalg.norm(indata) * 10
        if volume_norm < THRESHOLD:
            return
            
        # Add to buffer (sliding window)
        self.audio_buffer = np.roll(self.audio_buffer, -frames)
        self.audio_buffer[-frames:] = indata[:, 0]  # Take first channel
        
        # Predict command
        command = self.predict_command(self.audio_buffer)
        
        # Execute action based on command
        if command == "start":
            self.on_start()
        elif command == "stop":
            self.on_stop()
        elif command == "yes":
            self.on_yes()
        elif command == "no":
            self.on_no()
    
    def on_start(self):
        print("START command detected - starting process")
        # Add your start action here
        
    def on_stop(self):
        print("STOP command detected - stopping process")
        # Add your stop action here
        
    def on_yes(self):
        print("YES command detected - affirmative action")
        # Add your yes action here
        
    def on_no(self):
        print("NO command detected - negative action")
        # Add your no action here
    
    def listen(self):
        print("Listening for commands... Say 'start', 'stop', 'yes', or 'no'")
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            callback=self.callback,
            blocksize=int(SAMPLE_RATE * 0.5),  # Process audio in 0.5s chunks
            dtype=np.float32
        ):
            while True:
                time.sleep(0.1)

if __name__ == "__main__":
    recognizer = VoiceCommandRecognizer()
    recognizer.listen()
  #THIS NEEDS A PRETRAINED MODEL TO WORK

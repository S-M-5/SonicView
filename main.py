import pyaudio as pa
import numpy as np
from faster_whisper import WhisperModel
from colorama import init, Fore, Style
import torch
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix for macOS/KMP error
init(autoreset=True)                         # Initialize colorama for colored terminal output

CHUNK = 1024            # Number of audio samples per frame
FORMAT = pa.paInt16     # Audio format (16-bit PCM)
CHANNELS = 1            # 1 for mono audio, 2 for stereo
RATE = 16000            # Model expects 16,000 Hz audio
WINDOW_SECONDS = 2.0    # Duration of audio window for processing
CHUNKS_PER_WINDOW = int((RATE * WINDOW_SECONDS) / CHUNK)  # Number of chunks in the window

print("Loading Whisper model... (Might take a while on first run. ~140MB download)")

if torch.cuda.is_available():       # Check for GPU availability
    device = "cuda"
    gpu_name = torch.cuda.get_device_name(0)
    print(f"GPU Detected: {gpu_name}")
    print("Mode: High Performance (GPU)")
else:
    device = "cpu"
    print("GPU not detected (or not NVIDIA CUDA compatible).")
    print("Mode: Standard (CPU)")

print("---------------------------\n")

model = WhisperModel("small", device=device, compute_type="int8", cpu_threads=8)  # Load Whisper model

print("Model loaded.")
p = pa.PyAudio()        # Initialize PyAudio   
name = input("Enter your name: ")  # Get user's name for detection


stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,     # Open stream for input (microphone)
                frames_per_buffer=CHUNK)

print("SonicView is Listening...\n(press Ctrl+C to stop)")

try:
    while True:
        frames = []     # Buffer to hold audio frames

        for _ in range(CHUNKS_PER_WINDOW):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.int16))
            
        audio_data = np.concatenate(frames)  # Combine frames into a single array
        audio_float32 = audio_data.astype(np.float32) / 32768.0  # Convert Int16 (-32768 to 32767) to Float32 (-1.0 to 1.0) (Whisper expects Float32 input)
        volume = np.abs(audio_data).mean()
        if volume < 50:  # Adjust threshold as needed for silence detection
            continue  # Skip processing if audio is too quiet

        segments, info = model.transcribe(audio_float32, beam_size=1)  # Transcribe audio using Whisper model

        for segment in segments:
            text = segment.text.lower().strip()
            

            if not text:   # Skip silent segments
                continue   

            print(f"[{info.language}] Heard: '{text}'")
            if name and name.lower() in text:
                print(f"{Fore.RED}Your name {name} was called.{Style.RESET_ALL}")

except KeyboardInterrupt:
    print("Stopping...")        # Handle Ctrl+C to stop
    stream.stop_stream()
    stream.close()
    p.terminate()   


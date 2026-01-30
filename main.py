import pyaudiowpatch as pa
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
GAIN = 3.0              # Gain factor for audio amplification
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

print("\nAvailable Audio Input Devices:")
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:        # Filter for devices that actually have input channels
        print(f"{Fore.GREEN}[{i}] {info['name']}{Style.RESET_ALL}")

try:        # Get user's choice of audio input device
    device_choice = input("Select Device ID (default is 0): ")
    DEVICE_INDEX = int(device_choice) if device_choice else 0
except ValueError:
    DEVICE_INDEX = 0    

selected_info = p.get_device_info_by_index(DEVICE_INDEX)
CHANNELS = selected_info['maxInputChannels']
print(f"Using: {selected_info['name']} ({CHANNELS} channels)")
hardware_rate = int(selected_info['defaultSampleRate'])

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=hardware_rate,
                input=True,     # Open stream for input (microphone)
                input_device_index=DEVICE_INDEX,
                frames_per_buffer=CHUNK)

print("SonicView is Listening...\n(press Ctrl+C to stop)")

chunks_to_record = int((hardware_rate * WINDOW_SECONDS) / CHUNK)
try:
    while True:
        frames = []     # Buffer to hold audio frames
        

        for _ in range(CHUNKS_PER_WINDOW):
            data = stream.read(CHUNK, exception_on_overflow=False)
            frames.append(np.frombuffer(data, dtype=np.int16))
            
        audio_data = np.concatenate(frames)  # Combine frames into a single array
        if CHANNELS > 1:
            audio_data = audio_data.reshape(-1, CHANNELS).mean(axis=1).astype(np.int16)   # Convert to mono by averaging channels

        if hardware_rate != RATE:       # Resample audio if hardware rate differs from model rate
            indices = np.round(np.arange(0, len(audio_data), hardware_rate / RATE))
            indices = indices[indices < len(audio_data)].astype(int)
            audio_data = audio_data[indices]
            
        frames.append(audio_data)
        audio_data = np.clip(audio_data * GAIN).clip(-32768, 32767).astype(np.int16)  # Apply gain and ensure values are within Int16 range
        audio_float32 = audio_data.astype(np.float32) / 32768.0  # Convert Int16 (-32768 to 32767) to Float32 (-1.0 to 1.0) (Whisper expects Float32 input)
        volume = np.abs(audio_data).mean()
        if volume < 150:  # Adjust threshold as needed for silence detection
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


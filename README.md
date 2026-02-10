# SonicView (Work in Progress)

**SonicView** is a real-time audio transcription and keyword detection tool. It uses the **Faster-Whisper** engine to process audio from your microphone or desktop speakers, transcribing speech instantly and alerting you whenever your name is mentioned.


![SonicView Architecture](Flowchart.png)

---

## How It Works

SonicView follows a strict digital signal processing (DSP) pipeline to ensure high AI accuracy:

1. **Capture**  
   Pulls raw audio chunks using **PyAudio**.

2. **Mono Conversion**  
   If a stereo device (such as Desktop Audio) is used, the left and right channels are averaged into a single mono stream.

3. **Resampling**  
   Uses **NumPy** to downsample hardware sample rates (e.g., 48,000 Hz) to **16,000 Hz**, which is Whisper’s native sampling rate.

4. **Gain & Clip**  
   Amplifies the signal and clips peak values to prevent digital distortion.

5. **Inference**  
   Sends the 2-second audio window to the **Faster-Whisper** model for transcription.
   
---

## Features

* **Real-Time Transcription:** High-speed speech-to-text using the `faster-whisper` engine.
* **Keyword Detection:** Instantly highlights your name in **bold red** when detected in the audio stream.
* **Dual Audio Sources:** Support for both physical Microphones and Desktop Loopback (system audio).
* **Hardware Acceleration:** Automatically detects and utilizes NVIDIA GPUs (CUDA) for lightning-fast processing.
* **Auto-Resampling:** Intelligent handling of different hardware sample rates (44.1k/48k) to match Whisper's 16kHz requirement.
* **Digital Gain:** Built-in volume amplification for low-output microphones.

---

## Tuning Constants

You can adjust these variables at the top of the script:

| Constant            | Default | Description                                                                 |
|---------------------|---------|-----------------------------------------------------------------------------|
| `GAIN`              | 3.0     | Multiplier for input volume.                                                |
| `WINDOW_SECONDS`    | 2.0     | The duration of audio processed at once.                                    |
| `MODEL_SIZE`        | `small` | Options: `tiny`, `base`, `small`, `medium`, `large-v3`.                     |
| `SILENCE_THRESHOLD` | 150     | Minimum volume level required to trigger AI processing.                     |

---

## Installation

### Prerequisites
* **OS:** Windows 10/11 (Required for `PyAudioWPatch` loopback support)
* **Python:** 3.10+
* **FFmpeg:** Required for audio processing.
    * *Install via Winget:* `winget install -e --id Gyan.FFmpeg`
    * *Or download manually* and add to your System PATH.
 
---

### Setup
1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/S-M-5/SonicView.git](https://github.com/S-M-5/SonicView.git)
    cd SonicView
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    
---

## Usage

1.  Run the main application:
    ```bash
    python main.py
    ```
    *(Note: Replace `main.py` with your actual entry filename if different)*

2.  Select your audio input device index when prompted.
3.  The transcription will appear in the console real-time.

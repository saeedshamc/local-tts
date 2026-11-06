"""Voice test: record -> Whisper (Persian STT, CPU) -> Piper (Persian TTS)."""
import os
import glob
import wave
import winsound

import sounddevice as sd
from faster_whisper import WhisperModel
from piper import PiperVoice

# ---- Settings -------------------------------------------------------------
WHISPER_DIR = r"D:\Saeed\STT-Models\faster-whisper-small"   # folder with model.bin etc.
PIPER_DIR = r"D:\Saeed\TTS-Models\Ganji_Adabi"                     # folder with Piper .onnx + .onnx.json
SAMPLE_RATE = 16000
SECONDS = 5
# ---------------------------------------------------------------------------

# Find the Piper voice file automatically inside the folder
onnx_files = glob.glob(os.path.join(PIPER_DIR, "*.onnx"))
if not onnx_files:
    raise SystemExit(f"No .onnx voice file found in: {PIPER_DIR}")
VOICE_FILE = onnx_files[0]
CONFIG_FILE = VOICE_FILE + ".json"

# Check files before loading, so errors are clear
if not os.path.isfile(os.path.join(WHISPER_DIR, "model.bin")):
    raise SystemExit(f"model.bin not found in: {WHISPER_DIR}")
if not os.path.isfile(CONFIG_FILE):
    raise SystemExit(f"Piper config not found: {CONFIG_FILE}")

print("Loading Whisper...")
stt = WhisperModel(WHISPER_DIR, device="cpu", compute_type="int8")

print(f"Loading Piper voice: {os.path.basename(VOICE_FILE)}")
tts = PiperVoice.load(VOICE_FILE, config_path=CONFIG_FILE)


def record():
    print(f"\nSpeak now for {SECONDS} seconds...")
    audio = sd.rec(int(SECONDS * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                   channels=1, dtype="float32")
    sd.wait()
    return audio.flatten()


def listen(audio):
    segments, _ = stt.transcribe(audio, language="fa", vad_filter=True)
    return " ".join(s.text.strip() for s in segments).strip()


def speak(text, path="reply.wav"):
    with wave.open(path, "wb") as f:
        tts.synthesize_wav(text, f)
    winsound.PlaySound(path, winsound.SND_FILENAME)


while True:
    input("Press Enter to record (Ctrl+C to quit)...")
    text = listen(record())
    print("Heard:", text or "(nothing)")
    if text:
        speak(text)
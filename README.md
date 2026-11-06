# Local Persian Voice Assistant

A fully **offline** Persian voice toolkit for Windows: speak into the microphone, get your speech transcribed with **Whisper** (STT), and have it spoken back with **Piper** (TTS). Everything runs locally on CPU, with no API keys and no internet needed after the models are downloaded.

This is the foundation for a Persian "Jarvis"-style assistant (wake word + voice commands are planned, see [Roadmap](#roadmap)).

## Versions

| File | Description |
|------|-------------|
| `VoiceTest-Gui.py` | Graphical version (Tkinter): model lists, ON/OFF buttons, record-time setting, live step log |
| `VoiceTest-Cli.py` | Minimal terminal version: press Enter, speak, hear it back |

## How it works

```
Microphone → faster-whisper (Persian STT) → text → Piper (Persian TTS) → speakers
```

## Requirements

- Windows 10/11 (uses `winsound` for playback)
- Python 3.10 or newer (developed on Python 3.14)
- A working microphone
- About 1 GB of free disk space for the models

## 1. Install

Clone the repo and install the dependencies:

```powershell
git clone <your-repo-url>
cd "local tts"

python -m venv .venv
.venv\Scripts\Activate.ps1

pip install faster-whisper piper-tts sounddevice numpy
```

> Tkinter (needed by the GUI) ships with Python. If it is missing, re-run the Python installer and enable **tcl/tk and IDLE**.
>
> If a package fails to install on a very new Python version (no prebuilt wheel yet), create the virtual environment with Python 3.11 or 3.12 instead.

## 2. Download the models

Models are **not** included in this repo (they are large). Download two things:

### Speech-to-text: faster-whisper-small

Source: [Systran/faster-whisper-small](https://huggingface.co/Systran/faster-whisper-small)

```powershell
pip install huggingface_hub
hf download Systran/faster-whisper-small --local-dir "D:\Saeed\STT-Models\faster-whisper-small"
```

The folder must contain `model.bin`, `config.json`, `tokenizer.json` and `vocabulary.txt`.

### Text-to-speech: Piper Persian voice (Amir)

Source: [rhasspy/piper-voices, fa_IR/amir/medium](https://huggingface.co/rhasspy/piper-voices/tree/main/fa/fa_IR/amir/medium)

Download **both** files into one folder:

- `fa_IR-amir-medium.onnx`
- `fa_IR-amir-medium.onnx.json`

```powershell
mkdir "D:\Saeed\TTS-Models\Amir"
cd "D:\Saeed\TTS-Models\Amir"
curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/main/fa/fa_IR/amir/medium/fa_IR-amir-medium.onnx
curl -L -O https://huggingface.co/rhasspy/piper-voices/resolve/main/fa/fa_IR/amir/medium/fa_IR-amir-medium.onnx.json
```

> The `.onnx.json` config is required. Piper will not load without it.

### Recommended folder layout

```
D:\Saeed\
├── STT-Models\
│   └── faster-whisper-small\     ← model.bin, config.json, ...
└── TTS-Models\
    └── Amir\                     ← fa_IR-amir-medium.onnx + .onnx.json
```

You can use any location. Just point the app to it (see below). Larger Whisper models (`medium`, `large-v3`) are more accurate but slower on CPU, and you can put them next to `faster-whisper-small`.

## 3. Run

### GUI

```powershell
python VoiceTest-Gui.py
```

1. Set the **STT folder** and **TTS folder** (type the path or use *Browse...*). Every model found inside is listed automatically.
2. Pick a model in each list and press **Turn ON**. Loading takes a few seconds, and the status turns green when ready.
3. Set the **record time** in seconds (1 to 120).
4. Press **Record**, speak, and watch the heard text appear. Tick *Speak the heard text back* to hear it with Piper.
5. Use the *Speak a text* box to test the voice without recording.
6. Press **Turn OFF** to unload a model and free RAM.

Models with missing files are marked in the list, e.g. `[no model.bin]` or `[no .json config]`.

### CLI

Edit the paths at the top of `VoiceTest-Cli.py`:

```python
WHISPER_DIR = r"D:\Saeed\STT-Models\faster-whisper-small"
PIPER_DIR   = r"D:\Saeed\TTS-Models\Amir"
SECONDS     = 5
```

Then run:

```powershell
python VoiceTest-Cli.py
```

Press **Enter** to record, speak, and the recognized text is printed and spoken back. Press `Ctrl+C` to quit.

## Generated files

These are created next to the script and are listed in `.gitignore`:

| File | Purpose |
|------|---------|
| `voice_log.json` | Step-by-step log of every run (GUI): model loading, recording, transcription, synthesis, timings, errors |
| `voice_settings.json` | Remembers folders, record time and options (GUI) |
| `reply.wav` | Last synthesized audio |

Example log entry:

```json
{
  "n": 7,
  "time": "2026-11-06T14:20:31.512",
  "step": "stt_done",
  "status": "ok",
  "message": "Heard: سلام",
  "text": "سلام",
  "seconds": 2.1,
  "language": "fa",
  "language_probability": 0.98
}
```

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Invalid argument` when running from VS Code | Run manually from the terminal: `python VoiceTest-Gui.py` (the path contains a space, which can confuse the Run button) |
| `model.bin not found` | The STT folder must be the one that directly contains `model.bin` |
| `Piper config not found` | Put the `.onnx.json` file next to the `.onnx` file with the same name |
| Nothing is heard | Check the `peak` value in the log. If it is near 0, the wrong microphone is selected in Windows sound settings |
| Garbled Persian in the terminal | Use Windows Terminal or run `chcp 65001` |
| Slow transcription | Use `faster-whisper-small` or `base`, and keep record times short |
| Package install fails | Use a virtual environment with Python 3.11 or 3.12 |

## Roadmap

- [ ] Wake word ("جارویس")
- [ ] Voice commands (open apps, system actions) in Persian
- [ ] Continuous listening with voice activity detection
- [ ] Command mapping via a simple JSON config

## Credits

- [faster-whisper](https://github.com/SYSTRAN/faster-whisper), a CTranslate2 reimplementation of OpenAI Whisper
- [Piper](https://github.com/OHF-Voice/piper1-gpl), a fast local neural TTS
- [piper-voices](https://huggingface.co/rhasspy/piper-voices), the voice models

## License

See [LICENSE](LICENSE).
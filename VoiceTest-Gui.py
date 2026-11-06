"""Voice Assistant (GUI): record -> Whisper (Persian STT) -> Piper (Persian TTS).

Every step is logged to voice_log.json next to this script.
"""
import gc
import json
import os
import queue
import threading
import time
import tkinter as tk
import wave
import winsound
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from piper import PiperVoice

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_FILE = os.path.join(HERE, "voice_log.json")
SETTINGS_FILE = os.path.join(HERE, "voice_settings.json")
REPLY_WAV = os.path.join(HERE, "reply.wav")
SAMPLE_RATE = 16000

DEFAULT_SETTINGS = {
    "stt_dir": r"D:\Saeed\STT-Models",
    "tts_dir": r"D:\Saeed\TTS-Models",
    "seconds": 5,
    "echo": True,
}


# ---------------------------------------------------------------- JSON log --
class StepLog:
    """Writes every step into voice_log.json (a list of sessions)."""

    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        self.counter = 0
        self.data = self._read()
        self.session = {
            "session_id": datetime.now().strftime("%Y%m%d-%H%M%S"),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "steps": [],
        }
        self.data.append(self.session)

    def _read(self):
        if not os.path.isfile(self.path):
            return []
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                return data
        except Exception:
            pass
        try:  # corrupted file: keep a backup instead of losing it
            os.replace(self.path, self.path + ".bak")
        except OSError:
            pass
        return []

    def _write(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)

    def add(self, step, status="ok", **details):
        with self.lock:
            self.counter += 1
            entry = {
                "n": self.counter,
                "time": datetime.now().isoformat(timespec="milliseconds"),
                "step": step,
                "status": status,
            }
            entry.update(details)
            self.session["steps"].append(entry)
            try:
                self._write()
            except OSError:
                pass
            return entry


# ------------------------------------------------------------ model scanning --
def scan_stt(folder):
    """Whisper models = sub-folders (valid if they contain model.bin)."""
    items = {}
    if not os.path.isdir(folder):
        return items
    if os.path.isfile(os.path.join(folder, "model.bin")):
        items[os.path.basename(folder) + " (this folder)"] = folder
    for name in sorted(os.listdir(folder)):
        p = os.path.join(folder, name)
        if os.path.isdir(p):
            ok = os.path.isfile(os.path.join(p, "model.bin"))
            items[name if ok else f"{name}  [no model.bin]"] = p
    return items


def scan_tts(folder):
    """Piper voices = every .onnx file (valid if .onnx.json exists next to it)."""
    items = {}
    if not os.path.isdir(folder):
        return items
    for root, _dirs, files in os.walk(folder):
        for fn in sorted(files):
            if fn.lower().endswith(".onnx"):
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, folder)
                ok = os.path.isfile(full + ".json")
                items[rel if ok else f"{rel}  [no .json config]"] = full
    return items


# ---------------------------------------------------------------------- App --
class App:
    def __init__(self, root):
        self.root = root
        root.title("Persian Voice Assistant")
        root.geometry("820x640")
        root.minsize(720, 560)

        self.settings = self._load_settings()
        self.steplog = StepLog(LOG_FILE)
        self.q = queue.Queue()
        self.stt = None
        self.tts = None
        self.busy = False
        self.stt_items = {}
        self.tts_items = {}

        self._build_ui()
        self.refresh_lists()
        self.root.after(100, self._poll)
        self.log("app_start", message="Program started", script_dir=HERE)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

    # ---------------- settings ----------------
    def _load_settings(self):
        s = dict(DEFAULT_SETTINGS)
        try:
            with open(SETTINGS_FILE, "r", encoding="utf-8") as f:
                s.update(json.load(f))
        except Exception:
            pass
        return s

    def _save_settings(self):
        self.settings.update(
            stt_dir=self.stt_dir_var.get(),
            tts_dir=self.tts_dir_var.get(),
            seconds=self.seconds_var.get(),
            echo=self.echo_var.get(),
        )
        try:
            with open(SETTINGS_FILE, "w", encoding="utf-8") as f:
                json.dump(self.settings, f, ensure_ascii=False, indent=2)
        except OSError:
            pass

    # ---------------- UI ----------------
    def _build_ui(self):
        pad = {"padx": 8, "pady": 4}
        main = ttk.Frame(self.root, padding=10)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)

        # --- STT box
        self.stt_dir_var = tk.StringVar(value=self.settings["stt_dir"])
        self.stt_box = self._model_box(
            main, 0, "STT (Whisper) - speech to text",
            self.stt_dir_var, self.browse_stt, self.toggle_stt)

        # --- TTS box
        self.tts_dir_var = tk.StringVar(value=self.settings["tts_dir"])
        self.tts_box = self._model_box(
            main, 1, "TTS (Piper) - text to speech",
            self.tts_dir_var, self.browse_tts, self.toggle_tts)

        # --- recording box
        rec = ttk.LabelFrame(main, text="Recording")
        rec.grid(row=2, column=0, sticky="ew", **pad)
        rec.columnconfigure(3, weight=1)
        ttk.Label(rec, text="Record time (seconds):").grid(row=0, column=0, **pad)
        self.seconds_var = tk.IntVar(value=int(self.settings["seconds"]))
        ttk.Spinbox(rec, from_=1, to=120, width=6,
                    textvariable=self.seconds_var).grid(row=0, column=1, **pad)
        self.echo_var = tk.BooleanVar(value=bool(self.settings["echo"]))
        ttk.Checkbutton(rec, text="Speak the heard text back",
                        variable=self.echo_var).grid(row=0, column=2, **pad)
        self.rec_btn = ttk.Button(rec, text="🎙  Record", command=self.start_record)
        self.rec_btn.grid(row=0, column=4, **pad)
        self.progress = ttk.Progressbar(rec, maximum=100)
        self.progress.grid(row=1, column=0, columnspan=5, sticky="ew", **pad)

        # --- manual speak box
        spk = ttk.LabelFrame(main, text="Speak a text (TTS test)")
        spk.grid(row=3, column=0, sticky="ew", **pad)
        spk.columnconfigure(0, weight=1)
        self.speak_var = tk.StringVar()
        self.speak_entry = ttk.Entry(spk, textvariable=self.speak_var, justify="right")
        self.speak_entry.grid(row=0, column=0, sticky="ew", **pad)
        self.speak_btn = ttk.Button(spk, text="🔊  Speak", command=self.start_speak)
        self.speak_btn.grid(row=0, column=1, **pad)

        # --- heard text
        heard = ttk.LabelFrame(main, text="Last heard text")
        heard.grid(row=4, column=0, sticky="ew", **pad)
        heard.columnconfigure(0, weight=1)
        self.heard_var = tk.StringVar(value="-")
        ttk.Label(heard, textvariable=self.heard_var, font=("Segoe UI", 13),
                  anchor="e", wraplength=760).grid(row=0, column=0, sticky="ew", **pad)

        # --- log view
        logf = ttk.LabelFrame(main, text=f"Steps (saved in {os.path.basename(LOG_FILE)})")
        logf.grid(row=5, column=0, sticky="nsew", **pad)
        main.rowconfigure(5, weight=1)
        logf.columnconfigure(0, weight=1)
        logf.rowconfigure(0, weight=1)
        self.log_box = ScrolledText(logf, height=10, state="disabled",
                                    font=("Consolas", 9))
        self.log_box.grid(row=0, column=0, sticky="nsew", **pad)
        self.log_box.tag_config("error", foreground="#c0392b")
        self.log_box.tag_config("ok", foreground="#1e8449")
        self.log_box.tag_config("warn", foreground="#b9770e")

    def _model_box(self, parent, row, title, dir_var, browse_cmd, toggle_cmd):
        pad = {"padx": 8, "pady": 4}
        box = ttk.LabelFrame(parent, text=title)
        box.grid(row=row, column=0, sticky="ew", **pad)
        box.columnconfigure(1, weight=1)

        ttk.Label(box, text="Folder:").grid(row=0, column=0, **pad)
        ttk.Entry(box, textvariable=dir_var).grid(row=0, column=1, sticky="ew", **pad)
        ttk.Button(box, text="Browse...", command=browse_cmd).grid(row=0, column=2, **pad)
        ttk.Button(box, text="⟳ Refresh", command=self.refresh_lists).grid(row=0, column=3, **pad)

        ttk.Label(box, text="Model:").grid(row=1, column=0, **pad)
        combo = ttk.Combobox(box, state="readonly")
        combo.grid(row=1, column=1, sticky="ew", **pad)
        btn = ttk.Button(box, text="Turn ON", width=12, command=toggle_cmd)
        btn.grid(row=1, column=2, **pad)
        status = tk.Label(box, text="● OFF", fg="#c0392b", width=14, anchor="w")
        status.grid(row=1, column=3, **pad)
        return {"combo": combo, "btn": btn, "status": status}

    # ---------------- thread -> UI helpers ----------------
    def ui(self, fn, *args, **kwargs):
        self.q.put((fn, args, kwargs))

    def _poll(self):
        try:
            while True:
                fn, args, kwargs = self.q.get_nowait()
                fn(*args, **kwargs)
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def log(self, step, status="ok", message="", **details):
        """Write to JSON + show in the UI list."""
        self.steplog.add(step, status, message=message, **details)
        tag = status if status in ("ok", "error", "warn") else ""
        line = f"[{datetime.now():%H:%M:%S}] {step}: {message}\n"
        self.ui(self._append_log, line, tag)

    def _append_log(self, line, tag):
        self.log_box.configure(state="normal")
        self.log_box.insert("end", line, tag)
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def set_status(self, box, text, color):
        box["status"].configure(text=text, fg=color)

    def refresh_buttons(self):
        busy = self.busy
        for box, model in ((self.stt_box, self.stt), (self.tts_box, self.tts)):
            box["btn"].configure(text="Turn OFF" if model else "Turn ON",
                                 state="disabled" if busy else "normal")
            box["combo"].configure(state="disabled" if (busy or model) else "readonly")
        self.rec_btn.configure(state="disabled" if (busy or not self.stt) else "normal")
        self.speak_btn.configure(state="disabled" if (busy or not self.tts) else "normal")

    def run_bg(self, target, *args):
        if self.busy:
            return
        self.busy = True
        self.refresh_buttons()

        def wrapper():
            try:
                target(*args)
            except Exception as e:  # last-resort safety net
                self.log("unexpected_error", "error", str(e))
            finally:
                self.busy = False
                self.ui(self.refresh_buttons)

        threading.Thread(target=wrapper, daemon=True).start()

    # ---------------- folders ----------------
    def browse_stt(self):
        d = filedialog.askdirectory(initialdir=self.stt_dir_var.get() or HERE)
        if d:
            self.stt_dir_var.set(os.path.normpath(d))
            self.refresh_lists()

    def browse_tts(self):
        d = filedialog.askdirectory(initialdir=self.tts_dir_var.get() or HERE)
        if d:
            self.tts_dir_var.set(os.path.normpath(d))
            self.refresh_lists()

    def refresh_lists(self):
        self.stt_items = scan_stt(self.stt_dir_var.get())
        self.tts_items = scan_tts(self.tts_dir_var.get())
        for box, items in ((self.stt_box, self.stt_items), (self.tts_box, self.tts_items)):
            names = list(items.keys())
            box["combo"]["values"] = names
            if names and box["combo"].get() not in names:
                # prefer a valid model as the default selection
                valid = [n for n in names if "[" not in n]
                box["combo"].set((valid or names)[0])
            if not names:
                box["combo"].set("")
        self._save_settings()
        self.log("folder_scan", message=(
            f"STT: {len(self.stt_items)} item(s) | TTS: {len(self.tts_items)} item(s)"),
            stt_dir=self.stt_dir_var.get(), stt_found=list(self.stt_items.keys()),
            tts_dir=self.tts_dir_var.get(), tts_found=list(self.tts_items.keys()))
        self.refresh_buttons()

    # ---------------- STT on/off ----------------
    def toggle_stt(self):
        if self.stt:
            self.stt = None
            gc.collect()
            self.set_status(self.stt_box, "● OFF", "#c0392b")
            self.log("stt_unload", message="Whisper turned off")
            self.refresh_buttons()
            return
        label = self.stt_box["combo"].get()
        path = self.stt_items.get(label)
        if not path:
            messagebox.showwarning("STT", "No model selected.")
            return
        self.set_status(self.stt_box, "● Loading...", "#b9770e")
        self.run_bg(self._load_stt, label, path)

    def _load_stt(self, label, path):
        self.log("stt_load_start", message=f"Loading Whisper: {label}", path=path)
        if not os.path.isfile(os.path.join(path, "model.bin")):
            self.log("stt_load", "error", "model.bin not found", path=path)
            self.ui(self.set_status, self.stt_box, "● OFF", "#c0392b")
            return
        t = time.perf_counter()
        try:
            self.stt = WhisperModel(path, device="cpu", compute_type="int8")
        except Exception as e:
            self.stt = None
            self.log("stt_load", "error", f"Load failed: {e}", path=path)
            self.ui(self.set_status, self.stt_box, "● OFF", "#c0392b")
            return
        dt = round(time.perf_counter() - t, 2)
        self.log("stt_load", message=f"Whisper is ON ({dt}s)", path=path, seconds=dt)
        self.ui(self.set_status, self.stt_box, "● ON", "#1e8449")

    # ---------------- TTS on/off ----------------
    def toggle_tts(self):
        if self.tts:
            self.tts = None
            gc.collect()
            self.set_status(self.tts_box, "● OFF", "#c0392b")
            self.log("tts_unload", message="Piper turned off")
            self.refresh_buttons()
            return
        label = self.tts_box["combo"].get()
        path = self.tts_items.get(label)
        if not path:
            messagebox.showwarning("TTS", "No voice selected.")
            return
        self.set_status(self.tts_box, "● Loading...", "#b9770e")
        self.run_bg(self._load_tts, label, path)

    def _load_tts(self, label, path):
        self.log("tts_load_start", message=f"Loading Piper: {label}", path=path)
        config = path + ".json"
        if not os.path.isfile(config):
            self.log("tts_load", "error", "Config .onnx.json not found", config=config)
            self.ui(self.set_status, self.tts_box, "● OFF", "#c0392b")
            return
        t = time.perf_counter()
        try:
            self.tts = PiperVoice.load(path, config_path=config)
        except Exception as e:
            self.tts = None
            self.log("tts_load", "error", f"Load failed: {e}", path=path)
            self.ui(self.set_status, self.tts_box, "● OFF", "#c0392b")
            return
        dt = round(time.perf_counter() - t, 2)
        self.log("tts_load", message=f"Piper is ON ({dt}s)", path=path, seconds=dt)
        self.ui(self.set_status, self.tts_box, "● ON", "#1e8449")

    # ---------------- record -> STT -> (TTS) ----------------
    def start_record(self):
        try:
            seconds = int(self.seconds_var.get())
            if not 1 <= seconds <= 120:
                raise ValueError
        except (ValueError, tk.TclError):
            messagebox.showwarning("Record time", "Enter a number between 1 and 120.")
            return
        self._save_settings()
        self.run_bg(self._pipeline, seconds, self.echo_var.get())

    def _pipeline(self, seconds, echo):
        # 1) record
        self.log("record_start", message=f"Recording for {seconds}s - speak now",
                 seconds=seconds, sample_rate=SAMPLE_RATE)
        t0 = time.perf_counter()
        try:
            audio = sd.rec(int(seconds * SAMPLE_RATE), samplerate=SAMPLE_RATE,
                           channels=1, dtype="float32")
            while (elapsed := time.perf_counter() - t0) < seconds:
                self.ui(self.progress.configure, value=elapsed / seconds * 100)
                time.sleep(0.05)
            sd.wait()
        except Exception as e:
            self.log("record", "error", f"Recording failed: {e}")
            self.ui(self.progress.configure, value=0)
            return
        self.ui(self.progress.configure, value=0)
        audio = audio.flatten()
        peak = round(float(np.abs(audio).max()), 4)
        status = "warn" if peak < 0.005 else "ok"
        self.log("record_done", status, f"Recording finished (peak level {peak})", peak=peak)

        # 2) speech to text
        self.log("stt_start", message="Transcribing...")
        t = time.perf_counter()
        try:
            segments, info = self.stt.transcribe(audio, language="fa", vad_filter=True)
            text = " ".join(s.text.strip() for s in segments).strip()
        except Exception as e:
            self.log("stt", "error", f"Transcription failed: {e}")
            return
        dt = round(time.perf_counter() - t, 2)
        self.log("stt_done", message=f"Heard: {text or '(nothing)'}", text=text, seconds=dt,
                 language=info.language, language_probability=round(info.language_probability, 3))
        self.ui(self.heard_var.set, text or "(nothing heard)")

        # 3) speak back
        if text and echo:
            if self.tts:
                self._speak(text)
            else:
                self.log("tts_skip", "warn", "TTS is off, so the text was not spoken")

    # ---------------- speak ----------------
    def start_speak(self):
        text = self.speak_var.get().strip()
        if not text:
            return
        self.run_bg(self._speak, text)

    def _speak(self, text):
        self.log("tts_start", message=f"Synthesizing: {text}", text=text)
        t = time.perf_counter()
        try:
            with wave.open(REPLY_WAV, "wb") as f:
                self.tts.synthesize_wav(text, f)
        except Exception as e:
            self.log("tts", "error", f"Synthesis failed: {e}")
            return
        dt = round(time.perf_counter() - t, 2)
        self.log("tts_done", message=f"Audio ready ({dt}s), playing", file=REPLY_WAV, seconds=dt)
        try:
            winsound.PlaySound(REPLY_WAV, winsound.SND_FILENAME)
        except Exception as e:
            self.log("play", "error", f"Playback failed: {e}")
            return
        self.log("play_done", message="Playback finished")

    # ---------------- exit ----------------
    def on_close(self):
        self._save_settings()
        self.log("app_exit", message="Program closed")
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
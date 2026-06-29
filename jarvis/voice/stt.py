import asyncio
import queue
import re
import threading
import time

import edge_tts
import pyaudio
import speech_recognition as sr

from io import BytesIO
from typing import Optional
from pydub import AudioSegment

class JarvisEars:
    def __init__(self):
        self.rec = sr.Recognizer()
        self.mic = sr.Microphone()

        # Tune for responsiveness
        self.rec.energy_threshold         = 1400
        self.rec.dynamic_energy_threshold = True
        self.rec.pause_threshold          = 0.75   # seconds of silence to end phrase

        print("  Calibrating microphone… ", end="", flush=True)
        with self.mic as source:
            self.rec.adjust_for_ambient_noise(source, duration=1.5)
        print(f"threshold = {self.rec.energy_threshold:.0f}  ✓")

    def listen(self) -> Optional[str]:
        """
        Listen for one utterance (up to 20 s), return text or None.
        None is returned on silence timeout or unrecognised audio — not an error.
        """
        try:
            with self.mic as source:
                audio = self.rec.listen(source, timeout=10, phrase_time_limit=20)
            return getattr(self.rec, "recognize_google")(audio)
        except (sr.WaitTimeoutError, sr.UnknownValueError):
            return None
        except sr.RequestError as exc:
            print(f"\n  [STT Error] {exc}")
            return None


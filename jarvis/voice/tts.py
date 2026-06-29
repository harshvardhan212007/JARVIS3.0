import asyncio
import queue
import re
import threading
import time

import edge_tts
import pyaudio

from io import BytesIO
from typing import Optional
from pydub import AudioSegment
from voice.sentence_splitter import extract_sentences
import config 


class JarvisVoiceEngine:
    """
    Low-latency, gap-free TTS playback.

    Pipeline (two background threads):
      ┌─────────────────────────────────────────────────────────────────┐
      │  _sentence_q  →  TTS worker  →  _audio_q  →  Play worker       │
      │  (str)           edge-tts          (PCM bytes)   PyAudio write  │
      └─────────────────────────────────────────────────────────────────┘

    Key design choices:
      • Collect ALL edge-tts MP3 bytes for each sentence before decoding.
        Chunk-by-chunk MP3→PCM conversion causes audible gaps because individual
        MP3 frames are incomplete; full-sentence conversion is seamless.
      • TTS worker runs concurrently with Play worker — sentence N+1 is
        being synthesised while sentence N is still playing.
      • _audio_q has maxsize=4 to cap memory usage and provide backpressure.
    """

    def __init__(self):
        self._pa     = pyaudio.PyAudio()
        self._stream = self._pa.open(
            format=pyaudio.paInt16, channels=1,
            rate=config.AUDIO_RATE, output=True,
            frames_per_buffer=2048,
        )
        self._sentence_q = queue.Queue()           # str  → TTS worker
        self._audio_q    = queue.Queue(maxsize=4)  # PCM  → Play worker
        self._playing    = False
        self._active     = True

        threading.Thread(target=self._tts_worker,  daemon=True, name="JARVIS-TTS").start()
        threading.Thread(target=self._play_worker, daemon=True, name="JARVIS-Play").start()

    # ── TTS Worker ────────────────────────────────────────────────────────────

    def _tts_worker(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        while self._active:
            try:
                sentence = self._sentence_q.get(timeout=0.2)
                pcm = loop.run_until_complete(self._tts(sentence))
                if pcm:
                    self._audio_q.put(pcm)   # Blocks if audio_q is full → backpressure
                self._sentence_q.task_done()
            except queue.Empty:
                continue

    async def _tts(self, text: str) -> Optional[bytes]:
        """Convert one sentence to raw PCM bytes via edge-tts (full MP3 collected first)."""
        try:
            buf = BytesIO()
            comm = edge_tts.Communicate(text, config.VOICE, rate=config.SPEECH_RATE)
            async for chunk in comm.stream():
                if chunk["type"] == "audio":
                    buf.write(chunk.get("data", b""))

            buf.seek(0)
            if buf.getbuffer().nbytes == 0:
                return None

            # Decode full MP3 → PCM in one shot (no gaps, no partial frames)
            seg = (
                AudioSegment.from_file(buf, format="mp3")
                            .set_frame_rate(config.AUDIO_RATE)
                            .set_channels(1)
                            .set_sample_width(2)
            )
            return seg.raw_data

        except Exception as exc:
            print(f"\n  [TTS Error] {exc}")
            return None

    # ── Play Worker ───────────────────────────────────────────────────────────

    def _play_worker(self):
        while self._active:
            try:
                pcm = self._audio_q.get(timeout=0.05)
                self._playing = True
                self._stream.write(pcm)
                self._playing = False
                self._audio_q.task_done()
            except queue.Empty:
                self._playing = False

    # ── Public Interface ──────────────────────────────────────────────────────

    @property
    def is_speaking(self) -> bool:
        """True while ANY audio is in the pipeline or playing."""
        return (
            self._playing
            or not self._sentence_q.empty()
            or not self._audio_q.empty()
        )

    def enqueue(self, sentence: str):
        """Push one pre-split sentence straight into the TTS pipeline."""
        sentence = sentence.strip()
        if len(sentence) > 4:
            self._sentence_q.put(sentence)

    def speak(self, text: str):
        """Clean, split, and enqueue text — use for greetings / short messages."""
        cleaned = re.sub(r'[*_#`]', '', text).strip()
        sentences, _ = extract_sentences(cleaned + " ", min_len=5)
        # If no sentence boundary found, treat whole text as one sentence
        if not sentences:
            sentences = [cleaned]
        for s in sentences:
            self._sentence_q.put(s)

    def wait_done(self):
        """Block until every queued sentence has finished playing."""
        self._sentence_q.join()            # Wait for all TTS jobs to complete
        while self._audio_q.qsize() > 0 or self._playing:
            time.sleep(0.04)

    def interrupt(self):
        """Immediately clear all queued speech."""
        for q in (self._sentence_q, self._audio_q):
            while not q.empty():
                try:
                    q.get_nowait()
                    q.task_done()
                except queue.Empty:
                    break
        self._playing = False


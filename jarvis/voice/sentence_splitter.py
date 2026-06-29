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

import config
def extract_sentences(buffer: str, min_len: int = config.MIN_SENTENCE_LEN) -> tuple[list, str]:
    """
    Extract complete sentences from the START of `buffer`.

    Algorithm:
      For each sentence-boundary match (.!? followed by whitespace) in order,
      skip if the text before it is shorter than `min_len` (catches abbreviations
      like "Dr. " or "e.g. ").  On the first valid boundary, extract the sentence
      and restart search in the remainder.

    Returns (list_of_sentences, remaining_buffer).
    """
    sentences: list[str] = []

    while True:
        # `for…else` in Python: else block runs only if the loop wasn't `break`-ed
        for m in re.finditer(r'(?<=[.!?])\s+', buffer):
            if m.start() >= min_len:                      # Valid boundary
                sentence = buffer[:m.start()].strip()     # Text up to (not incl.) space
                buffer   = buffer[m.end():]               # Remainder starts after space
                if sentence:
                    sentences.append(sentence)
                break                                     # Restart search in new buffer
        else:
            break                                         # No valid boundary found → done

    return sentences, buffer

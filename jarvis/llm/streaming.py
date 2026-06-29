from voice.sentence_splitter import extract_sentences
from voice.tts import JarvisVoiceEngine
from langchain_groq import ChatGroq

def stream_and_speak(
    llm: ChatGroq,
    messages: list,
    voice: JarvisVoiceEngine,
) -> str:
    """
    Stream LLM tokens one by one.  The moment a sentence boundary is detected,
    the sentence is immediately pushed to the TTS pipeline — so JARVIS begins
    speaking ~300–500 ms into the LLM response, well before it finishes.

    Returns the full response text.
    """
    print("\n  J.A.R.V.I.S.: ", end="", flush=True)

    buffer    = ""
    full_text = ""

    for chunk in llm.stream(messages):
        tok = chunk.content
        if not tok:
            continue

        print(tok, end="", flush=True)
        buffer    =  buffer + str(tok)
        full_text = full_text + str(tok)

        # Extract any complete sentences that have formed in the buffer
        sentences, buffer = extract_sentences(buffer)
        for sentence in sentences:
            voice.enqueue(sentence)

    # Flush the remaining buffer (no trailing punctuation)
    tail = buffer.strip()
    if len(tail) > 4:
        voice.enqueue(tail)

    print()       # Newline after streaming output
    return full_text


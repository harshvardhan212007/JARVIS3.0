import os
import time
import warnings
warnings.filterwarnings("ignore")

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from voice.tts import JarvisVoiceEngine 
from voice.sentence_splitter import extract_sentences
from voice.stt import JarvisEars

from llm.streaming import stream_and_speak
from llm.rag import build_context
from llm.prompts import SYSTEM_PROMPT
from llm.memory import ConversationMemory

import config 


def boot_jarvis():
    width = 58
    print("\n" + "═" * width)
    print("  J.A.R.V.I.S.  v2.0  |  RAG + Live Web + Voice Stream")
    print("═" * width)

    if not os.path.exists(config.INDEX_PATH):
        print(
            f"\n  [ERROR] Vector index not found at '{config.INDEX_PATH}'.\n"
            "  Index your documents first:\n"
            "    python index_documents.py your_file.pdf\n"
        )
        return

    # ── Load components ───────────────────────────────────────────────────────
    print("\n  Loading embeddings…")
    embeddings = HuggingFaceEmbeddings(
        model_name=config.EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("  Loading vector memory…")
    vector_db = FAISS.load_local(
        config.INDEX_PATH, embeddings, allow_dangerous_deserialization=True
    )

    print(f"  Connecting to Groq  [{config.LLM_MODEL}]…")
    os.environ["GROQ_API_KEY"] = config.GROQ_API_KEY
    llm = ChatGroq(temperature=0.7, model=config.LLM_MODEL, streaming=True)

    print("  Initialising voice engine…")
    voice = JarvisVoiceEngine()

    print("  Initialising microphone…")
    ears = JarvisEars()

    print("\n  All systems nominal.")
    print("═" * width + "\n")

    # ── Greeting ─────────────────────────────────────────────────────────────
    voice.speak(
        "All systems online. Good day, Sir. "
        "J.A.R.V.I.S. is at your service. "
        "You may ask me about your documents or anything else."
    )
    voice.wait_done()

    # ── Conversation history  (LangChain message objects) ─────────────────────
    # Stored as (HumanMessage, AIMessage) pairs; we roll the last N turns.
    memory = ConversationMemory(config.HISTORY_TURNS)
    # ── Main loop ─────────────────────────────────────────────────────────────
    while True:
        try:
            # ── Gate: don't listen while speaking ─────────────────────────────
            while voice.is_speaking:
                time.sleep(0.08)

            print("\n  [Listening…]", flush=True)
            query = ears.listen()

            if not query:
                continue

            query = query.strip()
            print(f"  You › {query}")

            # ── Exit command ──────────────────────────────────────────────────
            if any(p in query.lower() for p in config.EXIT_PHRASES):
                voice.speak(
                    "Initiating shutdown sequence. "
                    "It has been a pleasure, Sir. Goodbye."
                )
                voice.wait_done()
                break

            # ── Build context  (RAG + optional web) ──────────────────────────
            context = build_context(query, vector_db)

            # ── Build message list with rolling history ───────────────────────
            messages = memory.build_messages(
                    SYSTEM_PROMPT,
                    context,
                    query,
                )

            # ── Stream response + start speaking immediately ──────────────────
            response_text = stream_and_speak(llm, messages, voice)

            # ── Save to history ───────────────────────────────────────────────
            memory.add(query, response_text)

            # ── Wait for all speech to finish before listening again ──────────
            voice.wait_done()

        except KeyboardInterrupt:
            print("\n\n  [Interrupted — standing by]")
            voice.interrupt()
            voice.speak("Interrupted. Standing by, Sir.")
            voice.wait_done()
            # Do not break — keep listening after Ctrl-C

        except Exception as exc:
            import traceback
            print(f"\n  [Error] {exc}")
            traceback.print_exc()
            voice.interrupt()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    boot_jarvis()
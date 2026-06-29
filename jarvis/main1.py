"""
J.A.R.V.I.S. Document Indexer  v2.0
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Ingests PDFs, TXT, and Markdown files into a local FAISS vector store
that JARVIS uses for Retrieval-Augmented Generation.

Usage
─────
    python index_documents.py file.pdf notes.txt
    python index_documents.py docs/*.pdf               # glob patterns work
    python index_documents.py report.pdf paper.pdf     # multiple files
    python index_documents.py --reset report.pdf       # wipe old index first

Index is saved to  jarvis_vector_index/  in the current directory.
Run this once (or whenever your documents change), then launch jarvis.py.
"""

import os
import sys
import glob
import shutil
import argparse
import warnings
warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ── Settings ──────────────────────────────────────────────────────────────────

INDEX_PATH    = "jarvis_vector_index"
EMBED_MODEL   = "all-MiniLM-L6-v2"
CHUNK_SIZE    = 900
CHUNK_OVERLAP = 150

# ── Helpers ───────────────────────────────────────────────────────────────────

def _banner(msg: str):
    width = 58
    print("\n" + "═" * width)
    print(f"  {msg}")
    print("═" * width)


def _load_file(path: str) -> list:
    """Load a single file. Returns list of LangChain Document objects."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        loader = PyPDFLoader(path)
    elif ext in (".txt", ".md", ".markdown"):
        loader = TextLoader(path, encoding="utf-8")
    else:
        return []          # Unsupported — caller will warn

    return loader.load()


# ── Main ──────────────────────────────────────────────────────────────────────

def index_documents(raw_paths: list[str], reset: bool = False) -> bool:
    _banner("J.A.R.V.I.S. MEMORY INDEXING  v2.0")

    # ── Resolve glob patterns → flat list of real paths ───────────────────────
    file_paths: list[str] = []
    for pattern in raw_paths:
        matches = sorted(glob.glob(pattern, recursive=True))
        if matches:
            file_paths.extend(matches)
        else:
            file_paths.append(pattern)   # Let the loader emit the "not found" error

    # ── Load every file ───────────────────────────────────────────────────────
    all_docs = []
    print()
    for path in file_paths:
        if not os.path.exists(path):
            print(f"  ⚠  Not found  → {path}")
            continue

        ext = os.path.splitext(path)[1].lower()
        if ext not in (".pdf", ".txt", ".md", ".markdown"):
            print(f"  ✗  Unsupported ({ext})  → {path}")
            continue

        print(f"  ►  Loading  {path}", end="", flush=True)
        try:
            docs = _load_file(path)
            all_docs.extend(docs)
            label = "pages" if ext == ".pdf" else "segments"
            print(f"   [{len(docs)} {label}]  ✓")
        except Exception as exc:
            print(f"   — ERROR: {exc}")

    if not all_docs:
        print("\n  [ERROR] No documents could be loaded. Aborting.\n")
        return False

    print(f"\n  Total raw segments  : {len(all_docs)}")

    # ── Chunk ─────────────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(all_docs)
    print(f"  Neural clusters     : {len(chunks)}")

    # ── Embed ─────────────────────────────────────────────────────────────────
    print("\n  Loading embedding model  (first run downloads ~90 MB) …")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBED_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    # ── Build / merge FAISS index ─────────────────────────────────────────────
    if reset and os.path.exists(INDEX_PATH):
        print(f"  Wiping old index at '{INDEX_PATH}' …")
        shutil.rmtree(INDEX_PATH)

    if os.path.exists(INDEX_PATH) and not reset:
        print("  Merging with existing index …")
        existing = FAISS.load_local(
            INDEX_PATH, embeddings, allow_dangerous_deserialization=True
        )
        new_db = FAISS.from_documents(chunks, embeddings)
        existing.merge_from(new_db)
        existing.save_local(INDEX_PATH)
    else:
        print("  Building FAISS index …")
        db = FAISS.from_documents(chunks, embeddings)
        db.save_local(INDEX_PATH)

    print(f"\n  ✓  Index saved → '{INDEX_PATH}/'")
    print("  JARVIS is ready.  Run  python jarvis.py  to boot.\n")
    print("═" * 58 + "\n")
    return True


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Index PDFs/text files for J.A.R.V.I.S.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "files", nargs="+",
        help="PDF or text files to index (glob patterns supported)"
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete existing index before indexing (default: merge)"
    )
    args = parser.parse_args()

    ok = index_documents(args.files, reset=args.reset)
    sys.exit(0 if ok else 1)
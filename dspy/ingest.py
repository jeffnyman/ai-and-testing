import os
import re
import ollama
import chromadb
from pypdf import PdfReader

# --- Configuration ---
# The PDF must be in the same directory as this script.
PDF_PATH = "brain-on-chatgpt.pdf"

# The Chroma collection and persistence directory. Running this script
# again will delete and recreate the collection, so ingestion is always
# a clean rebuild rather than an append.
CHROMA_PATH = "chroma_store"
COLLECTION_NAME = "brain_on_chatgpt"

# The embedding model. This must already be pulled in Ollama:
#   ollama pull nomic-embed-text
EMBED_MODEL = "nomic-embed-text"

# Chunking parameters. CHUNK_SIZE is the target character length per
# chunk (not tokens — characters are simpler to reason about and close
# enough for this purpose). CHUNK_OVERLAP is how many characters are
# shared between adjacent chunks. Overlap ensures that sentences or
# ideas that fall on a chunk boundary aren't lost to either side.
CHUNK_SIZE = 1500
CHUNK_OVERLAP = 150


def extract_text(pdf_path: str) -> str:
  """
  Extract all text from the PDF as a single string. pypdf handles
  the page iteration; we join with a space to avoid words being
  fused across page boundaries.
  """
  reader = PdfReader(pdf_path)
  pages = []
  for page in reader.pages:
    text = page.extract_text()
    if text:
      pages.append(text)
  return " ".join(pages)


def clean_text(text: str) -> str:
  """
  Light cleanup pass. PDF extraction from multi-column academic
  papers often produces extra whitespace, hyphenated line breaks,
  and artifacts from figure captions being interleaved with body
  text. This doesn't eliminate all noise, but reduces it enough
  that chunks are readable and embeddable.
  """
  # Collapse runs of whitespace (but preserve paragraph breaks)
  text = re.sub(r"[ \t]+", " ", text)
  # Rejoin words split by hyphen-newline (common in academic PDFs)
  text = re.sub(r"-\s*\n\s*", "", text)
  # Normalize remaining newlines to spaces
  text = re.sub(r"\n+", " ", text)
  return text.strip()


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
  """
  Split text into overlapping fixed-size chunks. This is the
  simplest possible chunking strategy: slide a window of
  `chunk_size` characters across the text, stepping by
  `chunk_size - overlap` each time.

  More sophisticated strategies exist — splitting on sentence
  boundaries, on section headers, or using semantic similarity to
  group related content — but fixed-size chunking with overlap is
  a reliable baseline and keeps the focus here on the DSPy pipeline
  rather than chunking theory.
  """
  chunks = []
  start = 0
  step = chunk_size - overlap

  while start < len(text):
    end = start + chunk_size
    chunks.append(text[start:end])
    start += step

  return chunks


def embed(texts: list[str]) -> list[list[float]]:
  """
  Call Ollama's embedding endpoint for a list of texts. Returns a
  list of embedding vectors, one per input text. Ollama's Python
  client handles the HTTP call; we just iterate over the results.
  """
  vectors = []
  for text in texts:
    response = ollama.embed(model=EMBED_MODEL, input=text)
    vectors.append(response.embeddings[0])
  return vectors


def ingest(pdf_path: str, chroma_path: str, collection_name: str):
  print(f"Reading: {pdf_path}")
  raw = extract_text(pdf_path)
  cleaned = clean_text(raw)
  print(f"Extracted {len(cleaned):,} characters of text.")

  chunks = chunk_text(cleaned, CHUNK_SIZE, CHUNK_OVERLAP)
  print(f"Created {len(chunks)} chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}).")

  # Set up Chroma with a local persistent store. If the directory
  # already exists, Chroma will open it; we then delete and recreate
  # the collection to ensure a clean ingest.
  client = chromadb.PersistentClient(path=chroma_path)

  # Delete the collection if it already exists so re-running ingest
  # always produces a clean store rather than appending duplicates.
  existing = [c.name for c in client.list_collections()]
  if collection_name in existing:
    client.delete_collection(collection_name)
    print(f"Deleted existing collection '{collection_name}'.")

  collection = client.create_collection(collection_name)

  print(f"Embedding {len(chunks)} chunks via {EMBED_MODEL}. This may take a minute...")
  vectors = embed(chunks)

  # Each chunk gets a simple sequential ID. Chroma also stores the
  # original text as a document so we can retrieve it at query time
  # without needing a separate lookup.
  ids = [f"chunk_{i}" for i in range(len(chunks))]
  collection.add(
    ids=ids,
    embeddings=vectors,
    documents=chunks,
  )

  print(f"Ingested {len(chunks)} chunks into collection '{collection_name}'.")
  print(f"Chroma store saved to: {os.path.abspath(chroma_path)}")


if __name__ == "__main__":
  if not os.path.exists(PDF_PATH):
    raise FileNotFoundError(
      f"'{PDF_PATH}' not found. Make sure the paper PDF is in the same "
      "directory as this script and is named 'paper.pdf'."
    )
  ingest(PDF_PATH, CHROMA_PATH, COLLECTION_NAME)

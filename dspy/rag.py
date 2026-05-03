import sys
import ollama
import chromadb
import dspy

# --- Configuration ---
# These must match the values used in ingest.py.
CHROMA_PATH = "chroma_store"
COLLECTION_NAME = "brain_on_chatgpt"
EMBED_MODEL = "nomic-embed-text"

# How many chunks to retrieve for each question. A larger k gives the
# model more context but increases the chance of irrelevant passages
# being included. 5 is a reasonable starting point for a paper of
# this length.
TOP_K = 5

# --- Layer 1: Backend ---
lm = dspy.LM(
  "ollama_chat/jeffnyman/ts-reasoner", api_base="http://localhost:11434", api_key=""
)
dspy.configure(lm=lm)


# --- Layer 2: Retrieval ---
# Retrieval is kept as a plain function rather than a DSPy module.
# This keeps the Chroma mechanics visible and separates the retrieval
# concern from the generation concern cleanly. The function embeds the
# query using the same model used during ingestion, queries Chroma for
# the top-k nearest chunks, and returns them as a list of strings.
def retrieve(question: str, k: int = TOP_K) -> list[str]:
  response = ollama.embed(model=EMBED_MODEL, input=question)
  query_vector = response.embeddings[0]

  client = chromadb.PersistentClient(path=CHROMA_PATH)
  collection = client.get_collection(COLLECTION_NAME)

  results = collection.query(
    query_embeddings=[query_vector],
    n_results=k,
  )

  # results["documents"] is a list of lists (one list per query).
  # We sent one query, so index 0 gives us our k retrieved chunks.
  docs = results["documents"]
  assert docs is not None
  return docs[0]


# --- Layer 3: Signatures ---
# RAGSignature is where field descriptions become load-bearing rather
# than optional. Without them, DSPy would compile a prompt that treats
# `context` as an opaque string field with no guidance on how to use
# it. The descriptions tell the model what the context is, where it
# came from, and what faithfulness constraint applies to the answer.
class RAGSignature(dspy.Signature):
  """Answer questions about the paper using only the provided context."""

  context: str = dspy.InputField(
    desc="Relevant passages retrieved from the paper "
    "'Your Brain on ChatGPT: Accumulation of Cognitive Debt when "
    "Using an AI Assistant for Essay Writing Task'."
  )
  question: str = dspy.InputField(
    desc="A question about the paper's findings, methodology, or conclusions."
  )
  answer: str = dspy.OutputField(
    desc="A precise answer grounded in the provided context. "
    "Do not introduce information not present in the context."
  )


# --- Layer 4: Module ---
# The module owns a single ChainOfThought predictor. The retrieval
# step runs before the DSPy call in forward(), making the pipeline
# explicit: retrieve context, then generate an answer grounded in it.
# This two-phase structure is the core RAG pattern.
class RAGPipeline(dspy.Module):
  def __init__(self):
    super().__init__()
    self.generate = dspy.ChainOfThought(RAGSignature)

  def forward(self, question: str):
    # Phase 1: retrieve relevant chunks from the paper.
    chunks = retrieve(question)

    # Join the retrieved chunks into a single context string.
    # A numbered list makes it easier for the model to distinguish
    # passage boundaries, which helps when the answer spans multiple
    # retrieved chunks.
    context = "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(chunks))

    # Phase 2: generate an answer grounded in the retrieved context.
    return self.generate(context=context, question=question)


if __name__ == "__main__":
  q = (
    sys.argv[1]
    if len(sys.argv) > 1
    else "What did the EEG data reveal about the LLM group compared to the Brain-only group?"
  )
  print(f"Question: {q}\n")

  pipeline = RAGPipeline()
  result = pipeline(question=q)

  print("=== Prediction ===")
  print(result)

  print("\n=== Generated Prompt ===")
  dspy.inspect_history(n=1)
  print("========================\n")

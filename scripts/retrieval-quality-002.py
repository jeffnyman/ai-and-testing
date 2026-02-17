# Reference:
# https://testerstories.com/2026/02/ai-and-testing-improving-retrieval-quality-part-2/

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from deepeval.metrics import ContextualPrecisionMetric, FaithfulnessMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from deepeval import evaluate

def create_rag_system(chunk_size=1000, chunk_overlap=200, k=3):
  """Create a RAG system with configurable parameters."""
  loader = PyPDFLoader("./arXiv-jnyman-051011v3.pdf")
  documents = loader.load()

  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap
  )

  chunks = text_splitter.split_documents(documents)

  embeddings = OllamaEmbeddings(model="nomic-embed-text")
  vectorstore = Chroma.from_documents(chunks, embeddings)

  retriever = vectorstore.as_retriever(search_kwargs={"k": k})

  return retriever, len(chunks)

def create_rag_system_semantic(k=3):
  """Create a RAG system with semantically-aware chunking."""
  loader = PyPDFLoader("./arXiv-jnyman-051011v3.pdf")
  documents = loader.load()

  # Use separators that respect document structure
  text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=150,
    separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
    length_function=len
  )

  chunks = text_splitter.split_documents(documents)

  embeddings = OllamaEmbeddings(model="nomic-embed-text")
  vectorstore = Chroma.from_documents(chunks, embeddings)

  retriever = vectorstore.as_retriever(search_kwargs={"k": k})

  return retriever, len(chunks)

def run_test(retriever, question, expected_output, show_chunks=True):
  """Run a complete test with both metrics."""

  execution_model = ChatOllama(model="jeffnyman/ts-reasoner")
  judge_model = OllamaModel(model="jeffnyman/ts-evaluator")

  # Get relevant context
  retrieved_docs = retriever.invoke(question)
  context = [doc.page_content for doc in retrieved_docs]

  # Generate chunks
  if show_chunks:
    print("\n" + "-" * 60)
    print("RETRIEVED CHUNKS:")
    print("-" * 60)

    for i, chunk in enumerate(context, 1):
      print(f"\n--- Chunk {i} ---")
      print(chunk)

    print("-" * 60 + "\n")

  # Generate response
  prompt = f"Based on this context: {context}\n\nQuestion: {question}"
  response = execution_model.invoke(prompt).content

  # Create test case
  test_case = LLMTestCase(
    input=question,
    actual_output=response,
    expected_output=expected_output,
    retrieval_context=context
  )

  # Create metrics
  precision_metric = ContextualPrecisionMetric(
    model=judge_model,
    verbose_mode=True
  )

  faithfulness_metric = FaithfulnessMetric(
    model=judge_model,
    verbose_mode=True
  )

  # Evaluate with both metrics
  results = evaluate(
    test_cases=[test_case],
    metrics=[precision_metric, faithfulness_metric]
  )

  return results, context, response

def get_scores(results):
  """Safely extract scores from results."""
  if results is not None:
    metrics_data = results.test_results[0].metrics_data
    if metrics_data is not None:
      return {m.name: m.score for m in metrics_data}

  return {}

def print_scores(label, results, baseline_results=None):
  """Print scores with optional comparison to baseline."""
  print(f"\n{label} Scores:")
  scores = get_scores(results)

  if scores:
    print(f"Contextual Precision: {scores.get("Contextual Precision")}")
    print(f"Faithfulness: {scores.get("Faithfulness")}")

    if baseline_results is not None:
      baseline_scores = get_scores(baseline_results)

      if baseline_scores:
        precision_change = scores.get("Contextual Precision", 0) \
          - baseline_scores.get("Contextual Precision", 0)
        faithfulness_change = scores.get("Faithfulness", 0) \
          - baseline_scores.get("Faithfulness", 0)

        print("\nComparison to Baseline:")
        print(f"Contextual Precision: {precision_change:+.2f}")
        print(f"Faithfulness: {faithfulness_change:+.2f}")
  else:
    print("No metrics data available.")

question = """Please consider Jeff Nyman's warp drive paper.
What energy source does the paper propose would be needed to
generate the warp bubble for faster-than-light travel?"""

expected_output = """Matter/antimatter annihilation, requiring
approximately 10^28 kg of antimatter (equivalent to Jupiter's
mass-energy)."""

# =========================================================
# BASELINE
# =========================================================
print("=" * 60)
print("BASELINE: chunk_size=1000, chunk_overlap=200, k=3")
print("=" * 60)

retriever, num_chunks = create_rag_system(
  chunk_size=1000,
  chunk_overlap=200,
  k=3
)

print(f"Document split into {num_chunks} chunks")

baseline_results, baseline_context, baseline_response = run_test(
  retriever,
  question,
  expected_output
)

print_scores("Baseline", baseline_results)

# =========================================================
# EXPERIMENT 1: Smaller Chunks
# =========================================================
print("\n" + "=" * 60)
print("EXPERIMENT 1: chunk_size=500, chunk_overlap=100, k=3")
print("=" * 60)

retriever_exp1, num_chunks_exp1 = create_rag_system(
  chunk_size=500,
  chunk_overlap=100,
  k=3
)

print(f"Document split into {num_chunks_exp1} chunks")

exp1_results, exp1_context, exp1_response = run_test(
  retriever_exp1,
  question,
  expected_output
)

print_scores("Experiment 1", exp1_results, baseline_results)

# =========================================================
# EXPERIMENT 2: More Chunks
# =========================================================
print("\n" + "=" * 60)
print("EXPERIMENT 2: chunk_size=1000, chunk_overlap=200, k=5")
print("=" * 60)

retriever_exp2, num_chunks_exp2 = create_rag_system(
  chunk_size=1000,
  chunk_overlap=200,
  k=5
)

print(f"Document split into {num_chunks_exp2} chunks")

exp2_results, exp2_context, exp2_response = run_test(
  retriever_exp2,
  question,
  expected_output
)

print_scores("Experiment 2", exp2_results, baseline_results)

# =========================================================
# EXPERIMENT 3: Combined (Smaller + More)
# =========================================================
print("\n" + "=" * 60)
print("EXPERIMENT 3: chunk_size=500, chunk_overlap=100, k=5")
print("=" * 60)

retriever_exp3, num_chunks_exp3 = create_rag_system(
  chunk_size=500,
  chunk_overlap=100,
  k=5
)

print(f"Document split into {num_chunks_exp3} chunks")

exp3_results, exp3_context, exp3_response = run_test(
  retriever_exp3,
  question,
  expected_output
)

print_scores("Experiment 3", exp3_results, baseline_results)

# =========================================================
# EXPERIMENT 4: Semantic Chunking
# =========================================================
print("\n" + "=" * 60)
print("EXPERIMENT 4: Semantic chunking, k=3")
print("=" * 60)

retriever_exp4, num_chunks_exp4 = create_rag_system_semantic(k=3)
print(f"Document split into {num_chunks_exp4} chunks")

exp4_results, exp4_context, exp4_response = run_test(
  retriever_exp4,
  question,
  expected_output
)

print_scores("Experiment 4", exp4_results, baseline_results)

# =========================================================
# RESULTS SUMMARY
# =========================================================
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"{'Configuration':<40} {'Precision':>12} {'Faithfulness':>12} {'Change':>12}")
print("-" * 60)

configs = [
  ("Baseline (1000/200/k=3)", baseline_results, None),
  ("Exp 1: Smaller chunks (500/100/k=3)", exp1_results, baseline_results),
  ("Exp 2: More chunks (1000/200/k=5)", exp2_results, baseline_results),
  ("Exp 3: Both (500/100/k=5)", exp3_results, baseline_results),
  ("Exp 4: Semantic (800/150/k=3)", exp4_results, baseline_results)
]

baseline_scores = get_scores(baseline_results)

for name, results, baseline in configs:
  scores = get_scores(results)
  precision = scores.get("Contextual Precision", 0.0)
  faithfulness = scores.get("Faithfulness", 0.0)

  if baseline is None:
    change = "baseline"
  else:
    precision_change = precision \
      - baseline_scores.get("Contextual Precision", 0.0)
    faithfulness_change = faithfulness \
      - baseline_scores.get("Faithfulness", 0.0)
    change = f"P:{precision_change:+.2f} F:{faithfulness_change:+.2f}"

  print(f"{name:<40} {precision:>12.2f} {faithfulness:>12.2f} {change:>12}")

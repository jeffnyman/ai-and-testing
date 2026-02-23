# Reference:
# https://testerstories.com/2026/02/ai-and-testing-improving-retrieval-quality-part-4/

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

def run_test(retriever, question, expected_output, show_chunks=True):
  """Run a complete test with both metrics."""

  execution_model = ChatOllama(model="jeffnyman/ts-reasoner")
  judge_model = OllamaModel(model="jeffnyman/ts-evaluator")

  # Get relevant context
  retrieved_docs = retriever.invoke(question)
  context = [doc.page_content for doc in retrieved_docs]

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

def print_scores(label, results):
  """Print scores."""
  print(f"\n{label} Scores:")
  scores = get_scores(results)

  if scores:
    print(f"Contextual Precision: {scores.get('Contextual Precision')}")
    print(f"Faithfulness: {scores.get('Faithfulness')}")
  else:
    print("No metrics data available.")

# =========================================================
# Setup baseline RAG system
# =========================================================
print("=" * 60)
print("BASELINE CONFIGURATION: chunk_size=1000, chunk_overlap=200, k=3")
print("=" * 60)

retriever, num_chunks = create_rag_system(
  chunk_size=1000,
  chunk_overlap=200,
  k=3
)

print(f"Document split into {num_chunks} chunks\n")

# =========================================================
# TEST: Specific Factual Query (like Part 2's energy question)
# =========================================================
print("=" * 60)
print("TEST: Specific Factual Query - Power-law Exponent")
print("=" * 60)

question = """What is the power-law exponent for the extinction
distribution in Jeff Nyman's extinction model?"""

expected = """The power-law exponent α is 2.183±0.007, meaning the
frequency of an extinction P(s) is related to its size s by P(s) ∝ s^(-α)."""

results, context, response = run_test(
  retriever,
  question,
  expected
)

print_scores("Specific Factual Query", results)

# =========================================================
# TEST: Conceptual Query - Model Mechanism
# =========================================================
print("\n" + "=" * 60)
print("TEST: Conceptual Query - Coevolutionary Avalanches")
print("=" * 60)

question = """How does Jeff Nyman's model explain the relationship
between coevolutionary avalanches and mass extinctions?"""

expected = """The model proposes that mass extinctions arise from
the coincidence of two events: coevolutionary avalanches (which
reduce species fitness and increase susceptibility to environmental
stress) and external environmental catastrophes. Neither alone is
sufficient; the avalanches create vulnerability, and environmental
stress exploits it."""

results, context, response = run_test(
  retriever,
  question,
  expected
)

print_scores("Conceptual Query", results)

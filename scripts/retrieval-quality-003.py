# Reference:
# https://testerstories.com/2026/02/ai-and-testing-improving-retrieval-quality-part-3/

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
# TEST 1: Extra Dimensions and Warp Bubble Creation
# =========================================================
print("=" * 60)
print("TEST 1: Extra Dimensions Question")
print("=" * 60)

question_1 = """How does Jeff Nyman propose that manipulating
extra dimensions creates a warp bubble?"""

expected_1 = """By locally manipulating the radius of extra
dimensions, which creates an asymmetry in the cosmological
constant that expands and contracts space-time around the
spacecraft."""

results_1, context_1, response_1 = run_test(
  retriever,
  question_1,
  expected_1
)

print_scores("Test 1", results_1)

# =========================================================
# TEST 2: Kaluza-Klein Modes
# =========================================================
print("\n" + "=" * 60)
print("TEST 2: Kaluza-Klein Modes Question")
print("=" * 60)

question_2 = """What role do Kaluza-Klein modes play in Jeff Nyman's
warp drive concept?"""

expected_2 = """Kaluza-Klein graviton modes contribute to the
Casimir energy in higher dimensions, which is associated with
the cosmological constant. This relationship between the
compactified extra dimensions and the cosmological constant is
fundamental to the warp drive mechanism."""

results_2, context_2, response_2 = run_test(
  retriever,
  question_2,
  expected_2
)

print_scores("Test 2", results_2)

# =========================================================
# TEST 3: Cosmological Constant Relationship
# =========================================================
print("\n" + "=" * 60)
print("TEST 3: Cosmological Constant Question")
print("=" * 60)

question_3 = """What is the relationship between the cosmological
constant and warp bubble formation in Jeff Nyman's paper?"""

expected_3 = """The cosmological constant is linked to the radius
of extra dimensions through Casimir energy. By manipulating the
extra dimension radius, the local cosmological constant can be
adjusted, creating expansion and contraction of space-time that
forms the warp bubble."""

results_3, context_3, response_3 = run_test(
  retriever,
  question_3,
  expected_3
)

print_scores("Test 3", results_3)

# =========================================================
# RESULTS SUMMARY
# =========================================================
print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)
print(f"{'Test':<50} {'Precision':>12} {'Faithfulness':>12}")
print("-" * 60)

tests = [
  ("Test 1: Extra Dimensions", results_1),
  ("Test 2: Kaluza-Klein Modes", results_2),
  ("Test 3: Cosmological Constant", results_3)
]

for name, results in tests:
  scores = get_scores(results)
  precision = scores.get("Contextual Precision", 0.0)
  faithfulness = scores.get("Faithfulness", 0.0)
  print(f"{name:<50} {precision:>12.2f} {faithfulness:>12.2f}")

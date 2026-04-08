"""
evaluate_coherence_essay.py

Evaluates the "Coherence at the Edge" essay using three DeepEval metrics:
  - ContextualRecallMetric
  - ContextualRelevancyMetric
  - GEval

The script is structured in three parts, mirroring the pattern established
in the DeepEval blog series:

  Part 1: Controlled cases (hand-crafted contexts) for Recall and Relevancy
  Part 2: Live RAG cases (actual retrieval from the essay URL) for all three
  Part 3: G-Eval with custom criteria targeting the essay's philosophical nuance

Source document: https://testerstories.com/files/ai_testing/coherence-at-the-edge.html
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os

os.environ["USER_AGENT"] = "coherence-essay-evaluator/1.0"

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from deepeval.metrics import ContextualRecallMetric, ContextualRelevancyMetric, GEval
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from bs4.filter import SoupStrainer

# =============================================================================
# MODELS
# =============================================================================

execution_model = ChatOllama(model="jeffnyman/ts-reasoner")
judge_model = OllamaModel(model="jeffnyman/ts-evaluator")

# =============================================================================
# METRICS
# =============================================================================

recall_metric = ContextualRecallMetric(model=judge_model, verbose_mode=True)
relevancy_metric = ContextualRelevancyMetric(model=judge_model, verbose_mode=True)

# G-Eval requires defining what we're judging. We use three criteria that
# fit a philosophical essay better than any RAG-specific metric could.
#
# Criterion A: Epistemic Humility
#   Does the response distinguish between the essay's speculative claims
#   and its stronger argumentative claims?
#
# Criterion B: Argumentative Fidelity
#   Does the response accurately represent the essay's actual position
#   without flattening nuance? The essay deliberately hovers between
#   positions rather than committing to one.
#
# Criterion C: Conceptual Precision
#   Does the response use the essay's key terms correctly in context?

geval_metric = GEval(
  name="PhilosophicalEssayFidelity",
  model=judge_model,
  verbose_mode=True,
  criteria="""Evaluate the response against these three criteria:

    1. Epistemic Humility: The response should acknowledge where the essay
    is explicitly speculative (using hedges like 'perhaps', 'might', 'one
    possibility is') versus where it makes stronger claims. Responses that
    present speculative positions as settled conclusions should score lower.

    2. Argumentative Fidelity: The response should accurately represent the
    essay's actual position without collapsing its deliberate nuance. For
    example, the essay explicitly positions itself between pantheism and
    panentheism rather than committing to either. A response that simply
    says 'the essay argues the universe is God' misrepresents this.

    3. Conceptual Precision: The response should use the essay's key terms
    (such as atemporality, teleological monism, modal structure, participatory
    ontology) correctly and in the sense the essay intends them, not in a
    generic philosophical sense.""",
  evaluation_params=[
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
    LLMTestCaseParams.EXPECTED_OUTPUT,
  ],
)

# =============================================================================
# PART 1: CONTROLLED CASES
# =============================================================================
#
# These cases use hand-crafted retrieval contexts, just as the blog post's
# warp drive examples did. This gives us a known baseline before we introduce
# the unpredictability of live retrieval.
#
# We use two questions:
#
#   Q1 (for Recall): "What three interpretations does the essay offer for
#      why physical limits like the speed of light exist?"
#      Answer lives in "Guardrails or Geometry". Requires all three
#      interpretations to be present for a complete answer.
#
#   Q3 (for Relevancy): "What does the essay mean when it says history might
#      be 'constitutive' of a Creator rather than merely something a Creator
#      interacts with?"
#      Answer lives in "History as the Medium of Divinity". The word
#      "history" and "Creator" appear across many sections -- a strong
#      vocabulary-trap candidate for a semantic retriever.

# -----------------------------------------------------------------------------
# Q1 context sets
# -----------------------------------------------------------------------------

q1_question = """In the essay "Coherence at the Edge", what three
interpretations does the author offer for why physical limits like
the speed of light exist?"""

q1_expected = """The essay offers three interpretations. First, a structural
interpretation: the limits are consequences of deep mathematical consistency
and emerge from the internal coherence of the equations rather than being
imposed from outside. Second, an evolutionary interpretation: life can only
arise in universes with such constraints because without them entropy
gradients would not form stable structures and complexity could not stabilize.
Third, a metaphysical or theological interpretation: reality has built-in
finitude, meaning creaturely knowledge is bounded by design, which resonates
with a long philosophical intuition even if it does not imply simplistic
design arguments."""

# High recall: all three interpretations present and well-ordered
q1_high_recall_context = [
  """One interpretation is purely structural: these limits are simply
    consequences of deep mathematical consistency. Causality preservation,
    energy conditions, quantum stability: these aren't imposed from outside;
    they emerge from the internal coherence of the equations.""",
  """Another interpretation is evolutionary: perhaps life can only arise
    in universes with such constraints. If causality were loose or time
    travel trivial, entropy gradients might not form stable structures.
    Information might not accumulate. Complexity might not stabilize. In
    that sense, the containment could be a precondition for life rather
    than a restriction against it.""",
  """A third interpretation is more metaphysical or theological: that
    reality has built-in finitude. We are not meant to have unbounded
    access to spacetime. There are epistemic horizons. That doesn't imply
    design in a simplistic sense, but it does resonate with a long
    philosophical intuition that creaturely knowledge is bounded.""",
]

# Low recall: only two of the three interpretations present, third is missing
# and replaced with a chunk that discusses limits in a different context
q1_low_recall_context = [
  """One interpretation is purely structural: these limits are simply
    consequences of deep mathematical consistency. Causality preservation,
    energy conditions, quantum stability: these aren't imposed from outside;
    they emerge from the internal coherence of the equations.""",
  """Another interpretation is evolutionary: perhaps life can only arise
    in universes with such constraints. If causality were loose or time
    travel trivial, entropy gradients might not form stable structures.""",
  """Take the speed-of-light limit again. In relativity, it's not a
    throttle; it's the geometry of spacetime itself. Remove that constraint
    and you don't just gain faster travel, you destabilize causality. You
    invite paradox. Logical consistency would be non-negotiable.""",
]

# -----------------------------------------------------------------------------
# Q3 context sets
# -----------------------------------------------------------------------------

q3_question = """What does the essay "Coherence at the Edge" mean when it
says history might be 'constitutive' of a Creator rather than merely
something a Creator interacts with?"""

q3_expected = """The essay argues that just as a human cannot exist apart
from physics (because human embodiment is constituted by physical processes,
not merely contained by them), a Creator might similarly require history not
as an external container but as what makes 'Creator' intelligible at all.
Without history there is nothing to ground the concept of atemporality,
because atemporality is only meaningful in contrast to temporality -- just
as silence is only meaningful in contrast to sound. History would then be
ontologically basic, with atemporality being something achieved or abstracted
from that process rather than an intrinsic divine property."""

# High relevancy: chunks are directly from the relevant section
q3_high_relevancy_context = [
  """A human cannot exist apart from physics because human embodiment is
    constituted by physical processes. Physics is not an external container;
    it is part of what makes a human be a human. So I'm wondering whether
    history is similarly constitutive for a Creator. Not just something the
    Creator interacts with, but something without which 'Creator' becomes
    unintelligible.""",
  """Is timelessness parasitic on time the way silence is parasitic on
    sound? If there were never any sound at all, would 'silence' even be
    intelligible? Atemporality only makes sense in contrast to temporality.
    If there were no change, no becoming anywhere, the concept of 'outside
    time' might dissolve into meaninglessness.""",
  """My model inverts the classical view. It suggests that history might
    be ontologically basic -- that becoming is fundamental -- and that
    atemporality is something achieved or abstracted from that process.""",
]

# Low relevancy: chunks discuss history and Creator but from other sections,
# not the constitutive argument -- the vocabulary trap in action
q3_low_relevancy_context = [
  """The 'God at the End of Time' idea preserves teleology without
    requiring temporal priority. The end explains the beginning not causally
    but logically. The final integrated state makes the whole trajectory
    coherent.""",
  """If the Creator guarantees the outcome by guiding the evolutionary
    arc, then the loop is no longer precarious. The system isn't gambling
    on blind contingency; it is internally directed.""",
  """The Creator is the emergent totality; perhaps the final integrated
    consciousness of the whole. Evolution becomes the mechanism by which
    divinity realizes itself.""",
]

# -----------------------------------------------------------------------------
# Build controlled test cases
# -----------------------------------------------------------------------------

q1_high_recall_case = LLMTestCase(
  input=q1_question,
  actual_output=q1_expected,  # using expected as actual for controlled baseline
  expected_output=q1_expected,
  retrieval_context=q1_high_recall_context,
)

q1_low_recall_case = LLMTestCase(
  input=q1_question,
  actual_output=q1_expected,
  expected_output=q1_expected,
  retrieval_context=q1_low_recall_context,
)

q3_high_relevancy_case = LLMTestCase(
  input=q3_question,
  actual_output=q3_expected,
  expected_output=q3_expected,
  retrieval_context=q3_high_relevancy_context,
)

q3_low_relevancy_case = LLMTestCase(
  input=q3_question,
  actual_output=q3_expected,
  expected_output=q3_expected,
  retrieval_context=q3_low_relevancy_context,
)

# -----------------------------------------------------------------------------
# Run controlled cases
# -----------------------------------------------------------------------------

print("=" * 60)
print("PART 1: CONTROLLED CASES")
print("=" * 60)

print("\n" + "=" * 60)
print("CONTEXTUAL RECALL — High Recall Case (Q1)")
print("=" * 60)
recall_metric.measure(q1_high_recall_case)
high_recall_score = recall_metric.score
high_recall_reason = recall_metric.reason

print("\n" + "=" * 60)
print("CONTEXTUAL RECALL — Low Recall Case (Q1)")
print("=" * 60)
recall_metric.measure(q1_low_recall_case)
low_recall_score = recall_metric.score
low_recall_reason = recall_metric.reason

print("\n" + "=" * 60)
print("CONTEXTUAL RELEVANCY — High Relevancy Case (Q3)")
print("=" * 60)
relevancy_metric.measure(q3_high_relevancy_case)
high_relevancy_score = relevancy_metric.score
high_relevancy_reason = relevancy_metric.reason

print("\n" + "=" * 60)
print("CONTEXTUAL RELEVANCY — Low Relevancy Case (Q3)")
print("=" * 60)
relevancy_metric.measure(q3_low_relevancy_case)
low_relevancy_score = relevancy_metric.score
low_relevancy_reason = relevancy_metric.reason

# =============================================================================
# PART 2: LIVE RAG CASES
# =============================================================================
#
# Now we load the actual essay from the web, chunk it, embed it, and retrieve
# against it. The key differences from the warp drive script:
#
#   - WebBaseLoader instead of PyPDFLoader (HTML source, not PDF)
#   - bs4.SoupStrainer to preserve h2 section headers in the extracted text
#     so the retriever has section-level context in each chunk
#   - Same RecursiveCharacterTextSplitter and Chroma setup otherwise
#
# We run all three questions through the live RAG pipeline:
#   Q1 -> Contextual Recall
#   Q3 -> Contextual Relevancy
#   Q5 -> G-Eval (the nuanced pantheism/panentheism question)

ESSAY_URL = "https://testerstories.com/files/ai_testing/coherence-at-the-edge.html"

print("\n" + "=" * 60)
print("PART 2: LIVE RAG SETUP")
print("=" * 60)
print(f"Loading essay from: {ESSAY_URL}")

# Load the HTML, preserving h2 headers and paragraph text.
# SoupStrainer restricts BS4 to only parse these tags, which keeps section
# header text in the extracted content rather than stripping it away.
loader = WebBaseLoader(
  web_paths=[ESSAY_URL],
  bs_kwargs={"parse_only": SoupStrainer(["h2", "p", "li"])},
)
documents = loader.load()

print(f"Loaded {len(documents)} document(s).")
print(f"Total characters: {sum(len(d.page_content) for d in documents)}")

# Chunk the essay. The chunk size is deliberately smaller than the warp
# drive example (600 vs 1000) because the essay's paragraphs are short
# and the argument is tightly interwoven. Larger chunks would blur section
# boundaries and reduce the retriever's ability to distinguish between
# sections that share vocabulary.
text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=600,
  chunk_overlap=100,
)
chunks = text_splitter.split_documents(documents)

print(f"Split into {len(chunks)} chunks.")

# Embed and store
embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# -----------------------------------------------------------------------------
# Q1 live RAG — Contextual Recall
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("CONTEXTUAL RECALL — Live RAG Case (Q1)")
print("=" * 60)

retrieved_q1 = retriever.invoke(q1_question)

print("\nRetrieved chunks:")
for i, doc in enumerate(retrieved_q1, 1):
  print(f"\n--- Chunk {i} ---")
  print(doc.page_content[:300], "..." if len(doc.page_content) > 300 else "")

context_q1 = [doc.page_content for doc in retrieved_q1]

prompt_q1 = f"Based on this context:\n{context_q1}\n\nQuestion: {q1_question}"
response_q1 = execution_model.invoke(prompt_q1).content

print("\nGENERATED RESPONSE:")
print(response_q1)

rag_recall_case = LLMTestCase(
  input=q1_question,
  actual_output=response_q1,
  expected_output=q1_expected,
  retrieval_context=context_q1,
)

recall_metric.measure(rag_recall_case)
rag_recall_score = recall_metric.score
rag_recall_reason = recall_metric.reason

# -----------------------------------------------------------------------------
# Q3 live RAG — Contextual Relevancy
# -----------------------------------------------------------------------------

print("\n" + "=" * 60)
print("CONTEXTUAL RELEVANCY — Live RAG Case (Q3)")
print("=" * 60)

retrieved_q3 = retriever.invoke(q3_question)

print("\nRetrieved chunks:")
for i, doc in enumerate(retrieved_q3, 1):
  print(f"\n--- Chunk {i} ---")
  print(doc.page_content[:300], "..." if len(doc.page_content) > 300 else "")

context_q3 = [doc.page_content for doc in retrieved_q3]

prompt_q3 = f"Based on this context:\n{context_q3}\n\nQuestion: {q3_question}"
response_q3 = execution_model.invoke(prompt_q3).content

print("\nGENERATED RESPONSE:")
print(response_q3)

rag_relevancy_case = LLMTestCase(
  input=q3_question,
  actual_output=response_q3,
  expected_output=q3_expected,
  retrieval_context=context_q3,
)

relevancy_metric.measure(rag_relevancy_case)
rag_relevancy_score = relevancy_metric.score
rag_relevancy_reason = relevancy_metric.reason

# =============================================================================
# PART 3: G-EVAL
# =============================================================================
#
# Q5 is specifically designed for G-Eval because it requires tracking a
# deliberate nuance in the essay's argument -- one that a model summarizing
# too quickly will almost certainly flatten.
#
# The essay explicitly positions itself *between* pantheism (God = universe)
# and panentheism (universe is in God but God exceeds it). A response that
# collapses this into "yes, the essay argues the universe is God" would
# fail Criterion B (Argumentative Fidelity) even if it's not entirely wrong
# about one strand of the argument.
#
# This is the kind of failure mode that RAG-specific metrics cannot catch
# but G-Eval can, because we define what fidelity means for this document.

q5_question = """Does the essay "Coherence at the Edge" argue that the
universe is God, or does it take a more nuanced position? What exactly
is that position?"""

q5_expected = """The essay explicitly resists the simple identification of
God with the universe. It acknowledges the pull toward both pantheism
(God and universe are identical) and panentheism (the universe is in God
but God exceeds the universe), but positions its own proposal somewhere
between them. The Creator in the essay's model is not prior to the universe
in time, but neither is the Creator reducible to any single stage within
it. Rather, the Creator is described as the emergent totality -- perhaps
the final integrated consciousness of the whole cosmic history. This
preserves teleology without requiring temporal priority, and preserves
transcendence without requiring a static deity external to the process.
The essay is careful to frame this as speculative rather than argued."""

print("\n" + "=" * 60)
print("PART 3: G-EVAL")
print("=" * 60)

retrieved_q5 = retriever.invoke(q5_question)

print("\nRetrieved chunks:")
for i, doc in enumerate(retrieved_q5, 1):
  print(f"\n--- Chunk {i} ---")
  print(doc.page_content[:300], "..." if len(doc.page_content) > 300 else "")

context_q5 = [doc.page_content for doc in retrieved_q5]

prompt_q5 = f"Based on this context:\n{context_q5}\n\nQuestion: {q5_question}"
response_q5 = execution_model.invoke(prompt_q5).content

print("\nGENERATED RESPONSE:")
print(response_q5)

geval_case = LLMTestCase(
  input=q5_question,
  actual_output=response_q5,
  expected_output=q5_expected,
)

geval_metric.measure(geval_case)
geval_score = geval_metric.score
geval_reason = geval_metric.reason

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

print("\nControlled Cases:")
print(
  f"  Recall     — High Recall (Q1):      {high_recall_score:.2f} | {high_recall_reason}"
)
print(
  f"  Recall     — Low Recall (Q1):       {low_recall_score:.2f} | {low_recall_reason}"
)
print(
  f"  Relevancy  — High Relevancy (Q3):   {high_relevancy_score:.2f} | {high_relevancy_reason}"
)
print(
  f"  Relevancy  — Low Relevancy (Q3):    {low_relevancy_score:.2f} | {low_relevancy_reason}"
)

print("\nLive RAG Cases:")
print(
  f"  Recall     — Live RAG (Q1):         {rag_recall_score:.2f} | {rag_recall_reason}"
)
print(
  f"  Relevancy  — Live RAG (Q3):         {rag_relevancy_score:.2f} | {rag_relevancy_reason}"
)

print("\nG-Eval:")
print(f"  PhilosophicalEssayFidelity (Q5):    {geval_score:.2f} | {geval_reason}")

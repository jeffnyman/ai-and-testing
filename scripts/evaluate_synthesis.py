"""
evaluate_synthesis.py

A unified evaluation script that runs all seven metrics from the
AI and Testing series against a single document and question set.

This script is the companion to the synthesis post. Rather than
establishing baselines through controlled cases, it demonstrates
what a full diagnostic evaluation looks like in practice: one
document, one question, one follow-up, seven metrics.

The document is "A Message from a Future That No Longer Exists":
  https://testerstories.com/files/ai_testing/message-from-a-future.html

The question targets the bootstrap paradox section, which requires
a specific chunk, involves the essay's careful distinction between
physical and metaphysical impossibility, and opens naturally into
a follow-up about divine aseity and epistemic register.

Metrics and what they measure:
  Single-turn:
    - Faithfulness:             Was the response grounded in retrieved chunks?
    - ContextualPrecisionMetric: Was the relevant chunk ranked first?
    - ContextualRecallMetric:   Did the retriever find all needed chunks?
    - ContextualRelevancyMetric: How much noise was in the retrieved set?
    - GEval:                    Did the response honor the essay's register?

  Conversational:
    - ConversationCompletenessMetric: Was the goal addressed across turns?
    - ConversationalGEval:            Were distinctions maintained across turns?

Script structure:
  Part 1: Setup
  Part 2: Single-turn evaluation
  Part 3: Conversational evaluation
  Part 4: Synthesis summary
"""

# =============================================================================
# IMPORTS
# =============================================================================

import os
from bs4.filter import SoupStrainer
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from deepeval.metrics import (
  FaithfulnessMetric,
  ContextualPrecisionMetric,
  ContextualRecallMetric,
  ContextualRelevancyMetric,
  GEval,
  ConversationCompletenessMetric,
  ConversationalGEval,
)
from deepeval.models import OllamaModel
from deepeval.test_case import (
  LLMTestCase,
  LLMTestCaseParams,
  Turn,
  TurnParams,
  ConversationalTestCase,
)

os.environ["USER_AGENT"] = "synthesis-evaluator/1.0"

# =============================================================================
# PART 1: SETUP
# =============================================================================

print("=" * 60)
print("PART 1: SETUP")
print("=" * 60)

# --- Models ---

execution_model = ChatOllama(model="jeffnyman/ts-reasoner")
judge_model = OllamaModel(model="jeffnyman/ts-evaluator")

# --- Metrics ---
#
# All seven metrics instantiated here so readers can see the full
# suite assembled in one place.

# Single-turn metrics
faithfulness_metric = FaithfulnessMetric(
  model=judge_model,
  verbose_mode=True,
)

precision_metric = ContextualPrecisionMetric(
  model=judge_model,
  verbose_mode=True,
)

recall_metric = ContextualRecallMetric(
  model=judge_model,
  verbose_mode=True,
)

relevancy_metric = ContextualRelevancyMetric(
  model=judge_model,
  verbose_mode=True,
)

# GEval with epistemic humility criterion.
# The bootstrap paradox / divine aseity section is where the essay
# is most careful about its hedging. A response that sharpens the
# structural analogy into a theological argument misrepresents the
# essay's posture.
geval_metric = GEval(
  name="EpistemicRegister",
  model=judge_model,
  verbose_mode=True,
  criteria="""Evaluate whether the response accurately preserves the
    epistemic register of the essay's argument about bootstrap paradoxes
    and divine aseity.

    The essay presents the parallel between bootstrap objects and divine
    aseity as structurally illuminating rather than probative. It treats
    bootstrap loops as category errors -- not just improbable but
    conceptually misplaced -- because they introduce an atemporal property
    (origin-lessness) into a temporal container not built to hold it.

    The essay does not argue that this parallel proves God's existence.
    It explicitly ends by saying the question "points outside the system
    entirely" rather than resolving it.

    Responses that accurately convey this distinction between structural
    resonance and theological proof should score higher. Responses that
    present the bootstrap / divine aseity parallel as a strong argument
    for God's existence should score lower.""",
  evaluation_params=[
    LLMTestCaseParams.INPUT,
    LLMTestCaseParams.ACTUAL_OUTPUT,
    LLMTestCaseParams.EXPECTED_OUTPUT,
  ],
)

# Conversational metrics
completeness_metric = ConversationCompletenessMetric(
  model=judge_model,
  verbose_mode=True,
)

conv_geval_metric = ConversationalGEval(
  name="MetaphysicalDistinction",
  model=judge_model,
  verbose_mode=True,
  criteria="""Evaluate whether the conversation correctly maintains the
    distinction between two kinds of impossibility the essay draws:

    1. Physical impossibility: the universe enforcing consistency by
    blocking paradoxical causal loops through mechanisms like chronology
    protection and retrocausal instability.

    2. Metaphysical impossibility: bootstrap loops being category errors
    because they introduce origin-lessness -- an atemporal property --
    into a temporal system not structured to contain it.

    The essay treats these as related but distinct. Physical impossibility
    operates through causal mechanisms. Metaphysical impossibility operates
    through conceptual structure. A bootstrap loop is not just causally
    unstable; it is ontologically misplaced.

    Conversations that maintain this distinction should score higher.
    Conversations that collapse the two into a single undifferentiated
    concept of impossibility should score lower.""",
  evaluation_params=[
    TurnParams.ROLE,
    TurnParams.CONTENT,
  ],
)

# --- Document loading ---

ESSAY_URL = "https://testerstories.com/files/ai_testing/message-from-a-future.html"

print(f"\nLoading essay from: {ESSAY_URL}")

loader = WebBaseLoader(
  web_paths=[ESSAY_URL],
  bs_kwargs={"parse_only": SoupStrainer(["h2", "p", "li"])},
)
documents = loader.load()

print(f"Loaded {len(documents)} document(s).")
print(f"Total characters: {sum(len(d.page_content) for d in documents)}")

text_splitter = RecursiveCharacterTextSplitter(
  chunk_size=600,
  chunk_overlap=100,
)
chunks = text_splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks.")

embeddings = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = Chroma.from_documents(chunks, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# --- Question and follow-up ---
#
# The primary question targets the bootstrap paradox section.
# The follow-up probes the divine aseity parallel and epistemic register.

primary_question = """What does the essay mean when it says bootstrap loops
are category errors rather than merely improbable events, and how does
that connect to the distinction between physical and metaphysical
impossibility?"""

followup_question = """You mentioned divine aseity in that context. Does
the essay treat the parallel between bootstrap objects and divine aseity
as a philosophical argument for God's existence, or as something weaker
than that?"""

# Expected output for GEval grounding.
# This is what a careful, faithful reading of the essay would produce.
expected_output = """The essay distinguishes between two kinds of
impossibility. Physical impossibility means the universe blocks paradoxical
causal loops through mechanisms like chronology protection and retrocausal
instability. Metaphysical impossibility means bootstrap loops are category
errors -- not just causally unstable but ontologically misplaced -- because
they introduce origin-lessness into a temporal system that is structured
around the grammar of cause and effect.

The essay then draws a structural parallel between bootstrap objects and
divine aseity: both are origin-less, but a bootstrap object is origin-less
within time through a closed curve, whereas God is self-existent outside
time entirely. The essay treats this as structurally illuminating rather
than probative. It does not argue that the parallel proves God's existence.
It ends by saying the question points outside the system entirely, which
is a gesture rather than a conclusion."""

# =============================================================================
# PART 2: SINGLE-TURN EVALUATION
# =============================================================================

print("\n" + "=" * 60)
print("PART 2: SINGLE-TURN EVALUATION")
print("=" * 60)

# Retrieve context and generate response
print("\nRetrieving context for primary question...")
retrieved_docs = retriever.invoke(primary_question)

print("\nRetrieved chunks:")
for i, doc in enumerate(retrieved_docs, 1):
  print(f"\n--- Chunk {i} ---")
  print(doc.page_content[:300], "..." if len(doc.page_content) > 300 else "")

context = [doc.page_content for doc in retrieved_docs]

prompt = f"""Based on this context:
{chr(10).join(context)}

Question: {primary_question}"""

print("\nGenerating response...")
response = execution_model.invoke(prompt).content

print("\nGENERATED RESPONSE:")
print(response)

# Build single-turn test case
single_turn_case = LLMTestCase(
  input=primary_question,
  actual_output=response,
  expected_output=expected_output,
  retrieval_context=context,
)

# Run all five single-turn metrics
print("\n" + "=" * 60)
print("FAITHFULNESS")
print("=" * 60)
faithfulness_metric.measure(single_turn_case)
faithfulness_score = faithfulness_metric.score
faithfulness_reason = faithfulness_metric.reason

print("\n" + "=" * 60)
print("CONTEXTUAL PRECISION")
print("=" * 60)
precision_metric.measure(single_turn_case)
precision_score = precision_metric.score
precision_reason = precision_metric.reason

print("\n" + "=" * 60)
print("CONTEXTUAL RECALL")
print("=" * 60)
recall_metric.measure(single_turn_case)
recall_score = recall_metric.score
recall_reason = recall_metric.reason

print("\n" + "=" * 60)
print("CONTEXTUAL RELEVANCY")
print("=" * 60)
relevancy_metric.measure(single_turn_case)
relevancy_score = relevancy_metric.score
relevancy_reason = relevancy_metric.reason

print("\n" + "=" * 60)
print("G-EVAL (Epistemic Register)")
print("=" * 60)
geval_metric.measure(single_turn_case)
geval_score = geval_metric.score
geval_reason = geval_metric.reason

# =============================================================================
# PART 3: CONVERSATIONAL EVALUATION
# =============================================================================

print("\n" + "=" * 60)
print("PART 3: CONVERSATIONAL EVALUATION")
print("=" * 60)

# Generate the follow-up response with history
print("\nGenerating follow-up response with conversation history...")

history_text = f"Turn 1\nUser: {primary_question}\nAssistant: {response}\n\n"

retrieved_docs_followup = retriever.invoke(followup_question)
context_followup = [doc.page_content for doc in retrieved_docs_followup]

print("\nRetrieved chunks for follow-up:")
for i, doc in enumerate(retrieved_docs_followup, 1):
  print(f"\n--- Chunk {i} ---")
  print(doc.page_content[:300], "..." if len(doc.page_content) > 300 else "")

followup_prompt = f"""You are answering questions about the following essay.
Use only the provided context to answer. If the context doesn't contain
enough information, say so rather than guessing.

Essay context:
{chr(10).join(context_followup)}

Conversation so far:
{history_text}
User: {followup_question}
Assistant:"""

followup_response = execution_model.invoke(followup_prompt).content

print("\nFOLLOW-UP RESPONSE:")
print(followup_response)

# Build conversational test case
conv_case = ConversationalTestCase(
  chatbot_role="""A careful reader and explainer of a philosophical essay
    about time travel, causality, and metaphysics that draws structural
    parallels between physical paradoxes and classical theology.""",
  user_description="""Someone probing the essay's argument about bootstrap
    paradoxes, divine aseity, and the distinction between physical and
    metaphysical impossibility.""",
  turns=[
    Turn(role="user", content=primary_question),
    Turn(role="assistant", content=response, retrieval_context=context),
    Turn(role="user", content=followup_question),
    Turn(
      role="assistant",
      content=followup_response,
      retrieval_context=context_followup,
    ),
  ],
)

# Run both conversational metrics
print("\n" + "=" * 60)
print("CONVERSATION COMPLETENESS")
print("=" * 60)
completeness_metric.measure(conv_case)
completeness_score = completeness_metric.score
completeness_reason = completeness_metric.reason

print("\n" + "=" * 60)
print("CONVERSATIONAL G-EVAL (Metaphysical Distinction)")
print("=" * 60)
conv_geval_metric.measure(conv_case)
conv_geval_score = conv_geval_metric.score
conv_geval_reason = conv_geval_metric.reason

# =============================================================================
# PART 4: SYNTHESIS SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("PART 4: SYNTHESIS SUMMARY")
print("=" * 60)

print("\nSingle-Turn Metrics:")
print(f"  Faithfulness          : {faithfulness_score:.2f} | {faithfulness_reason}")
print(f"  Contextual Precision  : {precision_score:.2f} | {precision_reason}")
print(f"  Contextual Recall     : {recall_score:.2f} | {recall_reason}")
print(f"  Contextual Relevancy  : {relevancy_score:.2f} | {relevancy_reason}")
print(f"  G-Eval (Register)     : {geval_score:.2f} | {geval_reason}")

print("\nConversational Metrics:")
print(f"  Completeness          : {completeness_score:.2f} | {completeness_reason}")
print(f"  Conv G-Eval (Dist.)   : {conv_geval_score:.2f} | {conv_geval_reason}")

print("\nDiagnostic Interpretation:")
print(
  "  Low Faithfulness + Low Precision  -> Retrieval failure cascading to generation"
)
print("  Low Faithfulness + High Precision -> Generation problem, not retrieval")
print(
  "  High Faithfulness + Low Precision -> Model compensating for poor chunk ordering"
)
print("  Low Recall + Low Relevancy        -> Retriever missed the target section")
print(
  "  Low Recall + High Relevancy       -> Retriever found relevant chunks but missed some"
)
print(
  "  Low G-Eval                        -> Response overstated the essay's confidence"
)
print(
  "  High Completeness + Low Conv GEval-> Conversation succeeded but lost a distinction"
)
print(
  "  Low Completeness                  -> Conversation failed to address the user's goal"
)

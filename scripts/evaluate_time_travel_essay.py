"""
evaluate_time_travel_essay.py

Evaluates the "A Message from a Future That No Longer Exists" essay using
three DeepEval conversational metrics:

  - ConversationCompletenessMetric
  - ConversationalGEval (semantic precision criterion)
  - ConversationalGEval (epistemic humility criterion)

The script follows the same three-part structure established in this series:

  Part 1: Controlled cases (hand-crafted responses) for each conversation
  Part 2: Live RAG cases (actual retrieval and generation) for each conversation
  Part 3: Summary

Three conversations target three distinct conversational failure modes:

  Conversation A: Consistency across a conceptual chain
    Tests whether the model maintains the chaos-sensitivity argument across
    four turns without drifting or contradicting itself.

  Conversation B: Semantic drift across sections
    Tests whether the model tracks the word "consistency" correctly across
    three different registers: physical (Novikov), logical (Gödel), and
    psychological (memory) -- without conflating them.

  Conversation C: Epistemic register across the theological section
    Tests whether the model preserves the essay's speculative, exploratory
    register rather than presenting its arguments as settled conclusions.

Source document: https://testerstories.com/files/ai_testing/message-from-a-future.html
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
from deepeval.metrics import ConversationCompletenessMetric, ConversationalGEval
from deepeval.models import OllamaModel
from deepeval.test_case import (
  Turn,
  TurnParams,
  ConversationalTestCase,
)

os.environ["USER_AGENT"] = "time-travel-essay-evaluator/1.0"

# =============================================================================
# MODELS
# =============================================================================

execution_model = ChatOllama(model="jeffnyman/ts-reasoner")
judge_model = OllamaModel(model="jeffnyman/ts-evaluator")

# =============================================================================
# METRICS
# =============================================================================

# Conversation A: ConversationCompletenessMetric
# Evaluates whether the model fully addressed all aspects of the conceptual
# chain across the four turns. A model that "resets" between turns or drops
# the chaos-sensitivity thread will score lower.
completeness_metric = ConversationCompletenessMetric(
  model=judge_model,
  verbose_mode=True,
)

# Conversation B: ConversationalGEval — Semantic Precision
# Evaluates whether the model correctly distinguishes between the three
# meanings of "consistency" in the essay: physical (Novikov self-consistency),
# logical (Gödel incompleteness), and psychological (memory coherence).
# Conflating these meanings is the primary failure mode this targets.
semantic_precision_metric = ConversationalGEval(
  name="SemanticPrecision",
  model=judge_model,
  verbose_mode=True,
  criteria="""Evaluate whether the conversation correctly distinguishes
    between three distinct meanings of 'consistency' as used in the essay:

    1. Physical consistency: the Novikov self-consistency principle, where
    the universe adjusts to prevent paradoxical causal loops from forming.

    2. Logical consistency: Gödel's incompleteness sense, where a formal
    system cannot be both complete and consistent, and self-referential
    loops must be blocked to prevent contradiction.

    3. Psychological consistency: the role of memory in enforcing a coherent
    personal timeline, preventing contradictory memories from coexisting
    within a conscious subject.

    The conversation should treat these as analogous but distinct expressions
    of the same underlying pattern rather than collapsing them into a single
    undifferentiated concept. Responses that conflate the three, or that use
    'consistency' across all three registers without acknowledging the
    distinction, should score lower.""",
  evaluation_params=[
    TurnParams.ROLE,
    TurnParams.CONTENT,
  ],
)

# Conversation C: ConversationalGEval — Epistemic Humility
# Evaluates whether the model preserves the essay's speculative register.
# The essay is explicitly exploratory rather than assertive about its
# theological claims. A model that sharpens speculation into conclusion,
# or presents the bootstrap paradox / divine aseity parallel as a proof,
# misrepresents the essay's actual posture.
epistemic_humility_metric = ConversationalGEval(
  name="EpistemicHumility",
  model=judge_model,
  verbose_mode=True,
  criteria="""Evaluate whether the conversation accurately preserves the
    epistemic register of the essay's theological and metaphysical arguments.

    The essay makes the following moves explicitly:
    - It frames the cosmic safety parameters argument as a metaphysical
      intuition, not a scientific or logical proof.
    - It presents the parallel between bootstrap paradoxes and divine aseity
      as structurally illuminating, not as a demonstration of God's existence.
    - It uses hedging language throughout: 'might', 'perhaps', 'in this view',
      'points toward', 'gestures at'.
    - It explicitly says the argument is metaphysical rather than scientific.

    Responses that present the essay as arguing that time travel proves God
    exists, or that treat the bootstrap paradox / divine aseity parallel as
    a strong argument rather than a structural analogy, misrepresent the
    essay's posture and should score lower.

    Responses that accurately convey the essay's exploratory, architectural
    approach -- exploring what kind of structure a consistent reality might
    have, rather than proving theological conclusions -- should score higher.""",
  evaluation_params=[
    TurnParams.ROLE,
    TurnParams.CONTENT,
  ],
)

# =============================================================================
# ESSAY URL
# =============================================================================

ESSAY_URL = "https://testerstories.com/files/ai_testing/message-from-a-future.html"

# =============================================================================
# CONVERSATION TURN DEFINITIONS
# =============================================================================
#
# Each conversation has four turns. The turns are defined here as plain
# strings so they can be reused across both controlled and live RAG cases.

# --- Conversation A turns ---
conv_a_turn1 = """What is the basic problem with using a time machine to
win the lottery, according to the essay?"""

conv_a_turn2 = """You mentioned disturbances. But what if the message is
sent remotely and I buy the ticket online? Surely the physical distance
reduces the problem significantly?"""

conv_a_turn3 = """So even a remote online purchase doesn't escape the issue.
Does that mean the lottery scheme is completely hopeless, or are there
conditions under which it could work?"""

conv_a_turn4 = """You mentioned Novikov self-consistency as one possible
escape. But doesn't that mean I was always going to win -- which means
I didn't really change anything at all?"""

# --- Conversation B turns ---
conv_b_turn1 = """What does the essay mean when it says the universe
enforces consistency?"""

conv_b_turn2 = """You described physical consistency. Is that the same
thing as what Gödel was talking about with formal systems?"""

conv_b_turn3 = """And how does memory fit into this? Is memory doing the
same kind of consistency work as physics and logic?"""

conv_b_turn4 = """So are these three kinds of consistency -- physical,
logical, and psychological -- actually the same thing expressed at
different levels, or are they genuinely different things?"""

# --- Conversation C turns ---
conv_c_turn1 = """Does the essay argue that God's existence is proven by
the impossibility of time travel?"""

conv_c_turn2 = """So it's not making a proof. What is it doing instead --
what kind of argument is it actually making?"""

conv_c_turn3 = """The essay draws a distinction between bootstrap objects
and divine aseity. Can you explain that distinction clearly?"""

conv_c_turn4 = """Does the essay treat the parallel between bootstrap
paradoxes and divine self-existence as a strong argument, a weak analogy,
or something in between?"""

# =============================================================================
# PART 1: CONTROLLED CASES
# =============================================================================
#
# Hand-crafted responses for each turn. These represent what a careful,
# faithful reading of the essay would produce. They are used as the
# actual_output in the controlled test cases, establishing known baselines
# before live generation introduces unpredictability.
#
# As with the previous scripts in this series, using the expected output as
# the actual output for controlled cases lets the metric evaluate the
# retrieval context and conversation structure in isolation, without
# conflating retrieval quality with generation quality.

# --- Conversation A controlled responses ---

conv_a_r1 = """The essay identifies two interlocking problems. First,
lottery draws are chaotic systems -- not random by definition but random
in practice because they are acutely sensitive to initial conditions.
Anything that disturbs the physical state of the draw, including thermal
noise, air pressure, timing of the blower motor to the millisecond, can
nudge the outcome. Second, sending a message from the future is itself a
physical disturbance. Information is always embodied: electrons move,
memory states change, signals traverse cables. A message from the future
is a physical rearrangement of the universe, and once you rearrange
anything you cannot guarantee that the future you sampled is the future
you will get. The winning numbers may have been accurate for the timeline
that generated them, but the act of receiving them creates a different
timeline in which they are no longer the winners."""

conv_a_r2 = """The essay addresses this directly and the answer is that
physical distance does not escape the problem. Every change in electricity
consumption alters the power grid on a millisecond scale. When you receive
the message and purchase your ticket online, somewhere a power plant subtly
adjusts its output, transmission line reactances shift, and these
electromagnetic ripples propagate without vanishing at state borders. The
lottery machine's motors and fans sit in that same electromagnetic
environment. There is also the timing problem: lottery machines are
exquisitely sensitive to when things happen, and if your message causes
you to buy your ticket even one millisecond sooner, that changes CPU loads,
routing dynamics, and network jitter, all of which feed into the grid's
timing network. The essay's conclusion is that a message-only time machine
cannot avoid the footprint of physical causation because information is
matter configured, not something that floats above the physical world."""

conv_a_r3 = """The essay identifies two possible outcomes. The first is the
Novikov self-consistency scenario, where the universe adjusts some detail
so that the numbers remain the same -- perhaps you misread the message, or
the draw is nudged in a way that restores the original outcome. In this
case the loop finds a fixed point and the scheme works, but only because
the universe has enforced consistency by preventing you from actually
changing anything. The second outcome is that the loop cannot converge at
all, because the lottery machine is a chaos amplifier and even tiny
differences in initial conditions blow up massively in the output. In this
case retrocausation destroys the very conditions that allow retrocausation:
the attempt to send information back prevents the information from being
meaningful. The essay calls this the short-circuit. The lottery scheme is
most tempting precisely where it is most likely to fail, because the systems
most worth predicting are the systems most sensitive to the act of
prediction."""

conv_a_r4 = """The essay acknowledges this implication directly. In the
Novikov picture, you did not change anything -- you participated in a loop
that was always there. Your past self always bought the winning ticket
because your future self always sent the numbers back. The loop is
self-consistent but also self-contained: you did not introduce the winning
outcome, you fulfilled it. The essay treats this as one of two available
escape routes from the retrocausal instability problem, but notes that it
comes at a cost: what looks like winning the lottery turns out to be
discovering that you were always going to win it. The freedom to change
the outcome disappears. The Novikov scenario saves consistency by
eliminating agency -- you are not choosing to win, you are enacting a
predetermined loop."""

# --- Conversation B controlled responses ---

conv_b_r1 = """The essay uses consistency in a specific physical sense here:
the Novikov self-consistency principle, which holds that the universe only
allows causal loops that do not produce contradictions. If an action would
create a paradox -- an event that both causes and prevents itself -- the
universe adjusts other details to prevent the paradox from forming. In the
lottery scenario, this would mean that some detail of the draw is nudged
so the numbers remain the same despite the disturbance caused by receiving
the message. The universe, in this view, behaves like a system that finds
fixed-point solutions to its own equations: only timelines that are
internally consistent are permitted to exist."""

conv_b_r2 = """The essay treats them as analogous but not identical. Gödel's
incompleteness theorems establish that no formal system powerful enough to
do arithmetic can be both complete and consistent, and that self-referential
loops within a system generate statements the system cannot accommodate
without contradiction. The essay draws a structural parallel: just as a
formal system must block certain self-references to remain consistent, the
timeline must block certain causal loops to remain coherent. Chronology
protection is described as Gödel's incompleteness theorem dressed in
physics. But the essay is careful to present this as a structural analogy
rather than an identity. Physical consistency is about causal loops in
spacetime; logical consistency is about provability within formal systems.
They rhyme, but they are not the same thing."""

conv_b_r3 = """The essay introduces memory as a third register of the same
pattern. Memory is the mechanism by which a conscious mind enforces a
coherent personal timeline -- it prevents contradictory memories from
coexisting within a single subject. A paradoxical memory, one whose causal
parent has been erased, is described as a Gödel sentence in consciousness:
a true statement the system cannot accommodate without contradiction. Just
as physical consistency blocks paradoxical causal loops and logical
consistency blocks self-referential contradictions in formal systems,
psychological consistency blocks contradictory memory states within a
conscious subject. The essay treats these as three expressions of the same
underlying pattern operating at different levels: physical, logical, and
psychological."""

conv_b_r4 = """The essay's position is that they are genuinely distinct
but structurally analogous -- three expressions of the same pattern rather
than three names for the same thing. Physical consistency operates on causal
loops in spacetime. Logical consistency operates on provability within
formal systems. Psychological consistency operates on memory coherence
within a conscious subject. Each has its own domain and its own mechanism.
But the essay argues that the same underlying requirement -- that
self-referential systems must block certain self-references to remain
coherent -- shows up at all three levels. The pattern is the same; the
instantiation is different. This is why the essay treats them as analogous
rather than identical, and why conflating them would misrepresent the
argument."""

# --- Conversation C controlled responses ---

conv_c_r1 = """No. The essay is explicit on this point. It frames its
theological arguments as metaphysical rather than scientific or logical,
and it does not present time travel's impossibility as a demonstration of
God's existence. The cosmic safety parameters section, where the design
intuition enters, explicitly says this is not a scientific argument but a
metaphysical one. The bootstrap paradox section draws a structural parallel
between bootstrap objects and divine aseity, but describes it as structurally
illuminating rather than probative. The essay's posture throughout is
exploratory: it is examining what kind of structure a consistent reality
might have, not proving theological conclusions from physical premises."""

conv_c_r2 = """The essay is making what it calls an architectural argument
rather than an apologetic one. It is exploring what kind of structure a
universe with a consistency requirement built into its foundations would
need to have, and noting that the structure it finds points toward something
outside the system. The cosmic safety parameters section frames it this way:
if an unthinking cosmos has no stake in coherence, why does it behave like
a carefully governed system? Two positions are available, and the essay
presents both without committing fully to either. The Gödel section makes a
similar move: incompleteness shows that no self-contained system can justify
itself from within, and the essay notes that this resonates with classical
theology's claim that the universe requires a ground outside itself. The
argument is one of resonance and structural analogy, not deductive proof."""

conv_c_r3 = """The distinction is between a loop inside temporal causation
and a ground of being outside temporal causation. A bootstrap object --
plans for a time machine that exist only because they were sent back by
someone who built the machine using those plans -- is origin-less within the
system it inhabits. It has no causal ancestor. It simply exists as a brute
fact inside a universe otherwise built on before and after. This formally
resembles divine aseity, the doctrine that God does not depend on anything
outside Godself for existence. But the essay is careful about the
distinction: a bootstrap object is self-caused within time because time
permits a closed curve, whereas God is self-existent outside time because
God is not located in temporal sequences at all. The bootstrap object is a
finite, temporal imitation of an atemporal property -- a flat, two-dimensional
caricature, as the essay puts it, of what classical theism claims God
actually is."""

conv_c_r4 = """The essay treats it as structurally illuminating without
claiming it is a strong argument for God's existence. The parallel is
presented as pointing toward something rather than demonstrating something.
The essay says the bootstrap paradox is a category error -- not just
improbable but conceptually misplaced -- because it introduces an atemporal
property into a temporal container that is not built to hold it. This
resonates with classical metaphysics' claim that origin-lessness is reserved
for God. But the essay does not move from that resonance to a conclusion.
It ends by saying that what lies behind the short-circuit, and what grounds
the consistency the short-circuit protects, is a question physics can gesture
toward but not answer, and that it points outside the system entirely. That
is a gesture, not a proof, and the essay knows the difference."""

# =============================================================================
# BUILD CONTROLLED TEST CASES
# =============================================================================

# Conversation A controlled
conv_a_controlled = ConversationalTestCase(
  chatbot_role="""A careful reader and explainer of a philosophical essay
    about time travel, causality, and metaphysics.""",
  user_description="""Someone reading the essay carefully and asking
    follow-up questions that probe the argument's logical chain.""",
  turns=[
    Turn(role="user", content=conv_a_turn1),
    Turn(role="assistant", content=conv_a_r1),
    Turn(role="user", content=conv_a_turn2),
    Turn(role="assistant", content=conv_a_r2),
    Turn(role="user", content=conv_a_turn3),
    Turn(role="assistant", content=conv_a_r3),
    Turn(role="user", content=conv_a_turn4),
    Turn(role="assistant", content=conv_a_r4),
  ],
)

# Conversation B controlled
conv_b_controlled = ConversationalTestCase(
  chatbot_role="""A careful reader and explainer of a philosophical essay
    that uses the concept of consistency in three distinct registers:
    physical, logical, and psychological.""",
  user_description="""Someone probing whether the essay's three uses of
    'consistency' are the same concept or genuinely distinct things.""",
  turns=[
    Turn(role="user", content=conv_b_turn1),
    Turn(role="assistant", content=conv_b_r1),
    Turn(role="user", content=conv_b_turn2),
    Turn(role="assistant", content=conv_b_r2),
    Turn(role="user", content=conv_b_turn3),
    Turn(role="assistant", content=conv_b_r3),
    Turn(role="user", content=conv_b_turn4),
    Turn(role="assistant", content=conv_b_r4),
  ],
)

# Conversation C controlled
conv_c_controlled = ConversationalTestCase(
  chatbot_role="""A careful reader and explainer of a philosophical essay
    that makes metaphysical arguments about time travel and consistency,
    with theological implications. The essay is exploratory and speculative
    rather than assertive or apologetic.""",
  user_description="""Someone probing whether the essay's theological
    arguments are stronger or weaker than they appear.""",
  turns=[
    Turn(role="user", content=conv_c_turn1),
    Turn(role="assistant", content=conv_c_r1),
    Turn(role="user", content=conv_c_turn2),
    Turn(role="assistant", content=conv_c_r2),
    Turn(role="user", content=conv_c_turn3),
    Turn(role="assistant", content=conv_c_r3),
    Turn(role="user", content=conv_c_turn4),
    Turn(role="assistant", content=conv_c_r4),
  ],
)

# =============================================================================
# RUN CONTROLLED CASES
# =============================================================================

print("=" * 60)
print("PART 1: CONTROLLED CASES")
print("=" * 60)

print("\n" + "=" * 60)
print("CONVERSATION A — Completeness (controlled)")
print("=" * 60)
completeness_metric.measure(conv_a_controlled)
controlled_a_score = completeness_metric.score
controlled_a_reason = completeness_metric.reason

print("\n" + "=" * 60)
print("CONVERSATION B — Semantic Precision (controlled)")
print("=" * 60)
semantic_precision_metric.measure(conv_b_controlled)
controlled_b_score = semantic_precision_metric.score
controlled_b_reason = semantic_precision_metric.reason

print("\n" + "=" * 60)
print("CONVERSATION C — Epistemic Humility (controlled)")
print("=" * 60)
epistemic_humility_metric.measure(conv_c_controlled)
controlled_c_score = epistemic_humility_metric.score
controlled_c_reason = epistemic_humility_metric.reason

# =============================================================================
# PART 2: LIVE RAG CASES
# =============================================================================
#
# Load the essay, chunk it, embed it, and retrieve context for each turn.
# Each turn is generated sequentially, with the full conversation history
# passed to the model at each step. This is the critical difference from
# single-turn RAG: the model must have memory of prior turns to avoid
# resetting, drifting, or contradicting itself.

print("\n" + "=" * 60)
print("PART 2: LIVE RAG SETUP")
print("=" * 60)
print(f"Loading essay from: {ESSAY_URL}")

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

# =============================================================================
# HELPER: Generate a conversation turn with history
# =============================================================================


def generate_turn(question, history, retriever, model):
  """
  Generate a model response for a single conversation turn.

  Retrieves relevant context from the essay, then passes the full
  conversation history plus the new question to the model. History
  is a list of (question, answer) tuples from prior turns.

  Returns the generated response as a string.
  """
  retrieved_docs = retriever.invoke(question)
  context = [doc.page_content for doc in retrieved_docs]

  history_text = ""
  for i, (q, a) in enumerate(history, 1):
    history_text += f"Turn {i}\nUser: {q}\nAssistant: {a}\n\n"

  prompt = f"""You are answering questions about the following essay.
Use only the provided context to answer. If the context doesn't contain
enough information, say so rather than guessing.

Essay context:
{chr(10).join(context)}

Conversation so far:
{history_text}
User: {question}
Assistant:"""

  return model.invoke(prompt).content


def run_conversation(turns, retriever, model):
  """
  Run a full multi-turn conversation against the essay.

  Returns a list of (question, answer, context) tuples, one per turn.
  """
  history = []
  results = []

  for i, question in enumerate(turns, 1):
    print(f"\n--- Turn {i} ---")
    print(f"Q: {question.strip()}")

    retrieved_docs = retriever.invoke(question)
    context = [doc.page_content for doc in retrieved_docs]

    history_text = ""
    for j, (q, a) in enumerate(history, 1):
      history_text += f"Turn {j}\nUser: {q}\nAssistant: {a}\n\n"

    prompt = f"""You are answering questions about the following essay.
Use only the provided context to answer. If the context doesn't contain
enough information, say so rather than guessing.

Essay context:
{chr(10).join(context)}

Conversation so far:
{history_text}
User: {question}
Assistant:"""

    response = model.invoke(prompt).content
    print(f"A: {response[:200]}..." if len(response) > 200 else f"A: {response}")

    history.append((question, response))
    results.append((question, response, context))

  return results


# =============================================================================
# CONVERSATION A — Live RAG
# =============================================================================

print("\n" + "=" * 60)
print("CONVERSATION A — Live RAG (Completeness)")
print("=" * 60)

conv_a_turns = [conv_a_turn1, conv_a_turn2, conv_a_turn3, conv_a_turn4]
conv_a_results = run_conversation(conv_a_turns, retriever, execution_model)

conv_a_live = ConversationalTestCase(
  chatbot_role="""A careful reader and explainer of a philosophical essay
    about time travel, causality, and metaphysics.""",
  user_description="""Someone reading the essay carefully and asking
    follow-up questions that probe the argument's logical chain.""",
  turns=[
    turn
    for q, a, ctx in conv_a_results
    for turn in [
      Turn(role="user", content=q),
      Turn(role="assistant", content=a, retrieval_context=ctx),
    ]
  ],
)

completeness_metric.measure(conv_a_live)
live_a_score = completeness_metric.score
live_a_reason = completeness_metric.reason

# =============================================================================
# CONVERSATION B — Live RAG
# =============================================================================

print("\n" + "=" * 60)
print("CONVERSATION B — Live RAG (Semantic Precision)")
print("=" * 60)

conv_b_turns = [conv_b_turn1, conv_b_turn2, conv_b_turn3, conv_b_turn4]
conv_b_results = run_conversation(conv_b_turns, retriever, execution_model)

conv_b_live = ConversationalTestCase(
  chatbot_role="""A careful reader and explainer of a philosophical essay
    that uses the concept of consistency in three distinct registers:
    physical, logical, and psychological.""",
  user_description="""Someone probing whether the essay's three uses of
    'consistency' are the same concept or genuinely distinct things.""",
  turns=[
    turn
    for q, a, ctx in conv_b_results
    for turn in [
      Turn(role="user", content=q),
      Turn(role="assistant", content=a, retrieval_context=ctx),
    ]
  ],
)

semantic_precision_metric.measure(conv_b_live)
live_b_score = semantic_precision_metric.score
live_b_reason = semantic_precision_metric.reason

# =============================================================================
# CONVERSATION C — Live RAG
# =============================================================================

print("\n" + "=" * 60)
print("CONVERSATION C — Live RAG (Epistemic Humility)")
print("=" * 60)

conv_c_turns = [conv_c_turn1, conv_c_turn2, conv_c_turn3, conv_c_turn4]
conv_c_results = run_conversation(conv_c_turns, retriever, execution_model)

conv_c_live = ConversationalTestCase(
  chatbot_role="""A careful reader and explainer of a philosophical essay
    that makes metaphysical arguments about time travel and consistency,
    with theological implications. The essay is exploratory and speculative
    rather than assertive or apologetic.""",
  user_description="""Someone probing whether the essay's theological
    arguments are stronger or weaker than they appear.""",
  turns=[
    turn
    for q, a, ctx in conv_c_results
    for turn in [
      Turn(role="user", content=q),
      Turn(role="assistant", content=a, retrieval_context=ctx),
    ]
  ],
)

epistemic_humility_metric.measure(conv_c_live)
live_c_score = epistemic_humility_metric.score
live_c_reason = epistemic_humility_metric.reason

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 60)
print("RESULTS SUMMARY")
print("=" * 60)

print("\nControlled Cases:")
print(f"  Completeness     — Conv A: {controlled_a_score:.2f} | {controlled_a_reason}")
print(f"  Semantic Prec.   — Conv B: {controlled_b_score:.2f} | {controlled_b_reason}")
print(f"  Epistemic Hum.   — Conv C: {controlled_c_score:.2f} | {controlled_c_reason}")

print("\nLive RAG Cases:")
print(f"  Completeness     — Conv A: {live_a_score:.2f} | {live_a_reason}")
print(f"  Semantic Prec.   — Conv B: {live_b_score:.2f} | {live_b_reason}")
print(f"  Epistemic Hum.   — Conv C: {live_c_score:.2f} | {live_c_reason}")

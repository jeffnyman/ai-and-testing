# Reference:
# ???

from deepeval.metrics import ContextualPrecisionMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

judge_model = OllamaModel(model="jeffnyman/ts-evaluator")

metric = ContextualPrecisionMetric(model=judge_model, verbose_mode=True)

question = """Please consider Jeff Nyman's warp drive paper.
What energy source does the paper propose would be needed to
generate the warp bubble for faster-than-light travel?"""

expected = """Matter/antimatter annihilation, requiring approximately
10^28 kg of antimatter (equivalent to Jupiter's mass-energy)."""

actual = """According to Nyman's paper, the most efficient energy
production method would be matter/antimatter annihilation. The
paper calculates that generating a warp bubble would require
approximately 10^28 kg of antimatter, roughly equivalent to
Jupiter's mass-energy."""

high_precision_context = [
  """Jeff Nyman's paper proposes that an arbitrarily advanced
  civilization would utilize matter/antimatter annihilation as
  the most efficient energy production method. The paper determines
  this warp bubble would require around 10^28 Kg of antimatter to
  generate, roughly the mass-energy of the planet Jupiter.""",

  """The paper notes this energy requirement would drop dramatically
  if using a thin-shell of modified space-time instead of a bubble
  encompassing the volume of the craft.""",

  """Through calculations based on the cosmological constant and
  spacecraft volume, the energy requirements for faster-than-light
  travel are determined."""
]

low_precision_context = [
  """The paper discusses the Kaluza-Klein model as an early
  attempt to unify gravitation with electromagnetism. Theodor
  Kaluza initially postulated that a fifth spatial dimension
  could be introduced into Einstein's equations.""",

  """Numerous papers discussing the idea of warp drives have
  emerged in the literature in recent years. The basic idea is
  to formulate a solution to Einstein's equations whereby a warp
  bubble is driven by expansion and contraction of space-time.""",

  """Jeff Nyman's paper proposes that an arbitrarily advanced
  civilization would utilize matter/antimatter annihilation as
  the most efficient energy production method, requiring around
  10^28 Kg of antimatter."""
]

high_precision_case = LLMTestCase(
  input=question,
  actual_output=actual,
  expected_output=expected,
  retrieval_context=high_precision_context
)

low_precision_case = LLMTestCase(
  input=question,
  actual_output=actual,
  expected_output=expected,
  retrieval_context=low_precision_context
)

print("=" * 60)
print("HIGH PRECISION EXAMPLE (relevant chunks, well-ordered)")
print("=" * 60)
metric.measure(high_precision_case)

print("\n" + "=" * 60)
print("LOW PRECISION EXAMPLE (noise + relevant chunk buried)")
print("=" * 60)
metric.measure(low_precision_case)

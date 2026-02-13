# Reference:
# https://testerstories.com/2026/02/ai-and-testing-faithfulness/

from deepeval.metrics import FaithfulnessMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

judge_model = OllamaModel(model="jeffnyman/ts-evaluator")

metric = FaithfulnessMetric(model=judge_model, verbose_mode=True)

question = """Please consider Jeff Nyman's warp drive paper.
What energy source does the paper propose would be needed to
generate the warp bubble for faster-than-light travel?"""

research = [
  """Jeff Nyman's paper on warp drive propulsion calculates the
  energy requirements for faster-than-light travel. The paper
  proposes that an arbitrarily advanced civilization would utilize
  matter/antimatter annihilation as the most efficient energy
  production method. Through calculations based on the cosmological
  constant and spacecraft volume, the paper determines this warp
  bubble would require around 10^28 Kg of antimatter to generate,
  roughly the mass-energy of the planet Jupiter. The paper notes
  this energy requirement would drop dramatically if using a
  thin-shell of modified space-time instead of a bubble
  encompassing the volume of the craft."""
]

valid = LLMTestCase(
  input=question,
  actual_output=(
    """According to Nyman's paper, the most efficient energy
    production method would be matter/antimatter annihilation. The
    paper calculates that generating a warp bubble would require
    approximately 10^28 kg of antimatter, roughly equivalent to
    Jupiter's mass-energy. However, this requirement would decrease
    significantly if using a thin-shell configuration rather than
    a full volume bubble."""
  ),
  retrieval_context=research
)

invalid = LLMTestCase(
  input=question,
  actual_output=(
    """Nyman's paper proposes using zero-point energy extraction
    from the quantum vacuum to power the warp drive. The paper
    calculates this would require harnessing the Casimir effect
    across a surface area roughly equivalent to Earth's diameter,
    generating approximately 10^45 joules of energy almost
    instantaneously."""
  ),
  retrieval_context=research
)

metric.measure(valid)
print()
metric.measure(invalid)

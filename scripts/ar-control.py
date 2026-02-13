# Reference:
# https://testerstories.com/2026/02/ai-and-testing-answer-relevancy/

from deepeval.metrics import AnswerRelevancyMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase

judge_model = OllamaModel(model="jeffnyman/ts-evaluator")

metric = AnswerRelevancyMetric(model=judge_model)

question = "What does the Higgs boson explain in particle physics?"

good_case = LLMTestCase(
  input=question,
  actual_output=(
    "It explains how particles acquire mass via the Higgs field."
  )
)

rambly_case = LLMTestCase(
  input=question,
  actual_output=(
    "It explains how particles acquire mass via the Higgs field. "
    "Particle physics is a branch of physics. "
    "The Large Hadron Collider is in Europe."
  )
)

for label, case in [
  ("GOOD", good_case),
  ("RAMBLY", rambly_case)
]:
  metric.measure(case)

  print()
  print("*" * 60)
  print(f"{label} ANSWER")
  print("*" * 60)

  print("score: ", metric.score)
  print("pass:  ", metric.is_successful())
  print("reason:", metric.reason)

  print("-" * 40)
  print("Extracted statements:")
  print("-" * 40)

  for i, stmt in enumerate(metric.statements, start=1):
    print(f"{i}. {stmt}")

  print("-" * 40)
  print("Verdicts:")
  print("-" * 40)

  for verdict in metric.verdicts:
    print(verdict)

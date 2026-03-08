from deepeval.metrics import FaithfulnessMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from pathlib import Path
from extraction import run_variants


def compare_variants(passage: str, verbose: bool = False) -> list:
  print("\n  Running all prompt variants...")
  variant_results = run_variants(passage, verbose=verbose)

  scores = []
  for variant_name, extraction in variant_results.items():
    print(f"\n  Evaluating variant: {variant_name}")
    score = evaluate_extraction(
      passage=passage, extraction=extraction, variant_name=variant_name, verbose=verbose
    )
    scores.append(score)

  return scores


def print_comparison(scores: list) -> None:
  print("\n" + "=" * 60)
  print("  EXTRACTION FAITHFULNESS COMPARISON")
  print("=" * 60)

  for result in scores:
    print(f"\n  Variant: {result['variant']}")
    print(f"  Triples extracted : {result['triple_count']}")
    print(f"  Faithfulness score: {result['faithfulness_score']:.2f}")
    print(f"  Reason: {result['reason']}")

  print("\n" + "=" * 60)


def triple_to_claim(triple: dict) -> str:
  subject = triple["subject"]["label"]
  predicate = triple["predicate"]["label"]
  obj = triple["object"]["label"]
  confidence = triple.get("confidence", "unknown")
  source_claim = triple.get("source_claim", "unknown")
  return (
    f"{subject} {predicate} {obj} (confidence: {confidence}, source: {source_claim})"
  )


def evaluate_extraction(
  passage: str, extraction: dict, variant_name: str, verbose: bool = False
) -> dict:
  judge_model = OllamaModel(model="jeffnyman/ts-evaluator")
  metric = FaithfulnessMetric(model=judge_model, verbose_mode=verbose)

  claims = [triple_to_claim(t) for t in extraction.get("triples", [])]
  combined_claims = " | ".join(claims)

  test_case = LLMTestCase(
    input="Extract knowledge graph triples from this passage.",
    actual_output=combined_claims,
    retrieval_context=[passage],
  )

  metric.measure(test_case)

  return {
    "variant": variant_name,
    "triple_count": len(extraction.get("triples", [])),
    "faithfulness_score": metric.score,
    "reason": metric.reason,
  }


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser(
    description="Evaluate extraction faithfulness across prompt variants."
  )
  parser.add_argument(
    "--passage",
    type=Path,
    default=Path(__file__).parent / "passage.txt",
    help="Path to the passage text file",
  )
  parser.add_argument(
    "--verbose", action="store_true", help="Show full model output at each stage"
  )

  args = parser.parse_args()

  passage = args.passage.read_text(encoding="utf-8").strip()
  print(f"  Loaded passage: {args.passage.name} ({len(passage)} characters)")

  scores = compare_variants(passage, verbose=args.verbose)
  print_comparison(scores)

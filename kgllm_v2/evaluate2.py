from deepeval.metrics import FaithfulnessMetric
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from pathlib import Path
from extraction import run_variants
import json
# from pathlib import Path


def normalize(text: str) -> str:
  return text.lower().strip().replace(" ", "_").replace("-", "_")


def triple_matches_reference(extracted: dict, reference: dict) -> bool:
  return (
    normalize(extracted["subject"]["label"]) == normalize(reference["subject"])
    and normalize(extracted["predicate"]["label"]) == normalize(reference["predicate"])
    and normalize(extracted["object"]["label"]) == normalize(reference["object"])
  )


def score_against_reference(
  extraction: dict, reference_triples: list, variant_name: str
) -> dict:
  extracted = extraction.get("triples", [])

  matched_reference = set()
  matched_extracted = set()

  for i, ref in enumerate(reference_triples):
    for j, ext in enumerate(extracted):
      if triple_matches_reference(ext, ref):
        matched_reference.add(i)
        matched_extracted.add(j)

  recall = len(matched_reference) / len(reference_triples) if reference_triples else 0.0
  precision = len(matched_extracted) / len(extracted) if extracted else 0.0
  f1 = (
    2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
  )

  missed = [
    reference_triples[i]
    for i in range(len(reference_triples))
    if i not in matched_reference
  ]
  extra = [extracted[j] for j in range(len(extracted)) if j not in matched_extracted]

  return {
    "variant": variant_name,
    "extracted_count": len(extracted),
    "reference_count": len(reference_triples),
    "matched_count": len(matched_reference),
    "recall": recall,
    "precision": precision,
    "f1": f1,
    "missed_triples": missed,
    "extra_triples": extra,
  }


def load_reference(path: Path) -> list:
  with open(path, encoding="utf-8") as f:
    data = json.load(f)
  return data.get("triples", [])


def compare_variants(passage: str, reference_path: Path, verbose: bool = False) -> list:
  print("\n  Running all prompt variants...")
  variant_results = run_variants(passage, verbose=verbose)

  reference_triples = load_reference(reference_path)
  print(f"  Reference graph loaded: {len(reference_triples)} triples")

  scores = []
  for variant_name, extraction in variant_results.items():
    print(f"\n  Evaluating variant: {variant_name}")

    faithfulness = evaluate_extraction(
      passage=passage, extraction=extraction, variant_name=variant_name, verbose=verbose
    )

    reference_scores = score_against_reference(
      extraction=extraction,
      reference_triples=reference_triples,
      variant_name=variant_name,
    )

    scores.append({**faithfulness, **reference_scores})

  return scores


def print_comparison(scores: list) -> None:
  print("\n" + "=" * 60)
  print("  EXTRACTION EVALUATION SUMMARY")
  print("=" * 60)

  for result in scores:
    print(f"\n  Variant       : {result['variant']}")
    print(
      f"  Triples       : {result['extracted_count']} extracted, "
      f"{result['matched_count']} matched reference"
    )
    print(f"  Faithfulness  : {result['faithfulness_score']:.2f}")
    print(f"  Recall        : {result['recall']:.2f}")
    print(f"  Precision     : {result['precision']:.2f}")
    print(f"  F1            : {result['f1']:.2f}")

    if result["missed_triples"]:
      print(f"\n  Missed ({len(result['missed_triples'])}):")
      for t in result["missed_triples"]:
        print(f"    {t['subject']} --[{t['predicate']}]--> {t['object']}")

    if result["extra_triples"]:
      print(f"\n  Extra ({len(result['extra_triples'])}):")
      for t in result["extra_triples"]:
        print(
          f"    {t['subject']['label']} "
          f"--[{t['predicate']['label']}]--> "
          f"{t['object']['label']}"
        )

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
    "--reference",
    type=Path,
    default=Path(__file__).parent / "reference.json",
    help="Path to the reference triples JSON file",
  )
  parser.add_argument(
    "--verbose", action="store_true", help="Show full model output at each stage"
  )

  args = parser.parse_args()

  passage = args.passage.read_text(encoding="utf-8").strip()
  print(f"  Loaded passage : {args.passage.name} ({len(passage)} characters)")

  scores = compare_variants(
    passage=passage, reference_path=args.reference, verbose=args.verbose
  )
  print_comparison(scores)

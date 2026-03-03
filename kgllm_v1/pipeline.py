import argparse
import textwrap
import requests
import sys
from pathlib import Path
from config import MODEL_NAME, OLLAMA_ENDPOINT
from extraction import call_ollama
from graph import build_graph, graph_summary, save_graph
from queries import (
  format_results,
  query_argumentative_triples,
  query_entity_neighborhood,
  query_inferred_triples,
  query_multihop,
)

DEFAULT_PASSAGE = Path(__file__).parent / "passage.txt"

DEFAULT_QUESTION = (
  "Based on the knowledge graph, what is the relationship between "
  "Chronicles and the permission structure, and which later texts "
  "are connected to that structure through the graph?"
)

SEPARATOR = "=" * 60

GROUNDED_SYSTEM_PROMPT = """You are a research assistant that answers
questions strictly from structured query results extracted from a
knowledge graph. You do not draw on any knowledge beyond what appears
in the results provided.

Your answer must:
- Be grounded exclusively in the query results given
- Clearly indicate which query result supports each claim
- Explicitly state if the results do not contain enough information
  to answer the question fully
- Be written in clear, readable prose — not bullet points

If a query returned no results, say so and explain what that absence
might mean for the question being asked."""


def stage(label: str) -> None:
  print(f"\n{SEPARATOR}")
  print(f"  STAGE: {label}")
  print(SEPARATOR)


def load_passage(path: Path) -> str:
  if not path.exists():
    print(f"[Error] Passage file not found: {path}")
    sys.exit(1)

  text = path.read_text(encoding="utf-8").strip()
  print(f"  Loaded passage: {path.name} ({len(text)} characters)")

  return text


def build_grounded_prompt(question: str, query_results: dict) -> str:
  results_block = "\n\n".join(
    f"{name}:\n{result}" for name, result in query_results.items()
  )

  return textwrap.dedent(f"""
        QUESTION:
        {question}

        KNOWLEDGE GRAPH QUERY RESULTS:
        {results_block}

        Answer the question using only the query results above.
        Do not use any knowledge beyond what appears in these results.
        If the results are insufficient, say so explicitly.
    """).strip()


def call_grounded_answer(
  question: str, query_results: dict, verbose: bool = False
) -> str:
  payload = {
    "model": MODEL_NAME,
    "prompt": build_grounded_prompt(question, query_results),
    "system": GROUNDED_SYSTEM_PROMPT,
    "stream": False,
    # Deliberately no "format": "json" here —
    # we want prose, not structured output
  }

  response = requests.post(OLLAMA_ENDPOINT, json=payload)
  response.raise_for_status()

  answer = response.json().get("response", "").strip()

  if verbose:
    print("\n--- RAW GROUNDED ANSWER ---")
    print(answer)
    print("---------------------------\n")

  return answer


def run_pipeline(
  passage_path: Path,
  question: str,
  save_graph_path: Path = None,
  verbose: bool = False,
) -> None:
  # Stage 1: Extraction
  stage("1 of 4 — Triple Extraction")
  passage = load_passage(passage_path)

  print("\n  Sending passage to model for triple extraction...")
  print(f"  Model: {MODEL_NAME}")

  extraction = call_ollama(passage, verbose=verbose)

  print("\n  Reasoning trace (first 400 chars):")

  reasoning = extraction.get("reasoning", "none provided")

  print(textwrap.indent(textwrap.fill(reasoning[:400], width=56), "    "))

  if len(reasoning) > 400:
    print("    [... truncated ...]")

  triples = extraction.get("triples", [])

  print(f"\n  Triples extracted: {len(triples)}")

  if not triples:
    print(
      "\n  [Error] No valid triples extracted. "
      "Check your model and prompt configuration."
    )
    sys.exit(1)

  # Stage 2: Graph construction
  stage("2 of 4 — Graph Construction")
  print("\n  Building RDF graph from extracted triples...")

  g = build_graph(extraction)
  graph_summary(g)

  if save_graph_path:
    save_graph(g, str(save_graph_path))

  # Stage 3: SPARQL queries
  stage("3 of 4 — SPARQL Queries")
  print("\n  Running Query 1: Entity Neighborhood (Chronicles)...")

  q1 = query_entity_neighborhood(g, "Chronicles")
  q1_formatted = format_results(q1, "Q1 — Entity Neighborhood")
  print(f"\n{textwrap.indent(q1_formatted, '  ')}")

  print("\n  Running Query 2: Argumentative Triples...")
  q2 = query_argumentative_triples(g)
  q2_formatted = format_results(q2, "Q2 — Argumentative Triples")
  print(f"\n{textwrap.indent(q2_formatted, '  ')}")

  print("\n  Running Query 3: Multi-hop from Permission Structure...")
  q3 = query_multihop(g, "permission structure")
  q3_formatted = format_results(q3, "Q3 — Multi-hop Traversal")
  print(f"\n{textwrap.indent(q3_formatted, '  ')}")

  print("\n  Running Query 4: Inferred Triples...")
  q4 = query_inferred_triples(g)
  q4_formatted = format_results(q4, "Q4 — Inferred Triples")
  print(f"\n{textwrap.indent(q4_formatted, '  ')}")

  # Collect all formatted results for the grounded answer prompt
  query_results = {
    "Query 1 — Entity Neighborhood": q1_formatted,
    "Query 2 — Argumentative Triples": q2_formatted,
    "Query 3 — Multi-hop Traversal": q3_formatted,
    "Query 4 — Inferred Triples": q4_formatted,
  }

  # Stage 4: Grounded answer
  stage("4 of 4 — Grounded Answer")
  print(f"\n  Question:\n  {question}\n")
  print("  Sending query results to model for grounded answer...")

  answer = call_grounded_answer(question, query_results, verbose=verbose)

  print(f"\n{SEPARATOR}")
  print("  ANSWER (grounded in graph query results)")
  print(SEPARATOR)
  print(textwrap.fill(answer, width=60, initial_indent="  ", subsequent_indent="  "))
  print(f"\n{SEPARATOR}\n")

  # Summary
  print("  Pipeline complete.")
  print(f"  Triples extracted : {len(triples)}")
  print(f"  RDF statements    : {len(g)}")
  print(f"  Query 1 results   : {len(q1)}")
  print(f"  Query 2 results   : {len(q2)}")
  print(f"  Query 3 results   : {len(q3)}")
  print(f"  Query 4 results   : {len(q4)}")
  print()


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Knowledge graph pipeline: extract, build, query, answer.",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=textwrap.dedent("""
            Examples:
              python pipeline.py
              python pipeline.py --passage my_passage.txt
              python pipeline.py --question "What texts does Chronicles revise?"
              python pipeline.py --save-graph --verbose
        """),
  )

  parser.add_argument(
    "--passage",
    type=Path,
    default=DEFAULT_PASSAGE,
    help=f"Path to the passage text file (default: {DEFAULT_PASSAGE})",
  )

  parser.add_argument(
    "--question",
    type=str,
    default=DEFAULT_QUESTION,
    help="Natural language question for the grounded answer stage",
  )

  parser.add_argument(
    "--save-graph",
    action="store_true",
    help="Serialize the RDF graph to faithfulness_kg.ttl in Turtle format",
  )

  parser.add_argument(
    "--verbose", action="store_true", help="Show full model output at each stage"
  )

  return parser.parse_args()


if __name__ == "__main__":
  args = parse_args()

  graph_path = Path("faithfulness_kg.ttl") if args.save_graph else None

  run_pipeline(
    passage_path=args.passage,
    question=args.question,
    save_graph_path=graph_path,
    verbose=args.verbose,
  )

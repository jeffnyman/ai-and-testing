import json
import re
import requests
from config import (
  CONFIDENCE_LEVELS,
  ENTITY_TYPES,
  MODEL_NAME,
  OLLAMA_ENDPOINT,
  SOURCE_CLAIM_TYPES,
)

SYSTEM_PROMPT = f"""
You are a knowledge graph extraction assistant working with
scholarly biblical and historical texts. Your task is to identify
entities and relationships in the passage provided and output them
as structured JSON triples.

ENTITY TYPES you may assign:
{", ".join(ENTITY_TYPES)}

PREDICATE CATEGORIES you may assign:
- textual: direct relationships between texts (revises, cites,
  draws_on, omits, adds_to, contradicts, preserves)
- argumentative: claims about what texts or figures establish or
  authorize (establishes, authorizes, extends, refines, confirms,
  challenges)
- historical: relationships involving communities, periods, or
  events (written_for, composed_during, influenced, received_by,
  canonized_by)

CONFIDENCE LEVELS: {", ".join(CONFIDENCE_LEVELS)}
SOURCE CLAIM TYPES: {", ".join(SOURCE_CLAIM_TYPES)}
- stated: the relationship is explicitly present in the text
- inferred: the relationship requires interpretive reasoning

OUTPUT SCHEMA (strict JSON, no other text after the JSON block):
{{
  "reasoning": "your step-by-step analysis before extraction",
  "triples": [
    {{
      "subject": {{
        "id": "snake_case_identifier",
        "label": "Human Readable Label",
        "type": "EntityType"
      }},
      "predicate": {{
        "id": "snake_case_identifier",
        "label": "Human Readable Label",
        "category": "textual|argumentative|historical"
      }},
      "object": {{
        "id": "snake_case_identifier",
        "label": "Human Readable Label",
        "type": "EntityType"
      }},
      "confidence": "high|medium|low",
      "source_claim": "stated|inferred",
      "section": "section reference if identifiable"
    }}
  ]
}}

EXAMPLE TRIPLE:
{{
  "subject": {{
    "id": "chronicles",
    "label": "Chronicles",
    "type": "Text"
  }},
  "predicate": {{
    "id": "revises",
    "label": "revises",
    "category": "textual"
  }},
  "object": {{
    "id": "samuel_kings",
    "label": "Samuel-Kings",
    "type": "Text"
  }},
  "confidence": "high",
  "source_claim": "stated",
  "section": "II.A"
}}

Before outputting triples, reason through:
1. What are the key entities in this passage?
2. What are the most significant relationships between them?
3. Which relationships are explicitly stated versus inferred?

Output only valid JSON. No preamble or explanation outside the JSON.
"""


def build_user_prompt(passage: str) -> str:
  return f"""
  Analyze the following passage and extract knowledge graph triples
  according to the schema and instructions provided.

  PASSAGE:
  {passage}
  """


def validate_triples(triples: list) -> list:
  valid = []

  required_keys = {"subject", "predicate", "object", "confidence", "source_claim"}

  for i, triple in enumerate(triples):
    missing = required_keys - set(triple.keys())

    if missing:
      print(f"  [Warning] Triple {i} dropped — missing keys: {missing}")
      continue

    if triple["confidence"] not in CONFIDENCE_LEVELS:
      print(
        f"  [Warning] Triple {i} has invalid confidence value: "
        f"'{triple['confidence']}' — dropped."
      )
      continue

    if triple["source_claim"] not in SOURCE_CLAIM_TYPES:
      print(
        f"  [Warning] Triple {i} has invalid source_claim value: "
        f"'{triple['source_claim']}' — dropped."
      )
      continue

    valid.append(triple)

  print(
    f"  Extracted {len(valid)} valid triples ({len(triples) - len(valid)} dropped)."
  )

  return valid


def parse_extraction(raw: str) -> dict:
  # Strip markdown code fences if present
  cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip()
  cleaned = cleaned.rstrip("`").strip()

  try:
    parsed = json.loads(cleaned)
  except json.JSONDecodeError as e:
    raise ValueError(
      f"Model output could not be parsed as JSON.\n"
      f"Error: {e}\n"
      f"Raw output (first 300 chars): {raw[:300]}"
    )

  # Basic structural validation
  if "triples" not in parsed:
    raise ValueError(
      "Parsed JSON is missing required 'triples' key. "
      f"Keys found: {list(parsed.keys())}"
    )

  validated = validate_triples(parsed["triples"])
  parsed["triples"] = validated

  return parsed


def call_ollama(passage: str, verbose: bool = True) -> dict:
  payload = {
    "model": MODEL_NAME,
    "prompt": build_user_prompt(passage),
    "system": SYSTEM_PROMPT,
    "stream": False,
    "format": "json",
  }

  response = requests.post(OLLAMA_ENDPOINT, json=payload)
  response.raise_for_status()

  raw = response.json().get("response", "")

  if verbose:
    print("\n--- RAW MODEL OUTPUT ---")
    print(raw[:500], "..." if len(raw) > 500 else "")
    print("------------------------\n")

  return parse_extraction(raw)


if __name__ == "__main__":
  sample_passage = """
The two versions of the census account begin in strikingly different
ways. Second Samuel 24:1 opens with YHWH inciting David to take the
census. First Chronicles 21:1 begins differently: a satan stood up
against Israel and incited David to count the people. The Chronicler
grounds this revision in the Balaam narrative of Numbers 22, where
the angel of YHWH acts as an adversary. Stokes (2009) identifies the
structural parallels between these two narratives as uncanny.
"""

  result = call_ollama(sample_passage, verbose=True)

  print(f"Reasoning trace:\n{result.get('reasoning', 'none')}\n")
  print(f"Triples extracted: {len(result['triples'])}")

  for t in result["triples"]:
    print(
      f"  {t['subject']['label']} "
      f"--[{t['predicate']['label']}]--> "
      f"{t['object']['label']} "
      f"({t['confidence']}, {t['source_claim']})"
    )

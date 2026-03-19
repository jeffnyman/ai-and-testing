"""
generate.py

Sends the Z-Machine header parsing prompt to a local Ollama model,
injecting the three source documents (normative spec, conventional spec,
and ontology) into the prompt at runtime.

The model's response is printed to stdout and also saved to
zmachine_header.py in the same directory as this script.

Usage:
    python generate.py

Requirements:
    - Ollama running locally (default: http://localhost:11434)
    - The following files in the same directory as this script:
        extraction_input_sect11.txt
        extraction_input_appb.txt
        zmachine_header.ttl

Configuration:
    Edit the constants below to change the model or Ollama endpoint.
"""

import json
import re
import sys
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_ENDPOINT = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5-coder:14b"  # Change to match your local model name

SCRIPT_DIR = Path(__file__).parent

SECT11_FILE = SCRIPT_DIR / "extraction_input_sect11.txt"
APPB_FILE = SCRIPT_DIR / "extraction_input_appb.txt"
ONTOLOGY_FILE = SCRIPT_DIR / "zmachine_header.ttl"
OUTPUT_FILE = SCRIPT_DIR / "zmachine_header.py"

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are an expert Python programmer and Z-Machine specialist. You will be
given three sources that together describe the structure of the Z-Machine
header:

  1. NORMATIVE SPEC: Section 11 of the Z-Machine Standard 1.1. This is the
     authoritative definition of what a conforming interpreter must handle.

  2. CONVENTIONAL SPEC: Appendix B of the Z-Machine Standard 1.1. These
     fields are not normatively required but appear by convention in all
     real Infocom and Inform story files. A parser that ignores them will
     misread real files.

  3. ONTOLOGY: A formal OWL/RDF ontology (Turtle format) that reconciles
     both sources into a structured knowledge graph. It captures field
     offsets, byte sizes, field types, version applicability
     (applicableFrom / applicableUntil), access authority (mutatedBy /
     mustResetOnRestore), flags register structure (containsBit), pointer
     semantics (pointerTo), and supersession relationships (supersededBy)
     where the meaning at an offset changes across versions.

Your task is to write a single Python module, zmachine_header.py, that
implements a complete Z-Machine header parser. The parser must:

  - Accept a bytes object (the raw story file contents) as input.
  - Read the version byte at offset 0x00 FIRST, since almost every other
    field is version-conditional.
  - Parse all normative fields from Section 11 correctly, including:
      * All scalar fields with their correct byte offsets and sizes.
      * Flags 1 (offset 0x01): the bit meanings differ between Versions 1-3
        and Versions 4+. Branch on version and decode each set of bits
        separately, labeling them clearly.
      * Flags 2 (offset 0x10): a two-byte flags register with per-bit
        version applicability. Include all bits defined in the ontology.
      * The font dimension fields at offsets 0x26 and 0x27: in Version 5,
        0x26 is font width and 0x27 is font height; in Version 6, these
        meanings are SWAPPED. The ontology models this with supersededBy
        relationships. The code must branch correctly.
      * The header extension table (pointed to by offset 0x36): read it
        only from Version 5 onward, handle the case where the pointer is 0
        or the table is absent, read words by index with fallback to 0 for
        reads beyond the table length, and parse Flags 3 (word index 4) and
        the true default colours (word indices 5 and 6).
  - Also parse the conventional fields from Appendix B:
      * Release number (offset 0x02, 2 bytes)
      * Serial code (offset 0x12, 6 bytes, ASCII)
      * Tandy bit (Flags 1, bit 3, Version 3 only)
      * Flags 2 bit 10 (conventional printer error flag)
      * Player username (offset 0x38, 8 bytes, Version 6 only)
      * Inform compiler version (offset 0x3C, 4 bytes, ASCII)
  - Return a structured Python dictionary that groups fields logically:
    scalar fields, flags registers (with their decoded bits as sub-
    dictionaries), the extension table fields, and conventional fields.
  - Include a __main__ block that accepts a story file path as a command-
    line argument, parses the header, and pretty-prints the result.
  - Handle edge cases gracefully:
      * Story files shorter than 64 bytes should raise a clear ValueError.
      * Missing extension table (V5+ but pointer is 0) should not crash.
      * Reads beyond the extension table length should return 0.

Do not import any third-party libraries. Use only the Python standard
library.

Reconcile the three sources as follows: where the ontology and the spec
text agree, implement that behavior. Where they appear to differ, prefer
the normative spec text, but add a comment noting the discrepancy. Where
a field is marked hasSourceAuthority zm:conventional in the ontology,
implement it but mark it clearly in the output dictionary under a
"conventional" key.

CRITICAL OUTPUT RULE: Your entire response must be valid Python source
code and nothing else. Do not write any English prose, headers, summaries,
or explanation — before, during, or after the code. Do not use markdown.
Do not wrap the code in backticks or code fences. The very first character
of your response must be a # comment or the word "import" or '\"\"\"'.
If you include any English prose outside of Python comments, the output
will be unusable.
"""

USER_PROMPT_TEMPLATE = """\
Using the three sources below, generate the complete zmachine_header.py
module. Before writing code, work through the following:

  1. List all fields you will parse, grouped as: scalar fields, flags
     registers, extension table fields, conventional fields.
  2. Identify the cases where version-conditional branching is required
     not just version-gating (fields that appear from version N onward)
     but cases where the MEANING at an offset changes across versions.
  3. Then write the complete module.

---

NORMATIVE SPEC (Section 11):
{sect11}

---

CONVENTIONAL SPEC (Appendix B):
{appb}

---

ONTOLOGY (Turtle):
{ontology}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load(path: Path) -> str:
  try:
    return path.read_text(encoding="utf-8")
  except FileNotFoundError:
    print(f"[Error] Required file not found: {path}", file=sys.stderr)
    sys.exit(1)


def extract_code(raw: str) -> str:
  """
  Strip any markdown code fences the model may have added.
  Returns the raw Python source.
  Exits with an error if the output contains no Python code at all.
  """
  cleaned = re.sub(r"```(?:python)?\s*", "", raw).strip()
  cleaned = cleaned.rstrip("`").strip()

  # Sanity check: if there are no def/import/class statements,
  # the model probably returned prose instead of code.
  python_indicators = ("def ", "import ", "class ", "if __name__")
  if not any(indicator in cleaned for indicator in python_indicators):
    print(
      "\n[Error] The model did not return Python code.\n"
      "        The response appears to be prose or a summary.\n"
      "        First 300 chars of response:\n"
      f"        {cleaned[:300]}\n\n"
      "        Suggestions:\n"
      "        - Try a larger model (e.g. qwen2.5-coder:32b)\n"
      "        - Reduce num_ctx if the model is running out of memory\n"
      "        - Check Ollama logs for context length errors",
      file=sys.stderr,
    )
    sys.exit(1)

  return cleaned


def call_ollama(system: str, user: str) -> str:
  payload = {
    "model": MODEL_NAME,
    "stream": True,  # Stream so the user sees progress
    "options": {
      "num_ctx": 32768,  # Extend context window — the prompt is large
    },
    "messages": [
      {"role": "system", "content": system},
      {"role": "user", "content": user},
    ],
  }

  request = urllib.request.Request(
    OLLAMA_ENDPOINT,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
  )

  chunks = []

  print(f"[generate_parser] Sending prompt to {MODEL_NAME} via Ollama...")
  print("[generate_parser] Streaming response:\n")

  with urllib.request.urlopen(request) as response:
    for line in response:
      if not line.strip():
        continue
      try:
        obj = json.loads(line.decode("utf-8"))
      except json.JSONDecodeError:
        continue

      token = obj.get("message", {}).get("content", "")
      print(token, end="", flush=True)
      chunks.append(token)

      if obj.get("done", False):
        break

  print("\n")
  return "".join(chunks)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
  # Load source documents
  sect11 = load(SECT11_FILE)
  appb = load(APPB_FILE)
  ontology = load(ONTOLOGY_FILE)

  # Build user prompt with sources injected
  user_prompt = USER_PROMPT_TEMPLATE.format(
    sect11=sect11,
    appb=appb,
    ontology=ontology,
  )

  # Call the model
  raw_output = call_ollama(SYSTEM_PROMPT, user_prompt)

  # Strip any markdown fences
  code = extract_code(raw_output)

  # Save the generated parser
  OUTPUT_FILE.write_text(code, encoding="utf-8")
  print(f"[generate_parser] Parser saved to: {OUTPUT_FILE}")
  print(f"[generate_parser] Lines of code generated: {len(code.splitlines())}")


if __name__ == "__main__":
  main()

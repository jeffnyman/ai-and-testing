OLLAMA_ENDPOINT = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:latest"
KG_NAMESPACE = "http://example.org/faithfulness/"

ENTITY_TYPES = [
  "Person",
  "Text",
  "Community",
  "Concept",
  "Event",
  "Period",
]

PREDICATE_CATEGORIES = [
  "textual",
  "argumentative",
  "historical",
]

CONFIDENCE_LEVELS = ["high", "medium", "low"]

SOURCE_CLAIM_TYPES = ["stated", "inferred"]

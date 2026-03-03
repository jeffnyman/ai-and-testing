from rdflib import Graph, Literal, Namespace, RDF, RDFS, URIRef
from config import KG_NAMESPACE

KG = Namespace(KG_NAMESPACE)


def build_graph(extraction: dict) -> Graph:
  g = Graph()
  g.bind("kg", KG)
  g.bind("rdf", RDF)
  g.bind("rdfs", RDFS)

  triples = extraction.get("triples", [])

  for i, triple in enumerate(triples):
    _add_triple(g, triple, index=i)

  print(f"  Graph constructed: {len(g)} RDF statements.")

  return g


def _entity_uri(entity: dict) -> URIRef:
  return KG[entity["id"]]


def _add_entity(g: Graph, entity: dict) -> URIRef:
  uri = _entity_uri(entity)

  # Type assertion: kg:chronicles rdf:type kg:Text
  entity_type = KG[entity.get("type", "Unknown")]
  g.add((uri, RDF.type, entity_type))

  # Human-readable label
  g.add((uri, KG.label, Literal(entity["label"])))

  return uri


def _add_predicate(g: Graph, predicate: dict) -> URIRef:
  uri = KG[predicate["id"]]

  g.add((uri, KG.label, Literal(predicate["label"])))
  g.add((uri, KG.category, Literal(predicate.get("category", "unknown"))))

  return uri


def _add_triple(g: Graph, triple: dict, index: int) -> URIRef:
  triple_uri = KG[f"triple_{index:03d}"]

  # Mark this node as a reified statement
  g.add((triple_uri, RDF.type, RDF.Statement))

  # Add subject and object entities to graph and link to triple node
  subject_uri = _add_entity(g, triple["subject"])
  object_uri = _add_entity(g, triple["object"])
  predicate_uri = _add_predicate(g, triple["predicate"])

  # Core reified triple links
  g.add((triple_uri, KG.subject, subject_uri))
  g.add((triple_uri, KG.predicate, predicate_uri))
  g.add((triple_uri, KG.object, object_uri))

  # Metadata properties on the triple node
  g.add((triple_uri, KG.confidence, Literal(triple.get("confidence", "unknown"))))
  g.add((triple_uri, KG.sourceClaim, Literal(triple.get("source_claim", "unknown"))))

  if "section" in triple and triple["section"]:
    g.add((triple_uri, KG.section, Literal(triple["section"])))

  return triple_uri


def save_graph(g: Graph, path: str, format: str = "turtle") -> None:
  g.serialize(destination=path, format=format)
  print(f"  Graph saved to {path} ({format} format).")


def graph_summary(g: Graph) -> None:
  type_query = """
SELECT ?type (COUNT(?entity) AS ?count)
WHERE {
  ?entity rdf:type ?type .
  FILTER(?type != rdf:Statement)
}
GROUP BY ?type
ORDER BY DESC(?count)
"""

  print("\n--- GRAPH SUMMARY ---")
  print(f"Total RDF statements: {len(g)}\n")
  print("Entity types:")

  for row in g.query(type_query, initNs={"rdf": RDF}):
    type_label = str(row.type).replace(KG_NAMESPACE, "kg:")
    print(f"  {type_label}: {row.count}")

  print("---------------------\n")


if __name__ == "__main__":
  # Minimal synthetic extraction for testing graph construction
  # without needing a live Ollama instance
  synthetic_extraction = {
    "triples": [
      {
        "subject": {"id": "chronicles", "label": "Chronicles", "type": "Text"},
        "predicate": {"id": "revises", "label": "revises", "category": "textual"},
        "object": {"id": "samuel_kings", "label": "Samuel-Kings", "type": "Text"},
        "confidence": "high",
        "source_claim": "stated",
        "section": "II.A",
      },
      {
        "subject": {
          "id": "permission_structure",
          "label": "permission structure",
          "type": "Concept",
        },
        "predicate": {
          "id": "established_by",
          "label": "established by",
          "category": "argumentative",
        },
        "object": {"id": "chronicles", "label": "Chronicles", "type": "Text"},
        "confidence": "high",
        "source_claim": "stated",
        "section": "III.A",
      },
      {
        "subject": {"id": "jubilees", "label": "Jubilees", "type": "Text"},
        "predicate": {"id": "extends", "label": "extends", "category": "argumentative"},
        "object": {
          "id": "permission_structure",
          "label": "permission structure",
          "type": "Concept",
        },
        "confidence": "medium",
        "source_claim": "inferred",
        "section": "IV.C",
      },
    ]
  }

  g = build_graph(synthetic_extraction)
  graph_summary(g)
  save_graph(g, "./faithfulness_kg.ttl")

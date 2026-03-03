from rdflib import Graph, Namespace, RDF
from config import KG_NAMESPACE

KG = Namespace(KG_NAMESPACE)

INIT_NS = {
  "kg": KG,
  "rdf": RDF,
}

# Query 1 (Entity Neighborhood)

QUERY_1 = """
SELECT ?targetLabel ?predicateLabel ?predicateCategory
WHERE {
  ?subject kg:label "Chronicles" .

  ?triple kg:subject   ?subject ;
          kg:predicate ?predicate ;
          kg:object    ?target .

  ?predicate kg:label    ?predicateLabel ;
             kg:category ?predicateCategory .
  
  ?target kg:label ?targetLabel .
}
ORDER BY ?predicateCategory ?targetLabel
"""


def query_entity_neighborhood(g: Graph, entity_label: str = "Chronicles") -> list:
  # Parameterize the entity label so this query is reusable
  query = QUERY_1.replace('"Chronicles"', f'"{entity_label}"')

  results = []

  for row in g.query(query, initNs=INIT_NS):
    results.append(
      {
        "target": str(row.targetLabel),
        "predicate": str(row.predicateLabel),
        "category": str(row.predicateCategory),
      }
    )

  return results


# Query 2 (Predicate Filter)

QUERY_2 = """
SELECT ?subjectLabel ?predicateLabel ?objectLabel ?confidence
WHERE {
  ?triple kg:subject   ?subject ;
          kg:predicate ?predicate ;
          kg:object    ?object ;
          kg:confidence ?confidence .
  
  ?predicate kg:category "argumentative" ;
             kg:label    ?predicateLabel .
  
  ?subject kg:label ?subjectLabel .
  ?object  kg:label ?objectLabel .
}
ORDER BY ?confidence ?subjectLabel
"""


def query_argumentative_triples(g: Graph) -> list:
  results = []

  for row in g.query(QUERY_2, initNs=INIT_NS):
    results.append(
      {
        "subject": str(row.subjectLabel),
        "predicate": str(row.predicateLabel),
        "object": str(row.objectLabel),
        "confidence": str(row.confidence),
      }
    )

  return results


# Query 3 (Multi-hop Traversal)

QUERY_3 = """
SELECT ?hopOneLabel ?hopOnePredicateLabel
       ?hopTwoLabel ?hopTwoPredicateLabel
       ?terminalLabel
WHERE {
  ?concept kg:label "permission structure" .

  ?tripleOne kg:subject   ?concept ;
             kg:predicate ?hopOnePredicate ;
             kg:object    ?hopOne .

  ?hopOnePredicate kg:label ?hopOnePredicateLabel .
  ?hopOne kg:label ?hopOneLabel .

  ?tripleTwo kg:subject   ?hopOne ;
             kg:predicate ?hopTwoPredicate ;
             kg:object    ?terminal .

  ?hopTwoPredicate kg:label ?hopTwoPredicateLabel .
  ?terminal kg:label ?terminalLabel .

  FILTER(?terminal != ?concept)
}
ORDER BY ?hopOneLabel ?terminalLabel
"""


def query_multihop(g: Graph, concept_label: str = "permission structure") -> list:
  query = QUERY_3.replace('"permission structure"', f'"{concept_label}"')

  results = []

  for row in g.query(query, initNs=INIT_NS):
    results.append(
      {
        "hop_one": str(row.hopOneLabel),
        "hop_one_predicate": str(row.hopOnePredicateLabel),
        "hop_two": str(row.hopTwoLabel),
        "hop_two_predicate": str(row.hopTwoPredicateLabel),
        "terminal": str(row.terminalLabel),
      }
    )

  return results


# Query 4 (Confidence and Source Diagnostic)
QUERY_4 = """
SELECT ?subjectLabel ?predicateLabel ?objectLabel
       ?sourceClaim ?section

WHERE {
  ?triple kg:subject    ?subject ;
          kg:predicate  ?predicate ;
          kg:object     ?object ;
          kg:sourceClaim "inferred" .

  ?subject  kg:label ?subjectLabel .
  ?predicate kg:label ?predicateLabel .
  ?object   kg:label ?objectLabel .

  OPTIONAL { ?triple kg:section ?section . }
}
ORDER BY ?section ?subjectLabel
"""


def query_inferred_triples(g: Graph) -> list:
  results = []

  for row in g.query(QUERY_4, initNs=INIT_NS):
    results.append(
      {
        "subject": str(row.subjectLabel),
        "predicate": str(row.predicateLabel),
        "object": str(row.objectLabel),
        "section": str(row.section) if row.section else "unknown",
      }
    )

  return results


def format_results(results: list, query_name: str) -> str:
  if not results:
    return f"[{query_name}]: No results found."

  lines = [f"[{query_name}] — {len(results)} result(s):"]

  for i, row in enumerate(results, 1):
    line = "  " + " | ".join(f"{k}: {v}" for k, v in row.items())
    lines.append(f"  {i}. {line}")

  return "\n".join(lines)


def print_results(results: list, query_name: str) -> None:
  print(format_results(results, query_name))


if __name__ == "__main__":
  from graph import build_graph, graph_summary

  # Use the same synthetic extraction as graph.py for isolated
  # testing
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

  print("\n=== Query 1: Entity Neighborhood (Chronicles) ===")
  print_results(query_entity_neighborhood(g, "Chronicles"), "Q1")

  print("\n=== Query 2: Argumentative Triples ===")
  print_results(query_argumentative_triples(g), "Q2")

  print("\n=== Query 3: Multi-hop from Permission Structure ===")
  print_results(query_multihop(g, "permission structure"), "Q3")

  print("\n=== Query 4: Inferred Triples ===")
  print_results(query_inferred_triples(g), "Q4")

"""
validate_ontology.py

Validation script for the Z-Machine header ontology.
Runs a sequence of checks in increasing specificity:

  1. Basic parse and triple count
  2. Individual counts by class
  3. Structural relationships (supersededBy, pointerTo)
  4. Access authority coverage (mutatedBy, mustResetOnRestore)
  5. Version applicability spot-checks
  6. Bit position diagnostics (flags for duplicate or misversioned bits)

Each section is independently useful. If you add or change individuals
in the ontology, re-run this script to confirm the expected counts and
relationships are still intact.
"""

from collections import Counter, defaultdict
from pathlib import Path

from rdflib import Graph, URIRef
from rdflib.namespace import RDF, RDFS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

script_dir = Path(__file__).parent
TTL_FILE = script_dir / "zmachine_header.ttl"

ZM = "http://example.org/zmachine/"
OWL_NS = "http://www.w3.org/2002/07/owl#"
RDFS_NS = "http://www.w3.org/2000/01/rdf-schema#"


def zm(local):
  return URIRef(ZM + local)


def short(uri):
  return (
    str(uri)
    .replace(ZM, "zm:")
    .replace(OWL_NS, "owl:")
    .replace(RDFS_NS, "rdfs:")
    .replace("http://www.w3.org/1999/02/22-rdf-syntax-ns#", "rdf:")
  )


def section(title):
  print()
  print("=" * 60)
  print(f"  {title}")
  print("=" * 60)


def subsection(title):
  print()
  print(f"--- {title} ---")


# ---------------------------------------------------------------------------
# Load graph
# ---------------------------------------------------------------------------

section("1. Parse")

g = Graph()
g.parse(str(TTL_FILE), format="turtle")

print(f"File   : {TTL_FILE.name}")
print(f"Triples: {len(g)}")

# ---------------------------------------------------------------------------
# Individual counts by class
# ---------------------------------------------------------------------------

section("2. Individual counts by class")

class_counts = Counter()

for s, p, o in g.triples((None, RDF.type, None)):
  class_counts[short(o)] += 1

for cls, count in sorted(class_counts.items()):
  print(f"  {cls}: {count}")

# ---------------------------------------------------------------------------
# Structural relationships
# ---------------------------------------------------------------------------

section("3. Structural relationships")

subsection("supersededBy")

found = False

for s, p, o in g.triples((None, zm("supersededBy"), None)):
  print(f"  {short(s)}")
  print(f"    -> {short(o)}")

  found = True

if not found:
  print("  (none found — check ontology if supersededBy links are expected)")

subsection("pointerTo")

found = False

for s, p, o in g.triples((None, zm("pointerTo"), None)):
  print(f"  {short(s)}")
  print(f"    -> {short(o)}")

  found = True

if not found:
  print("  (none found — check ontology if pointerTo links are expected)")

subsection("containsBit (FlagsRegister -> BitField)")

for reg in sorted(g.subjects(RDF.type, zm("FlagsRegister")), key=str):
  bits = list(g.objects(reg, zm("containsBit")))
  reg_label = next(g.objects(reg, RDFS.label), short(reg))

  print(f"  {reg_label}: {len(bits)} bit(s)")

# ---------------------------------------------------------------------------
# Access authority coverage
# ---------------------------------------------------------------------------

section("4. Access authority coverage")

subsection("Fields and bits with mutatedBy set")

mutable = []

for s, p, o in g.triples((None, zm("mutatedBy"), None)):
  label = next(g.objects(s, RDFS.label), short(s))
  authority = next(g.objects(o, RDFS.label), short(o))
  mutable.append((str(label), str(authority)))

for label, authority in sorted(mutable):
  print(f"  {label}: {authority}")

subsection("Fields and bits with mustResetOnRestore = true")

reset_count = 0

for s, p, o in g.triples((None, zm("mustResetOnRestore"), None)):
  if str(o).lower() == "true":
    reset_count += 1

print(f"  {reset_count} individual(s) marked mustResetOnRestore")

subsection("Fields with no mutatedBy (read-only candidates)")

all_fields = list(g.subjects(RDF.type, zm("HeaderField")))
all_bits = list(g.subjects(RDF.type, zm("BitField")))
all_ext = list(g.subjects(RDF.type, zm("ExtensionField")))

readonly_count = 0

for node in all_fields + all_bits + all_ext:
  has_mutated_by = any(True for _ in g.objects(node, zm("mutatedBy")))

  if not has_mutated_by:
    readonly_count += 1

print(f"  {readonly_count} individual(s) have no mutatedBy (read-only)")

# ---------------------------------------------------------------------------
# Version applicability spot-checks
# ---------------------------------------------------------------------------

section("5. Version applicability spot-checks")

subsection("Fields applicable from each version")

version_counts = Counter()

for s, p, o in g.triples((None, zm("applicableFrom"), None)):
  version_counts[short(o)] += 1

for ver, count in sorted(version_counts.items()):
  print(f"  {ver}: {count} individual(s)")

subsection("Fields with applicableUntil (bounded version range)")

bounded = []

for s, p, o in g.triples((None, zm("applicableUntil"), None)):
  label = next(g.objects(s, RDFS.label), short(s))
  af = next(g.objects(s, zm("applicableFrom")), None)
  au = o
  bounded.append((str(label), short(af) if af else "?", short(au)))

if bounded:
  for label, frm, until in sorted(bounded):
    print(f"  {label}")
    print(f"    applicable: {frm} to {until}")
else:
  print("  (none — all fields are open-ended or version-unbounded)")

# ---------------------------------------------------------------------------
# Bit position diagnostics
# ---------------------------------------------------------------------------

section("6. Bit position diagnostics")

subsection("All BitField individuals at bit position 0")

for s in sorted(g.subjects(RDF.type, zm("BitField")), key=str):
  for pos in g.objects(s, zm("bitPosition")):
    if int(pos) == 0:
      label = next(g.objects(s, RDFS.label), short(s))
      af = [short(x) for x in g.objects(s, zm("applicableFrom"))]
      au = [short(x) for x in g.objects(s, zm("applicableUntil"))]
      reg = next(g.objects(s, zm("withinRegister")), None)
      reg_label = next(g.objects(reg, RDFS.label), short(reg)) if reg else "?"

      print(f"  {label}")
      print(f"    register: {reg_label}")
      print(f"    applicableFrom: {af}  applicableUntil: {au if au else '(open)'}")

subsection("Duplicate bit positions within a single register (potential conflicts)")

conflicts_found = False

for reg in g.subjects(RDF.type, zm("FlagsRegister")):
  reg_label = next(g.objects(reg, RDFS.label), short(reg))
  position_map = defaultdict(list)

  for bit in g.objects(reg, zm("containsBit")):
    for pos in g.objects(bit, zm("bitPosition")):
      bit_label = next(g.objects(bit, RDFS.label), short(bit))
      af = next(g.objects(bit, zm("applicableFrom")), None)
      au = next(g.objects(bit, zm("applicableUntil")), None)

      position_map[int(pos)].append(
        (str(bit_label), short(af) if af else "?", short(au) if au else None)
      )

  for pos, entries in sorted(position_map.items()):
    if len(entries) > 1:
      conflicts_found = True

      print(f"  {reg_label}, bit {pos}: {len(entries)} individuals")

      for label, frm, until in entries:
        range_str = f"v{frm[-1]}" if frm != "?" else "?"

        if until:
          range_str += f" to v{until[-1]}"

        print(f"    [{range_str}] {label}")

if not conflicts_found:
  print("  No duplicate bit positions found.")

subsection("sourceAuthority distribution")

auth_counts = Counter()

for s, p, o in g.triples((None, zm("hasSourceAuthority"), None)):
  auth_counts[short(o)] += 1

for auth, count in sorted(auth_counts.items()):
  print(f"  {auth}: {count} individual(s)")

print()
print("=" * 60)
print("  Validation complete.")
print("=" * 60)

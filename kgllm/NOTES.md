# config.py
This is the central configuration for the knowledge graph pipeline. You should only need to modify this file to adapt the logic to a different model or domain.

The `KG_NAMESPACE` isn't a real URL that anything tries to connect to. It's purely a naming convention, a way of making sure every term in your graph has a unique, unambiguous identifier. RDF needs this but why?

RDF was designed to be a web-scale knowledge format where graphs from different sources could be merged without naming collisions. If your graph says chronicles and someone else's graph says chronicles, those could mean completely different things. Namespaces solve this by making every term globally unique: http://example.org/faithfulness/chronicles can't collide with anyone else's chronicles because the prefix is yours.

Nothing in your pipeline makes an HTTP request to `http://example.org/faithfulness/`. RDFLib just uses it as a string prefix when constructing identifiers internally.

The only constraint worth mentioning is consistency: whatever namespace you choose, it needs to match across `config.py`, the SPARQL queries, and the RDFLib graph construction. Since all three pull from `config.py` via the KG_NAMESPACE constant, changing it is a one-line edit.

A reasonable recommendation is to use a domain you control if you're building something real, use http://example.org/ if you're experimenting. The example.org domain is actually reserved by IANA specifically for illustrative use in documentation, so http://example.org/faithfulness/ is exactly the right choice for a demo context. It signals intent clearly without implying a real endpoint exists.

The `ENTITY_TYPES` are valid entity types that the extraction prompt will recognize.

The valid predicate categories (`PREDICATE_CATEGORIES`) provide three value: textual, for direct relationships between texts; argumentative, for claims about what texts establish or authorize; historical, for relationships involving communities, periods, events.

# graph.py
Constructs an RDFLib in-memory graph from extracted triples.

There is an RDF reification note: Standard RDF triples are three-part statements: subject, predicate, object. They cannot carry metadata, so you cannot attach a confidence level or a section reference to a triple directly. Reification solves this by treating each triple as a node in its own right. Instead of:

```
Chronicles --revises--> Samuel-Kings
```

We create a triple node and attach properties to it:

```
:triple_001 a rdf:Statement
:triple_001 kg:subject    :chronicles
:triple_001 kg:predicate  :revises
:triple_001 kg:object     :samuel_kings
:triple_001 kg:confidence "high"
:triple_001 kg:section    "II.A"
```

This is more verbose but it's what makes Queries 2 and 4 possible; querying by confidence level or source claim requires metadata on triples, not just on entities.

`graph.py: build_graph`
Constructs an RDFLib graph from the parsed extraction output. The extraction parameter is The dict returned by extraction.parse_extraction(), containing a 'triples' list. This function returns a populated rdflib.Graph instance ready for SPARQL queries.

`graph.py: _entity_uri`
Constructs a stable URI for an entity from its id field. snake_case ids become URIs like kg:chronicles.

`graph.py: _add_entity`
Adds an entity node to the graph with its type and label. This is safe to call multiple times for the same entity because RDFLib deduplicates statements automatically. The parameters are g (the graph to add to) and entity (a dict with 'id', 'label', and 'type' keys). This function returns the URI of the entity node.

`graph.py: _add_triple`
Adds a reified triple to the graph. The triple itself becomes a named node (kg:triple_000, etc.) with its subject, predicate, object, and metadata attached as properties of that node. The parameters are g (the graph to add to), triple (a validated triple dict from the extraction output), and index (integer index used to generate a unique triple URI). This function returns the URI of the triple node.

`graph.py: _add_predicate`
Adds a predicate node to the graph with its label and category. The parameters are g (the graph to add to) and predicate (dict with 'id', 'label', and 'category' keys). This function returns the URI of the predicate node.

`graph.py: save_graph`
Serializes the graph to a file. Turtle format is the most human-readable RDF serialization, useful for inspecting the raw graph output. The parameters are g (the graph to serialize), path (output file path), and format (RDF serialization format. Defaults to 'turtle').

`graph.py: graph_summary`
Prints a brief summary of the graph contents. Useful as a sanity check before running SPARQL queries.

# Test graph construction without needing Ollama
python kgllm/graph.py

# queries.py
The four SPARQL queries that demonstrate what the knowledge graph can answer. Each query is a standalone function returning results as a list of dicts. This is clean enough to pass directly into the grounded answer prompt in pipeline.py.

Query design notes:
- Q1 — Entity neighborhood: what is Chronicles connected to?
- Q2 — Predicate filter: which relationships are argumentative?
- Q3 — Multi-hop traversal: what chains out from permission structure?
- Q4 — Confidence diagnostic: which triples were inferred?

The queries use the reification pattern established in graph.py: each triple is a node (?triple) with subject, predicate, object, and metadata attached as properties of that node.

`queries.py: query_entity_neighborhood`
Returns every entity directly connected to the named entity, along with the predicate label and category for each connection. The parameters ae g (the populated RDF graph) and entity_label (the human-readable label of the focal entity). This function returns a list of dicts with keys: target, predicate, category.

`queries.py: query_argumentative_triples`
Returns all triples whose predicate is categorized as argumentative, ordered so lower-confidence triples appear first. The parameter is g (the populated RDF graph). This function returns a list of dicts with keys: subject, predicate, object, confidence.

`queries.py: query_multihop`
Walks two relationship hops out from the named concept and returns the full chain of connections found. This query demonstrates what graph traversal can do that vector similarity search cannot — following typed relationship chains rather than retrieving semantically similar passages. The parameters are g (the populated RDF graph) and concept_label (the human-readable label of the starting concept). The function returns a list of dicts with keys: hop_one, hop_one_predicate, hop_two, hop_two_predicate, terminal.

`queries.py: query_inferred_triples`
Returns all triples marked as inferred rather than directly stated, ordered by section so clustering patterns are immediately visible. The parameter is g (the populated RDF graph). The function return a list of dicts with keys: subject, predicate, object, source_claim, section.

`queries.py: format_results`
Formats query results as a readable string suitable for passing into the grounded answer prompt. The parameters are results (list of dicts from any query function) and query_name (label used in the header). The function returns a formatted string representation of the results.

# Test queries against the synthetic graph
python kgllm/queries.py

# extraction.py
Sends a passage to the local LLM and extracts structured entity-relationship triples as JSON. The extraction happens in two stages within a single prompt:

- The model reasons about what entities and relationships are present (exploiting Qwen's reasoning training)
- The model outputs those as structured JSON

This two-stage approach produces more reliable triples than asking for JSON directly, and gives readers a visible reasoning trace to inspect.

`extraction.py: call_ollama`
Sends the passage to the local Ollama model and returns the parsed JSON response. The parameters are passage (the text to extract triples from) and verbose (if True, prints the reasoning trace before the triples). The function returns a dict containing 'reasoning' and 'triples' keys. The function raises a ValueError if the response cannot be parsed as valid JSON and a requests.HTTPError if the Ollama API returns an error.

`extraction.py: parse_extraction`
Parses the model's raw output into a validated Python dict. This function handles two failure modes:

- The model wraps JSON in markdown code fences — strips them.
- The model outputs malformed JSON — raises with a clear message.

The parameter is raw (the raw string output from the model). The function returns a dict with 'reasoning' and 'triples' keys.

`extraction.py: validate_triples`
Validates each triple against the schema and filters out malformed entries rather than failing hard. This is intentionally lenient — a malformed triple is dropped with a warning rather than crashing the pipeline. For a blog post demo this is more instructive than a hard failure. The parameter is triples (list of triple dicts from the model). The function returns a list of valid triple dicts.

# pipeline.py
The end-to-end knowledge graph pipeline.

Usage:
- python pipeline.py                        # uses default passage.txt
- python pipeline.py --passage my_text.txt  # uses a custom passage file
- python pipeline.py --question "What texts does Chronicles draw on?"
- python pipeline.py --save-graph           # serializes graph to turtle
- python pipeline.py --verbose              # shows full model output

What this does:
Stage 1: Sends the passage to ts-reasoner for triple extraction
Stage 2: Builds an RDF graph from the extracted triples
Stage 3: Runs four SPARQL queries against the graph
Stage 4: Sends query results to ts-reasoner for a grounded answer

`pipeline.py: stage`
Prints a visible stage header so the pipeline output reads as a narrative rather than an undifferentiated log.

`pipeline.py: run_pipeline`
Runs the full four-stage pipeline and prints results at each stage. The parameters here are passage_path (path to the text file containing the passage), question (the natural language question for the grounded answer), save_graph_path (if provided, serializes the graph to this path), and verbose (if True, shows full model output at each stage).

`pipeline.py: build_grounded_prompt`
Constructs the grounded answer prompt by injecting all four query results into the user turn. The parameter are question (the natural language question to answer) and query_results (dict mapping query name to formatted result string). This function return the complete user prompt string.

`pipeline.py: call_grounded_answer`
Sends the query results and question to the local model and returns a grounded natural language answer. This call does NOT use JSON mode — we want flowing prose here, not structured output. This is the key difference from the extraction call in Stage 1. The parameters are question (the natural language question), query_results (dict of formatted query result strings), and verbose (if True, prints the raw model response). The function returns the grounded answer as a string.

# Run the full pipeline with verbose model output
python pipeline.py --verbose

# Save the RDF graph and inspect it
python pipeline.py --save-graph

then open faithfulness_kg.ttl — it's human-readable Turtle format

# Ask a different question
python pipeline.py --question "What is the role of Numbers 22 in the Chronicler's revision?"

# Try a different passage entirely
python pipeline.py --passage your_text.txt

The `--save-graph` flag is worth highlighting. The Turtle output is readable enough that you can see exactly what RDF reification looks like in practice — which is more instructive than any code explanation of it.

The one thing to watch when you run it: if Qwen's extraction produces zero valid triples, the pipeline exits cleanly at Stage 1 with a diagnostic message rather than crashing in Stage 2. That's where prompt tuning starts — the extraction prompt is deliberately designed to be iterated on.

What to watch for on first run is the most obvious: the first stage. The most likely failure point is Stage 1 — if ts-reasoner doesn't respect the JSON format constraint reliably on the first attempt, the extraction parser will either drop malformed triples with warnings or exit with a clear message. That's intentional — it tells you where to focus prompt tuning rather than producing a silent bad graph. If you get zero valid triples, the most productive next step is running with --verbose to see the raw model output. That shows you exactly what ts-reasoner returned before parsing, which tells you whether the problem is JSON structure, schema compliance, or something else entirely.

And if you want to save the Turtle file to inspect the raw graph:

python pipeline.py --save-graph

That's probably worth doing on the first successful run regardless — seeing the actual RDF output is instructive.

# CHECKING

Quick way to verify before running the full pipeline — run this curl to confirm /api/generate works with your model name: check.py

If that returns a response with a response field containing text, the pipeline should connect cleanly.

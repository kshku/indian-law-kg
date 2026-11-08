# Indian Law KG

A document-aware legal knowledge graph and LangChain agent for Indian Acts.
The ontology is general-purpose; the documents and provisions are instances
loaded from the manifest.

## Ontology

The schema is `ontology/legal-ontology.rdf` (RDF/XML, viewable in Protégé). It
models `LegalDocument`, `Act`, `Provision`, `TextPassage`, `ReferenceMention`,
and `LegalConcept` with relations such as `hasProvision`, `containsPassage`,
`referencesProvision`, and `mentionsConcept`. The `sectionNumber` property is a
string, so section IDs such as `2A`, `4-I`, and `53-O` are preserved.

Local concepts are defined in `data/concepts.json`: 53 labels with aliases,
short definitions, and optional `skos:broader` links. They are a local
vocabulary, not external concept IRIs. Generated provisions are annotated with
W3C Web Annotation records (`oa:Annotation`, `oa:hasBody`, `oa:hasTarget`) that
link a matched text span to a concept by character offsets. Each annotation also
adds a direct `legal:mentionsConcept` edge for SPARQL retrieval.

The combined generated graph is `ontology/generated/legal-documents.rdf`. Open
the schema and generated graph together in Protégé to inspect the ontology and
its individuals, or load either RDF/XML file with RDFLib.
When Protégé resolves the generated graph's import, map
`https://example.org/legal-ontology` to `../legal-ontology.rdf` in
`ontology/generated/catalog-v001.xml` (the generated graph is not the schema).

The manifest at `data/documents.json` currently registers:

| Document | PDF | Provisions |
|---|---|---:|
| The Railways Act, 1989 | `A1989-24.pdf` | 235 |
| The Multimodal Transportation of Goods Act, 1993 | `a1993-28.pdf` | 33 |
| The Competition Act, 2002 | `A2003-12.pdf` | 98 |
| The Food Safety and Standards Act, 2006 | `A2006-34.pdf` | 101 |
| The Legal Metrology Act, 2009 | `a2010-1.pdf` | 57 |
| The Bureau of Indian Standards Act, 2016 | `A2016-11.pdf` | 43 |
| The Consumer Protection Act, 2019 | `A2019-35.pdf` | 107 |
| The Jammu and Kashmir Reorganisation Act, 2019 | `a2019-34.pdf` | 107 |
| The Mediation Act, 2023 | `a2023-32.pdf` | 65 |

## Data extraction and rebuilding

The reproducible pipeline is:

```text
PDF -> page-text JSONL -> parsed section JSON -> combined RDF/XML
```

`extract` processes every manifest PDF. It can also process one document by ID;
the old PDF-path form remains available for PDFs already registered in the
manifest.

```bash
# Extract and parse all manifest documents
uv run python -m indian_law_kg extract

# Or just one document
uv run python -m indian_law_kg extract \
  --document competition-act-2002

# Combine all parsed documents into one RDF graph
uv run python -m indian_law_kg build

# Build the local concept, text, and graph vector index (after build)
uv run python -m indian_law_kg index

# Check parsed data and document-scoped RDF links/counts
uv run python scripts/validate.py
uv run pytest -q
```

The vector index lives in `data/vector_db/` and is not part of the RDF graph.
It is ignored by Git and can be rebuilt from the graph and concept catalog at
any time.

Page text is written to `data/extracted/<document-id>.jsonl` (one JSON record per
page). Parsed provisions are JSON arrays at `data/parsed/<document-id>.json`.
The generator combines these into `ontology/generated/legal-documents.rdf`.
The validator checks each Act independently, including section IDs, passages,
and document-to-provision links.

### Adding another Act

1. Download a text-extractable PDF and save it under `data/`.
2. Add an entry to `data/documents.json` with a stable lowercase hyphenated `id`,
   `title`, project-relative `pdf` path, `expected_sections` count, and
   `source_url` if available.
3. If its heading layout differs from the supported IndiaCode formats, add
   parser hints (`first_section_pattern`, `section_pattern`, or `end_marker`) to
   that entry.
4. Run `extract`, `build`, and `scripts/validate.py` as shown above.
5. If the new Act needs vocabulary not already in `data/concepts.json`, add
   concepts there, rebuild, and run `index`.

For example, a manifest entry looks like:

```json
{
  "id": "example-act-2025",
  "title": "The Example Act, 2025",
  "pdf": "data/example-act-2025.pdf",
  "source_url": "",
  "expected_sections": 42
}
```

The graph retains each document's own section IDs, so equal IDs in different
Acts do not collide. Reference resolution is within the source Act; cross-Act
linking is not currently performed.

## Querying the graph

Section queries require a document ID when more than one Act is loaded. `find`
is exact SPARQL search; `search` uses the ontology as middleware; `similar`
uses graph embeddings. All three can be filtered to one Act:

```bash
uv run python -m indian_law_kg query \
  --document consumer-protection-act-2019 --section 2
uv run python -m indian_law_kg query \
  --document competition-act-2002 --section 53-O

uv run python -m indian_law_kg find "refund"
uv run python -m indian_law_kg find "competition" \
  --document competition-act-2002

uv run python -m indian_law_kg search \
  "consumer complaint about a defective product" \
  --document consumer-protection-act-2019
uv run python -m indian_law_kg similar \
  --document consumer-protection-act-2019 --section 2
```

Runtime KG reads use SPARQL queries; the agent does not read the parsed JSON
files. `find` returns case-insensitive matches on section IDs, titles, and text
as `match_type: direct`. It also returns one-hop provisions linked by citations
as `match_type: related`, with `related_via` identifying the direction. Related
provisions are context, not direct wording matches.

`search` follows the ontology-as-middleware flow:

```text
query -> concept resolution -> SPARQL over annotated provisions -> ranking
```

1. Resolve the query to local concepts. Exact alias matches are tried first;
   concept vectors are used only when no alias matches.
2. Query `legal:mentionsConcept` annotations with SPARQL and collect the direct
   provisions plus their one-hop citation neighbours.
3. Rank only inside those ontology-grounded candidates with reciprocal rank
   fusion over the concept, exact-keyword, and text-vector signals.

Results carry `ontology_grounded`, `match_type` (`concept` or `related`),
`signals`, and a fused `score`. If no local concept resolves, `search` falls back
to exact keyword results and marks them `match_type="keyword_fallback"` with
`ontology_grounded=false`. Those results are not ontology-grounded. If a concept
resolves but no provision is annotated with it, `search` returns no sections
rather than arbitrary vector-only hits.

`similar` reads the seed provision's graph vector, which combines citation edges
and inverse-frequency-weighted concept co-occurrence, and returns structurally
similar provisions. Use it for related context, not as direct evidence.

### Vector index

The index uses Qdrant local mode. It has two collections:

| Collection | Vectors | Dimensions |
|---|---|---:|
| `provisions` | named `text` and `graph` vectors | 512 and 64 |
| `concepts` | named `text` vector | 512 |

Text vectors use FastEmbed with
`jinaai/jina-embeddings-v2-small-en`; the model downloads once on first use.
Graph vectors come from deterministic symmetric-normalised SVD over the
provision affinity graph. Embeddings stay in `data/vector_db/`, not in RDF. Run
`build` before `index` so the annotations and graph exist.

## Agent

The agent in `src/indian_law_kg/agent/` uses typed SPARQL-backed tools including
`kg_list_documents`, `kg_list_sections`, `kg_get_section`, `kg_find_sections`,
`kg_get_concepts`, and reference tools, plus read-only `kg_sparql`. When the
vector index exists it also gets ontology retrieval tools:

- `kg_resolve_concepts` — alias-first concept resolution with vector fallback.
- `kg_ontology_search` — grounded hybrid search returning concepts and results.
- `kg_similar_provisions` — graph-similar provisions for structural context.

The agent discovers Act IDs, scopes section queries to the right Act, grounds
answers in graph results, and cites the Act title and section ID (plus source URL
when present). Before SPARQL, it calls `kg_get_schema` to learn the available
ontology terms. Use `kg_find_sections` for exact wording and section IDs;
`kg_ontology_search` for paraphrases and concept-level questions; and
`kg_similar_provisions` as context, not direct evidence. Re-read a returned
provision with `kg_get_section` before quoting it.

The agent is built into the CLI; it is not a separate service or subcommand.
`ask` answers one question, while `chat` starts an interactive session with
streaming output and tool events. If the vector index is missing, both keep the
exact-search behavior, print a note, and suggest running `index`.

Set up the model in `.env` (see `.env.example`). For the OpenRouter provider,
set `KG_MODEL_PROVIDER=openrouter`, `KG_MODEL_NAME` to an OpenRouter model ID,
and `OPENROUTER_API_KEY`, then install its optional integration:

```bash
uv sync --extra openrouter
uv run indian-law-kg ask \
  "What are the rights of a consumer under the Consumer Protection Act?"
uv run indian-law-kg chat
```

The module form works too: `uv run python -m indian_law_kg ask "..."`.

`KG_MODEL_PROVIDER` and `KG_MODEL_NAME` select the model. Provider extras are
`openai`, `ollama`, `gemini`, `anthropic`, and `openrouter`.

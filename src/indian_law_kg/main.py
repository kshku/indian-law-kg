import argparse
import asyncio
import json
from pathlib import Path

import pymupdf
from rdflib.namespace import RDF

from . import config
from .annotations import OA
from .documents import get_document, load_documents
from .index_builder import build_index
from .kg import KnowledgeGraph
from .ontology_generator import LEGAL, generate_ontology
from .parser import parse
from .pdf_utils import extract_pdf
from .retrieval import OntologyRetriever
from .vector_store import FastEmbedder, VectorStore

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="indian-law-kg",
        description="Legal Acts knowledge graph and agent",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    extract = subparsers.add_parser(
        "extract", help="Extract and parse manifest PDFs into sections"
    )
    extract.add_argument(
        "pdf",
        nargs="?",
        type=Path,
        help="optional manifest PDF path (for compatibility)",
    )
    extract.add_argument("--document", help="extract one manifest document ID")

    subparsers.add_parser(
        "build", help="Generate the RDF ontology from parsed sections"
    )

    subparsers.add_parser(
        "index",
        help="Build the local concept, text, and graph vector index",
    )

    query = subparsers.add_parser(
        "query", help="Show one section from the knowledge graph"
    )
    query.add_argument("--section", required=True)
    query.add_argument("--document", help="filter by manifest document ID")

    find = subparsers.add_parser(
        "find", help="Find sections by title or text"
    )
    find.add_argument("query")
    find.add_argument("--document", help="filter by manifest document ID")

    search = subparsers.add_parser(
        "search", help="Search provisions through the ontology"
    )
    search.add_argument("query")
    search.add_argument("--document", help="filter by manifest document ID")

    similar = subparsers.add_parser(
        "similar", help="Find provisions with similar graph structure"
    )
    similar.add_argument("--section", required=True)
    similar.add_argument("--document", help="filter by manifest document ID")

    ask = subparsers.add_parser(
        "ask", help="Ask the agent one question"
    )
    ask.add_argument("question")

    subparsers.add_parser(
        "chat", help="Start an interactive agent session"
    )

    return parser


def _load_kg() -> KnowledgeGraph:
    if not config.GENERATED_ONTOLOGY_PATH.is_file():
        raise FileNotFoundError(
            "Generated ontology not found: "
            f"{config.GENERATED_ONTOLOGY_PATH}. Run `build` first."
        )

    return KnowledgeGraph(
        config.GENERATED_ONTOLOGY_PATH, config.ONTOLOGY_SCHEMA_PATH
    )


def _load_retriever() -> OntologyRetriever:
    if not config.VECTOR_DB_PATH.is_dir():
        raise FileNotFoundError(
            f"Vector index not found: {config.VECTOR_DB_PATH}. "
            "Run `index` first."
        )

    kg = _load_kg()
    store = VectorStore(config.VECTOR_DB_PATH)
    try:
        embedder = FastEmbedder()
    except Exception:
        store.close()
        raise
    return OntologyRetriever(kg, store, embedder)


def _load_optional_retriever() -> OntologyRetriever | None:
    if not config.VECTOR_DB_PATH.is_dir():
        return None
    return _load_retriever()


def command_extract(
    document_id: str | None = None, pdf_path: Path | None = None
) -> None:
    if document_id is not None and pdf_path is not None:
        raise ValueError("Use either --document or a PDF path, not both")

    documents = load_documents()
    if pdf_path is not None:
        requested_path = pdf_path.resolve()
        matching = [
            document
            for document in documents
            if document.pdf_path.resolve() == requested_path
        ]
        if not matching:
            raise ValueError(f"PDF is not registered in the document manifest: {pdf_path}")
        selected = matching
    elif document_id is not None:
        selected = [get_document(document_id, documents)]
    else:
        selected = documents

    for document in selected:
        if not document.pdf_path.is_file():
            raise FileNotFoundError(f"PDF not found: {document.pdf_path}")

        extracted_path = config.EXTRACTED_DIR / f"{document.id}.jsonl"
        parsed_path = config.PARSED_DIR / f"{document.id}.json"
        config.EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
        config.PARSED_DIR.mkdir(parents=True, exist_ok=True)

        extract_pdf(document.pdf_path, extracted_path)
        sections = parse(
            extracted_path,
            parsed_path,
            first_section_pattern=document.first_section_pattern,
            section_pattern=document.section_pattern,
            end_marker=document.end_marker,
        )
        if len(sections) != document.expected_sections:
            raise ValueError(
                f"Document {document.id} expected {document.expected_sections} "
                f"sections, parsed {len(sections)}"
            )


def command_build() -> None:
    documents = []
    total_sections = 0

    for document in load_documents():
        input_path = config.PARSED_DIR / f"{document.id}.json"
        with input_path.open(encoding="utf-8") as file:
            sections = json.load(file)
        documents.append((document, sections))
        total_sections += len(sections)

    graph = generate_ontology(documents)

    config.GENERATED_ONTOLOGY_PATH.parent.mkdir(
        parents=True, exist_ok=True
    )
    graph.serialize(
        destination=str(config.GENERATED_ONTOLOGY_PATH), format="xml"
    )

    print(f"Documents processed: {len(documents)}")
    print(f"Sections processed: {total_sections}")
    print(
        "Concepts generated: "
        f"{len(list(graph.subjects(RDF.type, LEGAL.LegalConcept)))}"
    )
    print(
        "Annotations generated: "
        f"{len(list(graph.subjects(RDF.type, OA.Annotation)))}"
    )
    print(f"RDF triples generated: {len(graph)}")
    print(f"Output: {config.GENERATED_ONTOLOGY_PATH}")


def _report_index_progress(stage: str, completed: int, total: int) -> None:
    width = 24
    ratio = completed / total if total else 1.0
    filled = min(width, max(0, int(width * ratio)))
    bar = "=" * filled + "-" * (width - filled)
    print(
        f"{stage:<24} [{bar}] {completed:>4}/{total:<4}",
        flush=True,
    )


def command_index() -> None:
    print("Loading knowledge graph...", flush=True)
    kg = _load_kg()
    print("Loading embedding model...", flush=True)
    embedder = FastEmbedder()
    device = getattr(embedder, "device", "cpu")
    max_length = getattr(embedder, "max_length", "?")
    print(
        f"Embedding device: {device} (max {max_length} tokens)",
        flush=True,
    )
    if device != "cuda":
        if not getattr(embedder, "cuda_available", False):
            print("CUDA provider not installed; using CPU.", flush=True)
        elif getattr(embedder, "device_error", None):
            print(
                f"CUDA unavailable ({embedder.device_error}); using CPU.",
                flush=True,
            )

    store = VectorStore(config.VECTOR_DB_PATH)
    try:
        store.recreate()
        counts = build_index(
            kg, embedder, store, progress=_report_index_progress
        )
    finally:
        store.close()

    print(f"Documents indexed: {counts['documents']}")
    print(f"Provisions indexed: {counts['provisions']}")
    print(f"Concepts indexed: {counts['concepts']}")
    print(f"Text vectors indexed: {counts['text_vectors']}")
    print(f"Graph vectors indexed: {counts['graph_vectors']}")
    print(f"Concept vectors indexed: {counts['concept_vectors']}")
    print(f"Index path: {config.VECTOR_DB_PATH}")


def command_query(
    section_id: str, document_id: str | None = None
) -> None:
    section = _load_kg().get_section(section_id, document_id)

    if section is None:
        if document_id:
            raise ValueError(
                f"Section {section_id} not found in document {document_id}"
            )
        raise ValueError(f"Section {section_id} not found")

    print(json.dumps(section, indent=2, ensure_ascii=False))


def command_find(query: str, document_id: str | None = None) -> None:
    print(
        json.dumps(
            _load_kg().find_sections(query, document_id),
            indent=2,
            ensure_ascii=False,
        )
    )


def command_search(query: str, document_id: str | None = None) -> None:
    retriever = _load_retriever()
    try:
        result = retriever.search(query, document_id)
    finally:
        retriever.store.close()
    print(json.dumps(result, indent=2, ensure_ascii=False))


def command_similar(
    section_id: str, document_id: str | None = None
) -> None:
    retriever = _load_retriever()
    try:
        results = retriever.similar(section_id, document_id)
    finally:
        retriever.store.close()
    print(json.dumps(results, indent=2, ensure_ascii=False))


def command_ask(question: str) -> None:
    from .agent import build_agent

    kg = _load_kg()
    retriever = _load_optional_retriever()
    if retriever is None:
        print(
            "Note: vector index not found; ontology retrieval is disabled. "
            "Run `index` to enable it."
        )
    try:
        agent = build_agent(kg, kg.graph, retriever=retriever)
        result = agent.invoke(
            {"messages": [{"role": "user", "content": question}]}
        )
    finally:
        if retriever is not None:
            retriever.store.close()

    print(result["messages"][-1].content)


async def _chat(
    kg: KnowledgeGraph, retriever: OntologyRetriever | None = None
) -> None:
    from .agent import build_agent

    agent = build_agent(kg, kg.graph, retriever=retriever)

    while True:
        try:
            question = input(">>> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if question.strip().lower() in {"exit", "quit"}:
            break

        if not question.strip():
            continue

        print()

        async for event in agent.astream_events(
            {"messages": [{"role": "user", "content": question}]},
            version="v2",
        ):
            event_type = event["event"]

            match event_type:
                case "on_chat_model_stream":
                    chunk = event["data"]["chunk"]
                    if chunk.content:
                        print(chunk.content, end="", flush=True)

                case "on_tool_start":
                    print(
                        f"\n[tool] {event['name']} "
                        f"{event['data'].get('input')}"
                    )

                case "on_tool_end":
                    print(f"[tool done] {event['name']}")

                case "on_chain_error":
                    print(f"\n[error] {event['data']}")

        print("\n")


def command_chat() -> None:
    kg = _load_kg()
    retriever = _load_optional_retriever()
    if retriever is None:
        print(
            "Note: vector index not found; ontology retrieval is disabled. "
            "Run `index` to enable it."
        )
    try:
        asyncio.run(_chat(kg, retriever))
    finally:
        if retriever is not None:
            retriever.store.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    try:
        match args.command:
            case "extract":
                command_extract(args.document, args.pdf)
            case "build":
                command_build()
            case "index":
                command_index()
            case "query":
                command_query(args.section, args.document)
            case "find":
                command_find(args.query, args.document)
            case "search":
                command_search(args.query, args.document)
            case "similar":
                command_similar(args.section, args.document)
            case "ask":
                command_ask(args.question)
            case "chat":
                command_chat()
    except (FileNotFoundError, ValueError, pymupdf.FileDataError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

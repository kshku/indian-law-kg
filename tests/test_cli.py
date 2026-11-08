import json
import re

import pytest
from rdflib import Graph

from indian_law_kg import config
from indian_law_kg.documents import Document
from indian_law_kg import main
from indian_law_kg.main import build_parser


def test_parser_has_subcommands():
    parser = build_parser()
    args = parser.parse_args(
        ["query", "--section", "53-O", "--document", "competition-act-2002"]
    )
    assert args.command == "query"
    assert args.section == "53-O"
    assert args.document == "competition-act-2002"


def test_extract_parser_defaults_to_all_manifest_documents():
    args = build_parser().parse_args(["extract"])

    assert args.document is None
    assert args.pdf is None


def test_extract_parser_accepts_document_and_legacy_manifest_pdf_path():
    parser = build_parser()
    by_id = parser.parse_args(["extract", "--document", "mediation-act-2023"])
    by_path = parser.parse_args(["extract", "data/A2019-35.pdf"])

    assert by_id.document == "mediation-act-2023"
    assert by_path.pdf.as_posix() == "data/A2019-35.pdf"


def make_document(document_id, pdf_path):
    return Document(
        id=document_id,
        title=f"{document_id} Act",
        pdf_path=pdf_path,
        source_url=None,
        expected_sections=1,
    )


def test_command_extract_processes_all_documents_or_one(tmp_path, monkeypatch):
    pdf_a = tmp_path / "a.pdf"
    pdf_b = tmp_path / "b.pdf"
    pdf_a.touch()
    pdf_b.touch()
    documents = [
        make_document("act-a", pdf_a),
        make_document("act-b", pdf_b),
    ]
    extracted_dir = tmp_path / "extracted"
    parsed_dir = tmp_path / "parsed"
    monkeypatch.setattr(main, "load_documents", lambda: documents)
    monkeypatch.setattr(config, "EXTRACTED_DIR", extracted_dir)
    monkeypatch.setattr(config, "PARSED_DIR", parsed_dir)
    extracted = []

    def fake_extract(pdf_path, output_path):
        extracted.append(pdf_path)
        output_path.write_text("{}\n", encoding="utf-8")

    def fake_parse(input_path, output_path, **kwargs):
        output_path.write_text("[]", encoding="utf-8")
        return [object()]

    monkeypatch.setattr(main, "extract_pdf", fake_extract)
    monkeypatch.setattr(main, "parse", fake_parse)

    main.command_extract()
    assert extracted == [pdf_a, pdf_b]
    assert (extracted_dir / "act-a.jsonl").is_file()
    assert (parsed_dir / "act-b.json").is_file()

    extracted.clear()
    main.command_extract("act-b")
    assert extracted == [pdf_b]


def test_command_extract_accepts_only_registered_legacy_pdf(tmp_path, monkeypatch):
    pdf_path = tmp_path / "consumer.pdf"
    pdf_path.touch()
    document = make_document("consumer-act", pdf_path)
    monkeypatch.setattr(main, "load_documents", lambda: [document])
    monkeypatch.setattr(config, "EXTRACTED_DIR", tmp_path / "extracted")
    monkeypatch.setattr(config, "PARSED_DIR", tmp_path / "parsed")
    monkeypatch.setattr(main, "extract_pdf", lambda *_: None)
    monkeypatch.setattr(main, "parse", lambda *_args, **_kwargs: [object()])

    main.command_extract(pdf_path=pdf_path)
    with pytest.raises(ValueError, match="not registered"):
        main.command_extract(pdf_path=tmp_path / "unknown.pdf")


def test_command_build_loads_all_parsed_documents(tmp_path, monkeypatch):
    documents = [
        make_document("act-a", tmp_path / "a.pdf"),
        make_document("act-b", tmp_path / "b.pdf"),
    ]
    parsed_dir = tmp_path / "parsed"
    parsed_dir.mkdir()
    for document in documents:
        (parsed_dir / f"{document.id}.json").write_text(
            json.dumps([{"section_id": "2A"}]), encoding="utf-8"
        )
    output_path = tmp_path / "generated" / "legal-documents.rdf"
    graph = Graph()
    received = {}
    monkeypatch.setattr(main, "load_documents", lambda: documents)
    monkeypatch.setattr(config, "PARSED_DIR", parsed_dir)
    monkeypatch.setattr(config, "GENERATED_ONTOLOGY_PATH", output_path)
    def fake_generate_ontology(inputs):
        received["inputs"] = inputs
        return graph

    monkeypatch.setattr(main, "generate_ontology", fake_generate_ontology)

    main.command_build()

    assert output_path.is_file()
    assert [document.id for document, _ in received["inputs"]] == [
        "act-a",
        "act-b",
    ]
    assert graph.serialize(format="xml")


def test_parser_index_command():
    parser = build_parser()
    args = parser.parse_args(["index"])
    assert args.command == "index"


def test_command_index_builds_both_collections(
    tmp_path, kg, fake_embedder, monkeypatch, capsys
):
    from indian_law_kg import index_builder

    real_store = main.VectorStore
    monkeypatch.setattr(main, "_load_kg", lambda: kg)
    monkeypatch.setattr(main, "FastEmbedder", lambda: fake_embedder)
    monkeypatch.setattr(
        main,
        "VectorStore",
        lambda path: real_store(
            path, text_dimension=4, graph_dimension=4
        ),
    )
    monkeypatch.setattr(
        main,
        "build_index",
        lambda kg, embedder, store: index_builder.build_index(
            kg, embedder, store, graph_dimension=4
        ),
    )
    monkeypatch.setattr(
        config, "VECTOR_DB_PATH", tmp_path / "vector_db"
    )

    main.command_index()

    output = capsys.readouterr().out
    assert "Provisions indexed: 8" in output
    assert "Text vectors indexed: 8" in output
    assert "Graph vectors indexed: 8" in output
    concept_count = int(
        re.search(r"Concepts indexed: (\d+)", output).group(1)
    )
    vector_count = int(
        re.search(r"Concept vectors indexed: (\d+)", output).group(1)
    )
    assert concept_count > 0
    assert vector_count == concept_count
    assert (tmp_path / "vector_db").is_dir()


def test_query_and_find_keep_document_filter(kg, monkeypatch, capsys):
    monkeypatch.setattr(main, "_load_kg", lambda: kg)

    with pytest.raises(ValueError, match="document_id is required"):
        main.command_query("2")

    main.command_query("53-O", "example-act-a")
    query_result = json.loads(capsys.readouterr().out)
    assert query_result["document_id"] == "example-act-a"
    assert query_result["section_id"] == "53-O"

    main.command_find("definitions", "example-act-b")
    find_result = json.loads(capsys.readouterr().out)
    assert find_result[0]["document_id"] == "example-act-b"


def test_parser_build_command():
    parser = build_parser()
    args = parser.parse_args(["build"])
    assert args.command == "build"

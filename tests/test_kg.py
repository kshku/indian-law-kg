import pytest


def test_list_documents(kg):
    assert kg.list_documents() == [
        {
            "document_id": "example-act-a",
            "title": "Example Act A",
            "source_url": "https://example.org/a",
            "section_count": 6,
        },
        {
            "document_id": "example-act-b",
            "title": "Example Act B",
            "source_url": None,
            "section_count": 2,
        },
    ]


def test_list_sections_includes_document_identity_and_string_ids(kg):
    sections = kg.list_sections()

    assert [
        (section["document_id"], section["section_id"])
        for section in sections
    ] == [
        ("example-act-a", "2"),
        ("example-act-a", "2A"),
        ("example-act-a", "10"),
        ("example-act-a", "53A"),
        ("example-act-a", "53-I"),
        ("example-act-a", "53-O"),
        ("example-act-b", "2"),
        ("example-act-b", "99"),
    ]
    assert sections[0]["document_title"] == "Example Act A"
    assert sections[0]["title"] == "Definitions"
    assert [
        section["section_id"]
        for section in kg.list_sections("example-act-a")
    ] == ["2", "2A", "10", "53A", "53-I", "53-O"]


def test_get_section_is_document_scoped_and_unknown_section_is_none(kg):
    section = kg.get_section("53-O", "example-act-a")
    assert section["document_title"] == "Example Act A"
    assert section["section_id"] == "53-O"
    assert section["title"] == "Procedure and powers"
    assert section["text"] == "53-O. Procedure and powers.—Body for 53-O."
    assert section["start_page"] == 8
    assert kg.get_section("99", "example-act-a") is None
    assert kg.get_section("99", "example-act-b")["title"] == "Only in Act B"


def test_get_concepts_returns_annotations_for_one_section(kg):
    assert kg.get_concepts("2", "example-act-a") == [
        {
            "concept": "ConceptDefinitions",
            "label": "definitions",
            "text": "Definitions",
            "start": 3,
            "end": 14,
        }
    ]
    assert [
        concept["concept"]
        for concept in kg.get_concepts("10", "example-act-a")
    ] == ["ConceptAppeal", "ConceptAppeal"]
    assert kg.get_concepts("99", "example-act-a") == []


def test_section_addressed_calls_require_document_when_ambiguous(kg):
    with pytest.raises(ValueError, match="document_id is required"):
        kg.get_section("2")
    with pytest.raises(ValueError, match="document_id is required"):
        kg.section_exists("2")
    with pytest.raises(ValueError, match="Unknown document ID"):
        kg.get_section("2", "unknown-act")

    for method in (
        kg.get_references,
        kg.get_referenced_by,
        kg.get_external_references,
    ):
        with pytest.raises(ValueError, match="document_id is required"):
            method("2")
        with pytest.raises(ValueError, match="Unknown document ID"):
            method("2", "unknown-act")


def test_references_resolve_only_within_the_document(kg):
    references = kg.get_references("2", "example-act-a")

    assert [reference["target"] for reference in references] == [
        "2A",
        "53-O",
        None,
        None,
    ]
    assert [reference["status"] for reference in references] == [
        "resolved",
        "resolved",
        "unresolved",
        "external",
    ]
    assert kg.get_referenced_by("2", "example-act-a") == ["2A"]
    assert kg.get_referenced_by("2A", "example-act-a") == ["2"]
    assert kg.get_referenced_by("2", "example-act-b") == ["99"]
    assert kg.get_external_references("2", "example-act-a") == [
        "Indian Evidence Act, 1872"
    ]


def test_get_schema(kg):
    schema = kg.get_schema()

    assert schema["schema_loaded"] is True
    assert schema["prefixes"]["legal"] == "https://example.org/legal-ontology#"
    assert "Provision" in schema["classes"]
    assert {
        "name": "containsPassage",
        "domain": "Provision",
        "range": "TextPassage",
    } in schema["object_properties"]
    assert {
        "name": "sectionNumber",
        "domain": "Provision",
        "range": "string",
    } in schema["datatype_properties"]
    assert schema["counts"] == {
        "provisions": 8,
        "text_passages": 8,
        "reference_mentions": 6,
    }


def test_runtime_kg_retrieval_uses_sparql_not_graph_traversal(
    monkeypatch, kg
):
    def fail_if_traversed(*args, **kwargs):
        raise AssertionError("runtime retrieval must use SPARQL")

    for graph in (kg.graph, kg.schema):
        monkeypatch.setattr(graph, "subjects", fail_if_traversed)
        monkeypatch.setattr(graph, "objects", fail_if_traversed)
        monkeypatch.setattr(graph, "value", fail_if_traversed)

    assert len(kg.get_schema()["classes"]) == 4
    assert len(kg.list_documents()) == 2
    assert len(kg.list_sections()) == 8
    assert kg.section_exists("53-O", "example-act-a") is True
    assert kg.get_section("53-O", "example-act-a")["title"] == (
        "Procedure and powers"
    )
    assert kg.get_references("2", "example-act-a")[0]["target"] == "2A"
    assert kg.get_referenced_by("2", "example-act-a") == ["2A"]
    assert kg.get_concepts("2", "example-act-a")[0]["concept"] == (
        "ConceptDefinitions"
    )


def test_missing_schema_is_reported(monkeypatch, tmp_path, kg):
    monkeypatch.setattr(
        "indian_law_kg.kg.config.ONTOLOGY_SCHEMA_PATH",
        tmp_path / "missing.rdf",
    )

    path = tmp_path / "kg.rdf"
    kg.graph.serialize(destination=path, format="xml")

    schema = type(kg)(path).get_schema()

    assert schema["schema_loaded"] is False
    assert schema["classes"] == []
    assert schema["counts"]["provisions"] == 8


def test_find_sections_searches_all_documents_and_can_filter(kg):
    matches = kg.find_sections("definitions")
    direct = [match for match in matches if match["match_type"] == "direct"]
    related = [match for match in matches if match["match_type"] == "related"]

    assert {
        (section["document_id"], section["section_id"])
        for section in direct
    } == {
        ("example-act-a", "2"),
        ("example-act-a", "2A"),
        ("example-act-b", "2"),
    }
    assert [(section["document_id"], section["section_id"]) for section in related] == [
        ("example-act-a", "53-O"),
        ("example-act-b", "99"),
    ]
    assert [section["related_via"] for section in related] == [
        "references",
        "referenced_by",
    ]
    assert "Procedure and powers" in related[0]["snippet"]
    assert len(
        [
            section
            for section in matches
            if section["document_id"] == "example-act-a"
            and section["section_id"] == "2A"
        ]
    ) == 1

    filtered = kg.find_sections("definitions", "example-act-b")
    assert [section["document_id"] for section in filtered] == [
        "example-act-b",
        "example-act-b",
    ]
    assert [section["match_type"] for section in filtered] == [
        "direct",
        "related",
    ]

    id_only = kg.find_sections("99", "example-act-b")
    assert [(item["section_id"], item["match_type"]) for item in id_only] == [
        ("99", "direct"),
        ("2", "related"),
    ]
    assert id_only[1]["related_via"] == "references"
    assert "Only in Act B" in id_only[0]["snippet"]
    assert "This section is only in Act B." in id_only[0]["snippet"]

    ordered = kg.find_sections("neutral", "example-act-a")
    assert [item["section_id"] for item in ordered] == ["10", "53A", "53-I"]
    assert {item["match_type"] for item in ordered} == {"direct"}

    inbound = kg.find_sections("Procedure", "example-act-a")
    assert [(item["section_id"], item["match_type"]) for item in inbound] == [
        ("53-O", "direct"),
        ("2", "related"),
    ]
    assert inbound[1]["related_via"] == "referenced_by"
    assert kg.find_sections("no-such-term") == []
    assert kg.find_sections("   ") == []


def test_find_sections_binds_search_text_as_a_literal(kg):
    query = '" } UNION { ?s ?p ?o'
    assert kg.find_sections(query) == []

    regex_query = r'[.*+?^${}()|\\] " } UNION { ?s ?p ?o'
    assert kg.find_sections(regex_query) == []

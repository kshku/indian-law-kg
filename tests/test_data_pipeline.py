from time import perf_counter

from rdflib import Literal, Namespace
from rdflib.namespace import SKOS, XSD

from indian_law_kg import config
from indian_law_kg.annotations import OA
from indian_law_kg.documents import Document
from indian_law_kg.kg import KnowledgeGraph
from indian_law_kg.ontology_generator import generate_ontology
from indian_law_kg.validation import (
    validate_annotations,
    validate_document,
    validate_parsed_sections,
)

LEGAL = Namespace(f"{config.ONTOLOGY_IRI}#")


def document(document_id, title, expected_sections=1):
    return Document(
        id=document_id,
        title=title,
        pdf_path=config.DATA_DIR / "unused.pdf",
        source_url=None,
        expected_sections=expected_sections,
    )


def section(section_id, title, text):
    return {
        "section_id": section_id,
        "title": title,
        "text": text,
        "start_page": 1,
        "end_page": 1,
    }


def kg_from_documents(tmp_path, docs_and_sections):
    graph = generate_ontology(docs_and_sections)
    graph_path = tmp_path / "documents.rdf"
    graph.serialize(destination=graph_path, format="xml")
    return KnowledgeGraph(graph_path)


def test_validate_two_document_parsed_data_and_rdf(tmp_path):
    act_a = document("act-a", "Act A")
    act_b = document("act-b", "Act B")
    section_a = section("2A", "Inserted section", "2A. Text A.")
    section_b = section("53-O", "Procedure", "53-O. Text B.")
    kg = kg_from_documents(
        tmp_path,
        [(act_a, [section_a]), (act_b, [section_b])],
    )

    assert validate_document(act_a, [section_a], kg) == []
    assert validate_document(act_b, [section_b], kg) == []


def test_validate_parsed_sections_reports_missing_duplicate_and_empty_ids():
    doc = document("act-a", "Act A", expected_sections=3)
    sections = [
        section("2", "Section two", "Text."),
        section("2", "Duplicate section", "Text."),
        section("", "Missing ID", "Text."),
        section("3", "Empty text", "  "),
    ]

    problems = validate_parsed_sections(doc, sections)

    assert any("empty/invalid section ID" in problem for problem in problems)
    assert any("duplicate section IDs: ['2']" in problem for problem in problems)
    assert any("section 3 has empty text" in problem for problem in problems)
    assert any("expected 3 parsed sections, got 4" in problem for problem in problems)


def test_validate_detects_wrong_document_links_despite_matching_totals(tmp_path):
    act_a = document("act-a", "Act A")
    act_b = document("act-b", "Act B")
    section_a = section("2", "Section A", "Text A.")
    section_b = section("2", "Section B", "Text B.")
    kg = kg_from_documents(
        tmp_path,
        [(act_a, [section_a]), (act_b, [section_b])],
    )
    kg.graph.remove(
        (LEGAL["act-a"], LEGAL.hasProvision, LEGAL["act-a_Section2"])
    )
    kg.graph.add(
        (LEGAL["act-a"], LEGAL.hasProvision, LEGAL["act-b_Section2"])
    )

    problems = validate_document(act_a, [section_a], kg)

    assert any("section 2 is not linked to its Act" in p for p in problems)
    assert any("foreign provision: act-b_Section2" in p for p in problems)


def test_validate_annotations_accepts_fixture_and_detects_corruption(kg):
    assert validate_annotations(kg) == []

    passage = LEGAL["example-act-a_Section2_Passage1"]
    concept = LEGAL.ConceptDefinitions
    kg.graph.remove((passage, LEGAL.mentionsConcept, concept))
    assert any(
        "missing its mentionsConcept edge" in problem
        for problem in validate_annotations(kg)
    )
    kg.graph.add((passage, LEGAL.mentionsConcept, concept))

    position = LEGAL[
        "Annotation_example-act-a_Section2_3_ConceptDefinitions_Position"
    ]
    for value in list(kg.graph.objects(position, OA.end)):
        kg.graph.remove((position, OA.end, value))
    kg.graph.add(
        (position, OA.end, Literal(999, datatype=XSD.nonNegativeInteger))
    )
    assert any(
        "quote does not match the passage text" in problem
        for problem in validate_annotations(kg)
    )

    kg.graph.add((concept, SKOS.broader, LEGAL.ConceptMissing))
    assert any(
        "unknown broader target" in problem
        for problem in validate_annotations(kg)
    )


def test_generated_graph_resolves_reviewed_inserted_section_ids():
    kg = KnowledgeGraph(config.GENERATED_ONTOLOGY_PATH)

    assert kg.section_exists("53A", "competition-act-2002")
    assert kg.section_exists("20-I", "railways-act-1989")
    assert any(
        reference["target"] == "53A" and reference["status"] == "resolved"
        for reference in kg.get_references("53B", "competition-act-2002")
    )
    assert any(
        reference["target"] == "20-I" and reference["status"] == "resolved"
        for reference in kg.get_references("20H", "railways-act-1989")
    )


def test_generated_graph_search_stays_fast_and_naturally_ordered():
    kg = KnowledgeGraph(config.GENERATED_ONTOLOGY_PATH)

    started = perf_counter()
    results = kg.find_sections("refund")
    elapsed = perf_counter() - started

    assert results
    assert elapsed < 15
    assert {result["match_type"] for result in results} <= {
        "direct",
        "related",
    }

    competition_53 = [
        section["section_id"]
        for section in kg.list_sections("competition-act-2002")
        if section["section_id"].startswith("53")
    ]
    assert competition_53[:5] == ["53", "53A", "53B", "53C", "53D"]
    assert "53-I" in competition_53
    assert competition_53.index("53H") < competition_53.index("53-I")
    assert competition_53.index("53-I") < competition_53.index("53J")
    assert competition_53.index("53N") < competition_53.index("53-O")
    assert competition_53.index("53-O") < competition_53.index("53P")
    assert competition_53.index("53R") < competition_53.index("53-S")

import json
from pathlib import Path

import pytest
from rdflib import BNode, Graph, Literal, Namespace
from rdflib.namespace import RDF, RDFS, SKOS, XSD

from indian_law_kg.annotations import (
    Concept,
    add_annotations,
    concept_iri,
    load_concepts,
    match_concepts,
)
from indian_law_kg.documents import Document

LEGAL = Namespace("https://example.org/legal-ontology#")
OA = Namespace("http://www.w3.org/ns/oa#")


def _write_catalog(tmp_path, entries):
    path = tmp_path / "concepts.json"
    path.write_text(json.dumps({"concepts": entries}), encoding="utf-8")
    return path


def _concept(concept_id, label, aliases, broader=None):
    return {
        "id": concept_id,
        "label": label,
        "definition": f"Definition of {label}.",
        "aliases": aliases,
        "broader": broader,
    }


def test_load_concepts_validates_ids_aliases_and_broader_targets(tmp_path):
    path = _write_catalog(
        tmp_path,
        [
            _concept("consumer", "consumer", ["consumer", "consumers"]),
            _concept(
                "refund",
                "refund",
                ["refund", "refunds"],
                broader="consumer",
            ),
        ],
    )

    concepts = load_concepts(path)

    assert [concept.id for concept in concepts] == ["consumer", "refund"]
    assert concepts[1].broader == "consumer"


@pytest.mark.parametrize(
    "entries, message",
    [
        (
            [
                _concept("consumer", "consumer", ["consumer"]),
                _concept("consumer", "other", ["other"]),
            ],
            "Duplicate concept ID",
        ),
        (
            [
                _concept("consumer", "consumer", ["consumer"]),
                _concept("refund", "refund", ["Consumer"]),
            ],
            "Alias",
        ),
        (
            [_concept("Bad ID", "bad", ["bad"])],
            "invalid ID",
        ),
        (
            [_concept("consumer", "consumer", ["consumer"], broader="missing")],
            "unknown broader concept",
        ),
    ],
)
def test_load_concepts_rejects_invalid_catalogs(tmp_path, entries, message):
    path = _write_catalog(tmp_path, entries)

    with pytest.raises(ValueError, match=message):
        load_concepts(path)


def test_real_catalog_is_valid_and_has_hierarchy():
    concepts = load_concepts()
    by_id = {concept.id: concept for concept in concepts}

    assert len(concepts) >= 30
    assert len(by_id) == len(concepts)
    assert any(concept.broader for concept in concepts)
    assert all(
        concept.broader is None or concept.broader in by_id
        for concept in concepts
    )


def test_match_concepts_uses_whole_words_and_longest_alias():
    concepts = [
        Concept(
            id="standard",
            label="standard",
            definition="A standard.",
            aliases=("standard",),
        ),
        Concept(
            id="standardisation",
            label="standardisation",
            definition="Standardisation.",
            aliases=("standards", "standardisation"),
        ),
        Concept(
            id="consumer",
            label="consumer",
            definition="A consumer.",
            aliases=("consumer", "consumer protection"),
        ),
    ]

    matches = match_concepts(
        "Consumer protection standards and a consumer, not consumers.",
        concepts,
    )

    assert [(match.concept.id, match.text) for match in matches] == [
        ("consumer", "Consumer protection"),
        ("standardisation", "standards"),
        ("consumer", "consumer"),
    ]


def test_match_concepts_returns_offsets_and_empty_text():
    concepts = [
        Concept(
            id="refund",
            label="refund",
            definition="A refund.",
            aliases=("refund",),
        )
    ]

    matches = match_concepts("A refund is due.", concepts)

    assert len(matches) == 1
    assert matches[0].start == 2
    assert matches[0].end == 8
    assert matches[0].text == "refund"
    assert match_concepts("", concepts) == []


def test_concept_iri_uses_a_local_legal_concept_name():
    assert concept_iri("consumer") == LEGAL.ConceptConsumer
    assert (
        concept_iri("abuse-of-dominant-position")
        == LEGAL.ConceptAbuseOfDominantPosition
    )


def _add_annotation_fixture(graph):
    document = Document(
        id="test-act",
        title="Test Act",
        pdf_path=Path("test.pdf"),
        source_url=None,
        expected_sections=1,
    )
    section = {
        "section_id": "2",
        "title": "Definitions",
        "text": "A consumer may seek a refund for a defect.",
        "start_page": 1,
    }
    concepts = [
        Concept(
            id="consumer-protection",
            label="consumer protection",
            definition="Protection for consumers.",
            aliases=("consumer protection",),
        ),
        Concept(
            id="consumer",
            label="consumer",
            definition="A buyer of goods or services.",
            aliases=("consumer", "consumers"),
            broader="consumer-protection",
        ),
        Concept(
            id="refund",
            label="refund",
            definition="Return of money.",
            aliases=("refund", "refunds"),
            broader="consumer-protection",
        ),
    ]
    count = add_annotations(graph, document, section, concepts)
    return count, section


def test_add_annotations_emits_w3c_annotations_and_mentions_concept():
    graph = Graph()
    count, section = _add_annotation_fixture(graph)

    assert count == 2
    annotations = list(graph.subjects(RDF.type, OA.Annotation))
    assert len(annotations) == 2

    consumer_annotation = next(
        annotation
        for annotation in annotations
        if (annotation, OA.hasBody, LEGAL.ConceptConsumer) in graph
    )
    target = graph.value(consumer_annotation, OA.hasTarget)
    source = graph.value(target, OA.hasSource)
    selectors = list(graph.objects(target, OA.hasSelector))

    assert source == LEGAL["test-act_Section2_Passage1"]
    assert len(selectors) == 2

    quote = next(
        selector
        for selector in selectors
        if (selector, RDF.type, OA.TextQuoteSelector) in graph
    )
    position = next(
        selector
        for selector in selectors
        if (selector, RDF.type, OA.TextPositionSelector) in graph
    )

    assert str(graph.value(quote, OA.exact)) == "consumer"
    assert graph.value(position, OA.start) == Literal(
        2, datatype=XSD.nonNegativeInteger
    )
    assert graph.value(position, OA.end) == Literal(
        10, datatype=XSD.nonNegativeInteger
    )
    assert (
        LEGAL["test-act_Section2_Passage1"],
        LEGAL.mentionsConcept,
        LEGAL.ConceptConsumer,
    ) in graph
    assert section["text"][2:10] == "consumer"


def test_add_annotations_adds_skos_concept_metadata_and_no_blank_nodes():
    graph = Graph()
    _add_annotation_fixture(graph)

    assert (LEGAL.ConceptConsumer, RDF.type, LEGAL.LegalConcept) in graph
    assert (
        LEGAL.ConceptConsumer,
        RDFS.label,
        Literal("consumer"),
    ) in graph
    assert (
        LEGAL.ConceptConsumer,
        SKOS.prefLabel,
        Literal("consumer"),
    ) in graph
    assert (
        LEGAL.ConceptConsumer,
        SKOS.broader,
        LEGAL.ConceptConsumerProtection,
    ) in graph
    assert not any(
        isinstance(term, BNode) for triple in graph for term in triple
    )


def test_add_annotations_is_deterministic():
    first = Graph()
    second = Graph()

    _add_annotation_fixture(first)
    _add_annotation_fixture(second)

    assert sorted(first) == sorted(second)

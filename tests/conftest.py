import hashlib
from pathlib import Path

import pytest
from rdflib import Graph, Literal, Namespace
from rdflib.namespace import OWL, RDF, RDFS, XSD

from indian_law_kg.annotations import Concept, add_annotations
from indian_law_kg.config import ONTOLOGY_IRI
from indian_law_kg.documents import Document
from indian_law_kg.kg import KnowledgeGraph

LEGAL = Namespace(f"{ONTOLOGY_IRI}#")


class FakeEmbedder:
    """Deterministic tiny embedder; never downloads a model."""

    def __init__(self, dimension=4):
        self.dimension = dimension

    def embed(self, texts):
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append(
                [
                    (digest[index % len(digest)] / 127.5) - 1.0
                    for index in range(self.dimension)
                ]
            )
        return vectors


@pytest.fixture
def fake_embedder():
    return FakeEmbedder()


def _add_section(graph, document_id, section_id, title, text, page):
    provision = LEGAL[f"{document_id}_Section{section_id}"]
    passage = LEGAL[f"{document_id}_Section{section_id}_Passage1"]
    graph.add((provision, RDF.type, LEGAL.Provision))
    graph.add(
        (
            provision,
            LEGAL.sectionNumber,
            Literal(section_id, datatype=XSD.string),
        )
    )
    graph.add((provision, LEGAL.sectionTitle, Literal(title)))
    graph.add((provision, LEGAL.containsPassage, passage))
    graph.add((passage, RDF.type, LEGAL.TextPassage))
    graph.add((passage, LEGAL.text, Literal(text)))
    graph.add((passage, LEGAL.pageNumber, Literal(page, datatype=XSD.integer)))
    graph.add((LEGAL[document_id], LEGAL.hasProvision, provision))


@pytest.fixture
def kg(tmp_path):
    graph = Graph()
    documents = [
        ("example-act-a", "Example Act A", "https://example.org/a"),
        ("example-act-b", "Example Act B", None),
    ]

    for document_id, title, source_url in documents:
        act = LEGAL[document_id]
        graph.add((act, RDF.type, LEGAL.Act))
        graph.add((act, LEGAL.title, Literal(title, datatype=XSD.string)))
        if source_url:
            graph.add(
                (
                    act,
                    LEGAL.sourceURL,
                    Literal(source_url, datatype=XSD.anyURI),
                )
            )

    for document_id, section_id, title, text, page in [
        (
            "example-act-a",
            "2",
            "Definitions",
            "2. Definitions.—See sections 2A and 53-O and section 99.",
            6,
        ),
        (
            "example-act-a",
            "2A",
            "Inserted definitions",
            "2A. Inserted definitions.—See section 2.",
            7,
        ),
        (
            "example-act-a",
            "10",
            "Appeals",
            "10. Appeals.—Neutral appeal text.",
            11,
        ),
        (
            "example-act-a",
            "53A",
            "Powers",
            "53A. Powers.—Neutral powers text.",
            12,
        ),
        (
            "example-act-a",
            "53-I",
            "Inspection",
            "53-I. Inspection.—Neutral inspection text.",
            13,
        ),
        (
            "example-act-a",
            "53-O",
            "Procedure and powers",
            "53-O. Procedure and powers.—Body for 53-O.",
            8,
        ),
        (
            "example-act-b",
            "2",
            "Definitions",
            "2. Definitions.—A different Act's definition.",
            9,
        ),
        (
            "example-act-b",
            "99",
            "Only in Act B",
            "This section is only in Act B.",
            10,
        ),
    ]:
        _add_section(graph, document_id, section_id, title, text, page)

    annotation_concepts = [
        Concept(
            id="definitions",
            label="definitions",
            definition="Statements that define terms used in an Act.",
            aliases=("definitions",),
        ),
        Concept(
            id="appeal",
            label="appeal",
            definition="A request for review of a decision.",
            aliases=("appeal", "appeals"),
        ),
    ]
    annotation_document = Document(
        id="example-act-a",
        title="Example Act A",
        pdf_path=Path("unused.pdf"),
        source_url="https://example.org/a",
        expected_sections=6,
    )
    add_annotations(
        graph,
        annotation_document,
        {
            "section_id": "2",
            "title": "Definitions",
            "text": "2. Definitions.—See sections 2A and 53-O and section 99.",
        },
        annotation_concepts,
    )
    add_annotations(
        graph,
        annotation_document,
        {
            "section_id": "10",
            "title": "Appeals",
            "text": "10. Appeals.—Neutral appeal text.",
        },
        annotation_concepts,
    )

    references = [
        ("example-act-a", "2", 1, "sections 2A", "resolved", "2A"),
        ("example-act-a", "2", 2, "section 53-O", "resolved", "53-O"),
        ("example-act-a", "2", 3, "section 99", "unresolved", None),
        ("example-act-a", "2A", 1, "section 2", "resolved", "2"),
        ("example-act-b", "99", 1, "section 2", "resolved", "2"),
    ]
    for document_id, source_id, index, text, status, target_id in references:
        mention = LEGAL[
            f"{document_id}_Section{source_id}_Reference{index}"
        ]
        graph.add((mention, RDF.type, LEGAL.ReferenceMention))
        graph.add((mention, LEGAL.referenceText, Literal(text)))
        graph.add((mention, LEGAL.resolutionStatus, Literal(status)))
        graph.add(
            (
                mention,
                LEGAL.occursIn,
                LEGAL[f"{document_id}_Section{source_id}_Passage1"],
            )
        )
        if target_id:
            graph.add(
                (
                    mention,
                    LEGAL.referencesProvision,
                    LEGAL[f"{document_id}_Section{target_id}"],
                )
            )

    external = LEGAL["example-act-a_Section2_Reference4"]
    graph.add((external, RDF.type, LEGAL.ReferenceMention))
    graph.add((external, LEGAL.referenceText, Literal("the Evidence Act")))
    graph.add((external, LEGAL.resolutionStatus, Literal("external")))
    graph.add((external, LEGAL.externalLaw, Literal("Indian Evidence Act, 1872")))
    graph.add(
        (external, LEGAL.occursIn, LEGAL["example-act-a_Section2_Passage1"])
    )

    path = tmp_path / "kg.rdf"
    graph.serialize(destination=path, format="xml")

    schema = Graph()
    for class_name in (
        "Provision",
        "TextPassage",
        "ReferenceMention",
        "Act",
    ):
        schema.add((LEGAL[class_name], RDF.type, OWL.Class))

    schema.add((LEGAL.containsPassage, RDF.type, OWL.ObjectProperty))
    schema.add((LEGAL.containsPassage, RDFS.domain, LEGAL.Provision))
    schema.add((LEGAL.containsPassage, RDFS.range, LEGAL.TextPassage))
    schema.add((LEGAL.sectionNumber, RDF.type, OWL.DatatypeProperty))
    schema.add((LEGAL.sectionNumber, RDFS.domain, LEGAL.Provision))
    schema.add((LEGAL.sectionNumber, RDFS.range, XSD.string))

    schema_path = tmp_path / "schema.rdf"
    schema.serialize(destination=schema_path, format="xml")

    return KnowledgeGraph(path, schema_path)

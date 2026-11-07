import json
from collections.abc import Sequence

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import OWL, RDF, RDFS, SKOS, XSD

from . import config
from .annotations import (
    OA,
    Concept,
    add_annotations,
    add_concepts,
    load_concepts,
)
from .documents import Document, load_documents
from .references import extract_references, resolution_status

LEGAL = Namespace(f"{config.ONTOLOGY_IRI}#")


class SortedGraph(Graph):
    """Graph whose serialization order does not depend on hash order."""

    def subjects(self, predicate=None, object=None):
        return iter(
            sorted(
                super().subjects(predicate, object), key=str
            )
        )

    def predicate_objects(self, subject=None, predicate=None):
        return iter(
            sorted(
                super().predicate_objects(subject, predicate),
                key=lambda item: (str(item[0]), str(item[1])),
            )
        )


def sorted_graph(graph: Graph) -> Graph:
    """Return a copy with a stable triple order for reproducible output."""
    ordered = SortedGraph()

    for prefix, namespace in graph.namespaces():
        ordered.bind(prefix, namespace)

    for triple in graph:
        ordered.add(triple)

    return ordered


def _provision_uri(document_id: str, section_id: str) -> URIRef:
    return LEGAL[f"{document_id}_Section{section_id}"]


def _validate_sections(document: Document, sections: Sequence[dict]) -> set[str]:
    section_ids = []

    for section in sections:
        section_id = section.get("section_id")
        if not isinstance(section_id, str) or not section_id:
            raise ValueError(
                f"Document {document.id} has an invalid section ID: {section_id!r}"
            )
        section_ids.append(section_id)

    if len(section_ids) != len(set(section_ids)):
        duplicates = sorted(
            {section_id for section_id in section_ids if section_ids.count(section_id) > 1}
        )
        raise ValueError(
            f"Document {document.id} has duplicate section IDs: {duplicates}"
        )

    if len(section_ids) != document.expected_sections:
        raise ValueError(
            f"Document {document.id} expected {document.expected_sections} "
            f"sections, got {len(section_ids)}"
        )

    return set(section_ids)


def generate_ontology(
    documents: Sequence[tuple[Document, Sequence[dict]]],
    concepts: Sequence[Concept] | None = None,
) -> Graph:
    """Generate a deterministic graph for each Act and its provisions."""
    graph = Graph()
    graph.bind("legal", LEGAL)
    graph.bind("oa", OA)
    graph.bind("owl", OWL)
    graph.bind("rdfs", RDFS)
    graph.bind("skos", SKOS)
    graph.bind("xsd", XSD)

    if concepts is None:
        concepts = load_concepts()
    add_concepts(graph, concepts)

    data_ontology = URIRef(config.DATA_ONTOLOGY_IRI)
    graph.add((data_ontology, RDF.type, OWL.Ontology))
    graph.add((data_ontology, OWL.imports, URIRef(config.ONTOLOGY_IRI)))

    seen_document_ids = set()
    ids_by_document = {}

    for document, sections in documents:
        if document.id in seen_document_ids:
            raise ValueError(f"Duplicate document ID: {document.id}")
        seen_document_ids.add(document.id)
        ids_by_document[document.id] = _validate_sections(document, sections)

        act = LEGAL[document.id]
        graph.add((act, RDF.type, LEGAL.Act))
        graph.add(
            (
                act,
                LEGAL.title,
                Literal(document.title, datatype=XSD.string),
            )
        )
        if document.source_url:
            graph.add(
                (
                    act,
                    LEGAL.sourceURL,
                    Literal(document.source_url, datatype=XSD.anyURI),
                )
            )

        for section in sections:
            section_id = section["section_id"]
            provision = _provision_uri(document.id, section_id)
            passage = LEGAL[
                f"{document.id}_Section{section_id}_Passage1"
            ]

            graph.add((provision, RDF.type, LEGAL.Provision))
            graph.add(
                (
                    provision,
                    LEGAL.sectionNumber,
                    Literal(section_id, datatype=XSD.string),
                )
            )
            graph.add(
                (
                    provision,
                    LEGAL.sectionTitle,
                    Literal(section["title"], datatype=XSD.string),
                )
            )
            graph.add((act, LEGAL.hasProvision, provision))

            graph.add((passage, RDF.type, LEGAL.TextPassage))
            graph.add(
                (
                    passage,
                    LEGAL.text,
                    Literal(section["text"], datatype=XSD.string),
                )
            )
            graph.add(
                (
                    passage,
                    LEGAL.pageNumber,
                    Literal(int(section["start_page"]), datatype=XSD.integer),
                )
            )
            graph.add((provision, LEGAL.containsPassage, passage))
            add_annotations(
                graph,
                document,
                section,
                concepts,
                add_metadata=False,
            )

    # Resolve citations only after provisions for their source Act exist.
    for document, sections in documents:
        section_ids = ids_by_document[document.id]
        for section in sections:
            source_id = section["section_id"]
            passage = LEGAL[
                f"{document.id}_Section{source_id}_Passage1"
            ]

            for index, reference in enumerate(
                extract_references(section["text"]), start=1
            ):
                mention = LEGAL[
                    f"{document.id}_Section{source_id}_Reference{index}"
                ]
                graph.add((mention, RDF.type, LEGAL.ReferenceMention))
                graph.add(
                    (
                        mention,
                        LEGAL.referenceText,
                        Literal(reference.text, datatype=XSD.string),
                    )
                )
                graph.add((mention, LEGAL.occursIn, passage))

                status = resolution_status(reference, section_ids)
                if status == "resolved":
                    graph.add(
                        (
                            mention,
                            LEGAL.referencesProvision,
                            _provision_uri(document.id, reference.target),
                        )
                    )
                elif status == "external" and reference.external_law:
                    graph.add(
                        (
                            mention,
                            LEGAL.externalLaw,
                            Literal(reference.external_law, datatype=XSD.string),
                        )
                    )

                graph.add(
                    (
                        mention,
                        LEGAL.resolutionStatus,
                        Literal(status, datatype=XSD.string),
                    )
                )

    return sorted_graph(graph)


def main() -> None:
    documents = []
    total_sections = 0

    for document in load_documents():
        input_path = config.PARSED_DIR / f"{document.id}.json"
        with input_path.open(encoding="utf-8") as file:
            sections = json.load(file)
        documents.append((document, sections))
        total_sections += len(sections)

    graph = generate_ontology(documents)
    config.GENERATED_ONTOLOGY_PATH.parent.mkdir(parents=True, exist_ok=True)
    graph.serialize(
        destination=str(config.GENERATED_ONTOLOGY_PATH), format="xml"
    )

    print(f"Documents processed: {len(documents)}")
    print(f"Sections processed: {total_sections}")
    print(f"RDF triples generated: {len(graph)}")
    print(f"Output: {config.GENERATED_ONTOLOGY_PATH}")


if __name__ == "__main__":
    main()

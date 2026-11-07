import json

from rdflib import Namespace
from rdflib.namespace import RDF, SKOS

from . import config
from .annotations import OA
from .documents import Document, load_documents
from .kg import KnowledgeGraph

LEGAL = Namespace(f"{config.ONTOLOGY_IRI}#")


def validate_parsed_sections(
    document: Document, sections: list[dict]
) -> list[str]:
    problems = []
    section_ids = []

    for index, section in enumerate(sections, start=1):
        if not isinstance(section, dict):
            problems.append(f"parsed record {index} is not an object")
            continue
        section_id = section.get("section_id")
        if not isinstance(section_id, str) or not section_id.strip():
            problems.append(f"parsed record {index} has an empty/invalid section ID")
            continue
        section_ids.append(section_id)
        if not str(section.get("text", "")).strip():
            problems.append(f"section {section_id} has empty text")

    duplicates = sorted(
        {item for item in section_ids if section_ids.count(item) > 1}
    )
    if duplicates:
        problems.append(f"duplicate section IDs: {duplicates}")
    if len(sections) != document.expected_sections:
        problems.append(
            f"expected {document.expected_sections} parsed sections, got {len(sections)}"
        )
    return problems


def validate_document(
    document: Document, sections: list[dict], kg: KnowledgeGraph
) -> list[str]:
    """Compare one document's parsed records against its RDF individuals."""
    problems = validate_parsed_sections(document, sections)
    graph_document = next(
        (item for item in kg.list_documents() if item["document_id"] == document.id),
        None,
    )
    if graph_document is None:
        return problems + ["Act individual is missing from the RDF graph"]
    if graph_document["title"] != document.title:
        problems.append(f"RDF Act title mismatch: {graph_document['title']!r}")
    if document.source_url and graph_document["source_url"] != document.source_url:
        problems.append("RDF source URL does not match the manifest")
    if graph_document["section_count"] != document.expected_sections:
        problems.append(
            f"expected {document.expected_sections} RDF provisions, "
            f"got {graph_document['section_count']}"
        )

    valid_sections = [
        item
        for item in sections
        if isinstance(item, dict)
        and isinstance(item.get("section_id"), str)
        and item["section_id"].strip()
    ]
    parsed_ids = {item["section_id"] for item in valid_sections}
    graph_sections = kg.list_sections(document.id)
    graph_ids = {item["section_id"] for item in graph_sections}
    if parsed_ids != graph_ids:
        problems.append(
            "parsed/RDF section ID mismatch: "
            f"missing in RDF={sorted(parsed_ids - graph_ids)}, "
            f"extra in RDF={sorted(graph_ids - parsed_ids)}"
        )

    act = LEGAL[document.id]
    expected_prefix = f"{document.id}_Section"
    for linked_provision in kg.graph.objects(act, LEGAL.hasProvision):
        local_name = str(linked_provision).rsplit("#", 1)[-1]
        if not local_name.startswith(expected_prefix):
            problems.append(
                f"Act is linked to a foreign provision: {local_name}"
            )

    for section in valid_sections:
        section_id = section["section_id"]
        provision = LEGAL[f"{document.id}_Section{section_id}"]
        if (act, LEGAL.hasProvision, provision) not in kg.graph:
            problems.append(f"section {section_id} is not linked to its Act")
        if (provision, RDF.type, LEGAL.Provision) not in kg.graph:
            problems.append(f"section {section_id} provision individual is missing")

        number = kg.graph.value(provision, LEGAL.sectionNumber)
        if number is None or str(number) != section_id:
            problems.append(f"section {section_id} has a mismatched RDF sectionNumber")
        title = kg.graph.value(provision, LEGAL.sectionTitle)
        if title is None or str(title) != str(section.get("title", "")):
            problems.append(f"section {section_id} has a mismatched RDF title")

        passages = set(kg.graph.objects(provision, LEGAL.containsPassage))
        if len(passages) != 1:
            problems.append(
                f"section {section_id} should have one text passage, got {len(passages)}"
            )
            continue
        passage = next(iter(passages))
        if (passage, RDF.type, LEGAL.TextPassage) not in kg.graph:
            problems.append(f"section {section_id} passage individual is missing")
        text = kg.graph.value(passage, LEGAL.text)
        if text is None or str(text) != str(section.get("text", "")):
            problems.append(f"section {section_id} has mismatched passage text")
        page = kg.graph.value(passage, LEGAL.pageNumber)
        if page is None or int(page) != int(section.get("start_page", 0)):
            problems.append(f"section {section_id} has mismatched passage page")

    return problems


def validate_annotations(kg: KnowledgeGraph) -> list[str]:
    """Validate W3C concept annotations and local SKOS hierarchy links."""
    problems = []
    graph = kg.graph

    for annotation in set(graph.subjects(RDF.type, OA.Annotation)):
        annotation_name = str(annotation).rsplit("#", 1)[-1]
        bodies = list(graph.objects(annotation, OA.hasBody))
        targets = list(graph.objects(annotation, OA.hasTarget))

        if len(bodies) != 1:
            problems.append(
                f"annotation {annotation_name} has {len(bodies)} bodies"
            )
        if not targets:
            problems.append(
                f"annotation {annotation_name} has no target"
            )

        body = bodies[0] if len(bodies) == 1 else None
        if body is not None and (
            body,
            RDF.type,
            LEGAL.LegalConcept,
        ) not in graph:
            problems.append(
                f"annotation {annotation_name} body is not a LegalConcept"
            )

        for target in targets:
            if (
                target,
                RDF.type,
                OA.SpecificResource,
            ) not in graph:
                problems.append(
                    f"annotation {annotation_name} target is not a "
                    "SpecificResource"
                )
            sources = list(graph.objects(target, OA.hasSource))
            if len(sources) != 1:
                problems.append(
                    f"annotation {annotation_name} target has "
                    f"{len(sources)} sources"
                )
                continue
            source = sources[0]
            passage_text = graph.value(source, LEGAL.text)
            if passage_text is None:
                problems.append(
                    f"annotation {annotation_name} source has no text"
                )
                continue

            selectors = list(graph.objects(target, OA.hasSelector))
            quotes = [
                selector
                for selector in selectors
                if (selector, RDF.type, OA.TextQuoteSelector) in graph
            ]
            positions = [
                selector
                for selector in selectors
                if (selector, RDF.type, OA.TextPositionSelector) in graph
            ]
            if len(quotes) != 1:
                problems.append(
                    f"annotation {annotation_name} has {len(quotes)} quote "
                    "selectors"
                )
            if len(positions) != 1:
                problems.append(
                    f"annotation {annotation_name} has {len(positions)} "
                    "position selectors"
                )
            if len(quotes) != 1 or len(positions) != 1:
                continue

            exact = graph.value(quotes[0], OA.exact)
            start_value = graph.value(positions[0], OA.start)
            end_value = graph.value(positions[0], OA.end)
            try:
                start = int(start_value)
                end = int(end_value)
            except (TypeError, ValueError):
                problems.append(
                    f"annotation {annotation_name} has non-integer offsets"
                )
                continue

            if start < 0 or end <= start:
                problems.append(
                    f"annotation {annotation_name} has invalid offsets "
                    f"{start}:{end}"
                )
            if exact is None:
                problems.append(
                    f"annotation {annotation_name} has no exact quote"
                )
            elif str(passage_text)[start:end] != str(exact):
                problems.append(
                    f"annotation {annotation_name} quote does not match the "
                    "passage text"
                )
            if body is not None and (
                source,
                LEGAL.mentionsConcept,
                body,
            ) not in graph:
                problems.append(
                    f"annotation {annotation_name} is missing its "
                    "mentionsConcept edge"
                )

    for concept in set(graph.subjects(RDF.type, LEGAL.LegalConcept)):
        concept_name = str(concept).rsplit("#", 1)[-1]
        for parent in graph.objects(concept, SKOS.broader):
            if (parent, RDF.type, LEGAL.LegalConcept) not in graph:
                problems.append(
                    f"concept {concept_name} has unknown broader target "
                    f"{str(parent).rsplit('#', 1)[-1]}"
                )

    return problems


def main() -> None:
    documents = load_documents()
    kg = KnowledgeGraph(config.GENERATED_ONTOLOGY_PATH)
    problems = []
    total_sections = 0

    for document in documents:
        parsed_path = config.PARSED_DIR / f"{document.id}.json"
        if not parsed_path.is_file():
            current = [f"parsed data file is missing: {parsed_path}"]
            section_count = 0
        else:
            try:
                sections = json.loads(parsed_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as exc:
                sections = []
                current = [f"parsed data is invalid JSON: {exc}"]
            else:
                if not isinstance(sections, list):
                    sections = []
                    current = ["parsed data must be a JSON array"]
                else:
                    current = validate_document(document, sections, kg)
            section_count = len(sections)

        total_sections += section_count
        status = "INVALID" if current else "VALID"
        print(
            f"{document.id}: {section_count}/{document.expected_sections} "
            f"sections — {status}"
        )
        for message in current:
            problems.append(f"{document.id}: {message}")
            print(f"  - {message}")

    for message in validate_annotations(kg):
        problems.append(f"annotations: {message}")
        print(f"  - {message}")

    print(f"Documents validated: {len(documents)}")
    print(f"Parsed sections: {total_sections}")
    print(
        "RDF totals: "
        f"{len(set(kg.graph.subjects(RDF.type, LEGAL.Provision)))} provisions, "
        f"{len(set(kg.graph.subjects(RDF.type, LEGAL.TextPassage)))} passages, "
        f"{len(set(kg.graph.subjects(RDF.type, LEGAL.ReferenceMention)))} references, "
        f"{len(set(kg.graph.subjects(RDF.type, LEGAL.LegalConcept)))} concepts, "
        f"{len(set(kg.graph.subjects(RDF.type, OA.Annotation)))} annotations"
    )
    if problems:
        raise SystemExit(1)
    print("VALID")

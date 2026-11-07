import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import RDF, RDFS, SKOS, XSD

from . import config
from .documents import Document

LEGAL = Namespace(f"{config.ONTOLOGY_IRI}#")
OA = Namespace("http://www.w3.org/ns/oa#")

CONCEPT_ID_PATTERN = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
CONTEXT_LENGTH = 60


@dataclass(frozen=True)
class Concept:
    id: str
    label: str
    definition: str
    aliases: tuple[str, ...]
    broader: str | None = None


@dataclass(frozen=True)
class ConceptMatch:
    concept: Concept
    start: int
    end: int
    text: str


def load_concepts(path: Path = config.CONCEPTS_PATH) -> list[Concept]:
    """Load and validate the local legal concept catalog."""
    catalog_path = Path(path)
    if not catalog_path.is_file():
        raise FileNotFoundError(f"Concept catalog not found: {catalog_path}")

    try:
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid concept catalog: {exc}") from exc

    if not isinstance(data, dict) or not isinstance(data.get("concepts"), list):
        raise ValueError("Concept catalog must contain a 'concepts' list")

    concepts = []
    seen_ids = set()
    aliases = {}

    for index, item in enumerate(data["concepts"], start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Concept entry {index} must be an object")

        concept_id = item.get("id")
        label = item.get("label")
        definition = item.get("definition")
        raw_aliases = item.get("aliases")
        broader = item.get("broader")

        if not isinstance(concept_id, str) or not CONCEPT_ID_PATTERN.fullmatch(
            concept_id
        ):
            raise ValueError(
                f"Concept entry {index} has an invalid ID: {concept_id!r}"
            )
        if concept_id in seen_ids:
            raise ValueError(f"Duplicate concept ID: {concept_id}")
        seen_ids.add(concept_id)

        if not isinstance(label, str) or not label.strip():
            raise ValueError(f"Concept {concept_id} requires a non-empty label")
        if not isinstance(definition, str) or not definition.strip():
            raise ValueError(
                f"Concept {concept_id} requires a non-empty definition"
            )
        if not isinstance(raw_aliases, list) or not raw_aliases:
            raise ValueError(
                f"Concept {concept_id} requires a non-empty aliases list"
            )

        concept_aliases = []
        for alias in raw_aliases:
            if not isinstance(alias, str) or not alias.strip():
                raise ValueError(
                    f"Concept {concept_id} has an invalid alias: {alias!r}"
                )
            normalized = alias.strip()
            key = normalized.casefold()
            if key in aliases:
                raise ValueError(
                    f"Alias {normalized!r} is used by both "
                    f"{aliases[key]} and {concept_id}"
                )
            aliases[key] = concept_id
            concept_aliases.append(normalized)

        if broader is not None and (
            not isinstance(broader, str) or not broader.strip()
        ):
            raise ValueError(
                f"Concept {concept_id} broader must be a concept ID or null"
            )

        concepts.append(
            Concept(
                id=concept_id,
                label=label.strip(),
                definition=definition.strip(),
                aliases=tuple(concept_aliases),
                broader=broader.strip() if broader else None,
            )
        )

    known_ids = {concept.id for concept in concepts}
    for concept in concepts:
        if concept.broader is not None and concept.broader not in known_ids:
            raise ValueError(
                f"Concept {concept.id} references unknown broader concept "
                f"{concept.broader}"
            )

    return concepts


def match_concepts(
    text: str, concepts: Sequence[Concept]
) -> list[ConceptMatch]:
    """Find non-overlapping whole-word concept mentions in text."""
    if not text or not concepts:
        return []

    aliases = [
        (alias, concept)
        for concept in concepts
        for alias in concept.aliases
    ]
    if not aliases:
        return []

    aliases.sort(key=lambda item: (-len(item[0]), item[0].casefold()))
    by_alias = {alias.casefold(): concept for alias, concept in aliases}
    pattern = re.compile(
        r"(?<!\w)(?:"
        + "|".join(re.escape(alias) for alias, _ in aliases)
        + r")(?!\w)",
        re.IGNORECASE,
    )

    matches = []
    for match in pattern.finditer(text):
        concept = by_alias.get(match.group(0).casefold())
        if concept is None:
            continue
        matches.append(
            ConceptMatch(
                concept=concept,
                start=match.start(),
                end=match.end(),
                text=match.group(0),
            )
        )
    return matches


def concept_iri(concept_id: str) -> URIRef:
    """Return the local legal concept IRI for a catalog ID."""
    parts = re.split(r"[^A-Za-z0-9]+", concept_id)
    local_name = "Concept" + "".join(part.title() for part in parts if part)
    return LEGAL[local_name]


def add_concepts(graph: Graph, concepts: Sequence[Concept]) -> None:
    """Add local concept individuals and SKOS labels to the graph."""
    for concept in concepts:
        concept_uri = concept_iri(concept.id)
        graph.add((concept_uri, RDF.type, LEGAL.LegalConcept))
        graph.add((concept_uri, RDFS.label, Literal(concept.label)))
        graph.add((concept_uri, SKOS.prefLabel, Literal(concept.label)))
        for alias in concept.aliases:
            graph.add((concept_uri, SKOS.altLabel, Literal(alias)))
        graph.add(
            (concept_uri, SKOS.definition, Literal(concept.definition))
        )
        if concept.broader is not None:
            graph.add(
                (concept_uri, SKOS.broader, concept_iri(concept.broader))
            )


def add_annotations(
    graph: Graph,
    document: Document,
    section: dict,
    concepts: Sequence[Concept],
    add_metadata: bool = True,
) -> int:
    """Add W3C annotations for one section; return the annotation count."""
    if add_metadata:
        add_concepts(graph, concepts)

    section_id = section["section_id"]
    passage = LEGAL[
        f"{document.id}_Section{section_id}_Passage1"
    ]
    text = section.get("text", "")
    matches = match_concepts(text, concepts)

    for match in matches:
        stem = (
            f"Annotation_{document.id}_Section{section_id}_"
            f"{match.start}_{concept_iri(match.concept.id).rsplit('#', 1)[-1]}"
        )
        annotation = LEGAL[stem]
        target = LEGAL[f"{stem}_Target"]
        quote_selector = LEGAL[f"{stem}_Quote"]
        position_selector = LEGAL[f"{stem}_Position"]
        body = concept_iri(match.concept.id)

        prefix = text[max(0, match.start - CONTEXT_LENGTH) : match.start]
        suffix = text[match.end : match.end + CONTEXT_LENGTH]

        graph.add((annotation, RDF.type, OA.Annotation))
        graph.add((annotation, OA.motivatedBy, OA.tagging))
        graph.add((annotation, OA.hasBody, body))
        graph.add((annotation, OA.hasTarget, target))

        graph.add((target, RDF.type, OA.SpecificResource))
        graph.add((target, OA.hasSource, passage))
        graph.add((target, OA.hasSelector, quote_selector))
        graph.add((target, OA.hasSelector, position_selector))

        graph.add((quote_selector, RDF.type, OA.TextQuoteSelector))
        graph.add(
            (
                quote_selector,
                OA.exact,
                Literal(match.text, datatype=XSD.string),
            )
        )
        graph.add(
            (
                quote_selector,
                OA.prefix,
                Literal(prefix, datatype=XSD.string),
            )
        )
        graph.add(
            (
                quote_selector,
                OA.suffix,
                Literal(suffix, datatype=XSD.string),
            )
        )

        graph.add((position_selector, RDF.type, OA.TextPositionSelector))
        graph.add(
            (
                position_selector,
                OA.start,
                Literal(match.start, datatype=XSD.nonNegativeInteger),
            )
        )
        graph.add(
            (
                position_selector,
                OA.end,
                Literal(match.end, datatype=XSD.nonNegativeInteger),
            )
        )

        graph.add((passage, LEGAL.mentionsConcept, body))

    return len(matches)

from langchain_core.tools import tool

from ..kg import KnowledgeGraph
from ..settings import settings
from ..sparql import run_query


def build_tools(kg: KnowledgeGraph, graph) -> list:
    """Create the read-only tools the agent can call."""

    @tool
    def kg_get_schema() -> dict:
        """Describe the ontology: classes, object and datatype properties
        with their domain and range, the legal prefix, and instance counts.
        Call this before writing a SPARQL query."""
        return kg.get_schema()

    @tool
    def kg_list_documents() -> list[dict]:
        """List loaded Acts with their document IDs, titles, source URLs,
        and section counts. Use a returned document ID for section tools."""
        return kg.list_documents()

    @tool
    def kg_list_sections(document_id: str | None = None) -> list[dict]:
        """List section IDs and titles for one Act, or all loaded Acts if no
        document ID is supplied. Results identify the source Act."""
        return kg.list_sections(document_id)

    @tool
    def kg_get_section(
        section_id: str, document_id: str | None = None
    ) -> dict | None:
        """Return a section's Act identity, title, full text, page, and
        references. Section IDs may be alphanumeric, such as `2A` or `53-O`.
        Supply a document ID when more than one Act is loaded."""
        return kg.get_section(section_id, document_id)

    @tool
    def kg_find_sections(
        query: str, document_id: str | None = None
    ) -> list[dict]:
        """Search section IDs, titles and text using SPARQL. Results include
        direct matches and one-hop sections related by citations; inspect
        match_type and related_via to distinguish them."""
        return kg.find_sections(query, document_id)

    @tool
    def kg_get_concepts(
        section_id: str, document_id: str | None = None
    ) -> list[dict]:
        """Return local legal concepts annotated in a section, including the
        matched text and character offsets. Concepts are labels, not legal
        conclusions. Supply the document ID when more than one Act is loaded."""
        return kg.get_concepts(section_id, document_id)

    @tool
    def kg_get_references(
        section_id: str, document_id: str | None = None
    ) -> list[dict]:
        """List the references made by a section, with resolution status,
        target section ID, or external law if any. Supply the document ID
        when more than one Act is loaded."""
        return kg.get_references(section_id, document_id)

    @tool
    def kg_get_referenced_by(
        section_id: str, document_id: str | None = None
    ) -> list[str]:
        """List section IDs in the same Act that reference this section.
        Supply the document ID when more than one Act is loaded."""
        return kg.get_referenced_by(section_id, document_id)

    @tool
    def kg_sparql(query: str) -> dict:
        """Run a read-only SPARQL SELECT/ASK/CONSTRUCT/DESCRIBE query against
        the RDF graph and return columns and rows."""
        return run_query(graph, query, settings.max_sparql_rows)

    return [
        kg_get_schema,
        kg_list_documents,
        kg_list_sections,
        kg_get_section,
        kg_find_sections,
        kg_get_concepts,
        kg_get_references,
        kg_get_referenced_by,
        kg_sparql,
    ]

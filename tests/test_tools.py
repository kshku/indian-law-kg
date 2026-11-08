import pytest

from indian_law_kg.agent.tools import build_tools


def test_tool_names_and_calls(kg):
    tools = {tool.name: tool for tool in build_tools(kg, kg.graph)}

    assert set(tools) == {
        "kg_get_schema",
        "kg_list_documents",
        "kg_list_sections",
        "kg_get_section",
        "kg_find_sections",
        "kg_get_concepts",
        "kg_get_references",
        "kg_get_referenced_by",
        "kg_sparql",
    }

    assert (
        "Provision"
        in tools["kg_get_schema"].invoke({})["classes"]
    )
    assert tools["kg_list_documents"].invoke({})[0]["document_id"] == (
        "example-act-a"
    )
    assert (
        tools["kg_get_section"]
        .invoke({"section_id": "53-O", "document_id": "example-act-a"})[
            "title"
        ]
        == "Procedure and powers"
    )
    assert (
        tools["kg_find_sections"]
        .invoke({"query": "definitions", "document_id": "example-act-b"})[
            0
        ]["document_id"]
        == "example-act-b"
    )
    assert tools["kg_get_referenced_by"].invoke(
        {"section_id": "2", "document_id": "example-act-a"}
    ) == ["2A"]
    assert tools["kg_get_concepts"].invoke(
        {"section_id": "2", "document_id": "example-act-a"}
    )[0]["concept"] == "ConceptDefinitions"


class FakeRetriever:
    def __init__(self):
        self.resolve_args = None
        self.search_args = None
        self.similar_args = None

    def resolve_concepts(self, query):
        self.resolve_args = query
        return [{"concept_id": "definitions", "method": "alias"}]

    def search(self, query, document_id=None, limit=10):
        self.search_args = (query, document_id, limit)
        return {
            "concepts": [{"concept_id": "definitions"}],
            "results": [{"section_id": "2", "ontology_grounded": True}],
        }

    def similar(self, section_id, document_id=None, limit=10):
        self.similar_args = (section_id, document_id, limit)
        return [{"section_id": "2A", "match_type": "graph_similar"}]


def test_ontology_tools_are_added_only_with_a_retriever(kg):
    without_retriever = {
        tool.name for tool in build_tools(kg, kg.graph)
    }
    assert "kg_ontology_search" not in without_retriever
    assert "kg_resolve_concepts" not in without_retriever
    assert "kg_similar_provisions" not in without_retriever

    retriever = FakeRetriever()
    tools = {
        tool.name: tool
        for tool in build_tools(kg, kg.graph, retriever=retriever)
    }
    assert {
        "kg_resolve_concepts",
        "kg_ontology_search",
        "kg_similar_provisions",
        "kg_get_concepts",
    } <= set(tools)

    assert tools["kg_resolve_concepts"].invoke(
        {"query": "definitions"}
    )[0]["concept_id"] == "definitions"
    assert retriever.resolve_args == "definitions"

    response = tools["kg_ontology_search"].invoke(
        {
            "query": "definitions",
            "document_id": "example-act-a",
            "limit": 3,
        }
    )
    assert response["results"][0]["ontology_grounded"] is True
    assert retriever.search_args == (
        "definitions",
        "example-act-a",
        3,
    )

    similar = tools["kg_similar_provisions"].invoke(
        {
            "section_id": "2",
            "document_id": "example-act-a",
            "limit": 2,
        }
    )
    assert similar[0]["match_type"] == "graph_similar"
    assert retriever.similar_args == ("2", "example-act-a", 2)


def test_tools_surface_document_ambiguity_and_identity(kg):
    tools = {tool.name: tool for tool in build_tools(kg, kg.graph)}

    matches = tools["kg_find_sections"].invoke({"query": "definitions"})
    assert {
        (match["document_id"], match["section_id"])
        for match in matches
    } == {
        ("example-act-a", "2"),
        ("example-act-a", "2A"),
        ("example-act-b", "2"),
        ("example-act-a", "53-O"),
        ("example-act-b", "99"),
    }
    assert [match["match_type"] for match in matches] == [
        "direct",
        "direct",
        "direct",
        "related",
        "related",
    ]
    assert [match["related_via"] for match in matches[3:]] == [
        "references",
        "referenced_by",
    ]

    with pytest.raises(ValueError, match="document_id is required"):
        tools["kg_get_section"].invoke({"section_id": "2"})

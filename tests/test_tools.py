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

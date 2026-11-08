from indian_law_kg import agent


def test_build_agent_uses_injected_model(monkeypatch, kg):
    captured = {}

    def fake_create_agent(**kwargs):
        captured.update(kwargs)
        return "agent"

    monkeypatch.setattr(agent, "create_agent", fake_create_agent)

    injected = object()
    retriever = object()

    result = agent.build_agent(
        kg,
        kg.graph,
        chat_model=injected,
        retriever=retriever,
    )

    assert result == "agent"
    assert captured["model"] is injected
    assert [tool.name for tool in captured["tools"]][0] == "kg_get_schema"
    tool_names = {tool.name for tool in captured["tools"]}
    assert {
        "kg_list_documents",
        "kg_get_concepts",
        "kg_resolve_concepts",
        "kg_ontology_search",
        "kg_similar_provisions",
    } <= tool_names
    prompt = captured["system_prompt"]
    assert "document_id" in prompt
    assert "53-O" in prompt
    assert 'match_type="related"' in prompt
    assert "ontology_grounded" in prompt
    assert "keyword_fallback" in prompt
    assert "alias" in prompt
    assert "embedding" in prompt
    assert "graph-similar" in prompt
    assert "no synonym/alias data" not in prompt

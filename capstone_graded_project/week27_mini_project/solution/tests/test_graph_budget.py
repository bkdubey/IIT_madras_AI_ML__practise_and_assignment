from graph import PolicyGraph


def test_graph_respects_budget_and_retries_once():
    graph = PolicyGraph(max_steps=5)
    state = graph.execute("Is MFA required for VPN? Cite and quote.")
    assert state.steps <= 5
    assert state.stop_reason in {"completed", "memory_written", "critic_failed_after_retry"}

from tools import PolicyNotFoundError, read_policy, safe_calculator, policy_search


def test_read_policy_success():
    text = read_policy("remote_work_v9.9")
    assert "Remote work limit" in text
    assert "two days" in text.lower()


def test_read_policy_missing_raises_typed_error():
    try:
        read_policy("missing_policy")
        assert False, "Expected PolicyNotFoundError"
    except PolicyNotFoundError:
        pass


def test_policy_search_returns_relevant_docs():
    results = policy_search("remote work limit days", k=2)
    assert results[0].doc_id == "remote_work_v9.9"


def test_safe_calculator():
    assert safe_calculator("12 * 4") == 48.0

from agent import PolicyAssistantAgent


def test_answer_has_doc_id_and_quote():
    agent = PolicyAssistantAgent()
    answer = agent.answer_question("What is our remote work limit? Cite and quote.")
    assert "remote_work_v9.9" in answer["answer"].lower()
    assert '"' in answer["answer"]


def test_direct_doc_id_request_is_grounded():
    agent = PolicyAssistantAgent()
    answer = agent.answer_question("Read policy remote_work_v9.9 and answer the limit.")
    assert "remote_work_v9.9" in answer["answer"].lower()
    assert '"' in answer["answer"]

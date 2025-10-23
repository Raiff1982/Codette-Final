from cognitive_auth import CognitiveAuthManager


def test_register_validate_collapse():
    mgr = CognitiveAuthManager()
    cid = mgr.register_user(
        username="JonathanH", password="Quantum#2025", metadata={"role": "developer", "access": "full"}
    )
    assert cid == "cocoon_JonathanH"
    assert mgr.validate_user("JonathanH", "Quantum#2025")
    assert mgr.collapse_user_node("JonathanH")

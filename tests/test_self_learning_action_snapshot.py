from fly_self_learning import SelfLearningLoop


class DummyMind:
    def __init__(self):
        self.boredom = 0.2
        self.social_interest = 0.1
        self.state = {"world": []}

    def learn_observation(self, *args, **kwargs):
        return None

    def experience(self, *args, **kwargs):
        return None

    def get_state(self):
        return self.state


def test_on_action_result_uses_pre_and_post_facts_not_same_snapshot():
    mind = DummyMind()
    loop = SelfLearningLoop(mind)

    before = [{
        "category": "location",
        "name": "room",
        "details": "baseline",
        "importance": 0.8,
        "confidence": 0.9,
    }]
    after = [
        {
            "category": "location",
            "name": "room",
            "details": "baseline",
            "importance": 0.8,
            "confidence": 0.9,
        },
        {
            "category": "object",
            "name": "door",
            "details": "opened",
            "importance": 0.95,
            "confidence": 0.9,
        },
    ]

    loop._last_facts = before
    loop._scene_fn = lambda: "door in room"
    loop._ocr_fn = lambda: []
    loop._last_action = "look_around"

    seen = {}
    def fake_log_action(action, facts_before, facts_after):
        seen["action"] = action
        seen["before"] = facts_before
        seen["after"] = facts_after
        return 0.5

    loop.evaluator.log_action = fake_log_action

    loop.on_action_result("look_around", "saw a door", outcome_score=0.1)

    assert seen["action"] == "look_around"
    assert seen["before"] == before
    assert seen["after"] == after
    assert seen["after"] is not seen["before"]

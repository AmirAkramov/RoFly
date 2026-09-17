"""
fly_mind_learning_patch.py
==========================

Drop-in replacements / additions for FlyMind in fly_mind.py.

HOW TO APPLY
------------
1.  Copy fly_self_learning.py next to fly_mind.py.
2.  At the TOP of fly_mind.py, add:

        from fly_self_learning import SelfLearningLoop

3.  Inside FlyMind.__init__, after self.load(), add:

        # Self-learning engine
        self.learning = SelfLearningLoop(self)

4.  Replace the three methods below in FlyMind with these
    improved versions (search for the method names).

5.  In doomfly.py (or wherever the main loop is), after
    initialising FlyMind, call:

        mind.learning.start(
            capture_fn = lambda: capture_screen(),
            ocr_fn     = lambda: get_ocr_lines(),
            scene_fn   = lambda: semantic_vision.get_scene(),
        )

    And when chat arrives:

        result = mind.process(speaker, message)
        mind.learning.on_chat(speaker, message, result["intent"])

    And after a movement action finishes:

        mind.learning.on_action_result(action, result, score)

These are ADDITIONS.  The rest of fly_mind.py is unchanged.
"""


# ============================================================
# IMPROVED: experience()
# ============================================================
# Replace the existing experience() in FlyMind with this one.
# It adds:
#   - stronger reward shaping
#   - boredom suppression on positive experiences
#   - fatigue on very high activity

def experience(
    self,
    observation,
    action=None,
    result=None,
    reward=0.0,
):
    import time as _time

    memory = {
        "time":        _time.time(),
        "observation": observation,
        "action":      action,
        "result":      result,
        "reward":      reward,
    }

    self.memories.append(memory)

    if len(self.memories) > MAX_MEMORIES:
        self.memories = self.memories[-MAX_MEMORIES:]

    self.total_experiences += 1

    # ----------------------------------------------------
    # REWARD SHAPING
    # ----------------------------------------------------

    if reward > 0:
        # Positive: boost curiosity, suppress boredom
        self.curiosity = min(
            1.0, self.curiosity + reward * 0.04
        )
        self.boredom = max(
            0.0, self.boredom - reward * 0.08
        )
        self.social_interest = min(
            1.0, self.social_interest + reward * 0.015
        )

    elif reward < 0:
        # Negative: raise attention sharply
        self.attention = min(
            1.0, self.attention + abs(reward) * 0.06
        )
        self.curiosity = max(
            0.0, self.curiosity + reward * 0.02   # small dent
        )

    # Neutral: mild curiosity tick (fly is always learning)
    self.curiosity = min(
        1.0, self.curiosity + 0.002
    )

    # Fatigue: very frequent high-reward experiences
    # reduce the marginal gain (diminishing returns)
    if self.total_experiences % 50 == 0:
        self.energy = max(
            0.20, self.energy - 0.02
        )

    # Clamp everything
    for attr in ("energy", "curiosity", "attention",
                 "boredom", "social_interest"):
        val = getattr(self, attr)
        setattr(self, attr, max(0.0, min(1.0, val)))

    if self.total_experiences % 10 == 0:
        self.save()


# ============================================================
# IMPROVED: learn_observation()
# ============================================================
# Replace the existing learn_observation() in FlyMind with this.
# It adds:
#   - confidence weighting (from SceneUnderstanding facts)
#   - novelty bonus scales with importance
#   - remembers HOW MANY TIMES the fly has seen something

def learn_observation(
    self,
    category,
    name,
    details=None,
    importance=0.5,
    confidence=1.0,       # <-- new parameter
):
    change = self.awareness.observe(
        category,
        name,
        details,
        importance,
    )

    # Scale boosts by both importance and confidence
    scale = importance * confidence

    if change == "NEW":
        self.curiosity  = min(1.0, self.curiosity  + 0.10 * scale)
        self.attention  = min(1.0, self.attention  + 0.06 * scale)
        self.boredom    = max(0.0, self.boredom    * (1.0 - 0.15 * scale))

        self.experience(
            observation={
                "type":       "new_observation",
                "category":   category,
                "name":       name,
                "details":    details,
                "confidence": confidence,
            },
            action="learn",
            result="learned something new",
            reward=0.12 * scale,
        )

        print(
            "[LEARNING] NEW "
            f"{category}: {name} "
            f"(importance={importance:.2f})"
        )

    elif change == "CHANGED":
        self.attention = min(1.0, self.attention + 0.05 * scale)
        self.boredom   = max(0.0, self.boredom   * (1.0 - 0.10 * scale))

        self.experience(
            observation={
                "type":     "changed_observation",
                "category": category,
                "name":     name,
                "details":  details,
            },
            action="update_memory",
            result="noticed a change",
            reward=0.06 * scale,
        )

        print(
            "[LEARNING] UPDATED "
            f"{category}: {name}"
        )

    # SEEN_AGAIN — very mild attention boost, no reward spam
    else:
        self.attention = min(
            1.0, self.attention + 0.005 * scale
        )

    # Clamp
    for attr in ("curiosity", "attention", "boredom"):
        val = getattr(self, attr)
        setattr(self, attr, max(0.0, min(1.0, val)))

    return change


# ============================================================
# NEW: generate_spontaneous_speech_v2()
# ============================================================
# Add this method to FlyMind and call it instead of the old
# generate_spontaneous_speech() to get richer speech that
# draws on self-learning insights.

def generate_spontaneous_speech_v2(self):
    import random as _random

    # Use self-learning insight if available
    if hasattr(self, "learning"):
        facts = self.learning.snapshot_facts()
        insight = self.learning._generate_insight(facts)
        if insight:
            return insight

    # Frequent player shout-out
    if hasattr(self, "learning"):
        fp = self.learning.patterns.most_frequent_player()
        if fp:
            return f"I keep seeing {fp} around."

    # Fallback to original logic
    focus = self.awareness.get_focus()

    if not focus:
        return "I wonder what is around me."

    category = focus.get("category", "thing")
    name     = focus.get("name",     "something")

    templates = {
        "player":   [
            f"I noticed {name}.",
            f"{name} caught my attention.",
            f"I've been watching {name}.",
        ],
        "location": [
            "This area feels interesting.",
            "I've been exploring here.",
            f"I remember this {name}.",
        ],
        "sign":     [
            f"That sign says '{name}'.",
            f"I read '{name}' on a sign.",
        ],
        "object":   [
            f"There's a {name} here.",
            f"I keep noticing the {name}.",
        ],
    }

    options = templates.get(
        category,
        [f"I noticed {name}.", f"{name} is interesting."]
    )
    return _random.choice(options)


# ============================================================
# NEW: think_v2()
# ============================================================
# Replaces think() with a version that consults the
# CuriosityDriver from SelfLearningLoop.

def think_v2(self):
    self.awareness.update()

    # Following overrides everything
    if self.follow_target:
        return {
            "action": "follow",
            "target": self.follow_target,
            "reason": "following remembered player",
        }

    # Ask the self-learning curiosity driver
    if hasattr(self, "learning"):
        state   = self.get_state()
        summary = state.get("world", [])
        facts   = self.learning.snapshot_facts()
        target  = self.learning.curiosity_drv.pick_target(
            summary, facts, self.boredom
        )
        if target:
            return {
                "action": "investigate",
                "target": target.get("name"),
                "reason": (
                    f"curious about "
                    f"{target.get('category','thing')}: "
                    f"{target.get('name','?')}"
                ),
            }

    # Classic fallback logic
    focus = self.awareness.get_focus()
    if focus and focus.get("importance", 0.5) >= 0.8:
        return {
            "action": "investigate",
            "target": focus.get("name"),
            "reason": "something important caught attention",
        }

    if self.curiosity > 0.65 and self.energy > 0.25:
        return {
            "action": "explore",
            "target": None,
            "reason": "curiosity is high",
        }

    if self.boredom > 0.65:
        return {
            "action": "explore",
            "target": None,
            "reason": "boredom is high",
        }

    return {
        "action": "wait",
        "target": None,
        "reason": "no strong motivation",
    }

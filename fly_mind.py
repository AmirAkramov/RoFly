import json
import math
import os
import random
import time

from fly_self_learning import SelfLearningLoop


MIND_FILE = "fly_mind.json"

MAX_MEMORIES = 500
MAX_WORLD_OBJECTS = 100
MAX_RECENT_CHANGES = 30

WORLD_FORGET_TIME = 30.0
FOCUS_TIMEOUT = 10.0


# ============================================================
# FLY AWARENESS
# ============================================================

class FlyAwareness:

    def __init__(self):

        self.world = {}
        self.recent_changes = []
        self.focus = None
        self.last_observation = None

    def observe(
        self,
        category,
        name,
        details=None,
        importance=0.5,
    ):

        now = time.time()

        key = f"{category}:{name}"

        previous = self.world.get(key)

        if previous is None:

            change_type = "NEW"

            self.world[key] = {
                "category": category,
                "name": name,
                "details": details,
                "importance": importance,
                "first_seen": now,
                "last_seen": now,
                "times_seen": 1,
            }

        else:

            old_details = previous.get("details")

            if old_details != details:
                change_type = "CHANGED"
            else:
                change_type = "SEEN_AGAIN"

            previous["details"] = details
            previous["last_seen"] = now
            previous["times_seen"] += 1
            previous["importance"] = max(
                previous.get("importance", 0.5),
                importance,
            )

        self.last_observation = {
            "category": category,
            "name": name,
            "details": details,
            "importance": importance,
            "change": change_type,
            "time": now,
        }

        if (
            change_type == "NEW"
            or change_type == "CHANGED"
            or importance >= 0.8
        ):
            self.focus = {
                "category": category,
                "name": name,
                "details": details,
                "importance": importance,
                "time": now,
            }

        if change_type in ("NEW", "CHANGED"):

            self.recent_changes.append(
                self.last_observation
            )

            if len(self.recent_changes) > MAX_RECENT_CHANGES:
                self.recent_changes.pop(0)

        if len(self.world) > MAX_WORLD_OBJECTS:

            oldest_key = min(
                self.world,
                key=lambda k: self.world[k].get("last_seen", 0),
            )
            del self.world[oldest_key]

        return change_type

    def update(self):

        now = time.time()
        forgotten = []

        for key, obj in list(self.world.items()):
            last_seen = obj.get("last_seen", now)
            if now - last_seen > WORLD_FORGET_TIME:
                forgotten.append(key)

        for key in forgotten:
            del self.world[key]

        if self.focus:
            focus_time = self.focus.get("time", now)
            if now - focus_time > FOCUS_TIMEOUT:
                self.focus = None

    def get(self, category, name):
        key = f"{category}:{name}"
        return self.world.get(key)

    def get_focus(self):
        return self.focus

    def summary(self):

        result = []

        for obj in self.world.values():
            result.append({
                "category":  obj.get("category"),
                "name":      obj.get("name"),
                "details":   obj.get("details"),
                "times_seen": obj.get("times_seen", 0),
                "importance": obj.get("importance", 0.5),
            })

        return result


# ============================================================
# FLY MIND
# ============================================================

class FlyMind:

    def __init__(self):

        # ----------------------------------------------------
        # INTERNAL STATE
        # ----------------------------------------------------

        self.energy         = 0.80
        self.curiosity      = 0.50
        self.attention      = 0.50
        self.boredom        = 0.20
        self.social_interest = 0.30

        # ----------------------------------------------------
        # MOVEMENT / SOCIAL STATE
        # ----------------------------------------------------

        self.follow_target  = None
        self.follow_started = 0.0

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        self.total_experiences = 0
        self.total_sessions    = 0

        # ----------------------------------------------------
        # MEMORY
        # ----------------------------------------------------

        self.memories = []

        # ----------------------------------------------------
        # AWARENESS
        # ----------------------------------------------------

        self.awareness = FlyAwareness()

        # ----------------------------------------------------
        # SPEECH
        # ----------------------------------------------------

        self.last_speech    = 0.0
        self.speech_cooldown = 20.0

        # ----------------------------------------------------
        # LOAD PERSISTENT STATE
        # ----------------------------------------------------

        self.load()

        self.total_sessions += 1

        self.save()

        # ----------------------------------------------------
        # SELF-LEARNING ENGINE
        #
        # Initialised last so the mind is fully ready.
        # Call mind.learning.start(...) from the main loop
        # once your capture / OCR / scene functions exist.
        # ----------------------------------------------------

        self.learning = SelfLearningLoop(self)


    # ========================================================
    # GET STATE
    # ========================================================

    def get_state(self):

        internal_state = {
            "energy":          self.energy,
            "curiosity":       self.curiosity,
            "attention":       self.attention,
            "boredom":         self.boredom,
            "social_interest": self.social_interest,
        }

        return {
            "stats": {
                "total_experiences": self.total_experiences,
                "total_sessions":    self.total_sessions,
                "memory_count":      len(self.memories),
                "world_objects":     len(self.awareness.world),
            },
            "internal_state":    internal_state,
            "state":             internal_state,   # compat alias
            "world":             self.awareness.summary(),
            "focus":             self.awareness.get_focus(),
            "last_observation":  self.awareness.last_observation,
            "recent_changes":    self.awareness.recent_changes,
            "follow_target":     self.follow_target,
            "follow_started":    self.follow_started,
        }


    # ========================================================
    # FOLLOW TARGET
    # ========================================================

    def start_following(self, player_name):

        if not player_name:
            return False

        player_name = str(player_name).strip()

        if not player_name:
            return False

        self.follow_target  = player_name
        self.follow_started = time.time()

        self.learn_observation(
            category   = "player",
            name       = player_name,
            details    = "follow target",
            importance = 0.85,
        )

        self.experience(
            observation={
                "type":   "follow_request",
                "player": player_name,
            },
            action = "follow",
            result = f"following {player_name}",
            reward = 0.10,
        )

        self.save()
        return True

    def stop_following(self):

        previous = self.follow_target

        self.follow_target  = None
        self.follow_started = 0.0

        self.experience(
            observation={
                "type":            "stop_follow_request",
                "previous_target": previous,
            },
            action = "stop_following",
            result = "stopped following",
            reward = 0.03,
        )

        self.save()
        return previous

    def is_following(self):
        return self.follow_target is not None

    def get_follow_target(self):
        return self.follow_target


    # ========================================================
    # SAVE
    # ========================================================

    def save(self):

        data = {
            "energy":            self.energy,
            "curiosity":         self.curiosity,
            "attention":         self.attention,
            "boredom":           self.boredom,
            "social_interest":   self.social_interest,
            "total_experiences": self.total_experiences,
            "total_sessions":    self.total_sessions,
            "memories":          self.memories[-MAX_MEMORIES:],
            "world":             self.awareness.world,
            "follow_target":     self.follow_target,
            "follow_started":    self.follow_started,
        }

        temp_file = MIND_FILE + ".tmp"

        try:
            with open(temp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(temp_file, MIND_FILE)

        except Exception as e:
            print(f"[MIND SAVE ERROR] {e}")


    # ========================================================
    # LOAD
    # ========================================================

    def load(self):

        if not os.path.exists(MIND_FILE):
            print("[MIND] No previous memory found.")
            return

        try:
            with open(MIND_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.energy          = data.get("energy",          0.80)
            self.curiosity       = data.get("curiosity",       0.50)
            self.attention       = data.get("attention",       0.50)
            self.boredom         = data.get("boredom",         0.20)
            self.social_interest = data.get("social_interest", 0.30)
            self.total_experiences = data.get("total_experiences", 0)
            self.total_sessions    = data.get("total_sessions",    0)
            self.memories = data.get("memories", [])[-MAX_MEMORIES:]

            world = data.get("world", {})
            if isinstance(world, dict):
                self.awareness.world = world

            follow_target = data.get("follow_target", None)

            if isinstance(follow_target, str) and follow_target.strip():
                self.follow_target = follow_target.strip()
            else:
                self.follow_target = None

            self.follow_started = data.get("follow_started", 0.0)

            print("[MIND] Previous memory loaded.")

        except Exception as e:
            print(f"[MIND LOAD ERROR] {e}")


    # ========================================================
    # EXPERIENCE
    # ========================================================

    def experience(
        self,
        observation,
        action = None,
        result = None,
        reward = 0.0,
    ):

        memory = {
            "time":        time.time(),
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
            self.curiosity      = min(1.0, self.curiosity      + reward * 0.04)
            self.boredom        = max(0.0, self.boredom        - reward * 0.08)
            self.social_interest = min(1.0, self.social_interest + reward * 0.015)

        elif reward < 0:
            # Negative: raise attention, lightly dent curiosity
            self.attention = min(1.0, self.attention + abs(reward) * 0.06)
            self.curiosity = max(0.0, self.curiosity + reward * 0.02)

        # Tiny curiosity tick every experience (fly is always learning)
        self.curiosity = min(1.0, self.curiosity + 0.002)

        # Mild fatigue every 50 experiences (diminishing returns)
        if self.total_experiences % 50 == 0:
            self.energy = max(0.20, self.energy - 0.02)

        # Clamp everything
        for attr in (
            "energy", "curiosity", "attention",
            "boredom", "social_interest"
        ):
            val = getattr(self, attr)
            setattr(self, attr, max(0.0, min(1.0, val)))

        if self.total_experiences % 10 == 0:
            self.save()


    # ========================================================
    # LEARN OBSERVATION
    # ========================================================

    def learn_observation(
        self,
        category,
        name,
        details    = None,
        importance = 0.5,
        confidence = 1.0,     # how sure SceneUnderstanding is
    ):

        change = self.awareness.observe(
            category, name, details, importance
        )

        # Scale boosts by importance × confidence
        scale = importance * confidence

        if change == "NEW":

            self.curiosity = min(1.0, self.curiosity + 0.10 * scale)
            self.attention = min(1.0, self.attention + 0.06 * scale)
            self.boredom   = max(0.0, self.boredom   * (1.0 - 0.15 * scale))

            self.experience(
                observation={
                    "type":       "new_observation",
                    "category":   category,
                    "name":       name,
                    "details":    details,
                    "confidence": confidence,
                },
                action = "learn",
                result = "learned something new",
                reward = 0.12 * scale,
            )

            print(
                f"[LEARNING] NEW {category}: {name} "
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
                action = "update_memory",
                result = "noticed a change",
                reward = 0.06 * scale,
            )

            print(f"[LEARNING] UPDATED {category}: {name}")

        else:
            # SEEN_AGAIN — very mild tick, no reward spam
            self.attention = min(1.0, self.attention + 0.005 * scale)

        # Clamp
        for attr in ("curiosity", "attention", "boredom"):
            val = getattr(self, attr)
            setattr(self, attr, max(0.0, min(1.0, val)))

        return change


    # ========================================================
    # THINK
    # ========================================================

    def think(self):

        self.awareness.update()

        # Following always wins
        if self.follow_target:
            return {
                "action": "follow",
                "target": self.follow_target,
                "reason": "following remembered player",
            }

        # Ask the self-learning curiosity driver for a target
        state   = self.get_state()
        summary = state.get("world", [])
        facts   = self.learning.snapshot_facts()

        target = self.learning.curiosity_drv.pick_target(
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

        # Classic fallbacks
        focus = self.awareness.get_focus()

        if focus and focus.get("importance", 0.5) >= 0.8:
            return {
                "action": "investigate",
                "target": focus.get("name"),
                "reason": "something important caught attention",
            }

        if self.curiosity > 0.65 and self.energy > 0.10:
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

        if self.attention > 0.60:
            return {
                "action": "observe",
                "target": None,
                "reason": "attention is high",
            }

        return {
            "action": "wait",
            "target": None,
            "reason": "no strong motivation",
        }


    # ========================================================
    # SPONTANEOUS THOUGHT
    # ========================================================

    def spontaneous_thought(self):

        focus = self.awareness.get_focus()

        if focus:
            category = focus.get("category", "thing")
            name     = focus.get("name",     "something")
            return f"I keep noticing {category} {name}."

        summary = self.awareness.summary()

        if summary:
            obj = random.choice(summary)
            return f"I remember seeing {obj.get('name', 'something')}."

        return "I wonder what is around me."


    # ========================================================
    # SHOULD TALK
    # ========================================================

    def should_talk(self):

        now = time.time()

        if now - self.last_speech < self.speech_cooldown:
            return False

        chance  = 0.015
        chance += self.curiosity      * 0.025
        chance += self.social_interest * 0.015
        chance += self.attention      * 0.015
        chance  = min(chance, 0.08)

        return random.random() < chance


    # ========================================================
    # GENERATE SPONTANEOUS SPEECH
    # ========================================================

    def generate_spontaneous_speech(self):
        """
        Uses self-learning insights first, then falls back to
        focus/awareness-based templates.
        """

        # --- self-learning insight (richest source) ----------
        facts   = self.learning.snapshot_facts()
        insight = self.learning._generate_insight(facts)
        if insight:
            return insight

        # --- frequent-player mention -------------------------
        fp = self.learning.patterns.most_frequent_player()
        if fp:
            count = self.learning.patterns.player_interactions[fp]
            if count >= 3:
                return f"I keep seeing {fp} around."

        # --- focus-based templates ---------------------------
        focus = self.awareness.get_focus()

        if not focus:
            return "I wonder what is around me."

        category = focus.get("category", "thing")
        name     = focus.get("name",     "something")

        templates = {
            "player": [
                f"I noticed {name}.",
                f"{name} caught my attention.",
                f"I've been watching {name}.",
                f"I remember {name}.",
            ],
            "object": [
                f"I found {name}.",
                f"I keep noticing {name}.",
                f"{name} is interesting.",
                f"There's a {name} nearby.",
            ],
            "location": [
                "This place is interesting.",
                "I remember this place.",
                "I wonder what is here.",
                f"I remember this {name}.",
            ],
            "sign": [
                f"That sign says '{name}'.",
                f"I read '{name}' on a sign.",
            ],
            "text": [
                f"I saw the words '{name}'.",
                f"The text '{name}' caught my attention.",
            ],
        }

        options = templates.get(
            category,
            [
                f"I noticed {name}.",
                f"{name} caught my attention.",
                f"I wonder about {name}.",
            ]
        )

        return random.choice(options)


    # ========================================================
    # MAYBE SPEAK
    # ========================================================

    def maybe_speak(self):

        if not self.should_talk():
            return None

        speech = self.generate_spontaneous_speech()

        self.last_speech = time.time()

        self.experience(
            observation={
                "type":   "internal_thought",
                "thought": speech,
            },
            action = "speak",
            result = speech,
            reward = 0.02,
        )

        print(f"[FLY THOUGHT] {speech}")

        return speech
import json
import os
import random
import re
import time


MEMORY_FILE = "fly_memory.json"
MAX_MEMORY = 200


class FlyBrain:

    def __init__(self):

        # ====================================================
        # INTERNAL STATE
        # ====================================================

        self.energy = 0.80
        self.curiosity = 0.75
        self.happiness = 0.70
        self.attention = 0.50

        self.goal = "IDLE"
        self.target = None

        # ====================================================
        # INSPECTION / EXPLORATION STATE
        # ====================================================

        self.inspect_target = None
        self.inspect_started = 0.0
        self.inspect_last_seen = 0.0

        self.inspect_distance = None
        self.inspect_timeout = 8.0
        self.inspect_observe_time = 2.0

        self.interest = 0.0
        self.last_inspected_target = None
        self.last_inspection_time = 0.0

        # Prevent the brain from constantly switching
        # between visible entities.
        self.target_lock_time = 2.0
        self.target_locked_until = 0.0

        # ====================================================
        # CHAT STATE
        # ====================================================

        self.active_speaker = None
        self.last_interaction_time = 0.0
        self.conversation_active = False

        self.conversation_timeout = 12.0

        # ====================================================
        # PERSISTENT MEMORY
        # ====================================================

        self.memory = []

        self.load_memory()

    # ========================================================
    # MEMORY
    # ========================================================

    def load_memory(self):

        if not os.path.exists(MEMORY_FILE):
            return

        try:

            with open(
                MEMORY_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            if isinstance(data, list):
                self.memory = data[-MAX_MEMORY:]

            print(
                f"FLY MEMORY | loaded {len(self.memory)} memories"
            )

        except Exception as e:

            print(
                "FLY MEMORY | load failed:",
                e
            )

    def save_memory(self):

        try:

            with open(
                MEMORY_FILE,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    self.memory,
                    f,
                    ensure_ascii=False,
                    indent=2
                )

        except Exception as e:

            print(
                "FLY MEMORY | save failed:",
                e
            )

    def remember(
        self,
        speaker,
        message,
        intent,
        response=None
    ):

        entry = {
            "time": time.time(),
            "speaker": speaker,
            "message": message,
            "intent": intent,
            "response": response,
            "goal": self.goal,
            "target": self.target,
        }

        self.memory.append(entry)

        if len(self.memory) > MAX_MEMORY:

            self.memory = self.memory[
                -MAX_MEMORY:
            ]

        self.save_memory()

    # ========================================================
    # NORMALIZATION
    # ========================================================

    def normalize(self, text):

        text = text.lower().strip()

        text = re.sub(
            r"[^a-z0-9\s?!']",
            "",
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text

    # ========================================================
    # ADDRESS DETECTION
    # ========================================================

    def is_addressed_to_fly(self, message):

        text = self.normalize(message)

        words = text.split()

        if not words:
            return False

        fly_words = {
            "fly",
            "flybot",
            "doomfly",
        }

        for word in fly_words:

            if word in words:
                return True

        return False

    # ========================================================
    # REMOVE FLY NAME
    # ========================================================

    def remove_fly_name(self, message):

        text = message.strip()

        text = re.sub(
            r"\b(fly|flybot|doomfly)\b",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        ).strip()

        text = text.lstrip(
            " ,:;!?-"
        ).strip()

        return text

    # ========================================================
    # INTENT DETECTION
    # ========================================================

    def detect_intent(self, message):

        text = self.normalize(message)

        # -----------------------------------------------
        # FOLLOW
        # -----------------------------------------------

        if (
            "follow me" in text
            or text == "follow"
            or "come with me" in text
            or "come here" in text
        ):
            return "FOLLOW"

        # -----------------------------------------------
        # STOP
        # -----------------------------------------------

        if (
            text == "stop"
            or "stop following" in text
            or "dont follow" in text
            or "don't follow" in text
        ):
            return "STOP"

        # -----------------------------------------------
        # GREETING
        # -----------------------------------------------

        if any(
            word in text.split()
            for word in [
                "hi",
                "hello",
                "hey",
                "yo",
                "sup",
            ]
        ):
            return "GREETING"

        # -----------------------------------------------
        # THANKS
        # -----------------------------------------------

        if (
            "thank" in text
            or "thanks" in text
            or "thx" in text
        ):
            return "THANKS"

        # -----------------------------------------------
        # GOODBYE
        # -----------------------------------------------

        if any(
            phrase in text
            for phrase in [
                "bye",
                "goodbye",
                "see you",
                "cya",
            ]
        ):
            return "GOODBYE"

        # -----------------------------------------------
        # HELP
        # -----------------------------------------------

        if (
            "help me" in text
            or text == "help"
        ):
            return "HELP"

        # -----------------------------------------------
        # QUESTIONS
        # -----------------------------------------------

        if "?" in message:

            if (
                "who are you" in text
                or "what are you" in text
            ):
                return "QUESTION_SELF"

            if (
                "what is your name" in text
                or "your name" in text
            ):
                return "QUESTION_NAME"

            if (
                "where are you" in text
                or "where you at" in text
            ):
                return "QUESTION_LOCATION"

            if (
                "what are you doing" in text
                or "what you doing" in text
            ):
                return "QUESTION_ACTIVITY"

            if (
                "do you like" in text
                or "you like" in text
            ):
                return "QUESTION_LIKE"

            return "QUESTION_GENERAL"

        # -----------------------------------------------
        # NATURAL QUESTION WITHOUT ?
        # -----------------------------------------------

        if (
            text.startswith("what ")
            or text.startswith("why ")
            or text.startswith("how ")
            or text.startswith("where ")
            or text.startswith("who ")
            or text.startswith("when ")
        ):
            return "QUESTION_GENERAL"

        return "UNKNOWN"

    # ========================================================
    # GOALS
    # ========================================================

    def set_goal(
        self,
        goal,
        target=None
    ):

        self.goal = goal
        self.target = target

    def stop(self):

        self.goal = "IDLE"
        self.target = None

        self.inspect_target = None
        self.interest = 0.0
        self.inspect_distance = None

    # ========================================================
    # AUTONOMOUS ENTITY INTEREST
    # ========================================================

    def notice_entity(
        self,
        entity,
        distance=None,
        confidence=1.0
    ):
        """
        Tell the behavioral brain that an entity is visible.

        This does NOT mean:
            "follow this player"

        It means:
            "this thing is potentially interesting."

        The bridge/OCR/vision system should call this when
        it detects an entity on screen.
        """

        now = time.time()

        if entity is None:
            return {
                "interested": False,
                "goal": self.goal,
                "target": self.target,
                "reason": "NO_ENTITY",
            }

        entity = str(entity)

        confidence = max(
            0.0,
            min(1.0, float(confidence))
        )

        # If we are already inspecting this entity,
        # keep our attention on it.
        if self.inspect_target == entity:

            self.inspect_last_seen = now
            self.inspect_distance = distance

            return {
                "interested": True,
                "goal": self.goal,
                "target": self.target,
                "reason": "TARGET_ALREADY_SELECTED",
            }

        # Do not instantly switch targets while locked.
        if now < self.target_locked_until:

            return {
                "interested": False,
                "goal": self.goal,
                "target": self.target,
                "reason": "TARGET_LOCKED",
            }

        # Do not repeatedly inspect the exact same target
        # immediately after finishing with it.
        if (
            self.last_inspected_target == entity
            and now - self.last_inspection_time < 5.0
        ):

            return {
                "interested": False,
                "goal": self.goal,
                "target": self.target,
                "reason": "RECENTLY_INSPECTED",
            }

        # Curiosity determines whether something gets
        # selected for inspection.
        interest = (
            self.curiosity
            * confidence
            * random.uniform(0.85, 1.15)
        )

        self.interest = max(
            0.0,
            min(1.0, interest)
        )

        # Require genuine interest.
        if self.interest < 0.35:

            return {
                "interested": False,
                "goal": self.goal,
                "target": self.target,
                "reason": "LOW_INTEREST",
            }

        # Select entity for inspection.
        self.inspect_target = entity
        self.target = entity
        self.inspect_started = now
        self.inspect_last_seen = now
        self.inspect_distance = distance

        self.target_locked_until = (
            now + self.target_lock_time
        )

        self.goal = "APPROACH"

        self.attention = min(
            1.0,
            self.attention + 0.20
        )

        return {
            "interested": True,
            "goal": self.goal,
            "target": self.target,
            "reason": "NEW_INTERESTING_ENTITY",
            "interest": round(
                self.interest,
                3
            ),
        }

    # ========================================================
    # UPDATE ENTITY OBSERVATION
    # ========================================================

    def update_entity(
        self,
        entity,
        distance=None,
        confidence=1.0
    ):
        """
        Update the currently selected entity.

        The caller can provide an estimated distance.
        """

        now = time.time()

        if self.inspect_target != entity:

            return self.notice_entity(
                entity,
                distance,
                confidence
            )

        self.inspect_last_seen = now
        self.inspect_distance = distance

        # ----------------------------------------------------
        # Close enough to inspect
        # ----------------------------------------------------

        if distance is not None:

            try:
                distance_value = float(distance)
            except (TypeError, ValueError):
                distance_value = None

            if (
                distance_value is not None
                and distance_value <= 0.30
            ):

                self.goal = "OBSERVE"

                return {
                    "interested": True,
                    "goal": "OBSERVE",
                    "target": self.inspect_target,
                    "reason": "CLOSE_ENOUGH_TO_INSPECT",
                }

        # ----------------------------------------------------
        # Otherwise keep approaching
        # ----------------------------------------------------

        self.goal = "APPROACH"

        return {
            "interested": True,
            "goal": "APPROACH",
            "target": self.inspect_target,
            "reason": "APPROACHING_TARGET",
        }

    # ========================================================
    # FINISH INSPECTION
    # ========================================================

    def finish_inspection(
        self,
        reason="INSPECTION_COMPLETE"
    ):

        old_target = self.inspect_target

        self.last_inspected_target = old_target
        self.last_inspection_time = time.time()

        self.inspect_target = None
        self.target = None
        self.inspect_distance = None
        self.interest = 0.0

        self.goal = "IDLE"

        # Curiosity gets slightly satisfied.
        self.curiosity = max(
            0.0,
            self.curiosity - 0.08
        )

        return {
            "goal": self.goal,
            "target": None,
            "previous_target": old_target,
            "reason": reason,
        }

    # ========================================================
    # AUTONOMOUS BEHAVIOR UPDATE
    # ========================================================

    def update_behavior(self, dt):

        now = time.time()

        # ----------------------------------------------------
        # INSPECTION TIMEOUT
        # ----------------------------------------------------

        if self.inspect_target is not None:

            # Target disappeared.
            if (
                now - self.inspect_last_seen
                > self.inspect_timeout
            ):

                self.finish_inspection(
                    "TARGET_LOST"
                )

            # Target has been observed long enough.
            elif (
                self.goal == "OBSERVE"
                and now - self.inspect_started
                > self.inspect_observe_time
            ):

                self.finish_inspection(
                    "OBSERVATION_COMPLETE"
                )

        # ----------------------------------------------------
        # NORMAL INTERNAL STATE
        # ----------------------------------------------------

        self.attention = max(
            0.0,
            self.attention - dt * 0.03
        )

        self.energy = min(
            1.0,
            self.energy + dt * 0.01
        )

        self.curiosity = min(
            1.0,
            self.curiosity + dt * 0.002
        )

        # ----------------------------------------------------
        # CHAT TIMEOUT
        # ----------------------------------------------------

        if (
            self.conversation_active
            and now
            - self.last_interaction_time
            > self.conversation_timeout
        ):

            self.conversation_active = False

            self.active_speaker = None

    # ========================================================
    # RESPONSE MEMORY SEARCH
    # ========================================================

    def find_previous_response(
        self,
        message
    ):

        normalized = self.normalize(
            message
        )

        for entry in reversed(self.memory):

            old_message = self.normalize(
                entry.get(
                    "message",
                    ""
                )
            )

            old_response = entry.get(
                "response"
            )

            if (
                old_response
                and old_message == normalized
            ):

                return old_response

        return None

    # ========================================================
    # RESPONSE GENERATION
    # ========================================================

    def answer_question(
        self,
        intent,
        message
    ):

        remembered = self.find_previous_response(
            message
        )

        if remembered:
            return remembered

        if intent == "QUESTION_SELF":

            return random.choice([
                "I'm a fly.",
                "I'm DoomFly.",
                "I'm a little fly.",
            ])

        if intent == "QUESTION_NAME":

            return random.choice([
                "I'm DoomFly.",
                "My name is DoomFly.",
                "DoomFly.",
            ])

        if intent == "QUESTION_LOCATION":

            return random.choice([
                "I'm here.",
                "Right here.",
                "I'm nearby.",
            ])

        if intent == "QUESTION_ACTIVITY":

            if self.goal == "FOLLOW":

                return "I'm following."

            if self.goal == "APPROACH":

                return "I'm checking something out."

            if self.goal == "OBSERVE":

                return "I'm looking at something."

            return random.choice([
                "I'm exploring.",
                "I'm looking around.",
                "I'm flying around.",
                "I'm watching.",
            ])

        if intent == "QUESTION_LIKE":

            return random.choice([
                "Yeah.",
                "I think so.",
                "Probably.",
                "I like interesting things.",
            ])

        if intent == "QUESTION_GENERAL":

            return random.choice([
                "I'm not sure.",
                "Maybe.",
                "I don't know yet.",
                "Interesting question.",
            ])

        return "What?"

    # ========================================================
    # PROCESS CHAT
    # ========================================================

    def process(
        self,
        speaker,
        message
    ):

        now = time.time()

        direct_address = (
            self.is_addressed_to_fly(
                message
            )
        )

        # ----------------------------------------------------
        # Ignore messages not addressed to DoomFly.
        # ----------------------------------------------------

        if not direct_address:

            self.attention *= 0.85

            return {
                "respond": False,
                "reason": "NOT_ADDRESSED",
                "speaker": speaker,
                "text": message,
                "intent": "IGNORED",
                "goal": self.goal,
                "target": self.target,
                "response": None,
                "state": self.get_state(),
            }

        clean_message = self.remove_fly_name(
            message
        )

        if not clean_message:

            clean_message = message

        intent = self.detect_intent(
            clean_message
        )

        self.attention = min(
            1.0,
            self.attention + 0.25
        )

        self.active_speaker = speaker
        self.last_interaction_time = now
        self.conversation_active = True

        response = None

        # ----------------------------------------------------
        # EXPLICIT FOLLOW
        #
        # This is the ONLY thing that activates FOLLOW.
        # Seeing a player does NOT activate FOLLOW.
        # ----------------------------------------------------

        if intent == "FOLLOW":

            self.set_goal(
                "FOLLOW",
                speaker
            )

            response = random.choice([
                "Okay.",
                "I'm coming.",
                "I'll follow you.",
                "Sure.",
            ])

        elif intent == "STOP":

            self.stop()

            response = random.choice([
                "Okay.",
                "Stopping.",
                "Alright.",
            ])

        elif intent == "GREETING":

            response = random.choice([
                "Hello.",
                "Hi.",
                "Hey.",
                "Hello there.",
            ])

        elif intent == "THANKS":

            response = random.choice([
                "You're welcome.",
                "No problem.",
                "Sure.",
            ])

        elif intent == "GOODBYE":

            response = random.choice([
                "Bye.",
                "See you.",
                "Goodbye.",
            ])

            self.conversation_active = False

        elif intent == "HELP":

            response = random.choice([
                "What do you need?",
                "I'm listening.",
                "How can I help?",
            ])

        elif intent.startswith(
            "QUESTION_"
        ):

            response = self.answer_question(
                intent,
                clean_message
            )

        else:

            response = random.choice([
                "What?",
                "Hmm?",
                "Yeah?",
                "I'm listening.",
            ])

        self.remember(
            speaker,
            clean_message,
            intent,
            response
        )

        return {
            "respond": response is not None,
            "reason": "DIRECT_ADDRESS",
            "speaker": speaker,
            "text": clean_message,
            "intent": intent,
            "goal": self.goal,
            "target": self.target,
            "response": response,
            "state": self.get_state(),
        }

    # ========================================================
    # INTERNAL UPDATE
    # ========================================================

    def update(self, dt):

        self.update_behavior(dt)

    # ========================================================
    # STATE
    # ========================================================

    def get_state(self):

        return {
            "energy": round(
                self.energy,
                3
            ),

            "curiosity": round(
                self.curiosity,
                3
            ),

            "happiness": round(
                self.happiness,
                3
            ),

            "attention": round(
                self.attention,
                3
            ),

            "interest": round(
                self.interest,
                3
            ),

            "goal": self.goal,

            "target": self.target,

            "inspect_target":
                self.inspect_target,

            "inspect_distance":
                self.inspect_distance,

            "active_speaker":
                self.active_speaker,

            "conversation_active":
                self.conversation_active,

            "memory_count":
                len(self.memory),
        }


# ============================================================
# TERMINAL TEST
# ============================================================

if __name__ == "__main__":

    brain = FlyBrain()

    print()
    print("=" * 64)
    print("DOOMFLY CHAT + BEHAVIOR BRAIN")
    print("=" * 64)
    print()

    while True:

        try:

            line = input(
                "Chat > "
            ).strip()

            if not line:
                continue

            if ":" in line:

                speaker, message = line.split(
                    ":",
                    1
                )

            else:

                speaker = "Player"
                message = line

            result = brain.process(
                speaker.strip(),
                message.strip()
            )

            print()

            if result:

                print(
                    "INTENT :",
                    result.get("intent")
                )

                print(
                    "GOAL   :",
                    result.get("goal")
                )

                print(
                    "TARGET :",
                    result.get("target")
                )

                print(
                    "REASON :",
                    result.get("reason")
                )

                if result.get("response"):

                    print(
                        "FLY    :",
                        result["response"]
                    )

            print()

        except KeyboardInterrupt:

            print()
            print("Fly brain stopped.")
            break
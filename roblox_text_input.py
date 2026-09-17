import json
import os
import random
import re
import time


# ============================================================
# CONFIGURATION
# ============================================================

FLY_NAMES = {
    "fly",
    "f1y",
    "bug",
    "insect",
    "little fly",
}

MIND_FILE = "fly_text_memory.json"

MAX_MEMORIES = 200
MAX_RECENT_CHANGES = 100
MAX_WORLD_OBJECTS = 100

WORLD_FORGET_TIME = 300
FOCUS_TIMEOUT = 8.0

# NOTE:
# This file is a text-command processor for chat input only. The active
# Roblox launcher uses the canonical runtime brains in fly_brain.py and
# fly_mind.py; this helper intentionally does not replace those classes.


# ============================================================
# HELPERS
# ============================================================

def normalize(text):
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def detect_call_name(text):
    """
    Returns the fly name used to call the fly.

    The fly must be explicitly called in the message.
    """

    normalized = normalize(text)

    for name in sorted(FLY_NAMES, key=len, reverse=True):
        pattern = rf"\b{re.escape(name)}\b"

        if re.search(pattern, normalized):
            return name

    return None


def is_back_fire_command(text):
    """
    BACK FIRE requires both:
        fly
        back fire

    in the same message.
    """

    normalized = normalize(text)

    return bool(
        re.search(r"\bback\s+fire\b", normalized)
    )


# ============================================================
# FLY AWARENESS
# ============================================================

class TextFlyAwareness:

    def __init__(self):
        self.focus = None
        self.focus_time = 0.0

        self.world_objects = {}
        self.recent_changes = []

        self.load()

    # --------------------------------------------------------
    # MEMORY
    # --------------------------------------------------------

    def load(self):
        if not os.path.exists(MIND_FILE):
            return

        try:
            with open(
                MIND_FILE,
                "r",
                encoding="utf-8"
            ) as f:
                data = json.load(f)

            self.focus = data.get("focus")

            self.world_objects = data.get(
                "world_objects",
                {}
            )

            self.recent_changes = data.get(
                "recent_changes",
                []
            )

        except Exception:
            self.focus = None
            self.world_objects = {}
            self.recent_changes = []

    def save(self):

        data = {
            "focus": self.focus,
            "world_objects": self.world_objects,
            "recent_changes": self.recent_changes,
        }

        try:
            with open(
                MIND_FILE,
                "w",
                encoding="utf-8"
            ) as f:
                json.dump(
                    data,
                    f,
                    indent=2,
                    ensure_ascii=False
                )

        except Exception as exc:
            print(
                "[TEXT] Memory save failed:",
                exc
            )

    # --------------------------------------------------------
    # FOCUS
    # --------------------------------------------------------

    def set_focus(self, target):

        self.focus = target
        self.focus_time = time.time()

        self.save()

    def clear_focus(self):

        self.focus = None
        self.focus_time = 0.0

        self.save()

    def get_focus(self):

        if self.focus is None:
            return None

        if time.time() - self.focus_time > FOCUS_TIMEOUT:
            return None

        return self.focus


# ============================================================
# TEXT COMMAND MIND
# ============================================================
# This is a separate text-driven follow / command helper. It is not the
# canonical FlyMind used by the active launcher; the runtime continues to
# use fly_mind.py:FlyMind.

class TextFlyMind:

    def __init__(self):

        self.awareness = TextFlyAwareness()

        self.follow_target = None
        self.follow_started = None

        self.load()

    # --------------------------------------------------------
    # PERSISTENCE
    # --------------------------------------------------------

    def load(self):

        if not os.path.exists(MIND_FILE):
            return

        try:

            with open(
                MIND_FILE,
                "r",
                encoding="utf-8"
            ) as f:

                data = json.load(f)

            self.follow_target = data.get(
                "follow_target"
            )

            self.follow_started = data.get(
                "follow_started"
            )

        except Exception:
            self.follow_target = None
            self.follow_started = None

    def save(self):

        data = {
            "follow_target": self.follow_target,
            "follow_started": self.follow_started,
        }

        try:

            with open(
                MIND_FILE,
                "w",
                encoding="utf-8"
            ) as f:

                json.dump(
                    data,
                    f,
                    indent=2,
                    ensure_ascii=False
                )

        except Exception as exc:

            print(
                "[TEXT] Follow save failed:",
                exc
            )

    # --------------------------------------------------------
    # FOLLOW
    # --------------------------------------------------------

    def start_following(self, target):

        self.follow_target = target
        self.follow_started = time.time()

        self.awareness.set_focus(target)

        self.save()

        print(
            f"[MIND] Following {target}"
        )

    def stop_following(self):

        if self.follow_target:

            print(
                f"[MIND] Stopped following "
                f"{self.follow_target}"
            )

        self.follow_target = None
        self.follow_started = None

        self.awareness.clear_focus()

        self.save()

    def is_following(self):

        return self.follow_target is not None

    def get_follow_target(self):

        return self.follow_target


# Compatibility alias kept for older imports. The active launcher still uses
# fly_mind.FlyMind and fly_brain.FlyBrain, not this text-command helper.
FlyAwareness = TextFlyAwareness
FlyMind = TextFlyMind


# ============================================================
# MESSAGE PROCESSOR
# ============================================================

def process_message(
    message,
    speaker="Unknown",
    mind=None
):

    if message is None:
        return {
            "called": False,
            "called_name": None,
            "speaker": speaker,
            "message": "",
            "command": None,
            "intent": "IGNORED",
            "response": None,
            "movement_event": None,
            "target": None,
        }

    message = str(message).strip()

    if not message:
        return {
            "called": False,
            "called_name": None,
            "speaker": speaker,
            "message": "",
            "command": None,
            "intent": "IGNORED",
            "response": None,
            "movement_event": None,
            "target": None,
        }

    normalized = normalize(message)

    # --------------------------------------------------------
    # CALL NAME CHECK
    # --------------------------------------------------------

    called_name = detect_call_name(normalized)

    if called_name is None:

        return {
            "called": False,
            "called_name": None,
            "speaker": speaker,
            "message": message,
            "command": None,
            "intent": "IGNORED",
            "response": None,
            "movement_event": None,
            "target": None,
        }

    # --------------------------------------------------------
    # BACK FIRE
    # --------------------------------------------------------

    if is_back_fire_command(normalized):

        print(
            f"[TEXT] BACK FIRE command from "
            f"{speaker}"
        )

        return {
            "called": True,
            "called_name": called_name,
            "speaker": speaker,
            "message": message,
            "command": "BACK_FIRE",
            "intent": "BACK_FIRE",
            "response": None,
            "movement_event": "BACK_FIRE",
            "target": None,
        }

    # --------------------------------------------------------
    # STOP
    # --------------------------------------------------------

    if re.search(
        r"\b(stop|stay|halt)\b",
        normalized
    ):

        if mind is not None:
            mind.stop_following()

        return {
            "called": True,
            "called_name": called_name,
            "speaker": speaker,
            "message": message,
            "command": "STOP",
            "intent": "STOP",
            "response": "Okay.",
            "movement_event": "STOP",
            "target": None,
        }

    # --------------------------------------------------------
    # FOLLOW
    # --------------------------------------------------------

    if re.search(
        r"\b(follow|come with me|come along)\b",
        normalized
    ):

        if mind is not None:
            mind.start_following(speaker)

        return {
            "called": True,
            "called_name": called_name,
            "speaker": speaker,
            "message": message,
            "command": "FOLLOW",
            "intent": "FOLLOW",
            "response": "Okay.",
            "movement_event": "FOLLOW",
            "target": speaker,
        }

    # --------------------------------------------------------
    # GREETING
    # --------------------------------------------------------

    if re.search(
        r"\b(hi|hello|hey|yo)\b",
        normalized
    ):

        return {
            "called": True,
            "called_name": called_name,
            "speaker": speaker,
            "message": message,
            "command": "TALK",
            "intent": "GREETING",
            "response": "Hello.",
            "movement_event": "TALK",
            "target": speaker,
        }

    # --------------------------------------------------------
    # WHAT DO YOU SEE?
    # --------------------------------------------------------

    if (
        "what do you see" in normalized
        or "what can you see" in normalized
        or "what do you see around" in normalized
    ):

        return {
            "called": True,
            "called_name": called_name,
            "speaker": speaker,
            "message": message,
            "command": "TALK",
            "intent": "QUESTION_SEE",
            "response": None,
            "movement_event": "TALK",
            "target": speaker,
        }

    # --------------------------------------------------------
    # NAME
    # --------------------------------------------------------

    if (
        "your name" in normalized
        or "who are you" in normalized
    ):

        return {
            "called": True,
            "called_name": called_name,
            "speaker": speaker,
            "message": message,
            "command": "TALK",
            "intent": "QUESTION_NAME",
            "response": "I'm the fly.",
            "movement_event": "TALK",
            "target": speaker,
        }

    # --------------------------------------------------------
    # DEFAULT CALLED MESSAGE
    # --------------------------------------------------------

    return {
        "called": True,
        "called_name": called_name,
        "speaker": speaker,
        "message": message,
        "command": "TALK",
        "intent": "UNKNOWN",
        "response": f"I hear you, {speaker}.",
        "movement_event": "TALK",
        "target": speaker,
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    tests = [
        ("fly back fire", "Player1"),
        ("hey fly back fire", "Player2"),
        ("back fire", "Player3"),
        ("hello everyone", "Player4"),
        ("fly follow me", "Player5"),
        ("fly stop", "Player5"),
    ]

    for text, speaker in tests:

        result = process_message(
            text,
            speaker
        )

        print()
        print("MESSAGE:", text)
        print("RESULT:", result)
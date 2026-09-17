"""Roblox social companion.

This process owns OCR and server-wide conversation only. It never presses
movement or camera keys; roblox_bridge.py remains the only movement controller.

Important runtime detail:
- roblox_social_bridge.py creates its own FlyMind and FlyBrain objects
- roblox_bridge.py creates a separate FlyBrain object in a different process
- neither process shares live memory, goals, attention, or reward state
  in-process; they only persist data via files and commands
"""

import difflib
import os
import random
import re
import shutil
import time
import ctypes

import mss
import numpy as np
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
from pynput.keyboard import Controller, Key, KeyCode

from fly_brain import FlyBrain
from fly_mind import FlyMind
from semantic_vision import SemanticVision
from player_behavior import PlayerBehavior
from roblox_bridge_commands import send_approach, send_say, send_stop


CHAT_REGION = {
    "left": 13,
    "top": 100,
    "width": 458,
    "height": 271,
}

OCR_INTERVAL = 0.5
MESSAGE_MEMORY = 25.0
SELF_MESSAGE_MEMORY = 8.0
RESPONSE_COOLDOWN = 3.0
SERVER_QUESTION_INTERVAL = 75.0
SERVER_QUESTION_INITIAL_DELAY = 30.0
SERVER_QUESTION_IDLE_DELAY = 120.0
STARTUP_GREETING = "Hi everyone, I'm awake."
STARTUP_GREETING_RETRY = 3.0

# Player detection was fully built (SemanticVision + PlayerBehavior)
# but nothing ever called it from this process, so roblox_bridge.py's
# "approach" command was permanently unused and the fly never moved
# toward or greeted anyone.
PLAYER_SCAN_INTERVAL = 1.0

TESSERACT_CANDIDATES = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    "tesseract",
)

OCR_AVAILABLE = False
for tesseract_path in TESSERACT_CANDIDATES:
    if tesseract_path == "tesseract":
        if shutil.which(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            OCR_AVAILABLE = True
            break
    elif os.path.exists(tesseract_path):
        pytesseract.pytesseract.tesseract_cmd = tesseract_path
        OCR_AVAILABLE = True
        break


sct = mss.MSS()
monitor = sct.monitors[1]
keyboard = Controller()
# Separate-process runtime state: these instances are local to the social
# OCR process and are not shared with roblox_bridge.py's movement brain.
mind = FlyMind()
chat_brain = FlyBrain()
semantic_vision = SemanticVision()
player_behavior = PlayerBehavior()

seen_messages = {}
outgoing_messages = []
last_ocr = 0.0
ocr_ignore_until = 0.0
last_player_scan = 0.0
last_server_question = time.time() + SERVER_QUESTION_INITIAL_DELAY
last_chat_seen = time.time()
last_response_time = 0.0
recent_incoming = []
startup_greeting_sent = False
last_greeting_attempt = 0.0


def roblox_is_foreground():
    """Prevent OCR from reading the launcher console instead of Roblox."""

    if os.name != "nt":
        return True

    user32 = ctypes.windll.user32
    window = user32.GetForegroundWindow()
    if not window:
        return False

    title = ctypes.create_unicode_buffer(512)
    user32.GetWindowTextW(window, title, len(title))
    return "roblox" in title.value.lower()


def capture_full():
    shot = sct.grab(monitor)
    return np.asarray(shot)[:, :, :3][:, :, ::-1]


def clean_text(text):
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text.strip("|[]{}")


def read_chat():
    if not OCR_AVAILABLE or not roblox_is_foreground():
        return ""

    try:
        frame = capture_full()
        crop = frame[
            CHAT_REGION["top"]:CHAT_REGION["top"] + CHAT_REGION["height"],
            CHAT_REGION["left"]:CHAT_REGION["left"] + CHAT_REGION["width"],
        ]
        image = Image.fromarray(crop).convert("L")
        image = ImageEnhance.Contrast(image).enhance(2.5)
        image = image.resize((image.width * 3, image.height * 3))
        image = ImageEnhance.Sharpness(image).enhance(2.0)
        text = pytesseract.image_to_string(image, config="--psm 6")
        if ":" in text:
            return text

        # Roblox UI scale can move the chat panel away from the fixed crop.
        # Retry the whole foreground game window when the crop has no chat rows.
        full = Image.fromarray(frame).convert("L")
        full = ImageEnhance.Contrast(full).enhance(1.8)
        full = full.resize((full.width // 2, full.height // 2))
        full = ImageEnhance.Sharpness(full).enhance(1.8)
        return pytesseract.image_to_string(full, config="--psm 11")
    except Exception as exc:
        print(f"\n[OCR ERROR] {exc}")
        return ""


def parse_chat(text):
    messages = []
    for raw in str(text).splitlines():
        line = clean_text(raw)
        lowered = line.lower()
        if (
            lowered.startswith("ps ")
            or "categoryinfo" in lowered
            or "fullyqualifiederrorid" in lowered
            or "objectnotfound" in lowered
            or re.search(r"^[a-z]:\\", lowered)
            or "python run_roblox_social" in lowered
        ):
            continue
        match = re.match(r"^([^:]{1,32}):\s*(.+)$", line)
        if match:
            speaker = clean_text(match.group(1))
            message = clean_text(match.group(2))
            if speaker and message:
                messages.append((speaker, message))
    return messages


def _normalize_for_dedup(text):
    """Strip everything but letters/digits so OCR noise in spacing,
    punctuation, or brackets (e.g. 'the real one] |tzz_Ninokii' vs
    '1tzz_Ninokii' vs 'therealonetzz_Ninokii') doesn't defeat matching."""

    return re.sub(r"[^a-z0-9]", "", text.lower())


def remember_outgoing(message):
    now = time.time()
    outgoing_messages.append((_normalize_for_dedup(message), now))
    outgoing_messages[:] = [
        item for item in outgoing_messages
        if now - item[1] <= SELF_MESSAGE_MEMORY
    ]


def is_our_message(message):
    value = _normalize_for_dedup(message)
    return any(
        value == previous or difflib.SequenceMatcher(
            None, value, previous
        ).ratio() >= 0.80
        for previous, timestamp in outgoing_messages
        if time.time() - timestamp <= SELF_MESSAGE_MEMORY
    )


def is_duplicate_incoming(speaker, message):
    value = _normalize_for_dedup(message)
    speaker_key = _normalize_for_dedup(speaker)
    now = time.time()

    recent_incoming[:] = [
        item for item in recent_incoming
        if now - item[2] <= MESSAGE_MEMORY
    ]

    for previous_speaker, previous_message, _ in recent_incoming:

        # Fuzzy-match the speaker too -- OCR jitters usernames just
        # as much as message text, so an exact-match requirement here
        # let the same real message from the same real player through
        # repeatedly under a slightly different "speaker" each time.
        speaker_matches = (
            speaker_key == previous_speaker
            or difflib.SequenceMatcher(
                None, speaker_key, previous_speaker
            ).ratio() >= 0.70
        )

        if not speaker_matches:
            continue

        if value == previous_message:
            return True

        if difflib.SequenceMatcher(
            None,
            value,
            previous_message,
        ).ratio() >= 0.70:
            return True

    recent_incoming.append((speaker_key, value, now))
    return False


def type_chat(message):
    global ocr_ignore_until
    message = str(message).strip()
    if not message:
        return False

    print(f"\n[TALK] Fly: {message}")
    send_say(message, pause=4.0)
    remember_outgoing(message)
    ocr_ignore_until = time.time() + 1.0
    mind.last_speech = time.time()
    return True


def handle_chat(speaker, message):
    global last_response_time

    result = chat_brain.process(speaker, message)
    if not result or not result.get("respond"):
        return False

    if time.time() - last_response_time < RESPONSE_COOLDOWN:
        return False

    send_stop(duration=4.0)
    try:
        response = str(result.get("response") or "").strip()
        if response and type_chat(response):
            mind.experience(
                observation={
                    "type": "chat",
                    "speaker": speaker,
                    "message": message,
                    "response": response,
                    "intent": result.get("intent"),
                },
                action="respond_to_chat",
                result="answered player",
                reward=0.2,
            )
            last_response_time = time.time()
            return True
    finally:
        # The bridge resumes automatically when the timed pause expires.
        pass

    return False


def ask_server_question():
    global last_server_question, last_response_time

    if not roblox_is_foreground():
        return

    state = mind.get_state().get("internal_state", {})
    bored = float(state.get("boredom", 0.0)) >= 0.65
    quiet = time.time() - last_chat_seen >= SERVER_QUESTION_IDLE_DELAY
    if not bored and not quiet:
        return

    if time.time() - last_response_time < RESPONSE_COOLDOWN:
        return

    question = random.choice([
        "Can anyone explain what is happening here?",
        "Does anyone know how I should move from here?",
        "Can someone tell me what I should try next?",
        "I am not sure what to do here. Can anyone help?",
    ])
    send_stop(duration=4.0)
    if type_chat(question):
        last_server_question = time.time()
        last_response_time = last_server_question
        mind.experience(
            observation={
                "type": "server_question",
                "message": question,
            },
            action="ask_server_question",
            result="asked the server",
            reward=0.1,
        )


def greet_server_once():
    global startup_greeting_sent, last_response_time, last_greeting_attempt

    now = time.time()
    if (
        startup_greeting_sent
        or not roblox_is_foreground()
        or now - last_greeting_attempt < STARTUP_GREETING_RETRY
    ):
        return

    last_greeting_attempt = now
    print("\n[STARTUP] Roblox focused; sending one greeting.")
    send_stop(duration=3.0)
    if type_chat(STARTUP_GREETING):
        startup_greeting_sent = True
        last_response_time = time.time()
        mind.experience(
            observation={
                "type": "startup_greeting",
                "message": STARTUP_GREETING,
            },
            action="greet_server",
            result="sent startup greeting",
            reward=0.1,
        )
        print("\n[STARTUP] Greeted the server once.")


def _horizontal_position_from(player):
    """Mirror PlayerBehavior.get_horizontal_position()'s conversion
    without touching PlayerBehavior's own target/state bookkeeping,
    since a sustained FOLLOW shouldn't be subject to its short
    approach_timeout/greet-cooldown logic."""

    for key in ("x", "center_x", "screen_x"):
        value = player.get(key)

        if isinstance(value, (int, float)):
            x = float(value)

            if x > 1.0:
                x = (x / 640.0) * 2.0 - 1.0
            else:
                x = (x - 0.5) * 2.0

            return max(-1.0, min(1.0, x))

    return 0.0


def scan_players():
    """
    Detect nearby players and, if one is worth approaching, tell
    roblox_bridge.py via the file-based command channel. This process
    never touches movement/camera keys itself -- roblox_bridge.py is
    the only thing that presses keys, per this file's own docstring.
    """

    if not roblox_is_foreground():
        return

    try:
        frame = capture_full()
        detected_players = semantic_vision.detect_players(frame)
    except Exception as exc:
        print(f"\n[PLAYER SCAN ERROR] {exc}")
        return

    # An explicit "fly follow me" set chat_brain.goal to FOLLOW and
    # the bot already said "Okay, I'm coming" -- this makes that
    # true instead of just verbal. Takes priority over the brief
    # greet-and-leave social approach below, and persists (no
    # approach_timeout) until the player says "fly stop".
    if chat_brain.goal == "FOLLOW" and chat_brain.target:

        follow_name = str(chat_brain.target).strip().lower()

        for candidate in detected_players:

            name = str(
                candidate.get("name")
                or candidate.get("player")
                or candidate.get("username")
                or ""
            ).strip().lower()

            if name and name == follow_name:

                send_approach(
                    x=_horizontal_position_from(candidate),
                    distance=candidate.get("distance"),
                    greet=False,
                    target=chat_brain.target,
                )

                return

        # Target isn't currently visible -- don't fall through to
        # the unrelated greet-and-leave logic below for some other
        # player while we're supposed to be following someone else.
        return

    behavior = player_behavior.update(detected_players)

    target = behavior.get("target")

    if not target:
        return

    target_name = str(
        target.get("name")
        or target.get("player")
        or target.get("username")
        or "Unknown Player"
    )

    # IMPORTANT: roblox_bridge.py's read_approach_signal() expects
    # "distance" to be one of the category strings SemanticVision
    # produces ("near"/"medium"/"far"), and checks
    # `approach_distance not in {"near", "close"}` directly. Sending
    # PlayerBehavior.get_distance()'s converted numeric value here
    # would make that check always true, so we forward the raw
    # string from the detection instead.
    raw_distance = target.get("distance")

    send_approach(
        x=behavior.get("x", 0.0),
        distance=raw_distance,
        greet=behavior.get("greet", False),
        target=target_name,
    )


def main():
    global last_ocr, last_chat_seen, last_player_scan
    print("ROBLOX SOCIAL BRIDGE READY")
    print("OCR/chat only; roblox_bridge owns movement and stuck detection.")
    if not OCR_AVAILABLE:
        print("[WARN] Tesseract was not found; chat OCR is disabled.")

    try:
        while True:
            now = time.time()

            greet_server_once()

            if now - last_server_question >= SERVER_QUESTION_INTERVAL:
                ask_server_question()

            if now - last_player_scan >= PLAYER_SCAN_INTERVAL:
                try:
                    scan_players()
                except Exception as exc:
                    print(f"\n[PLAYER SCAN ERROR] {exc}")
                last_player_scan = now

            if now - last_ocr >= OCR_INTERVAL and now >= ocr_ignore_until:
                for speaker, message in parse_chat(read_chat()):
                    key = (speaker.lower(), message.lower())
                    if is_our_message(message):
                        continue
                    if is_duplicate_incoming(speaker, message):
                        continue
                    if now - seen_messages.get(key, 0.0) < MESSAGE_MEMORY:
                        continue
                    seen_messages[key] = now
                    last_chat_seen = now
                    print(f"\n[CHAT] {speaker}: {message}")
                    try:
                        handle_chat(speaker, message)
                    except Exception as exc:
                        print(f"\n[CHAT ERROR] {exc}")
                last_ocr = now

            time.sleep(0.05)
    except KeyboardInterrupt:
        print("\nROBLOX SOCIAL BRIDGE STOPPED")


if __name__ == "__main__":
    main()
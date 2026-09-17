import json
import re
import threading
import time
import traceback

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText


# ============================================================
# CONFIG
# ============================================================

MODEL_NAME = "HuggingFaceTB/SmolVLM-500M-Instruct"

MAX_NEW_TOKENS = 180
SCENE_INTERVAL = 3.0

IMAGE_WIDTH = 768
IMAGE_HEIGHT = 432

# Player VLM scan frequency.
PLAYER_DETECTION_INTERVAL = 1.0

# Number of consecutive detections required.
# Lowering this lets the bot lock onto visible avatars quickly instead of
# waiting for several noisy detections and then never acting.
PLAYER_CONFIRMATIONS = 2

# Lower threshold for real-world Roblox player detection.
# The original value was too strict for a live world and caused the fly to
# miss targets and spin in place.
PLAYER_CONFIDENCE_THRESHOLD = 0.45

# How long a confirmed player remains available
# between VLM scans.
PLAYER_MEMORY_TIMEOUT = 3.0


# ============================================================
# UI TEXT THAT MUST NEVER BE A PLAYER
# ============================================================

UI_WORDS = {
    "outfit",
    "outfits",
    "community outfits",
    "saved outfits",
    "outfit loader",
    "loader",
    "emotes",
    "settings",
    "setting",
    "shop",
    "store",
    "inventory",
    "avatar",
    "avatars",
    "catalog",
    "menu",
    "play",
    "resume",
    "leave",
    "exit",
    "close",
    "cancel",
    "confirm",
    "search",
    "back",
    "next",
    "previous",
    "apply",
    "reset",
    "save",
    "load",
    "friends",
    "friend",
    "players",
    "player list",
    "leaderboard",
    "chat",
    "messages",
    "notifications",
    "controls",
    "help",
    "report",
    "mute",
    "block",
    "invite",
    "follow",
    "unfollow",
    "profile",
    "username",
    "display name",
    "robux",
    "premium",
}


# ============================================================
# HELPERS
# ============================================================

def _clean_name(name):
    if not name:
        return ""

    name = str(name).strip()
    name = re.sub(r"\s+", " ", name)
    name = name.strip("\"'`[]{}")

    return name


def _looks_like_ui(name):
    if not name:
        return True

    normalized = re.sub(
        r"[^a-z0-9 ]+",
        "",
        name.lower(),
    ).strip()

    if not normalized:
        return True

    if normalized in UI_WORDS:
        return True

    for word in UI_WORDS:
        if normalized == word:
            return True

        if normalized.startswith(word + " "):
            return True

        if normalized.endswith(" " + word):
            return True

    ui_patterns = [
        r"\boutfits?\b",
        r"\bemotes?\b",
        r"\bsettings?\b",
        r"\bmenu\b",
        r"\bbutton\b",
        r"\btab\b",
        r"\bloader\b",
        r"\bsaved\b",
        r"\bcommunity\b",
        r"\bshop\b",
        r"\bstore\b",
        r"\binventory\b",
        r"\bavatar\b",
        r"\bprofile\b",
        r"\bsearch\b",
        r"\bnotification\b",
        r"\bleaderboard\b",
    ]

    for pattern in ui_patterns:
        if re.search(pattern, normalized):
            return True

    return False


def _clamp_x(value):
    try:
        value = float(value)
    except Exception:
        return 0.5

    return max(0.0, min(1.0, value))


def _normalize_distance(value):
    if value is None:
        return "unknown"

    value = str(value).strip().lower()

    if value in {
        "near",
        "close",
        "very close",
        "close by",
    }:
        return "near"

    if value in {
        "medium",
        "mid",
        "middle",
        "moderate",
    }:
        return "medium"

    if value in {
        "far",
        "distant",
        "background",
    }:
        return "far"

    return "unknown"


# ============================================================
# SEMANTIC VISION
# ============================================================

class SemanticVision:

    def __init__(self):

        print("[VISION] Loading semantic vision model...")

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(f"[VISION] Device: {self.device}")

        self.processor = AutoProcessor.from_pretrained(
            MODEL_NAME
        )

        self.model = AutoModelForImageTextToText.from_pretrained(
            MODEL_NAME,
            torch_dtype=(
                torch.float16
                if self.device == "cuda"
                else torch.float32
            ),
        )

        self.model.to(self.device)
        self.model.eval()

        self.scene_description = ""
        self.last_analysis = 0.0

        self.lock = threading.Lock()

        self.analyzing = False

        # ----------------------------------------------------
        # Player tracking
        # ----------------------------------------------------

        self.player_candidates = {}

        self.confirmed_players = {}

        self.last_player_scan = 0.0
        self.last_player_detection = 0.0

        print("[VISION] Semantic vision ready.")

    # ========================================================
    # IMAGE PREPARATION
    # ========================================================

    def prepare_image(self, frame):

        if isinstance(frame, Image.Image):

            image = frame.convert("RGB")

        elif isinstance(frame, np.ndarray):

            if frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]

            image = Image.fromarray(
                frame.astype(np.uint8)
            ).convert("RGB")

        else:

            raise TypeError(
                f"Unsupported frame type: {type(frame)}"
            )

        image = image.resize(
            (IMAGE_WIDTH, IMAGE_HEIGHT)
        )

        return image

    # ========================================================
    # MODEL GENERATION
    # ========================================================

    def _generate(self, image, prompt):

        try:

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                        },
                        {
                            "type": "text",
                            "text": prompt,
                        },
                    ],
                }
            ]

            text_prompt = self.processor.apply_chat_template(
                messages,
                add_generation_prompt=True,
            )

            inputs = self.processor(
                text=text_prompt,
                images=image,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(self.device)
                for key, value in inputs.items()
            }

            with torch.no_grad():

                output = self.model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False,
                )

            generated = output[0]

            input_length = inputs["input_ids"].shape[-1]

            generated = generated[input_length:]

            result = self.processor.decode(
                generated,
                skip_special_tokens=True,
            )

            return result.strip()

        except Exception:

            traceback.print_exc()

            return ""

    # ========================================================
    # GENERAL SCENE ANALYSIS
    # ========================================================

    def analyze_scene(
        self,
        frame,
        screen_text="",
    ):

        image = self.prepare_image(frame)

        prompt = """
Analyze this Roblox game screenshot.

Describe only what is visually present.

Identify:
- player characters / avatars
- buildings
- roads
- vehicles
- trees
- terrain
- walls
- doors
- rooms
- furniture
- animals
- objects
- signs
- menus
- buttons
- status bars
- icons
- notifications
- visible text
- what appears to be happening

For visible objects, describe approximate position:
LEFT, CENTER, or RIGHT.

Distinguish carefully between:
- actual 3D game-world objects
- Roblox UI elements
- text-only interface elements

IMPORTANT:
Do not claim that a UI label is a person.
Do not claim that text itself is a player.

A player means an actual visible humanoid/avatar character
in the 3D game world.
"""

        if screen_text:

            prompt += f"""

OCR text detected on the screen:

{screen_text}

OCR text is NOT evidence that a player exists.
Use it only as contextual information.
"""

        result = self._generate(
            image,
            prompt,
        )

        with self.lock:

            self.scene_description = result
            self.last_analysis = time.time()

        return result

    # ========================================================
    # PARSE PLAYER JSON
    # ========================================================

    def _parse_player_json(self, text):

        if not text:
            return []

        text = text.strip()

        text = re.sub(
            r"```json\s*",
            "",
            text,
            flags=re.IGNORECASE,
        )

        text = re.sub(
            r"```\s*",
            "",
            text,
        )

        match = re.search(
            r"\[[\s\S]*\]",
            text,
        )

        if not match:
            return []

        raw = match.group(0)

        try:

            data = json.loads(raw)

        except Exception:

            try:

                raw = raw.replace(
                    "'",
                    '"',
                )

                data = json.loads(raw)

            except Exception:

                return []

        if not isinstance(data, list):
            return []

        return data

    # ========================================================
    # VALIDATE PLAYER
    # ========================================================

    def _validate_player(self, candidate):

        if not isinstance(candidate, dict):
            return None

        # ----------------------------------------------------
        # Must explicitly be an avatar.
        # ----------------------------------------------------

        is_avatar = candidate.get(
            "is_avatar",
            candidate.get(
                "is_player",
                candidate.get("is_character", False),
            ),
        )

        if isinstance(is_avatar, str):

            is_avatar = (
                is_avatar.lower()
                in {
                    "true",
                    "yes",
                    "1",
                }
            )

        if not is_avatar:
            return None

        # ----------------------------------------------------
        # Must explicitly be in the 3D world.
        # ----------------------------------------------------

        is_3d_character = candidate.get(
            "is_3d_character",
            candidate.get(
                "is_3d",
                candidate.get("is_character", False),
            ),
        )

        if isinstance(is_3d_character, str):

            is_3d_character = (
                is_3d_character.lower()
                in {
                    "true",
                    "yes",
                    "1",
                }
            )

        if not is_3d_character:
            return None

        # ----------------------------------------------------
        # Must not be UI.
        # ----------------------------------------------------

        is_ui = candidate.get(
            "is_ui",
            False,
        )

        if isinstance(is_ui, str):

            is_ui = (
                is_ui.lower()
                in {
                    "true",
                    "yes",
                    "1",
                }
            )

        if is_ui:
            return None

        # ----------------------------------------------------
        # Body must actually be visible.
        # ----------------------------------------------------

        body_visible = candidate.get(
            "body_visible",
            candidate.get(
                "visible",
                candidate.get("body", is_avatar and is_3d_character),
            ),
        )

        if isinstance(body_visible, str):

            body_visible = (
                body_visible.lower()
                in {
                    "true",
                    "yes",
                    "1",
                }
            )

        if not body_visible:
            return None

        # ----------------------------------------------------
        # Confidence.
        # ----------------------------------------------------

        try:

            confidence = float(
                candidate.get(
                    "confidence",
                    0.0,
                )
            )

        except Exception:

            confidence = 0.0

        confidence = max(
            0.0,
            min(1.0, confidence),
        )

        if confidence < PLAYER_CONFIDENCE_THRESHOLD:
            return None

        # ----------------------------------------------------
        # Name.
        # ----------------------------------------------------

        name = _clean_name(
            candidate.get("name")
            or candidate.get("username")
            or candidate.get("player")
            or ""
        )

        if _looks_like_ui(name):
            return None

        if not name:
            name = "Unknown Player"

        # ----------------------------------------------------
        # X position.
        # ----------------------------------------------------

        try:

            x = float(
                candidate.get(
                    "x",
                    candidate.get(
                        "center_x",
                        0.5,
                    ),
                )
            )

        except Exception:

            x = 0.5

        if x > 1.0:
            x = x / float(IMAGE_WIDTH)

        x = _clamp_x(x)

        # ----------------------------------------------------
        # Distance.
        # ----------------------------------------------------

        distance = _normalize_distance(
            candidate.get("distance")
        )

        return {
            "name": name,
            "x": x,
            "distance": distance,
            "confidence": confidence,
        }

    # ========================================================
    # PLAYER KEY
    # ========================================================

    def _candidate_key(self, player):

        name = player["name"]

        if name != "Unknown Player":

            return (
                "name:"
                + name.lower()
            )

        x_bucket = int(
            round(
                player["x"] * 10
            )
        )

        return f"unknown:{x_bucket}"

    # ========================================================
    # UPDATE PLAYER CONFIRMATION
    # ========================================================

    def _update_player_confirmation(
        self,
        detections,
    ):

        now = time.time()

        for player in detections:

            key = self._candidate_key(
                player
            )

            previous = self.player_candidates.get(
                key
            )

            if previous is None:

                self.player_candidates[key] = {
                    "player": player,
                    "count": 1,
                    "last_seen": now,
                }

                continue

            previous_player = previous["player"]

            position_difference = abs(
                previous_player["x"]
                - player["x"]
            )

            # If the supposed player suddenly jumps
            # somewhere else, don't count it as the same
            # character.
            if position_difference > 0.20:

                previous["count"] = 1

            else:

                previous["count"] += 1

            previous["player"] = player
            previous["last_seen"] = now

            if (
                previous["count"]
                >= PLAYER_CONFIRMATIONS
            ):

                self.confirmed_players[key] = {
                    "player": player,
                    "last_seen": now,
                    "count": previous["count"],
                }

        # ----------------------------------------------------
        # Remove stale candidates.
        # ----------------------------------------------------

        stale_candidates = []

        for key, candidate in (
            self.player_candidates.items()
        ):

            if (
                now - candidate["last_seen"]
                > PLAYER_MEMORY_TIMEOUT
            ):

                stale_candidates.append(key)

        for key in stale_candidates:

            self.player_candidates.pop(
                key,
                None,
            )

        # ----------------------------------------------------
        # Remove stale confirmed players.
        # ----------------------------------------------------

        stale_confirmed = []

        for key, data in (
            self.confirmed_players.items()
        ):

            if (
                now - data["last_seen"]
                > PLAYER_MEMORY_TIMEOUT
            ):

                stale_confirmed.append(key)

        for key in stale_confirmed:

            self.confirmed_players.pop(
                key,
                None,
            )

        # ----------------------------------------------------
        # Return all currently confirmed players.
        # ----------------------------------------------------

        confirmed = []

        for data in (
            self.confirmed_players.values()
        ):

            if (
                now - data["last_seen"]
                <= PLAYER_MEMORY_TIMEOUT
            ):

                confirmed.append(
                    data["player"]
                )

        return confirmed

    # ========================================================
    # PUBLIC PLAYER DETECTION
    # ========================================================

    def detect_players(self, frame):

        now = time.time()

        # ====================================================
        # IMPORTANT:
        #
        # DO NOT RETURN [] BETWEEN SCANS.
        #
        # Keep the last confirmed players alive so the
        # behavior controller can continuously approach them.
        # ====================================================

        if (
            now - self.last_player_scan
            < PLAYER_DETECTION_INTERVAL
        ):

            confirmed = []

            for data in (
                self.confirmed_players.values()
            ):

                if (
                    now - data["last_seen"]
                    <= PLAYER_MEMORY_TIMEOUT
                ):

                    confirmed.append(
                        data["player"]
                    )

            return confirmed

        self.last_player_scan = now

        image = self.prepare_image(
            frame
        )

        prompt = """
You are detecting REAL ROBLOX PLAYER CHARACTERS.

This is an extremely strict detection task.

A player is ONLY a visible 3D humanoid/avatar character
physically present in the Roblox game world.

DO NOT detect:
- UI text
- buttons
- menus
- tabs
- outfit names
- item names
- emote names
- leaderboard entries
- chat messages
- usernames displayed only as text
- signs
- icons
- thumbnails
- profile pictures
- NPCs
- decorative objects
- text labels
- interface panels

A text label saying someone's name is NOT enough.

You must actually see the avatar/body in the 3D game world.

For every possible player, verify ALL of these:

1. A humanoid/avatar body is visibly present.
2. The body is physically in the 3D game world.
3. It is not part of the Roblox interface.
4. A meaningful portion of the body is visible.
5. The detection is visually grounded in the screenshot.

If there are no clearly visible player avatars,
return [].

Return ONLY valid JSON.

Format:

[
  {
    "name": "username if visually identifiable, otherwise Unknown Player",
    "x": 0.50,
    "distance": "near",
    "confidence": 0.90,
    "is_avatar": true,
    "is_3d_character": true,
    "body_visible": true,
    "is_ui": false
  }
]

x:
0.0 = far left
0.5 = center
1.0 = far right

distance must be:
"near"
"medium"
"far"

IMPORTANT:

Do not infer players from OCR.

Do not infer players from text.

Do not invent player names.

If uncertain whether something is a player,
DO NOT report it.

NPCs are NOT players.
UI avatars/icons are NOT players.
Profile pictures are NOT players.
"""

        raw = self._generate(
            image,
            prompt,
        )

        parsed = self._parse_player_json(
            raw
        )

        validated = []

        for candidate in parsed:

            player = self._validate_player(
                candidate
            )

            if player is not None:

                validated.append(
                    player
                )

        confirmed = (
            self._update_player_confirmation(
                validated
            )
        )

        self.last_player_detection = now

        if confirmed:

            print(
                "[VISION] Confirmed players:",
                confirmed,
            )

        return confirmed

    # ========================================================
    # QUESTION ANSWERING
    # ========================================================

    def answer_question(
        self,
        frame,
        question,
        screen_text="",
    ):

        image = self.prepare_image(
            frame
        )

        with self.lock:

            previous_scene = (
                self.scene_description
            )

        prompt = f"""
You are helping a fly-like game agent understand
a Roblox screenshot.

Answer the player's question using only information
supported by the current screenshot and supplied scene
description.

Current question:
{question}

Previous scene description:
{previous_scene}

OCR text:
{screen_text}

Do not invent information.
Do not claim to see something that is not visible.
Keep the answer concise.
"""

        return self._generate(
            image,
            prompt,
        )

    # ========================================================
    # GET SCENE
    # ========================================================

    def get_scene(self):

        with self.lock:

            return self.scene_description

    # ========================================================
    # BACKGROUND SCENE ANALYSIS
    # ========================================================

    def start_background_analysis(
        self,
        capture_function,
        ocr_function,
    ):

        if self.analyzing:
            return

        self.analyzing = True

        def worker():

            while self.analyzing:

                try:

                    frame = capture_function()

                    screen_text = ""

                    try:

                        screen_text = (
                            ocr_function()
                        )

                    except Exception:

                        screen_text = ""

                    self.analyze_scene(
                        frame,
                        screen_text,
                    )

                except Exception:

                    traceback.print_exc()

                time.sleep(
                    SCENE_INTERVAL
                )

        thread = threading.Thread(
            target=worker,
            daemon=True,
        )

        thread.start()

    # ========================================================
    # STOP BACKGROUND ANALYSIS
    # ========================================================

    def stop_background_analysis(self):

        self.analyzing = False
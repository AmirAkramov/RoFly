"""
fly_self_learning.py
====================

Autonomous self-learning layer for DoomFly.

What this adds
--------------
1.  SceneUnderstanding  – converts raw SemanticVision descriptions
    and OCR lines into structured facts the mind can reason over.

2.  PatternMemory       – notices which sights repeatedly appear
    together, which players interact most, which chat intents follow
    each other.  Builds a lightweight associative map without any
    external ML library.

3.  BehaviourEvaluator  – gives every action the fly takes a
    retrospective reward signal (did it lead somewhere interesting?).
    Feeds that signal back into FlyMind.experience() so curiosity /
    attention / boredom update meaningfully.

4.  CuriosityDriver     – picks what the fly should investigate next
    based on what it has seen least, or what changed most recently.

5.  SelfLearningLoop    – background thread that ties everything
    together, calls FlyMind hooks, and saves a compact
    "fly_knowledge.json" that persists across sessions.

Usage
-----
    from fly_self_learning import SelfLearningLoop

    loop = SelfLearningLoop(mind)          # mind is a FlyMind
    loop.start(
        capture_fn=capture_screen,         # () -> np.ndarray
        ocr_fn=get_ocr_lines,              # () -> list[str]
        scene_fn=get_scene_description,    # () -> str | None
    )

    # In your main loop, when a chat message arrives:
    loop.on_chat(speaker, message, intent)

    # When an action finishes:
    loop.on_action_result(action, result, outcome_score)
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from collections import defaultdict, deque
from typing import Callable, Dict, List, Optional, Tuple


# ============================================================
# CONSTANTS
# ============================================================

KNOWLEDGE_FILE    = "fly_knowledge.json"
LEARN_INTERVAL    = 4.0    # seconds between full learning cycles
FORGET_HALF_LIFE  = 600.0  # seconds; halves strength of unseen patterns
MAX_PATTERNS      = 300    # cap on stored co-occurrence pairs
MAX_PLAYER_FACTS  = 50     # facts remembered per player
MAX_LOCATIONS     = 40     # distinct named locations to track


# ============================================================
# SCENE UNDERSTANDING
# ============================================================

class SceneUnderstanding:
    """
    Parses raw text (SemanticVision output + OCR lines) into
    structured facts: players, objects, locations, events, signs.

    Each fact is a dict:
        { "category": str, "name": str, "details": str,
          "importance": float, "confidence": float }
    """

    # Rough keyword groups for importance scoring
    _HIGH_IMPORTANCE = {
        "player", "players", "character", "npc", "boss",
        "door", "exit", "portal", "chest", "loot",
        "enemy", "monster", "danger", "fire", "explosion",
        "sign", "notice", "warning",
    }

    _LOCATIONS = {
        "room", "hall", "corridor", "field", "forest", "city",
        "street", "road", "building", "shop", "house", "cave",
        "mountain", "island", "beach", "park", "plaza", "square",
        "lobby", "spawn", "arena", "zone", "area", "town",
    }

    _OBJECTS = {
        "tree", "rock", "wall", "fence", "table", "chair",
        "car", "vehicle", "bus", "lamp", "post", "bridge",
        "water", "river", "lake", "grass", "path", "coin",
        "gem", "flag", "tower", "statue",
    }

    # Roblox username heuristic: all-caps segments, brackets, numbers
    _USERNAME_RE = re.compile(
        r"\[?[A-Z0-9_ ]{3,}\]?[A-Z][a-zA-Z0-9_]{2,}"
    )

    def parse(
        self,
        scene_text: str,
        ocr_lines: List[str],
    ) -> List[Dict]:
        """Return a list of structured facts extracted from the inputs."""

        facts: List[Dict] = []

        if scene_text:
            facts.extend(
                self._parse_description(scene_text)
            )

        if ocr_lines:
            facts.extend(
                self._parse_ocr(ocr_lines)
            )

        # Deduplicate by (category, name)
        seen = set()
        unique = []
        for f in facts:
            key = (f["category"], f["name"].lower())
            if key not in seen:
                seen.add(key)
                unique.append(f)

        return unique

    def _importance(self, word: str) -> float:
        w = word.lower()
        if w in self._HIGH_IMPORTANCE:
            return 0.85
        if w in self._LOCATIONS:
            return 0.60
        if w in self._OBJECTS:
            return 0.50
        return 0.40

    def _parse_description(self, text: str) -> List[Dict]:
        facts = []

        lines = text.split("\n")

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Player / character detection
            # SemanticVision tends to say "player character" or "NPC"
            if re.search(
                r"\bplayer\b|\bcharacter\b|\bnpc\b",
                line, re.I
            ):
                # Try to extract a name or position label
                position = self._extract_position(line)
                name_match = re.search(
                    r"named? ['\"]?(\w+)['\"]?",
                    line, re.I
                )
                name = (
                    name_match.group(1)
                    if name_match
                    else f"player_{position}"
                )
                facts.append({
                    "category":  "player",
                    "name":      name,
                    "details":   line[:120],
                    "importance": 0.80,
                    "confidence": 0.70,
                })
                continue

            # Location detection
            for loc_word in self._LOCATIONS:
                if loc_word in line.lower():
                    name = self._extract_noun_near(line, loc_word)
                    facts.append({
                        "category":  "location",
                        "name":      name or loc_word,
                        "details":   line[:120],
                        "importance": self._importance(loc_word),
                        "confidence": 0.65,
                    })
                    break

            # Object detection
            for obj_word in self._OBJECTS:
                if obj_word in line.lower():
                    facts.append({
                        "category":  "object",
                        "name":      obj_word,
                        "details":   line[:120],
                        "importance": self._importance(obj_word),
                        "confidence": 0.60,
                    })
                    break

        return facts

    # Known noise words that OCR frequently produces from the fly's own
    # debug/telemetry text on screen, and a minimum ratio of normal
    # letters/numbers a line must have to be treated as real content.
    _OCR_GARBAGE_WORDS = {
        "ocr", "chat", "response", "brain", "neural", "window",
        "reason", "target", "goal", "reward", "signal", "malecns",
    }

    @classmethod
    def _is_garbage_ocr_line(cls, text: str) -> bool:
        text = text.strip()

        if len(text) < 3:
            return True

        if text.lower() in cls._OCR_GARBAGE_WORDS:
            return True

        # Real words/usernames are mostly letters, digits, underscores,
        # and spaces. Misread OCR noise is heavy on stray punctuation.
        valid = len(re.findall(r"[A-Za-z0-9_ ]", text))
        ratio = valid / max(len(text), 1)

        if ratio < 0.75:
            return True

        # A line with no vowels at all ("nent", "So7a van" minus vowels,
        # "{erancy new vex") is almost always a misread rather than a
        # real word or username.
        if not re.search(r"[aeiouAEIOU]", text):
            return True

        return False

    def _parse_ocr(self, lines: List[str]) -> List[Dict]:
        facts = []

        for line in lines:
            line = line.strip()
            if not line or len(line) < 3:
                continue

            if self._is_garbage_ocr_line(line):
                continue

            # Detect usernames in chat
            if ":" in line:
                parts = line.split(":", 1)
                speaker = parts[0].strip()
                msg     = parts[1].strip() if len(parts) > 1 else ""

                if speaker and not self._is_garbage_ocr_line(speaker):
                    facts.append({
                        "category":  "player",
                        "name":      speaker,
                        "details":   f"said: {msg[:80]}",
                        "importance": 0.75,
                        "confidence": 0.85,
                    })

            # Signs / notices
            elif re.match(
                r"^[A-Z][A-Z0-9 !?.]{4,}$",
                line
            ):
                facts.append({
                    "category":  "sign",
                    "name":      line[:40],
                    "details":   line,
                    "importance": 0.70,
                    "confidence": 0.75,
                })

            # General text
            else:
                facts.append({
                    "category":  "text",
                    "name":      line[:40],
                    "details":   line,
                    "importance": 0.35,
                    "confidence": 0.60,
                })

        return facts

    @staticmethod
    def _extract_position(text: str) -> str:
        for word in ["left", "right", "center", "front",
                     "background", "foreground", "middle"]:
            if word in text.lower():
                return word
        return "unknown"

    @staticmethod
    def _extract_noun_near(text: str, keyword: str) -> str:
        """Try to pull the noun right before or after the keyword."""
        pattern = rf"(\w+)\s+{keyword}|{keyword}\s+(\w+)"
        m = re.search(pattern, text, re.I)
        if m:
            return (m.group(1) or m.group(2) or keyword).lower()
        return keyword


# ============================================================
# PATTERN MEMORY
# ============================================================

class PatternMemory:
    """
    Lightweight associative co-occurrence learner.

    Tracks which (category, name) pairs tend to appear together,
    which chat intents follow each other, and how often specific
    players interact.

    All weights decay over time (half-life = FORGET_HALF_LIFE).
    """

    def __init__(self):
        # co_occur[(a, b)] = strength float
        self.co_occur: Dict[Tuple, float] = {}
        self.last_updated: Dict[Tuple, float] = {}

        # intent_sequence[(prev_intent, next_intent)] = count
        self.intent_seq: Dict[Tuple, int] = defaultdict(int)

        # player_interactions[player_name] = count
        self.player_interactions: Dict[str, int] = defaultdict(int)

        # Recent window of seen facts for co-occurrence
        self._recent_window: deque = deque(maxlen=12)

        self.load()

    # --------------------------------------------------------
    # OBSERVE FACTS
    # --------------------------------------------------------

    def observe_facts(self, facts: List[Dict]) -> None:
        """Update co-occurrence from a new batch of scene facts."""

        keys = [
            (f["category"], f["name"].lower())
            for f in facts
        ]

        now = time.time()

        # Combine with recent window
        window = list(self._recent_window) + keys

        # Count co-occurrences within window
        for i, a in enumerate(window):
            for b in window[i+1 : i+4]:   # nearby pairs
                if a == b:
                    continue
                pair = tuple(sorted([str(a), str(b)]))
                self._decay(pair, now)
                self.co_occur[pair] = (
                    self.co_occur.get(pair, 0.0) + 1.0
                )
                self.last_updated[pair] = now

        # Slide window
        for k in keys:
            self._recent_window.append(k)

        # Trim
        if len(self.co_occur) > MAX_PATTERNS:
            weakest = sorted(
                self.co_occur,
                key=lambda k: self.co_occur[k]
            )
            for k in weakest[:MAX_PATTERNS // 4]:
                self.co_occur.pop(k, None)
                self.last_updated.pop(k, None)

    def observe_intent_sequence(
        self,
        prev_intent: Optional[str],
        next_intent: str,
    ) -> None:
        if prev_intent:
            self.intent_seq[(prev_intent, next_intent)] += 1

    def observe_player(self, player_name: str) -> None:
        self.player_interactions[player_name] += 1

    # --------------------------------------------------------
    # QUERY
    # --------------------------------------------------------

    def strongest_associations(
        self,
        category: str,
        name: str,
        top_n: int = 5,
    ) -> List[Tuple[str, float]]:
        """Return the top_n strongest co-occurrents of (category, name)."""

        now = time.time()
        target = str((category, name.lower()))
        results = []

        for pair, strength in self.co_occur.items():
            if target in pair:
                other = [p for p in pair if p != target]
                if other:
                    decayed = self._decayed(strength, pair, now)
                    results.append((other[0], decayed))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_n]

    def most_frequent_player(self) -> Optional[str]:
        if not self.player_interactions:
            return None
        return max(
            self.player_interactions,
            key=lambda k: self.player_interactions[k]
        )

    def likely_next_intent(self, current_intent: str) -> Optional[str]:
        candidates = {
            b: c
            for (a, b), c in self.intent_seq.items()
            if a == current_intent
        }
        if not candidates:
            return None
        return max(candidates, key=lambda k: candidates[k])

    # --------------------------------------------------------
    # DECAY
    # --------------------------------------------------------

    def _decay(self, pair: Tuple, now: float) -> None:
        last = self.last_updated.get(pair, now)
        elapsed = now - last
        if elapsed > 0 and pair in self.co_occur:
            factor = math.exp(
                -elapsed * math.log(2) / FORGET_HALF_LIFE
            )
            self.co_occur[pair] *= factor

    def _decayed(
        self, strength: float, pair: Tuple, now: float
    ) -> float:
        last = self.last_updated.get(pair, now)
        elapsed = now - last
        factor = math.exp(
            -elapsed * math.log(2) / FORGET_HALF_LIFE
        )
        return strength * factor

    # --------------------------------------------------------
    # PERSISTENCE
    # --------------------------------------------------------

    def save(self, data: dict) -> None:
        data["co_occur"] = {
            str(k): v for k, v in self.co_occur.items()
        }
        data["last_updated"] = {
            str(k): v for k, v in self.last_updated.items()
        }
        data["intent_seq"] = {
            str(k): v for k, v in self.intent_seq.items()
        }
        data["player_interactions"] = dict(
            self.player_interactions
        )

    def load(self) -> None:
        if not os.path.exists(KNOWLEDGE_FILE):
            return
        try:
            with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)

            for k, v in data.get("co_occur", {}).items():
                self.co_occur[k] = float(v)
            for k, v in data.get("last_updated", {}).items():
                self.last_updated[k] = float(v)
            for k, v in data.get("intent_seq", {}).items():
                self.intent_seq[k] = int(v)
            for k, v in data.get(
                "player_interactions", {}
            ).items():
                self.player_interactions[k] = int(v)

            print(
                f"[SELF-LEARNING] Loaded {len(self.co_occur)} "
                f"patterns, "
                f"{len(self.player_interactions)} players."
            )

        except Exception as e:
            print(f"[SELF-LEARNING] Pattern load failed: {e}")


# ============================================================
# BEHAVIOUR EVALUATOR
# ============================================================

class BehaviourEvaluator:
    """
    Retrospectively scores the fly's actions by comparing what
    was observed before and after the action.

    Score is fed back into FlyMind.experience() as a reward.
    """

    def __init__(self):
        self._action_log: deque = deque(maxlen=60)

    def log_action(
        self,
        action: str,
        facts_before: List[Dict],
        facts_after:  List[Dict],
    ) -> float:
        """
        Compute reward for an action based on how much the scene
        changed, scaled by the importance of new facts seen.
        """

        before_keys = {
            (f["category"], f["name"].lower())
            for f in facts_before
        }
        after_keys = {
            (f["category"], f["name"].lower())
            for f in facts_after
        }

        new_keys  = after_keys - before_keys
        lost_keys = before_keys - after_keys

        # Gather importance of truly new facts
        importance_map = {
            (f["category"], f["name"].lower()): f.get("importance", 0.5)
            for f in facts_after
        }

        novelty_reward = sum(
            importance_map.get(k, 0.4)
            for k in new_keys
        )

        # Penalise doing nothing (all same)
        stagnation_penalty = (
            -0.05
            if not new_keys and not lost_keys
            else 0.0
        )

        reward = min(
            0.40,
            novelty_reward * 0.15 + stagnation_penalty
        )

        self._action_log.append({
            "action":  action,
            "reward":  reward,
            "new":     len(new_keys),
            "lost":    len(lost_keys),
            "time":    time.time(),
        })

        return reward

    def average_recent_reward(self) -> float:
        if not self._action_log:
            return 0.0
        recent = list(self._action_log)[-20:]
        return sum(e["reward"] for e in recent) / len(recent)

    def best_action_lately(self) -> Optional[str]:
        if not self._action_log:
            return None
        best = max(
            self._action_log,
            key=lambda e: e["reward"]
        )
        return best["action"]


# ============================================================
# CURIOSITY DRIVER
# ============================================================

class CuriosityDriver:
    """
    Decides what the fly should investigate next.

    Priority:
        1. Newly seen facts with high importance
        2. Things seen very few times (exploration)
        3. Things that changed most recently
        4. Random choice when everything is familiar
    """

    def pick_target(
        self,
        awareness_summary: List[Dict],
        recent_facts: List[Dict],
        boredom: float,
    ) -> Optional[Dict]:

        if not awareness_summary and not recent_facts:
            return None

        # 1. Something brand new (in recent_facts but not summary)
        summary_names = {
            (o.get("category"), o.get("name", "").lower())
            for o in awareness_summary
        }
        for f in recent_facts:
            key = (f["category"], f["name"].lower())
            if key not in summary_names:
                return f

        # 2. Least-seen thing with above-average importance
        scored = sorted(
            awareness_summary,
            key=lambda o: (
                o.get("times_seen", 1),
                -o.get("importance", 0.5),
            )
        )
        candidates = [
            o for o in scored
            if o.get("importance", 0.5) >= 0.5
        ]
        if candidates:
            return candidates[0]

        # 3. If very bored, pick anything
        if boredom > 0.7 and awareness_summary:
            import random
            return random.choice(awareness_summary)

        return None


# ============================================================
# SELF-LEARNING LOOP
# ============================================================

class SelfLearningLoop:
    """
    Background thread that runs the full self-learning cycle.

    Connects SceneUnderstanding, PatternMemory, BehaviourEvaluator,
    and CuriosityDriver to FlyMind.

    Parameters
    ----------
    mind : FlyMind
        The fly's FlyMind instance.  Must have:
            .learn_observation(category, name, details, importance)
            .experience(observation, action, result, reward)
            .get_state()
            .boredom  (float attribute)
    """

    def __init__(self, mind):
        self.mind = mind

        self.understanding = SceneUnderstanding()
        self.patterns      = PatternMemory()
        self.evaluator     = BehaviourEvaluator()
        self.curiosity_drv = CuriosityDriver()

        self._running   = False
        self._thread    = None
        self._lock      = threading.Lock()

        # Callbacks
        self._capture_fn: Optional[Callable] = None
        self._ocr_fn:     Optional[Callable] = None
        self._scene_fn:   Optional[Callable] = None

        # Internal state
        self._last_facts:      List[Dict] = []
        self._last_intent:     Optional[str] = None
        self._last_action:     Optional[str] = None
        self._last_facts_time: float = 0.0

        # Knowledge persistence
        self._knowledge: Dict = self._load_knowledge()

        print("[SELF-LEARNING] Initialised.")

    # --------------------------------------------------------
    # START / STOP
    # --------------------------------------------------------

    def start(
        self,
        capture_fn: Callable,
        ocr_fn:     Callable,
        scene_fn:   Callable,
    ) -> None:
        """Start the background learning thread."""

        self._capture_fn = capture_fn
        self._ocr_fn     = ocr_fn
        self._scene_fn   = scene_fn

        self._running = True

        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
            name="SelfLearningLoop",
        )
        self._thread.start()

        print("[SELF-LEARNING] Background loop started.")

    def stop(self) -> None:
        self._running = False
        print("[SELF-LEARNING] Stopped.")

    # --------------------------------------------------------
    # EXTERNAL HOOKS (called from main code)
    # --------------------------------------------------------

    def on_chat(
        self,
        speaker:  str,
        message:  str,
        intent:   str,
    ) -> None:
        """
        Call whenever a chat message is processed.
        Updates pattern memory and mind awareness.
        """

        with self._lock:
            self.patterns.observe_player(speaker)
            self.patterns.observe_intent_sequence(
                self._last_intent,
                intent,
            )
            self._last_intent = intent

        # Record the player as a known entity
        self.mind.learn_observation(
            category   = "player",
            name       = speaker,
            details    = f"intent={intent} msg={message[:60]}",
            importance = 0.75,
        )

        # Adjust social interest based on interaction frequency
        freq = self.patterns.player_interactions.get(speaker, 1)
        social_boost = min(0.05, 0.01 * math.log1p(freq))
        self.mind.social_interest = min(
            1.0,
            self.mind.social_interest + social_boost
        )

        # Experience: being spoken to is rewarding
        self.mind.experience(
            observation={
                "type":    "chat",
                "speaker": speaker,
                "message": message[:80],
                "intent":  intent,
            },
            action="respond_to_chat",
            result="engaged with player",
            reward=0.06,
        )

    def _capture_facts_now(self) -> List[Dict]:
        """Capture a fresh observation snapshot from the current scene."""

        scene_text = None
        ocr_lines = []

        if self._scene_fn:
            try:
                scene_text = self._scene_fn()
            except Exception:
                scene_text = None

        if self._ocr_fn:
            try:
                ocr_lines = self._ocr_fn() or []
            except Exception:
                ocr_lines = []

        if not scene_text and not ocr_lines:
            return []

        return self.understanding.parse(scene_text or "", ocr_lines)

    def on_action_result(
        self,
        action: str,
        result: str,
        outcome_score: float = 0.0,
    ) -> None:
        """
        Call after any movement or behaviour action finishes.
        outcome_score: +1 = great, 0 = neutral, -1 = bad.

        The evaluator must compare a real snapshot captured before the
        action against a fresh observation captured after it; otherwise
        it compares the same object instance and always reports a
        stagnation penalty.
        """

        with self._lock:
            self._last_action = action
            facts_before = [dict(f) for f in self._last_facts]

        # Capture a fresh post-action observation; do not reuse the stale
        # pre-action state.
        facts_after = self._capture_facts_now()
        if not facts_after:
            facts_after = facts_before

        with self._lock:
            self._last_facts = [dict(f) for f in facts_after]
            self._last_facts_time = time.time()

        reward = self.evaluator.log_action(
            action,
            facts_before,
            facts_after,
        )
        reward += outcome_score * 0.10

        self.mind.experience(
            observation={
                "type":   "action_result",
                "action": action,
                "result": result,
            },
            action=action,
            result=result,
            reward=reward,
        )

        print(
            f"[SELF-LEARNING] Action '{action}' reward={reward:+.3f}"
        )

    # --------------------------------------------------------
    # BACKGROUND LOOP
    # --------------------------------------------------------

    def _loop(self) -> None:
        while self._running:
            try:
                self._cycle()
            except Exception as e:
                print(f"[SELF-LEARNING ERROR] {e}")
            time.sleep(LEARN_INTERVAL)

    def _cycle(self) -> None:
        """One full learning cycle."""

        # ====================================================
        # 1. CAPTURE CURRENT SCENE
        # ====================================================

        facts = self._capture_facts_now()
        if not facts:
            return

        # ====================================================
        # 2. PARSE FACTS
        # ====================================================

        with self._lock:
            previous_facts = self._last_facts
            self._last_facts = facts
            self._last_facts_time = time.time()

        # ====================================================
        # 3. UPDATE PATTERN MEMORY
        # ====================================================

        self.patterns.observe_facts(facts)

        # ====================================================
        # 4. FEED FACTS INTO MIND AWARENESS
        # ====================================================

        for fact in facts:
            self.mind.learn_observation(
                category   = fact["category"],
                name       = fact["name"],
                details    = fact.get("details", ""),
                importance = fact.get("importance", 0.5),
            )

        # ====================================================
        # 5. BEHAVIOUR EVALUATION (compare to previous cycle)
        # ====================================================

        if previous_facts and self._last_action:
            reward = self.evaluator.log_action(
                self._last_action,
                previous_facts,
                facts,
            )

            self.mind.experience(
                observation={
                    "type":  "scene_change",
                    "new":   len(facts),
                    "prev":  len(previous_facts),
                },
                action=self._last_action or "observe",
                result="scene updated",
                reward=reward,
            )

        # ====================================================
        # 6. CURIOSITY — PICK NEXT TARGET
        # ====================================================

        state   = self.mind.get_state()
        boredom = self.mind.boredom

        summary = state.get("world", [])

        target = self.curiosity_drv.pick_target(
            summary,
            facts,
            boredom,
        )

        if target:
            category = target.get("category", "thing")
            name     = target.get("name",     "unknown")

            print(
                f"[SELF-LEARNING] Curious about "
                f"{category}: {name}"
            )

            # Boost curiosity toward interesting targets
            importance = target.get("importance", 0.5)
            self.mind.curiosity = min(
                1.0,
                self.mind.curiosity + importance * 0.04
            )
            self.mind.boredom = max(
                0.0,
                self.mind.boredom - 0.06
            )

        # ====================================================
        # 7. GENERATE AUTONOMOUS INSIGHT
        # ====================================================

        insight = self._generate_insight(facts)
        if insight:
            print(f"[SELF-LEARNING] Insight: {insight}")

        # ====================================================
        # 8. SAVE
        # ====================================================

        self._save_knowledge(facts, target, insight)

    # --------------------------------------------------------
    # INSIGHT GENERATION
    # --------------------------------------------------------

    def _generate_insight(
        self, facts: List[Dict]
    ) -> Optional[str]:
        """
        Produce a short natural-language insight about what the
        fly has noticed — used for spontaneous speech.

        This is rule-based and fast (no LLM needed).
        """

        players = [
            f for f in facts
            if f["category"] == "player"
        ]
        locations = [
            f for f in facts
            if f["category"] == "location"
        ]
        objects = [
            f for f in facts
            if f["category"] == "object"
        ]
        signs = [
            f for f in facts
            if f["category"] == "sign"
        ]

        if players and locations:
            p = players[0]["name"]
            l = locations[0]["name"]
            return f"I see {p} near a {l}."

        if signs:
            s = signs[0]["name"]
            return f"There's a sign that says '{s}'."

        if len(players) > 1:
            names = ", ".join(
                p["name"] for p in players[:3]
            )
            return f"Multiple players around: {names}."

        if objects:
            o = objects[0]["name"]
            return f"I notice a {o} nearby."

        freq_player = self.patterns.most_frequent_player()
        if freq_player:
            count = self.patterns.player_interactions[freq_player]
            return (
                f"I have seen {freq_player} "
                f"{count} times. They feel familiar."
            )

        return None

    # --------------------------------------------------------
    # SNAPSHOT FOR BEFORE-AFTER COMPARISON
    # --------------------------------------------------------

    def snapshot_facts(self) -> List[Dict]:
        """Return a copy of current facts for before/after comparison."""
        with self._lock:
            return list(self._last_facts)

    # --------------------------------------------------------
    # KNOWLEDGE PERSISTENCE
    # --------------------------------------------------------

    def _load_knowledge(self) -> Dict:
        if not os.path.exists(KNOWLEDGE_FILE):
            return {}
        try:
            with open(KNOWLEDGE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"[SELF-LEARNING] Knowledge load failed: {e}")
            return {}

    def _save_knowledge(
        self,
        facts:   List[Dict],
        target:  Optional[Dict],
        insight: Optional[str],
    ) -> None:

        data = dict(self._knowledge)

        # Stats
        data["last_cycle"] = time.time()
        data["total_cycles"] = data.get("total_cycles", 0) + 1

        # Current scene summary
        data["last_scene_facts"] = [
            {
                "category":  f["category"],
                "name":      f["name"],
                "importance": f.get("importance", 0.5),
            }
            for f in facts[:20]
        ]

        # Current curiosity target
        data["curiosity_target"] = target

        # Latest insight
        if insight:
            insights = data.get("recent_insights", [])
            insights.append({
                "time":    time.time(),
                "insight": insight,
            })
            data["recent_insights"] = insights[-30:]

        # Pattern memory
        self.patterns.save(data)

        # Average reward
        data["avg_reward"] = round(
            self.evaluator.average_recent_reward(), 4
        )
        data["best_action"] = self.evaluator.best_action_lately()

        try:
            tmp = KNOWLEDGE_FILE + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(tmp, KNOWLEDGE_FILE)
            self._knowledge = data
        except Exception as e:
            print(f"[SELF-LEARNING] Save failed: {e}")


# ============================================================
# STANDALONE TEST (no game required)
# ============================================================

if __name__ == "__main__":

    import random

    # ---------- Minimal FlyMind stub ----------
    class StubMind:
        energy = 0.8
        curiosity = 0.5
        attention = 0.5
        boredom = 0.3
        social_interest = 0.3

        def learn_observation(self, category, name,
                              details=None, importance=0.5):
            print(f"  [MIND] observe {category}: {name}")

        def experience(self, observation, action=None,
                       result=None, reward=0.0):
            print(f"  [MIND] experience action={action} "
                  f"reward={reward:+.3f}")

        def get_state(self):
            return {"world": []}

    # ---------- Fake data sources ----------

    FAKE_SCENE = [
        "I can see two player characters in the center. "
        "There is a large building in the background. "
        "A road runs left to right.",
        "A player named PlayerOne is on the left. "
        "There is a tree in the foreground. "
        "Signs say WELCOME TO THE PLAZA.",
        "The room is empty. "
        "There is a chest near the door.",
    ]

    FAKE_OCR = [
        ["PlayerOne: hey fly follow me", "WELCOME TO THE PLAZA"],
        ["PlayerTwo: back fire fly", "100 coins"],
        [],
    ]

    mind = StubMind()
    loop = SelfLearningLoop(mind)

    cycle = 0

    def fake_scene():
        return FAKE_SCENE[cycle % len(FAKE_SCENE)]

    def fake_ocr():
        return FAKE_OCR[cycle % len(FAKE_OCR)]

    loop.start(
        capture_fn=lambda: None,
        ocr_fn=fake_ocr,
        scene_fn=fake_scene,
    )

    print()
    print("=" * 60)
    print("FLY SELF-LEARNING  —  standalone test")
    print("Running 3 learning cycles (Ctrl+C to stop early)")
    print("=" * 60)
    print()

    try:
        for i in range(3):
            cycle = i
            time.sleep(LEARN_INTERVAL + 0.5)
            print()
            loop.on_chat("PlayerOne", "fly follow me", "FOLLOW")
            loop.on_action_result("explore", "moved to new area", 0.5)
            print()

    except KeyboardInterrupt:
        pass

    loop.stop()
    print()
    print("Test complete.  Check fly_knowledge.json")
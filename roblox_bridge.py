import time
import traceback
import random
import numpy as np
import mss

from PIL import Image
from pynput.keyboard import Controller, KeyCode, Key

from doom_learning_v6.visual import VisualMemoryBrain
from fly_brain import FlyBrain
from roblox_bridge_commands import read_command


# ============================================================
# SETTINGS
# ============================================================

MODEL_FILE = "roblox_decoder.npz"

SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480

STIMULUS_MS = 50
UPDATE_HZ = 10

THRESHOLD_MOVE = 0.30
THRESHOLD_CAMERA = 0.40


# ------------------------------------------------------------
# CAMERA BEHAVIOR
# ------------------------------------------------------------

CAMERA_ENABLED = True

# Windows virtual key VK_OEM_2 is the physical slash/question-mark key.
# Character mapping can produce '.' under non-US keyboard layouts.
CHAT_OPEN_KEY = KeyCode.from_vk(0xBF)

# Short camera movements.
CAMERA_PULSE_TIME = 0.12

# Minimum time between automatic camera movements.
CAMERA_COOLDOWN = 0.35

# How often the fly spontaneously looks around.
LOOK_AROUND_INTERVAL = 2.8

# Extra time spent looking around when inspecting something.
OBSERVE_LOOK_INTERVAL = 0.9

# How many camera pulses make up an observation scan.
OBSERVE_SCAN_PULSES = 4


# ------------------------------------------------------------
# MOVEMENT STABILIZATION
# ------------------------------------------------------------

MOVE_CONFIRM_FRAMES = 1
TURN_CONFIRM_FRAMES = 2
FORWARD_GRACE_SECONDS = 0.70
UNCERTAINTY_SECONDS = 2.5
UNCERTAINTY_CHAT_COOLDOWN = 30.0


# ============================================================
# LOAD DECODER
# ============================================================

print("=" * 70)
print("DOOMFLY ROBLOX NEURAL BRIDGE")
print("=" * 70)
print()

print("Loading decoder model...")

model = np.load(MODEL_FILE)

mean = model["mean"]
std = model["std"]
components = model["components"]

W_fwd = model["weights_forward"]
b_fwd = model["bias_forward"]

W_trn = model["weights_turn"]
b_trn = model["bias_turn"]

W_cam = model["weights_camera"]
b_cam = model["bias_camera"]

print("Decoder loaded.")
print()


# ============================================================
# LOAD MALECNS
# ============================================================

print("Loading MaleCNS...")

neural_brain = VisualMemoryBrain()

print(
    f"MaleCNS loaded: "
    f"{neural_brain.n:,} neurons"
)

print()


# ============================================================
# LOAD BEHAVIOR BRAIN
# ============================================================
# This process owns movement + camera control. It runs its own in-memory
# FlyBrain instance, separate from the social/OCR process. The two
# processes do not share live state; they only communicate via files or
# external commands, so their goals and memory must be treated as
# independent runtime state.

print("Loading FlyBrain...")

fly_brain = FlyBrain()

print("FlyBrain loaded.")
print()


# ============================================================
# SCREEN CAPTURE
# ============================================================

sct = mss.MSS()
monitor = sct.monitors[1]

print(
    f"Screen: "
    f"{monitor['width']}x{monitor['height']}"
)


def capture_screen():

    shot = sct.grab(monitor)

    frame = np.asarray(
        shot,
        dtype=np.uint8
    )

    frame = frame[:, :, :3][:, :, ::-1]

    image = Image.fromarray(frame)

    image = image.resize(
        (
            SCREEN_WIDTH,
            SCREEN_HEIGHT
        )
    )

    return np.asarray(
        image,
        dtype=np.uint8
    )


# ============================================================
# KEYBOARD
# ============================================================

ctrl = Controller()

held = {
    "w": False,
    "a": False,
    "s": False,
    "d": False,
    "left": False,
    "right": False,
}


KEY_OBJECTS = {

    "w":
        KeyCode.from_char("w"),

    "a":
        KeyCode.from_char("a"),

    "s":
        KeyCode.from_char("s"),

    "d":
        KeyCode.from_char("d"),

    "left":
        Key.left,

    "right":
        Key.right,
}


def set_key(
    name,
    pressed
):

    if held[name] == pressed:
        return

    if pressed:

        ctrl.press(
            KEY_OBJECTS[name]
        )

    else:

        ctrl.release(
            KEY_OBJECTS[name]
        )

    held[name] = pressed


def release_all():

    for name in held:

        set_key(
            name,
            False
        )


def send_chat(message):
    message = str(message).strip()
    if not message:
        return

    try:
        ctrl.press(CHAT_OPEN_KEY)
        time.sleep(0.20)
        ctrl.type(message)
        time.sleep(0.08)
        ctrl.press(Key.enter)
    except Exception as exc:
        print(f"\n[CHAT ERROR] Could not send bridge message: {exc}")


approach_until = 0.0
approach_x = 0.0
approach_distance = None
last_greet_time = 0.0
social_stop_until = 0.0


def read_approach_signal():
    global approach_until, approach_x, approach_distance
    global last_greet_time, social_stop_until

    command = read_command()
    if not isinstance(command, dict):
        return

    action = command.get("action")
    if action == "say":
        message = str(command.get("message", "")).strip()
        if message:
            social_stop_until = time.time() + 4.0
            approach_until = 0.0
            release_all()
            send_chat(message)
        return

    if action == "stop":
        try:
            duration = max(0.5, min(15.0, float(command.get("duration", 5.0))))
        except (TypeError, ValueError):
            duration = 5.0
        social_stop_until = time.time() + duration
        approach_until = 0.0
        return

    if action == "resume":
        social_stop_until = 0.0
        return

    if action != "approach":
        return

    try:
        signal_x = float(command.get("x", 0.0))
    except (TypeError, ValueError):
        print("\n[BRIDGE] Ignoring approach signal with invalid x.")
        return

    approach_until = time.time() + 2.0
    approach_x = max(-1.0, min(1.0, signal_x))
    approach_distance = command.get("distance")

    if command.get("greet") and time.time() - last_greet_time >= 12.0:
        send_chat("hi")
        last_greet_time = time.time()


# ============================================================
# SOFTMAX
# ============================================================

def softmax(z):

    z = z - np.max(z)

    e = np.exp(z)

    return e / e.sum()


def decode(
    features_p,
    W,
    b
):

    logits = (
        W @ features_p
        + b
    )

    probs = softmax(
        logits
    )

    prediction = int(
        np.argmax(probs)
    )

    confidence = float(
        probs[prediction]
    )

    return (
        prediction,
        confidence
    )


# ============================================================
# CAMERA STATE
# ============================================================

last_camera_time = 0.0
last_look_time = time.time()

observe_pulses = 0
observe_direction = "left"

last_scene_signature = None
last_scene_change = time.time()
last_help_time = 0.0
uncertain_since = 0.0
last_uncertainty_chat = 0.0
last_forward_command = 0.0


def check_for_stuck(frame, forward_active, now):
    """Ask the whole server for help when forward movement makes no progress."""

    global last_scene_signature, last_scene_change, last_help_time

    signature = frame[::16, ::16, :3].astype(np.int16)

    if last_scene_signature is None:
        last_scene_signature = signature
        last_scene_change = now
        return

    change = float(np.mean(np.abs(signature - last_scene_signature)))
    last_scene_signature = signature

    if change > 1.5:
        last_scene_change = now

    if (
        forward_active
        and now - last_scene_change >= 3.0
        and now - last_help_time >= 20.0
    ):
        send_chat(random.choice([
            "Can someone help me? I seem to be stuck.",
            "Does anyone know how to get past this?",
            "Can someone explain what I should try here?",
        ]))
        last_help_time = now
        last_scene_change = now
        print("\n[HELP] Asked the whole server for movement advice.")


def check_for_uncertainty(fwd_conf, trn_conf, now):
    """Ask the server when the learned controller cannot choose an action."""

    global uncertain_since, last_uncertainty_chat

    unsure = (
        fwd_conf < THRESHOLD_MOVE
        and trn_conf < THRESHOLD_MOVE
    )

    if not unsure:
        uncertain_since = 0.0
        return

    if uncertain_since == 0.0:
        uncertain_since = now
        return

    if (
        now - uncertain_since >= UNCERTAINTY_SECONDS
        and now - last_uncertainty_chat >= UNCERTAINTY_CHAT_COOLDOWN
    ):
        send_chat(random.choice([
            "Can someone help me understand how to move here?",
            "I am unsure what to try next. Any advice?",
            "Does anyone know what I should do now?",
        ]))
        last_uncertainty_chat = now
        uncertain_since = now
        print("\n[HELP] Asked the whole server for movement guidance.")

def camera_pulse(
    direction,
    duration=CAMERA_PULSE_TIME
):

    global last_camera_time

    now = time.time()

    if (
        now - last_camera_time
        < CAMERA_COOLDOWN
    ):
        return False

    if direction not in (
        "left",
        "right"
    ):
        return False

    # Never allow both directions simultaneously.
    set_key(
        "left",
        False
    )

    set_key(
        "right",
        False
    )

    set_key(
        direction,
        True
    )

    time.sleep(
        duration
    )

    set_key(
        direction,
        False
    )

    last_camera_time = time.time()

    return True


# ============================================================
# ACTIVE LOOK-AROUND
# ============================================================

def spontaneous_look():

    global last_look_time

    now = time.time()

    if (
        now - last_look_time
        < LOOK_AROUND_INTERVAL
    ):
        return False

    # Alternate direction each time.
    if int(now) % 2 == 0:

        direction = "left"

    else:

        direction = "right"

    result = camera_pulse(
        direction,
        CAMERA_PULSE_TIME
    )

    if result:

        last_look_time = now

    return result


# ============================================================
# OBSERVATION SCAN
# ============================================================

def observation_scan():

    global observe_pulses
    global observe_direction

    if observe_direction == "left":

        direction = "left"

        observe_direction = "right"

    else:

        direction = "right"

        observe_direction = "left"

    success = camera_pulse(
        direction,
        CAMERA_PULSE_TIME
    )

    if success:

        observe_pulses += 1

    return success


# ============================================================
# NEURAL STATE
# ============================================================

previous_fwd = None
previous_fwd_count = 0

previous_turn = None
previous_turn_count = 0


def stable_prediction(
    prediction,
    previous,
    count
):

    if prediction == previous:

        count += 1

    else:

        previous = prediction
        count = 1

    return (
        prediction,
        count
    )


# ============================================================
# READY
# ============================================================

print()
print("=" * 70)
print("DOOMFLY READY")
print("=" * 70)
print()

print(
    "Movement threshold:",
    f"{THRESHOLD_MOVE * 100:.0f}%"
)

print(
    "Camera threshold:",
    f"{THRESHOLD_CAMERA * 100:.0f}%"
)

print("Sample-trained visual policy: ACTIVE")
print("Continuous screen watching: ACTIVE")

print()

print(
    "Camera behavior:"
)

print(
    "  Neural camera turning: ON"
)

print(
    "  Automatic look-around: ON"
)

print(
    "  Observation scanning: ON"
)

print()

print(
    "Ctrl+C = emergency stop"
)

print()

try:
    input(
        "Press ENTER to start..."
    )
except EOFError:
    print("\n[BRIDGE] Input stream closed; starting automatically.")

print()

print(
    "Starting in 3 seconds..."
)

time.sleep(3)

print(
    "DOOMFLY ACTIVE"
)

print()


# ============================================================
# MAIN LOOP
# ============================================================

interval = 1.0 / UPDATE_HZ

try:

    while True:

        loop_start = time.perf_counter()
        read_approach_signal()
        # ----------------------------------------------------
        # SCREEN
        # ----------------------------------------------------

        frame = capture_screen()

        # ----------------------------------------------------
        # MALECNS
        # ----------------------------------------------------

        neural_brain.reset()

        spike_counts, _ = (
            neural_brain.rgb_step(
                frame,
                duration_ms=STIMULUS_MS,
                learning=False
            )
        )

        raw_features = (
            spike_counts > 0
        ).astype(
            np.float32
        )

        feat_n = (
            raw_features - mean
        ) / std

        feat_p = (
            feat_n @ components
        )

        # ----------------------------------------------------
        # NEURAL DECODERS
        # ----------------------------------------------------

        fwd, fwd_conf = decode(
            feat_p,
            W_fwd,
            b_fwd
        )

        trn, trn_conf = decode(
            feat_p,
            W_trn,
            b_trn
        )

        cam, cam_conf = decode(
            feat_p,
            W_cam,
            b_cam
        )

        # ----------------------------------------------------
        # STABLE MOVEMENT PREDICTION
        # ----------------------------------------------------

        (
            previous_fwd,
            previous_fwd_count
        ) = stable_prediction(
            fwd,
            previous_fwd,
            previous_fwd_count
        )

        (
            previous_turn,
            previous_turn_count
        ) = stable_prediction(
            trn,
            previous_turn,
            previous_turn_count
        )

        fwd_ready = (
            previous_fwd_count
            >= MOVE_CONFIRM_FRAMES
        )

        turn_ready = (
            previous_turn_count
            >= TURN_CONFIRM_FRAMES
        )

        forward_active = (
            fwd == 1
            and fwd_conf >= THRESHOLD_MOVE
            and fwd_ready
        )

        now = time.time()
        if forward_active:
            last_forward_command = now

        check_for_stuck(frame, forward_active, now)
        check_for_uncertainty(fwd_conf, trn_conf, now)

        # ----------------------------------------------------
        # CURRENT FLY GOAL
        # ----------------------------------------------------

        goal = fly_brain.goal

        # ----------------------------------------------------
        # BRIDGE-MEDIATED PLAYER APPROACH
        # ----------------------------------------------------

        if time.time() < social_stop_until:
            release_all()

        elif time.time() < approach_until:
            set_key("w", approach_distance not in {"near", "close"})
            set_key("s", False)
            set_key("a", False)
            set_key("d", False)

            if approach_x < -0.22:
                set_key("left", True)
                set_key("right", False)
            elif approach_x > 0.22:
                set_key("left", False)
                set_key("right", True)
            else:
                set_key("left", False)
                set_key("right", False)

        # ----------------------------------------------------
        # OBSERVE MODE
        # ----------------------------------------------------

        elif goal == "OBSERVE":

            # Stop walking while inspecting.
            set_key(
                "w",
                False
            )

            set_key(
                "s",
                False
            )

            set_key(
                "a",
                False
            )

            set_key(
                "d",
                False
            )

            # Actively scan the surroundings.
            if (
                observe_pulses
                < OBSERVE_SCAN_PULSES
            ):

                observation_scan()

            else:

                # Tell FlyBrain the inspection
                # has finished.
                fly_brain.finish_inspection(
                    "OBSERVATION_COMPLETE"
                )

                observe_pulses = 0

        # ----------------------------------------------------
        # NORMAL MOVEMENT
        # ----------------------------------------------------

        else:

            # Forward / backward
            forward_hold = (
                now - last_forward_command
                <= FORWARD_GRACE_SECONDS
                and fwd != 2
            )

            if forward_active or forward_hold:

                set_key(
                    "w",
                    True
                )

                set_key(
                    "s",
                    False
                )

            elif (
                fwd == 2
                and fwd_conf
                >= THRESHOLD_MOVE
                and fwd_ready
            ):

                set_key(
                    "w",
                    False
                )

                set_key(
                    "s",
                    True
                )

            else:

                set_key(
                    "w",
                    False
                )

                set_key(
                    "s",
                    False
                )

            # A / D
            if (
                trn == 1
                and trn_conf
                >= THRESHOLD_MOVE
                and turn_ready
            ):

                set_key(
                    "a",
                    True
                )

                set_key(
                    "d",
                    False
                )

            elif (
                trn == 2
                and trn_conf
                >= THRESHOLD_MOVE
                and turn_ready
            ):

                set_key(
                    "a",
                    False
                )

                set_key(
                    "d",
                    True
                )

            else:

                set_key(
                    "a",
                    False
                )

                set_key(
                    "d",
                    False
                )

            # ------------------------------------------------
            # NEURAL CAMERA
            # ------------------------------------------------

            if CAMERA_ENABLED:

                if (
                    cam == 1
                    and cam_conf
                    >= THRESHOLD_CAMERA
                ):

                    camera_pulse(
                        "left"
                    )

                elif (
                    cam == 2
                    and cam_conf
                    >= THRESHOLD_CAMERA
                ):

                    camera_pulse(
                        "right"
                    )

                else:

                    # No strong neural camera signal.
                    # Let the fly look around itself.
                    spontaneous_look()

        # ----------------------------------------------------
        # DISPLAY
        # ----------------------------------------------------

        fwd_lbl = [
            "---",
            "W",
            "S"
        ][fwd]

        turn_lbl = [
            "---",
            "A",
            "D"
        ][trn]

        cam_lbl = [
            "---",
            "<--",
            "-->"
        ][cam]

        print(
            f"\r"
            f"goal={goal:<8} "
            f"fwd={fwd_lbl} "
            f"({fwd_conf * 100:4.1f}%)  "
            f"turn={turn_lbl} "
            f"({trn_conf * 100:4.1f}%)  "
            f"cam={cam_lbl} "
            f"({cam_conf * 100:4.1f}%)  "
            f"target={str(fly_brain.target):<16}",
            end="",
            flush=True
        )

        # ----------------------------------------------------
        # TIMING
        # ----------------------------------------------------

        elapsed = (
            time.perf_counter()
            - loop_start
        )

        remaining = (
            interval
            - elapsed
        )

        if remaining > 0:

            time.sleep(
                remaining
            )


except KeyboardInterrupt:

    print()
    print()

except Exception as exc:

    print(f"\n[BRIDGE CRASH] {type(exc).__name__}: {exc}")
    traceback.print_exc()

finally:

    release_all()

    print(
        "DOOMFLY STOPPED."
    )

    print(
        "All movement and camera keys released."
    )
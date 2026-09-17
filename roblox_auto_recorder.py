import time
import random
import numpy as np
import mss
from PIL import Image
from pynput import keyboard as kb
from pynput.keyboard import Controller, KeyCode, Key
from pathlib import Path

from doom_learning_v6.visual import VisualMemoryBrain


# ============================================================
# SETTINGS
# ============================================================

TOTAL_PER_KEY = 100

# Each action: (label, keys_to_press)
ACTIONS = [
    ("W",          ["w"]),
    ("A",          ["a"]),
    ("S",          ["s"]),
    ("D",          ["d"]),
    ("CAM_LEFT",   [Key.left]),
    ("CAM_RIGHT",  [Key.right]),
    ("W_CAM_LEFT", ["w", Key.left]),
    ("W_CAM_RIGHT",["w", Key.right]),
]

HOLD_TIME   = 0.8
REST_TIME   = 0.5
START_DELAY = 0.15

SCREEN_WIDTH  = 640
SCREEN_HEIGHT = 480

OUTPUT_FILE = Path("roblox_controlled_demonstrations.npz")


# ============================================================
# ESCAPE LISTENER
# ============================================================

escape_pressed = False

def on_press(key):
    global escape_pressed
    if key == kb.Key.esc:
        escape_pressed = True
        return False

listener = kb.Listener(on_press=on_press)
listener.start()


# ============================================================
# KEYBOARD CONTROLLER
# ============================================================

controller = Controller()


# ============================================================
# SCREEN CAPTURE
# ============================================================

sct = mss.MSS()
monitor = sct.monitors[1]

print()
print("Screen detected:")
print(f"  {monitor['width']} x {monitor['height']}")
print()

def capture_screen():
    shot = sct.grab(monitor)
    frame = np.asarray(shot, dtype=np.uint8)
    frame = frame[:, :, :3][:, :, ::-1]
    image = Image.fromarray(frame, "RGB")
    image = image.resize((SCREEN_WIDTH, SCREEN_HEIGHT))
    return np.asarray(image, dtype=np.uint8)


# ============================================================
# LOAD MALECNS
# ============================================================

print("=" * 70)
print("AUTO ROBLOX NEURAL RECORDER")
print("Keys are pressed AUTOMATICALLY - no human input needed")
print("=" * 70)
print()

print("Actions that will be recorded:")
for action_label, keys in ACTIONS:
    key_names = [k.name if isinstance(k, Key) else k.upper() for k in keys]
    print(f"  {action_label:15s} -> {' + '.join(key_names)}")
print()

print("Loading MaleCNS...")
brain = VisualMemoryBrain()
print(f"MaleCNS loaded: {brain.n:,} neurons")
print()


# ============================================================
# LOAD PREVIOUS DATA
# ============================================================

all_features = []
all_labels   = []

if OUTPUT_FILE.exists():
    print(f"Found existing recording file: {OUTPUT_FILE}")
    old = np.load(OUTPUT_FILE)
    all_features = list(old["features"])
    all_labels   = list(old["labels"])
    print(f"Loaded {len(all_features)} previous trials.")
    print()

action_labels = [a[0] for a in ACTIONS]

counts = {
    label: sum(str(x) == label for x in all_labels)
    for label in action_labels
}

print("Existing trials:")
for label in action_labels:
    print(f"  {label:15s}: {counts[label]}/{TOTAL_PER_KEY}")
print()


# ============================================================
# BUILD RANDOMIZED TRIAL ORDER
# ============================================================

remaining = []
for label in action_labels:
    n = TOTAL_PER_KEY - counts[label]
    if n > 0:
        remaining.extend([label] * n)

random.shuffle(remaining)
total_remaining = len(remaining)

if total_remaining == 0:
    print("All trials already complete!")
    listener.stop()
    exit(0)

estimated_minutes = (total_remaining * (HOLD_TIME + REST_TIME + START_DELAY)) / 60
print(f"Trials remaining: {total_remaining}")
print(f"Estimated time:   ~{estimated_minutes:.1f} minutes")
print()


# ============================================================
# READY
# ============================================================

print("=" * 70)
print("INSTRUCTIONS")
print("=" * 70)
print()
print("1. Open Roblox and get into an open area.")
print("2. Make sure Roblox window is FOCUSED.")
print("3. Press ENTER here to start.")
print()
print("The script presses keys automatically.")
print("Press ESC to stop safely at any time.")
print()

input("Press ENTER when Roblox is ready and focused...")
print()
print("Starting in 3 seconds - click on Roblox now!")
time.sleep(3)


# ============================================================
# TRIAL LOOP
# ============================================================

action_map = {label: keys for label, keys in ACTIONS}
completed_this_session = 0

for target_label in remaining:

    if escape_pressed:
        print()
        print("ESC detected. Stopping safely.")
        break

    completed    = len(all_features)
    total_target = TOTAL_PER_KEY * len(ACTIONS)
    keys_to_press = action_map[target_label]

    print()
    print("=" * 70)
    print(f"TRIAL {completed + 1}/{total_target}  |  Auto-pressing: {target_label}")
    print("=" * 70)

    # Press all keys for this action
    for k in keys_to_press:
        controller.press(k)

    time.sleep(START_DELAY)

    # Record neural activity
    print("Recording...")
    neural_samples = []
    start_time = time.time()

    while time.time() - start_time < HOLD_TIME:

        if escape_pressed:
            for k in keys_to_press:
                controller.release(k)
            break

        frame = capture_screen()
        brain.reset()
        spike_counts, _ = brain.rgb_step(frame, duration_ms=50, learning=False)
        frame_features = (spike_counts > 0).astype(np.float32)
        neural_samples.append(frame_features)

        time.sleep(0.05)

    # Release all keys
    for k in keys_to_press:
        controller.release(k)

    if escape_pressed:
        print("ESC detected. Stopping safely.")
        break

    if not neural_samples:
        print("No neural samples. Trial discarded.")
        continue

    feature_vector = np.mean(
        np.asarray(neural_samples), axis=0
    ).astype(np.float32)

    all_features.append(feature_vector)
    all_labels.append(target_label)
    counts[target_label] += 1
    completed_this_session += 1

    np.savez_compressed(
        OUTPUT_FILE,
        features=np.asarray(all_features, dtype=np.float32),
        labels=np.asarray(all_labels)
    )

    print(f"Saved. (session total: {completed_this_session})")
    time.sleep(REST_TIME)


# ============================================================
# FINISHED
# ============================================================

listener.stop()

print()
print("=" * 70)
print("RECORDING FINISHED")
print("=" * 70)
print()
print(f"Trials this session: {completed_this_session}")
print(f"Total trials saved:  {len(all_features)}")
print()
for label in action_labels:
    print(f"  {label:15s}: {counts[label]}/{TOTAL_PER_KEY}")
print()
print(f"Saved to: {OUTPUT_FILE}")
print()
print("Next step: run train_roblox_decoder.py")
print()
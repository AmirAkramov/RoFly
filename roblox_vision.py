import time
import mss
import numpy as np
import cv2


# ============================================================
# DOOMFLY VISION
# ============================================================
#
# Completely invisible screen observation.
#
# NO windows.
# NO screenshots displayed.
# NO mouse control.
# NO keyboard control.
# NO Roblox modification.
#
# The program silently captures the screen and analyzes it.
# ============================================================


SCREEN = {
    "left": 0,
    "top": 0,
    "width": 1920,
    "height": 1080,
}

SCAN_INTERVAL = 0.1


# ============================================================
# CAPTURE
# ============================================================

def capture_screen(sct):

    screenshot = sct.grab(SCREEN)

    frame = np.array(
        screenshot
    )

    # BGRA → BGR
    frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGRA2BGR
    )

    return frame


# ============================================================
# BASIC ANALYSIS
# ============================================================

def analyze_screen(frame):

    height, width = frame.shape[:2]

    return {
        "width": width,
        "height": height,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("========================================")
    print("       DOOMFLY VISION ONLINE")
    print("========================================")
    print()
    print("Screen capture: ACTIVE")
    print("Display window : NONE")
    print("Roblox control : NONE")
    print()
    print("Watching silently...")
    print("Press CTRL+C to stop.")
    print()

    with mss.MSS() as sct:

        try:

            while True:

                start = time.time()

                # Capture the screen silently.
                frame = capture_screen(
                    sct
                )

                # Analyze it.
                information = analyze_screen(
                    frame
                )

                elapsed = time.time() - start

                fps = (
                    1 / elapsed
                    if elapsed > 0
                    else 0
                )

                print(
                    f"\rVision FPS: {fps:5.1f} | "
                    f"Screen: "
                    f"{information['width']}x"
                    f"{information['height']}",
                    end=""
                )

                time.sleep(
                    SCAN_INTERVAL
                )

        except KeyboardInterrupt:

            print()
            print()
            print("DoomFly Vision stopped.")


if __name__ == "__main__":
    main()
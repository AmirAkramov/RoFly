"""Start the Roblox movement bridge and social companion together."""

import signal
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
CHILD_SCRIPTS = (
    ROOT / "roblox_bridge.py",
    ROOT / "roblox_social_bridge.py",
)


processes = []


def stop_processes(*_args):
    for process in processes:
        if process.poll() is None:
            process.terminate()

    deadline = time.time() + 5.0
    for process in processes:
        if process.poll() is None:
            remaining = max(0.0, deadline - time.time())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                process.kill()


def main():
    missing = [str(path.name) for path in CHILD_SCRIPTS if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Roblox runtime scripts: " + ", ".join(missing)
        )

    signal.signal(signal.SIGINT, stop_processes)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, stop_processes)

    print("Starting Roblox bridge and social companion...")
    print("Press Ctrl+C once to stop both processes.")

    try:
        for script in CHILD_SCRIPTS:
            processes.append(
                subprocess.Popen(
                    [sys.executable, str(script)],
                    cwd=str(ROOT),
                )
            )

        while True:
            for process, script in zip(processes, CHILD_SCRIPTS):
                return_code = process.poll()
                if return_code is not None:
                    print(
                        f"{script.name} stopped with exit code {return_code}."
                    )
                    stop_processes()
                    return return_code
            time.sleep(0.5)

    except KeyboardInterrupt:
        return 0
    finally:
        stop_processes()
        print("Roblox bridge and social companion stopped.")


if __name__ == "__main__":
    raise SystemExit(main())

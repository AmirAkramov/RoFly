"""Small file-based command channel shared by the Roblox runtime and bridge."""

import json
import os
import tempfile
import time


COMMAND_FILE = "roblox_bridge_command.json"


def _write_command(command):
    directory = os.path.dirname(os.path.abspath(COMMAND_FILE)) or "."
    fd, temporary = tempfile.mkstemp(
        prefix="roblox_bridge_",
        suffix=".tmp",
        dir=directory,
    )

    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(command, stream)
        os.replace(temporary, COMMAND_FILE)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def send_approach(x, distance, greet=False, target="Unknown Player"):
    command = {
        "action": "approach",
        "x": float(x),
        "distance": distance,
        "greet": bool(greet),
        "target": str(target),
        "time": time.time(),
    }

    _write_command(command)


def send_stop(duration=5.0):
    _write_command({
        "action": "stop",
        "duration": float(duration),
        "time": time.time(),
    })


def send_resume():
    _write_command({
        "action": "resume",
        "time": time.time(),
    })


def send_say(message, pause=4.0):
    _write_command({
        "action": "say",
        "message": str(message),
        "duration": float(pause),
        "time": time.time(),
    })


def read_command():
    try:
        with open(COMMAND_FILE, "r", encoding="utf-8") as stream:
            command = json.load(stream)
        os.remove(COMMAND_FILE)
        return command
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None

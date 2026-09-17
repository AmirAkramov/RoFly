# RoFLy 3.1

RoFLy 3.1 is an experimental Roblox agent built from DoomFly-derived
components. It connects a simulated MaleCNS-inspired neural controller to
Roblox screen input, keyboard control, OCR, semantic vision, social behavior,
persistent state, and a pretrained movement decoder.

## Roblox requirement

Roblox is **not included** with RoFLy.

Before running RoFLy, download and install the official Roblox application
manually from the official Roblox website:

<https://www.roblox.com/download>

You must launch Roblox, enter the desired game, and keep the Roblox client open
while RoFLy is running. RoFLy does not distribute, install, modify, or include
the Roblox application or Roblox game assets.

## Run RoFLy 3.1

1. Install Roblox separately.
2. Open Roblox and enter the desired game.
3. Keep the Roblox client open.
4. Open a terminal in the RoFLy folder.
5. Start the neural host if required by your installation:

```powershell
python -u -m doom.server --model experimental-v6 --port 8766
```

6. Open a second terminal in the same folder.
7. Run the RoFLy launcher:

```powershell
python run_roblox_social.py
```

8. Focus the Roblox client when prompted.

The included `roblox_decoder.npz` is the pretrained runtime decoder. Users do
not need to repeat the original demonstration-collection process for normal
operation.

## Software and external application

RoFLy is software that communicates with a separately installed Roblox client
through screen capture, OCR, and keyboard input. Roblox is an external
application and is not part of this distribution.

Users are responsible for installing Roblox, complying with Roblox's Terms of
Use and applicable game rules, and ensuring that automated input is permitted
in the experience they use.
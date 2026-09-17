# RoFLy 3.1

**RoFLy 3.1** is an advanced, bio-inspired experimental autonomous agent designed for the Roblox ecosystem. Built upon highly optimized, DoomFly-derived neural kernels, RoFLy implements a full-graph computational brain simulation utilizing structural data from the biologically reconstructed **MaleCNS v1.0** Drosophila connectome. 

By bridging real-time multi-modal sensory pipelines with a biomimetic neural controller, the agent translates complex environment states directly into human-like behavioral vectors and continuous motor outputs.

---

## Core System Architecture & The Loop

The agent operates in a continuous, high-frequency execution loop structured around four primary subsystems:

1. **Multi-Modal Perception Layer:**
   * **Semantic Vision & Screen Capture:** High-frequency screen-buffer polling provides raw frames used to construct real-time proxy matrices representing spatial orientation and localized entities.
   * **Optical Character Recognition (OCR):** An embedded text-extraction sub-routine continuously parses active chat logs, system prompts, and UI data streams, transforming textual indicators into abstract semantic stimuli.

2. **Connectome-Driven Neural Simulation:**
   * **Scale & Graph Topology:** The core controller hosts the structural blueprint of the MaleCNS v1.0 connectome, executing approximated, non-cropped network dynamics across thousands of biological nodes and millions of synaptic connections tracked via Git LFS.
   * **Signal Propagation:** Environmental stimuli and parsed text tokens trigger specific sensory input zones (simulated retinal arrays and high-order neuropils). Neural activity cascades dynamically through the retained biological graph structure.

3. **Pretrained Latent Decoding:**
   * **Movement Vector Inference:** Motor function is governed by an integrated, high-fidelity latency-matched neural movement decoder (`roblox_decoder.npz`). 
   * **Execution:** Instead of reliance on traditional heuristics or rule-based game macros, biological descending neuron outputs (DN paths) are processed by the pretrained decoder to dynamically modulate continuous continuous keystrokes, orientation updates, and actions.

4. **Persistent Behavioral Memory:**
   * High-order state loops and internal neural baselines persist seamlessly across environment resets or round transitions, ensuring stable, context-aware macro behavior and authentic social responses.

---

## Roblox Host Application Requirements

RoFLy is an independent research pipeline that interacts with the target environment completely externally. **The Roblox client application is not bundled, modified, or distributed with this source code.**

### Mandated Setup:
* Prior to launching the neural controller, download and install the official client platform directly from the [Roblox Distribution Portal](https://roblox.com).
* You must launch your chosen Roblox experience manually and maintain the client window in an active, unminimized state on your display. 
* RoFLy relies strictly on standard operating system hooks (Window Screen Capture, OCR Intercepts, and Simulated Key Events) to achieve closed-loop control without binary injection or memory modification.

## Setup & Data Integrity Verification

Executing full-graph biological networks requires substantial physical memory resources and high-performance instruction processing. RoFLy cannot be executed within lightweight edge runtimes or client browsers.

### 1. Environment Initialization
Ensure **Python 3.11** and a valid C++ compilation environment are present on the host OS:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 2. Connectome Synchronization
Due to size boundaries on traditional Git structures, the underlying biological graph topology files are stored via Git Large File Storage (LFS). Ensure your local index is fully unfurled and audited:
```bash
git lfs pull
python -m doom.audit_data
```

---

## Executing the Simulation Pipeline

1. Launch your standalone Roblox instance, join the target game server, and position the window visibly.
2. Initialize the background biological server node to compute full-graph synapse dynamics:
   ```bash
   python -u -m doom.server --model experimental-v6 --port 8766
   ```
3. Open a secondary terminal, execute the main multi-modal client handler, and immediately focus (click on) the active Roblox window when prompted:
   ```bash
   python run_roblox_social.py
   ```

*Note: Normal operations do not require reprocessing or collecting new demonstration data arrays; the deployed `roblox_decoder.npz` provides the fully optimized operational baseline.*

---

## Software Boundaries & Scientific Disclaimers

* **Independent Scope:** RoFLy 3.1 is an academic research software suite. It is entirely unaffiliated with, unendorsed by, and disconnected from Roblox Corporation, id Software, or upstream connectome research consortiums. All respective platform trademarks and digital rights remain with their legal holders.
* **Biological Validity:** While the graph architecture mirrors structural findings from the MaleCNS v1.0 biological survey, the artificial stimulation arrays, linguistic inputs, numerical dynamics, and target app actions are modeled engineering parameters. Passing integrated code tests represents structural and software stability, not verified biological correctness.

---

## License & Attribution

The core algorithmic adaptations, runtime components, and interfacing wrappers unique to RoFLy 3.1 are published under the **MIT License**. Upstream network datasets, integrated artwork components, and external library modules retain their specific parent licenses (including Creative Commons Attribution 4.0 International for MaleCNS v1.0 data arrays). Review the localized `LICENSE` records and `THIRD_PARTY_NOTICES.md` for explicit boundaries.

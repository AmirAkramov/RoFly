# RoFLy 3.1

<p align="center">
  <img src="Rofly.png" alt="RoFLy 3.1 Banner" width="750">
</p>

**RoFLy 3.1** is an experimental autonomous AI agent engineered specifically for the Roblox ecosystem. The project utilizes a full-graph simulation of the **MaleCNS v1.0** Drosophila (fruit fly) connectome as its core neural controller, bridging virtual synaptic dynamics with live gameplay through real-time screen capture, Optical Character Recognition (OCR), and simulated hardware input.

Instead of relying on rigid, hard-coded game macros or heuristic bot rules, RoFLy ingests multi-modal data streams directly from the Roblox client window, processes these stimuli through a multi-million-connection graph model, and outputs continuous behavioral and motor vectors.

---

## Architectural Blueprint: The Loop

The agent runs on a high-frequency execution loop structured across four proprietary core pillars:

*   **The Perception Layer:** The runtime continuously polls the active Roblox client desktop buffer. A semantic vision module extracts spatial object proximities, while a parallel OCR thread intercepts active chat logs, system prompts, and UI notification text—converting raw environmental changes into abstract neural stimuli.
*   **The Connectome Controller:** The system maps these sensory and linguistic inputs directly onto targeted entry-points within a full-scale simulation of the MaleCNS v1.0 biological graph structure. Signals cascade natively across millions of synaptic paths without aggressive circuit clipping.
*   **The Latent Decoder (`roblox_decoder.npz`):** At the network's boundary, descending neural firing patterns are intercepted by a high-fidelity, pretrained latent movement decoder. This translation layer converts raw graph activity into precise physical keystroke durations and navigation inputs.
*   **Social State & Persistence:** The internal neural baselines and memory thresholds persist seamlessly across character resets and environmental transitions. This allows the agent to maintain contextual awareness and deliver organic social responses to other players in the game chat.

---

## Technical Constraints & Boundaries

Simulating a full biological connectome requires considerable local compute infrastructure. The simulation cannot run inside lightweight edge routines or standard web browsers.

1. **Environment:** Requires **Python 3.11** along with a local C++ compilation toolchain to host the native high-performance execution kernel.
2. **External Client Application (Roblox):** The agent interacts with the target game completely externally. **This repository does not distribute, host, or modify Roblox assets or binaries.** Control is maintained entirely via operating system hooks without memory injection.

---

## Quick Start

### 1. Dependency Initialization
Set up your virtual environment and install the required dependencies directly inside the project root:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Because the network data topologies (`edges.feather` and `edges.arrow`) total over 1.3 GB, verify that your Git LFS index has successfully pulled down the raw assets:
```bash
git lfs pull
python -m doom.audit_data
```

### 2. Launching the Pipeline
1. Open your separately installed standalone **Roblox** client, enter your target experience/server, and position the window visibly on your screen.
2. In your first terminal window, spin up the biological backend node responsible for resolving synapse dynamics:
   ```bash
   python -u -m doom.server --model experimental-v6 --port 8766
   ```
3. In a second terminal window, initiate the primary multi-modal client orchestration script:
   ```bash
   python run_roblox_social.py
   ```
4. **Important:** As soon as the terminal script prompts you, immediately click on your Roblox window to transfer active hardware input focus to the game.

*Note: The included `roblox_decoder.npz` contains fully optimized weights out of the box. Users do not need to recapture behavioral demonstrations or rerun the training routines for basic operations.*

---

## Disclaimers & Legal Boundaries

*   **Affiliation:** RoFLy 3.1 is an independent research platform. It is completely unaffiliated with, unendorsed by, and disconnected from Roblox Corporation, id Software, Bethesda, or the scientific consortiums behind the original MaleCNS dataset. All respective trademarks belong to their legal owners.
*   **Scientific Validity:** While the core graph topology faithfully replicates structural biological connectivity matrices, the artificial input mapping, language processing routines, and target engine interactions are purely engineering designs. Passing software unit tests validates framework stability, not literal biological replication.

---

## License & Attribution

The custom orchestration architecture, input wrappers, and integration modules unique to RoFLy 3.1 are licensed under the **MIT License**. Underlying network datasets and specific third-party components retain their parent licenses (including Creative Commons Attribution 4.0 International for the MaleCNS v1.0 records). See `LICENSE` and `THIRD_PARTY_NOTICES.md` for individual scopes.

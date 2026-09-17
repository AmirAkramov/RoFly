# RoFLy 3.1 third-party notices

RoFLy 3.1 is an independent Roblox software project. It is not affiliated with
or endorsed by Google Research, the MaleCNS collaboration, Roblox, ViZDoom,
Freedoom, id Software, Bethesda, or ZeniMax.

This file describes third-party material retained in the RoFLy 3.1 release.

## DoomFly-derived components

RoFLy is based on DoomFly components. DoomFly-derived source retains the
original copyright and license notices. RoFLy-specific Roblox integration,
social behavior, OCR/vision wiring, runtime coordination, and persistent-state
changes are project modifications.

See the root `LICENSE` and `THIRD_PARTY.md`.

## MaleCNS v1.0

MaleCNS-derived data, annotations, connectivity information, and provenance
metadata retain their original CC BY 4.0 terms.

Source:

https://male-cns.janelia.org/download/

License text:

`licenses/CC-BY-4.0.txt`

The MaleCNS creators and affiliated institutions do not endorse RoFLy unless
they explicitly state otherwise.

RoFLy's neural dynamics, sensory mappings, decoder weights, control mappings,
and analysis code are engineering transformations. They are not measurements
supplied or biologically validated by the MaleCNS creators.

## Shiu model

The RoFLy neural simulator may include or depend on code derived from the Shiu
model. Where that material is included, retain the MIT notice:

`licenses/Shiu-model-MIT.txt`

Source:

https://github.com/philshiu/Drosophila_brain_model

## Roblox

Roblox is not included with RoFLy. Users must download and install the official
Roblox application separately:

https://www.roblox.com/download

RoFLy does not distribute Roblox binaries, Roblox game assets, or Roblox
content. Roblox and related trademarks belong to their respective owners.

Users are responsible for complying with Roblox's terms and any rules
applicable to automated input in the experience they use.

## Excluded DoomFly material

The following are not part of the RoFLy 3.1 Roblox-only release:

- ViZDoom runtime and interface
- Freedoom artwork and WAD assets
- Doom scenarios and arenas
- Doom spectator UI and frontend packages
- Doom training reports and experiment outputs
- Doom-specific deployment files
- Doom-specific test suites

Their notices belong with the archived full DoomFly distribution, not this
Roblox-only release.
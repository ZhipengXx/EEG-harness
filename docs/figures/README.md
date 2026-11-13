# EEG-harness visual system

The illustrations use the visual language of a scientific instrument: EEG signal traces, clear measurement paths, and readable research decisions. A cool canvas, navy text, blue model stages, and teal evidence paths distinguish what the models decide from what the runtime measures and saves.

PNG exports provide predictable GitHub rendering. SVG sources retain editable vectors, selectable text, and accessible title/description metadata. The canvas is 1440 pixels wide; each PNG export is 2160 pixels wide. The separate platform screenshots retain their original viewport sizes.

| Illustration | PNG export | Editable source |
| --- | --- | --- |
| Project identity | [00-brain-decoding-hero.png](00-brain-decoding-hero.png) | [00-brain-decoding-hero.svg](00-brain-decoding-hero.svg) |
| Two research workflows | [01-system-overview.png](01-system-overview.png) | [01-system-overview.svg](01-system-overview.svg) |
| Current research loop | [02-eeg-agent-collaboration.png](02-eeg-agent-collaboration.png) | [02-eeg-agent-collaboration.svg](02-eeg-agent-collaboration.svg) |
| fMRI screening | [03-fmri-screening-loop.png](03-fmri-screening-loop.png) | [03-fmri-screening-loop.svg](03-fmri-screening-loop.svg) |
| Planner and Critic selection | [04-planning-options.png](04-planning-options.png) | [04-planning-options.svg](04-planning-options.svg) |

## Reading the research figures

The research-loop figure shows the `single_plan_v1` new-method path: Planner → Critic → Designer → Coder/Reviewer → GPU experiment → Analysis → local Observation publication. Paper and experiment memory inform planning; the Critic can retrieve additional experimental history related to the proposals. Observation publication preserves the facts and analysis for the next round.

Confirmation and validation reuse an existing implementation and enter measurement directly. Diagnosis inspects existing evidence without starting a new training job. Auditor and Curator are separate optional paths; paid maintenance defaults to off in single-plan research. They are not mandatory stages between every Analysis and the next Planner.

The planning figure illustrates the supported task categories: new method, confirmation, validation, and diagnosis. A round proposes 2–4 distinct plans allowed by its current state; it need not include one of every category. The Critic compares the complete pool and selects one original proposal. For a new method, Designer receives the selected task, current settings, interfaces, parent source, and explicit execution materials. The planning history packet and comparison rationale are not forwarded wholesale.

Blue identifies model decisions and implementation work. Teal identifies memory, measurements, saved observations, and evidence feedback. Arrows express information or execution flow, not direct messages between independent agents. The runtime coordinates the handoffs.

## Palette and typography

| Token | Value | Use |
| --- | --- | --- |
| Canvas | `#F5F8FC` | Figure background |
| Navy | `#15324F` | Main text and titles |
| Blue | `#315BE8` | Model stages and selected plans |
| Teal | `#008478` | Measurements, memory, and evidence |
| Muted | `#5B6B7C` | Supporting text |
| White | `#FFFFFF` | Workflow nodes |
| Line | `#C9D6E4` | Node boundaries and separators |

DejaVu Sans supplies both headings and body text. Main figure headings use 48 pixels; body copy uses 24–30 pixels on the 1440-pixel canvas. Text is left aligned, labels use sentence case, and line lengths are kept short enough for GitHub's scaled display. The figures use no gradients or shadows.

## Recorded platform screenshots

[`ui-workspace/`](ui-workspace/) contains workbench captures committed with `14ec3e6` on 2026-10-01. Their records and values are synthetic DEMO data. They show that recorded interface revision rather than the current interface. See the [platform gallery](../platform.md) for provenance.

`ui-workspace/ui-workspace-1440-en.png` is an English text-localization edit of the recorded screenshot, not a new browser capture. The original image is retained. The illustration renderers do not refresh application screenshots.

## README navigation

[`navigation/`](navigation/) contains five optional SVG button assets in the same cool canvas, navy, and teal palette. They can be embedded as image links on GitHub. The main README uses compact text links for navigation.

## Re-render

With Python 3, Inkscape, and DejaVu fonts installed, run from the repository root:

```bash
python docs/figures/render_figures.py
python docs/figures/render_planning_options.py
```

The first script regenerates figures 00–03; the second regenerates figure 04 using the same drawing primitives. Both use the Python standard library and Inkscape, without models, API calls, or training data. Running them replaces manual edits to the corresponding SVGs.

To export a manually edited SVG:

```bash
inkscape docs/figures/02-eeg-agent-collaboration.svg \
  --export-type=png --export-width=2160 \
  --export-filename=docs/figures/02-eeg-agent-collaboration.png
```

Brain shapes, activity colors, signal traces, and gallery glyphs are schematic. They are not measured results or anatomical localization maps. The [architecture map](../architecture.md) connects the depicted workflow to the implementation.

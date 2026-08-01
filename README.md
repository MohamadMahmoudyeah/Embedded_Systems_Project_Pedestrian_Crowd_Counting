# Crowd Density Estimation

Real-time pedestrian counting and spatial density (heatmap) estimation from
a sidewalk camera feed. Four interchangeable backends are supported:

- **YOLO** - a YOLOv8 detector fine-tuned on MOT16 pedestrian data (see
  `embeded-project.ipynb`), used with a tracker for per-person boxes and IDs.
- **CSRNet** - specialized crowd-counting model (native density map, no
  boxes/IDs), fine-tuned on ShanghaiTech Part A (see the
  `csrnet-pytorch-parta-*-run-two` notebooks).
- **DM-Count** - a second specialized density-regression model
  ([cvlab-stonybrook/DM-Count](https://github.com/cvlab-stonybrook/DM-Count)),
  VGG19-based, trained with an Optimal-Transport loss instead of CSRNet's
  pixel-wise MSE.
- **P2PNet** - a specialized point-based model
  ([CrowdCounting-P2PNet](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet)),
  a middle ground between YOLO (full boxes) and CSRNet/DM-Count (pure
  density, no localization at all): it outputs one (x, y) point per head.

Only the `person` class is ever counted for YOLO, so unrelated objects in
the scene (cars, motorcycles, ...) are ignored by construction. The three
specialized models have no notion of object classes at all - they regress
crowd density (or head points) directly from pixels, which is both their
strength in dense/occluded crowds and the source of a real failure mode
discussed below and in `REPORT.md` (§8).

## Project structure

```
crowd_estimation/
├── main.py                      # entry point (interactive or headless)
├── core/
│   ├── config.py                 # RunConfig / defaults / JSON load-save
│   ├── source.py                 # camera / video file / image folder input
│   ├── detector.py               # YOLO wrapper -> list[Detection]
│   ├── csrnet.py                 # CSRNet architecture (vendored) + inference
│   ├── dmcount.py                # DM-Count wrapper (imports from a cloned repo)
│   ├── p2pnet.py                 # P2PNet wrapper (imports from a cloned repo) -> list[Detection]
│   ├── types.py                   # shared Detection dataclass (+ from_point for point models)
│   ├── heatmap.py                # instantaneous/cumulative/classic (boxes/points) + native-density renderer
│   ├── counter.py                 # EMA count smoothing
│   ├── renderer.py                # box/marker/HUD drawing
│   ├── output_writer.py           # video / image-sequence export, run naming
│   └── cli.py                      # interactive terminal wizard
├── weights/
│   ├── yolo/                      # best.pt, best.onnx
│   ├── csrnet/                    # *.pth.tar
│   ├── dmcount/                   # *.pth (from a cloned DM-Count repo)
│   └── p2pnet/                    # *.pth (from a cloned P2PNet repo)
├── config/settings.yaml           # default parameters, incl. external repo paths
├── videos/, images/, outputs/     # sample inputs / run outputs
├── deployment/crowd_monitor.service  # systemd template for the board
├── requirements.txt                # base requirements (YOLO backend)
└── requirements-specialized.txt     # extra requirements (torch/torchvision + notes) for the other three
```

## Setup

```bash
pip install -r requirements.txt
```

Place model files under the matching `weights/<backend>/` subfolder - see
`weights/README.txt`. If you plan to use any backend besides YOLO:

```bash
pip install -r requirements-specialized.txt
```

**DM-Count and P2PNet additionally need a local clone of their own repo**
(their model architecture is imported directly from the clone, not
reproduced by hand in this project - see the docstrings at the top of
`core/dmcount.py` and `core/p2pnet.py` for why). Clone them anywhere, then
either fill in `external_repos:` in `config/settings.yaml`, or paste the
path in when the CLI asks for it. P2PNet also needs an ImageNet-pretrained
`vgg16_bn` backbone available locally or downloadable - see
`core/p2pnet.py`'s docstring for the offline pre-download command.

## Usage

Interactive (what you'll run in front of the TA):

```bash
python main.py
```

You will be asked, in order: which backend, then which model file (from
that backend's `weights/` subfolder), the input source (camera / video
file / image folder), the heatmap mode (YOLO/P2PNet only - see below),
the output format, and whether to tweak advanced parameters. Every run
writes its exact configuration to `outputs/run_<timestamp>/run_config.json`.

Headless, for the embedded board later (no prompts, no GUI window):

```bash
python main.py --config outputs/run_20260729_143210/run_config.json --no-display
```

## Output streams

There is no "what to save" question - every run always writes everything
the selected backend can produce, based on what kind of output it's
actually capable of:

- **YOLO and P2PNet** (both localize individual people - boxes or points)
  produce three files: `combined` (markers + count + heatmap - also the
  live preview), `detection` (markers + count, full brightness, no
  heatmap), and `heatmap` (count + heatmap only, darkened background, no
  markers). P2PNet has no track IDs (it runs per-frame with no tracker),
  so its `detection`/`combined` markers are plain circles with no ID label.
- **CSRNet and DM-Count** (pure density regression, no localization at
  all) produce one file: `heatmap` (count + heatmap, darkened background).

Streams that include a heatmap have their background frame darkened
(`heatmap.background_darken`, default `0.6`, i.e. 40% darker) before the
heatmap is overlaid, so the JET colors read clearly; the `detection`-only
stream is left at full brightness.

## Design decisions

**Input source.** Unchanged from the base design: `core/source.py`'s
`FrameSource` abstraction (camera/video/image-folder) means a CCTV/RTSP
camera on the board later is a configuration change, not a code change.

**Heatmap - three modes for instance-based backends, one rejected.** A
single flat overlay derived only from the person count was rejected: it
can't show *where* people are, which the assignment explicitly asks for.
For YOLO and P2PNet (both give per-instance locations every frame),
`instantaneous`/`cumulative`/`classic` modes are available exactly as
before (see `core/heatmap.py`'s docstring for the full rationale on
anchor points and normalization). CSRNet and DM-Count skip this choice
entirely - their networks already output a density map natively, so
there's nothing to build a Gaussian-blob map *from*; their own output is
rendered as-is (through the same adaptive-ceiling normalization,
`core/heatmap.NativeDensityRenderer`, for a consistent look).

**P2PNet's Gaussian radius is fixed, not box-derived.** P2PNet gives a
point, not a box - its pseudo-detections (`Detection.from_point`) use a
small, arbitrary fixed marker size only so they can flow through the same
drawing code as real boxes. Deriving the heatmap's Gaussian radius from
that arbitrary marker size (the way box-based YOLO detections do)
produced visibly smaller, patchier blobs than a size tuned for how a
head-scale point should actually look. So for P2PNet, every heatmap mode
uses a single fixed `heatmap.point_sigma` (default 25px) instead - this
was tuned to match the visual quality of a reference point-heatmap
script (scatter points, blur with one large fixed sigma) that looked
substantially better than the box-derived approach.

**Why DM-Count and P2PNet aren't vendored like CSRNet.** CSRNet's
architecture is small and fully specified in the training notebook (a
VGG16 frontend plus a short dilated-conv backend), so copying it directly
into `core/csrnet.py` was safe. DM-Count and P2PNet's model code is not
something that can be safely retyped from memory - a subtly wrong
reproduction wouldn't necessarily crash, it would just quietly compute
wrong numbers against a checkpoint trained for the *real* architecture,
which is worse than an honest hard dependency. So both backends instead
insert a local clone of their real upstream repo onto `sys.path` and
import the model-building function straight from it
(`from models import vgg19` / `from models import build_model`), exactly
as running their own scripts from inside the cloned repo would.

**Known limitation - specialized-model false positives on unfamiliar
scenes.** CSRNet, DM-Count, and P2PNet were all trained on curated
crowd-counting datasets (ShanghaiTech, etc.) with a specific camera style,
angle, and crowd density. On footage that looks different from that
training distribution, a density-regression model has no semantic concept
of "person" to fall back on - it only learned a correlation between local
image texture and crowd density, so sufficiently person-texture-like
clutter (a flower basket, a cluster of bicycle wheels) can score just as
"real" to it as an actual head. This was directly observed while testing
CSRNet on footage outside MOT16/ShanghaiTech - see `REPORT.md` (§3.4, §8)
for the full writeup, screenshots, and discussion. It's a genuine
generalization/domain-shift limitation, not a bug in this project's code,
and normalization tuning cannot fix it (see `core/heatmap.py`'s
`blend_density_map` docstring for exactly why).

**Mitigation - CSRNet peak-based counting.** `CSRNetConfig.count_method =
"peak_detect"` (`core/csrnet.py`) targets this directly: instead of
summing the whole density map - where a diffuse patch of clutter-driven
density mass counts just as much as a concentrated, person-shaped peak -
it counts the number of distinct local maxima in a smoothed density map.
Diffuse clutter (like a wide, low-amplitude flower-basket blob) tends to
collapse into fewer, broader peaks than the same amount of density mass
spread across several sharp, person-shaped peaks would, so this typically
undercounts clutter relative to genuine heads. It is a heuristic, not a
guarantee: clutter that happens to produce one sharp, head-sized peak is
still counted as a person by this method too - see `REPORT.md` for the
full discussion, including what a complete fix would actually require
(fine-tuning on the target camera's own footage).

**Output format.** Video file vs. numbered image sequence remains a user
choice - video for continuous monitoring, image sequence for periodic
snapshot logging on a bandwidth/storage-constrained embedded device.

**Reproducibility / board readiness.** Every run saves the full
configuration it used as JSON next to its output.
`main.py --config <file> --no-display` replays it with zero terminal
interaction and never touches `cv2.imshow` - exactly what a systemd
service on the board needs (`deployment/crowd_monitor.service`).

## Parameters (`config/settings.yaml`)

| Parameter | Meaning | Applies to |
|---|---|---|
| `detection.conf_threshold` | Minimum confidence to accept a detection/point | YOLO, P2PNet |
| `counting.smoothing` | EMA factor for the displayed count | all backends |
| `heatmap.decay` (cumulative mode) | Fraction of past density kept per frame | YOLO, P2PNet |
| `heatmap.sigma_ratio` / `classic_sigma_ratio` | Gaussian blob radius as a fraction of box size | YOLO |
| `heatmap.point_sigma` | Fixed Gaussian blob radius (pixels), used in all modes since points have no box size | P2PNet |
| `heatmap.intensity_ceiling` | Starting reference for "fully red" | YOLO, P2PNet, DM-Count |
| `heatmap.background_darken` | Brightness multiplier before overlaying any heatmap | all backends |
| `heatmap.noise_floor` | Zeroes out sub-threshold speckle in the display (cosmetic only) | all backends |
| `csrnet.count_method` | `density_sum` (standard) or `peak_detect` (clutter-robustness mitigation) | CSRNet |
| `csrnet.blur_sigma` / `.peak_min_height` / `.peak_nms_radius` | Tuning for the peak_detect method | CSRNet |
| `csrnet.intensity_ceiling` | CSRNet-specific heatmap starting reference (its raw values are much smaller than other backends') | CSRNet |
| `external_repos.dmcount` / `.p2pnet` | Local clone path for that backend's real model code | DM-Count, P2PNet |
| `p2pnet.backbone` / `.row` / `.line` / `.marker_half_size` | P2PNet model-build args + pseudo-box size for rendering | P2PNet |

Only `conf_threshold`, `counting.smoothing`, `heatmap.decay` (cumulative
mode), and `csrnet.count_method` are exposed in the interactive "adjust
parameters?" step; the rest are stable defaults meant to be edited
directly in `config/settings.yaml` if needed.

## Model comparison (for the report)

All four backends can be run over the same source (MOT16 image sequence
or `TownCentreXVID.mp4`) for the report's required comparison - in
particular around occlusion/overlap handling (YOLO as a per-instance
detector vs. P2PNet's point-based localization vs. CSRNet/DM-Count's pure
density regression) and around generalization to unfamiliar camera
styles (§8/§9 of `REPORT.md`).

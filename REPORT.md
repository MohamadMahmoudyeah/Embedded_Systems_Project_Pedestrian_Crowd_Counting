# Sidewalk Crowd Density Estimation — Project Report

**Course:** Embedded Systems
**Deliverable:** Crowd density / pedestrian-counting pipeline with real-time heatmap visualization

---

## 1. Problem Statement and Scope

The assignment asks for a system that, given a single camera feed of a
sidewalk, estimates in real time (a) how many people are present and (b)
where on the sidewalk they are concentrated, via a heatmap. Vehicles and
other irrelevant objects must be filtered out. Off-the-shelf detection
models are allowed and encouraged to be fine-tuned; models trained
specifically and exclusively for this exact assignment are not.

**Clarification from the TA session.** The assignment text says to
*fine-tune* a detector. In the TA session, this was clarified to simply
mean: fine-tune (or otherwise adapt) an existing detector so that it
reliably picks out the `person` class and ignores everything else — a
full custom architecture is not required, and a straightforward
single-class fine-tune of a stock detector satisfies the requirement.

**Additional task from the TA.** Because the base assignment was
considered straightforward, the TA asked for an additional deliverable:
integrate specialized, purpose-built crowd-counting models (as opposed to
a general-purpose object detector repurposed for counting) and compare
their behavior against the YOLO-based approach — in particular, on how
each handles people who visually overlap/occlude one another in the
frame. This project ended up integrating **three** specialized models
rather than one, to get a broader picture of that comparison:

| # | Model | Kind | Output |
|---|---|---|---|
| 1 | YOLOv8 (fine-tuned) | general-purpose detector | bounding boxes + track IDs |
| 2 | CSRNet | specialized, density regression | one density value per pixel |
| 3 | DM-Count | specialized, density regression | one density value per pixel |
| 4 | P2PNet | specialized, point-based | one (x, y) point per head |

This report covers all four, the software architecture built around
them, a real failure mode discovered while testing the specialized models
on footage unlike their training data, and a mitigation implemented for
it.

**Deployment target.** The assignment is framed around an embedded
board reading frames from a mounted camera. For grading, everything is
run on a laptop instead — no board is available for this delivery — but
the code is written so that moving to a board later is a configuration
change, not a rewrite (see §9, "Board readiness").

---

## 2. Model 1: YOLO-based Detector

### 2.1 Data and training pipeline

Training data was MOT16 (`MOT16-03` and related sequences), which ships
with per-frame ground-truth object boxes. The `embeded-project.ipynb`
notebook converts this ground truth into YOLO-format labels (normalized
`class x_center y_center width height` per line) and keeps only the
`person` class, discarding all other MOT16 object categories so the
model never learns to key off vehicles or other non-person classes in
the first place — filtering happens at training-data level, not just at
inference time.

### 2.2 Training attempts

Three fine-tuning attempts were made, in this order:

1. **`yolov8m.pt`, quick attempt.** A first pass fine-tuning YOLOv8
   Medium directly. Performance was noticeably weak for this use case —
   in particular on partially-overlapping pedestrians — so this
   configuration was abandoned in favor of a smaller, faster-to-iterate
   model.

2. **`yolov8n.pt`, quick fine-tune.**

   ```python
   from ultralytics import YOLO

   model = YOLO('yolov8n.pt')

   results = model.train(
       data='/kaggle/working/yolo_dataset/data.yaml',
       epochs=15,
       imgsz=640,
       batch=32,
       project='/kaggle/working/runs',
       name='crowd_yolo',
       device='0'
   )
   ```

   This established a fast baseline and confirmed the data pipeline
   (label conversion, single-class filtering) was correct, but a Nano
   model has limited capacity for a crowded-scene task with a lot of
   partial occlusion.

3. **`yolov8m.pt`, robust fine-tune (final model).** Went back to the
   Medium model, but trained more carefully this time to avoid the
   problems seen in attempt 1 — early stopping to avoid overfitting on a
   fairly small fine-tuning set, a frozen backbone to preserve
   COCO-pretrained low-level features, and dropout for regularization:

   ```python
   from ultralytics import YOLO

   model = YOLO('yolov8m.pt')

   results = model.train(
       data='/kaggle/working/yolo_dataset/data.yaml',
       epochs=15,
       patience=5,
       imgsz=640,
       batch=16,
       freeze=10,
       dropout=0.2,
       project='/kaggle/working/runs',
       name='crowd_yolo_medium_robust',
       device='0'
   )
   ```

   Training stopped early via the `patience=5` early-stopping criterion
   after 6 epochs (best weights from epoch 1), consistent with the
   `crowd_yolo_medium_robust` run recorded in the notebook. At the best
   epoch, validation metrics were approximately:

   | Metric | Value |
   |---|---|
   | Precision (box) | 0.838 |
   | Recall (box) | 0.634 |
   | mAP50 | 0.726 |
   | mAP50-95 | 0.427 |

   This became the final model, exported to both PyTorch (`best.pt`) and
   ONNX (`best.onnx`) format. Both are provided in `weights/yolo/`.

*(Re-confirm these figures against the notebook's saved metrics table
before final submission — quote the run's own `results.csv` / validation
printout rather than this report if they differ.)*

**[SCREENSHOT: training curves / results.png from the `crowd_yolo_medium_robust` run, if available]**

### 2.3 Inference pipeline

At inference time, the model is run with Ultralytics' built-in ByteTrack
tracker (`model.track(..., persist=True, tracker="bytetrack.yaml")`) so
that each detected person keeps a stable ID across frames — this is used
for the `detection`/`combined` output overlays and does not affect the
per-frame person count, which is simply the number of boxes above the
confidence threshold in that frame. `classes=[0]` restricts detections to
the `person` class, which happens to be class index 0 in both our
fine-tuned single-class model and stock COCO-pretrained YOLO models —
this filter is what guarantees cars/motorcycles are never counted, per
the assignment's explicit requirement, regardless of which YOLO
checkpoint is loaded.

---

## 3. Model 2: CSRNet (Specialized Crowd-Counting Model)

### 3.1 Why CSRNet

CSRNet (Li, Zhang & Chen, *CSRNet: Dilated Convolutional Neural Networks
for Understanding the Highly Congested Scenes*, CVPR 2018) is a
purpose-built crowd-counting architecture: a VGG16 front-end extracts
features, and a dilated-convolution back-end (dilation preserves spatial
resolution while growing receptive field, avoiding further downsampling)
regresses a **density map** directly — one continuous value per output
pixel representing local crowd density, not a discrete set of bounding
boxes. The estimated person count is simply the sum of that map. This is
architecturally very different from YOLO: there is no notion of
"detecting an instance," only "how much crowd-mass is present at this
location," which is exactly the property expected to help in dense,
overlapping crowds where individual instance boundaries are ambiguous.

### 3.2 Training

The checkpoints used here (`task_two_model_best.pth.tar`) come from two
Kaggle notebooks provided alongside this report
(`csrnet-pytorch-parta-90-120-run-two.ipynb` and
`csrnet-pytorch-parta-120-150-run-two.ipynb`), each continuing training
from a checkpoint saved by an earlier notebook in the same chain
(`...-60-90-run-two` → `...-90-120-run-two` → `...-120-150-run-two`) —
training was split across multiple Kaggle sessions due to the platform's
per-session time limit. Data was ShanghaiTech Part A: images with
head-annotated ground truth, converted ahead of time into
Gaussian-blurred density maps (`ground-truth-h5/`), downsampled by 8x to
match the network's output resolution (three 2x max-pools in the
frontend), with the target rescaled by 64x after downsampling
specifically so that the *sum* of the downsampled target still equals
the original head count. Loss was standard pixel-wise MSE between
predicted and target density maps, optimized with SGD at a very small
learning rate (`1e-7`), which is typical for *continuing* an already
largely-converged CSRNet checkpoint rather than training from scratch.

**Methodological note on the reported numbers.** Each notebook computes
its own random 90/10 train/validation split of the *training* data fresh
on each run (`get_train_val_list`), so the "best MAE" reported during
training (≈16 in the 120–150 run's later epochs, ≈27 in the 90–120 run)
is measured against a different, easy, in-distribution validation split
each time — not the official ShanghaiTech Part A **test** set, and not
comparable between the two notebooks. The number that actually matters
for comparing against the other models is the held-out test-set
evaluation run at the end of each notebook:

| Checkpoint | Test MAE (last epoch) | Test MAE (best-val epoch) | Test MSE (last) | Test MSE (best-val) |
|---|---|---|---|---|
| `...-90-120-run-two` | 72.033 | 71.895 | 104.761 | 104.870 |
| `...-120-150-run-two` | 72.009 | 70.603 | 105.431 | 104.828 |

These numbers are consistent with the range reported in the original
CSRNet paper for ShanghaiTech Part A (MAE ≈ 68, MSE ≈ 115), so despite
looking large in absolute terms (crowds in this dataset can number in the
hundreds), the model is behaving in line with published results for this
architecture on this dataset — the large gap versus the in-training
validation numbers above is a property of the evaluation split, not a
bug or a sign of a broken checkpoint.

### 3.3 Integration into this project

CSRNet is wired in as a backend (`core/csrnet.py`), with its
architecture reproduced directly in this project — safe to do here
specifically because the architecture is small and fully specified in
the training notebook (VGG16 frontend + a short dilated-conv backend),
unlike DM-Count/P2PNet below. Because CSRNet has **no bounding boxes, no
track IDs, and no confidence threshold** — it is a direct
image-to-density-map regressor — the pipeline adapts around it: the
heatmap-mode question (§6) is skipped entirely, since it's specifically
about turning *per-person locations* into a Gaussian density map, which
doesn't apply when the network already outputs a dense map natively; and
only one output stream is produced (`heatmap`: count + heatmap), since
there's nothing to draw a `detection`-only or `combined` marker overlay
from.

`CSRNet(load_weights=True)` is used at inference time — deliberately
different from the training notebooks' default of `load_weights=False`
— because that flag controls whether ImageNet-pretrained VGG16 weights
are downloaded from the internet before being immediately overwritten by
the checkpoint being loaded. Skipping that download keeps this backend
usable offline, which matters for the board-readiness goal in §9.

### 3.4 Observed failure mode: false positives on unfamiliar footage

While testing CSRNet on footage that looks visually different from
ShanghaiTech Part A (its training distribution) — a Japanese street
scene from an elevated, angled CCTV-style camera, quite unlike
ShanghaiTech's more frontal crowd photography — the model correctly lit
up on most pedestrian heads, but **also** activated on non-person clutter
that happens to share head-like texture statistics: a hanging flower
basket and areas near clustered bicycle wheels both registered visible
heatmap intensity and inflated the count, as shown below.

**[SCREENSHOT: the provided example frame - Count: 54, with heatmap blobs correctly over pedestrian heads but also visibly over a flower basket]**

**Why this happens.** CSRNet (and DM-Count and P2PNet, being trained the
same way on the same kind of data — see §4 and §5) has no semantic
concept of "a person." It only learned a correlation between local image
texture/shape statistics and crowd density from its training set. A
tightly-packed cluster of small, roughly round, similarly-colored,
head-sized shapes — a flower basket, bicycle wheel spokes, foliage — can
trigger a very similar activation pattern to a cluster of heads, purely
because the low-level texture statistics are close enough. This is a
well-documented cross-scene generalization gap for crowd-counting
networks: models trained on one camera style/angle/density regime
predictably degrade on visually different scenes, and there is no
threshold or normalization tweak on the *output* that can fix it, because
the false-positive regions genuinely produce high activations internally
— they are not just numerically small noise sitting near the display
floor. `core/heatmap.py`'s `blend_density_map()` docstring documents this
explicitly, including for the cosmetic `noise_floor` option it exposes
(which only cleans up faint speckle, not this).

**What a full fix would require.** Properly correcting this would mean
either (a) fine-tuning CSRNet/DM-Count/P2PNet on footage from the actual
target camera(s) — which needs new labeled density-map ground truth for
that footage, well beyond this project's scope — or (b) adding an
explicit "is this actually a person-like region" semantic filter ahead of
the density regressor, which is essentially re-introducing an
object-detector concept and defeats the purpose of using a pure
density-based specialized model in the first place. Neither was
attempted here; instead, a narrower, practical mitigation was implemented
for the specific way this failure mode tends to manifest.

**Mitigation implemented: peak-based counting.** `CSRNetConfig` (see
`core/csrnet.py`) exposes `count_method`, with two options:

- `density_sum` (the standard CSRNet method): count = the sum of the
  entire density map. A diffuse patch of clutter-driven density mass
  (e.g. spread broadly and thinly across a flower basket) contributes to
  this sum exactly as much as the same amount of mass concentrated into
  sharp, person-shaped peaks would.
- `peak_detect`: count = the number of distinct local maxima ("peaks") in
  a Gaussian-smoothed density map, one per detected head, using
  non-maximum suppression via grayscale dilation. Diffuse clutter tends
  to collapse into fewer, broader peaks after smoothing than the same
  density mass spread across several genuinely sharp person-shaped peaks
  would, so this typically undercounts clutter relative to real heads.

Tested against a synthetic frame combining three sharp, head-scale
Gaussian peaks (standing in for genuine heads) and one broad, low-
amplitude Gaussian blob with roughly 5 heads' worth of raw integrated
mass (standing in for a diffuse clutter region): `density_sum` reported
≈12.7, while `peak_detect` reported 4 — the 3 genuine peaks plus exactly
1 for the entire diffuse blob, rather than counting its mass as ≈9-10
additional phantom people.

This is explicitly a **heuristic, not a guarantee**: it exploits a
tendency (clutter activations are often more diffuse/spread-out than a
genuine head's sharp response) rather than a certainty. Clutter that
happens to produce one sharp, compact, head-sized peak — which is
plausible for small, dense clutter — would still be counted by
`peak_detect` exactly as it would by `density_sum`. It also does not
change *where* the false-positive heat appears in the displayed heatmap,
only how it's converted to a count; visually, a clutter blob (however
diffuse) is still shown on the heatmap. `CSRNetEstimator` prints a
one-time diagnostic on the first processed frame (raw sum vs. smoothed
sum vs. peak count) specifically so this trade-off can be checked against
real footage rather than assumed.

An earlier version of this project also explored a static, per-camera
exclusion mask (blacking out a known-bad region like a specific flower
basket from both the heatmap and the count). It was removed: masking
requires a human to have already identified every problem region in a
specific fixed camera's exact field of view ahead of time, which is a
much narrower and more fragile precondition than a property of the
density map itself (like peak sharpness) that `peak_detect` can exploit
automatically, without per-camera setup, on any footage.

---

## 4. Model 3: DM-Count (Specialized Crowd-Counting Model)

### 4.1 Why a second density-regression model

[DM-Count](https://github.com/cvlab-stonybrook/DM-Count) (Wang, Zhao,
Deng, Ma, Zhang, Hu — *Distribution Matching for Crowd Counting*, NeurIPS
2020) answers the same question as CSRNet — a per-pixel density map, with
the count being its sum — but from a different backbone (VGG19) and,
more importantly, a fundamentally different training objective: instead
of pixel-wise MSE against a Gaussian-blurred ground-truth map, DM-Count
minimizes an Optimal-Transport-based distribution-matching loss between
predicted and true point distributions. Including it alongside CSRNet
gives a same-*kind*-of-model comparison point (density regression vs.
density regression, rather than only density regression vs. instance
detection vs. points), which helps separate "specialized density models
in general behave this way" from "this particular CSRNet checkpoint
behaves this way."

### 4.2 Integration into this project

Unlike CSRNet, DM-Count's architecture is **not** reproduced by hand in
this project (`core/dmcount.py`). Its model code is not small or simple
enough to safely retype from memory the way CSRNet's ~30-line
frontend+backend was — a subtly wrong reproduction here would not
necessarily crash, it would just silently compute the wrong numbers
against a checkpoint trained for the real architecture, which is a worse
failure than an honest hard dependency. Instead, `DMCountEstimator`
inserts a local clone of the DM-Count repo onto `sys.path` and imports
its model-building function directly (`from models import vgg19`),
exactly like running the repo's own inference script would. The
checkpoint used was `pretrained_models/model_qnrf.pth` from the DM-Count
repo's own released weights (trained on the UCF-QNRF dataset rather than
ShanghaiTech - a further point of comparison, since it means this
backend's training distribution differs from both YOLO/CSRNet's).

Preprocessing intentionally matches DM-Count's own inference script: a
plain `ToTensor()` with **no** ImageNet mean/std normalization (CSRNet
and P2PNet both do normalize) — this was not an oversight, DM-Count's
published models are trained without that normalization step, so adding
it would silently produce incorrect density values.

Like CSRNet, DM-Count has no per-instance localization, so it skips the
heatmap-mode question and produces a single `heatmap` output stream,
rendered through the same shared `core/heatmap.NativeDensityRenderer`
used by CSRNet for a consistent look across both.

### 4.3 Expected behavior vs. CSRNet

Since DM-Count shares CSRNet's core limitation — a learned texture-to-
density correlation with no semantic "person" concept — the same
clutter-triggered false-positive failure mode described in §3.4 is
expected to appear here too, on the same problematic footage, though
possibly with different sensitivity given the different backbone/training
data/loss. This is directly testable by running both backends over the
same source and comparing (see §8).

**[SCREENSHOT / DATA: DM-Count heatmap on the same flower-basket frame as §3.4, for direct comparison against CSRNet]**

---

## 5. Model 4: P2PNet (Specialized, Point-Based Crowd-Counting Model)

### 5.1 A middle ground between boxes and pure density

[P2PNet](https://github.com/TencentYoutuResearch/CrowdCounting-P2PNet)
(Song et al., *Rethinking Counting and Localization in Crowds: A Purely
Point-Based Framework*, ICCV 2021) sits architecturally between YOLO and
the two density-regression models above: it is still a specialized
crowd-counting network (trained on the same kind of crowd-counting data
as CSRNet/DM-Count, with the same lack of a general "vehicle vs. person"
class distinction), but instead of a density surface, it directly
regresses a set of individual **(x, y) point locations**, one per
predicted head, each with a confidence score. This gives it real
per-instance localization — closer in spirit to what YOLO provides —
while still being purpose-built for crowded scenes the way CSRNet/DM-Count
are, rather than being a repurposed general object detector.

### 5.2 Integration into this project

As with DM-Count, P2PNet's architecture is imported from a local clone of
the upstream repo rather than reproduced by hand
(`from models import build_model` in `core/p2pnet.py`), for the same
correctness-risk reasoning given in §4.2. The checkpoint used was
`SHTechA.pth` (ShanghaiTech Part A, matching CSRNet's training set, for a
cleaner controlled comparison between the two ShanghaiTech-trained
models).

Two P2PNet-specific setup requirements, both handled once per machine
rather than per run:

- **Input size.** P2PNet's backbone requires both frame dimensions to be
  a multiple of 128. Frames are resized up to the nearest multiple of 128
  before inference, and detected points are rescaled back to the original
  frame's coordinate system afterward, so downstream rendering lines up
  correctly with the real frame.
- **Backbone weights.** `build_model` loads an ImageNet-pretrained
  `vgg16_bn` backbone via torchvision, which needs to either download
  automatically (~528 MB) or already be present in torch's local model
  cache. On a machine without reliable internet access at run time (as
  was the case for part of this project's own testing, requiring a
  resumed multi-attempt download over a proxy), it can be pre-downloaded
  once and placed in torchvision's default cache location
  (`~/.cache/torch/hub/checkpoints/vgg16_bn-6c64b313.pth`) ahead of time
  — see the setup instructions at the top of `core/p2pnet.py`.

### 5.3 Because P2PNet localizes, it plugs into the instance-based pipeline

Because P2PNet gives a discrete point per person every frame — the same
shape of information YOLO's boxes provide, just without a box size or a
track ID — `core/p2pnet.py`'s `P2PNetEstimator.detect()` returns the same
`list[Detection]` type as YOLO's detector, using a small fixed-size
pseudo-box centered on each point (`Detection.from_point`, since the
Gaussian-blob heatmap math and marker drawing both expect *something*
resembling a box). This means P2PNet gets the full instance-based
treatment other backends can't: the `instantaneous`/`cumulative`/`classic`
heatmap-mode choice (§6) and all three output streams
(`combined`/`detection`/`heatmap`), the same as YOLO — just without track
IDs, since P2PNet runs independently per frame with no tracker.

**Heatmap quality fix: fixed point sigma instead of a box-derived one.**
Initially, P2PNet's heatmap reused YOLO's box-derived Gaussian radius
(`sigma` scaled from the pseudo-box's width), which is only a small,
arbitrary marker size (§5.3 above) rather than a real measurement of
anything - the result was a visibly patchier, less convincing heatmap
than a reference point-heatmap script (scatter points, then blur with one
large, fixed sigma) produced on the same output. `heatmap.point_sigma`
(default 25px, `core/heatmap.py`) fixes this: for P2PNet specifically,
every heatmap mode uses this single fixed radius instead of deriving one
from the pseudo-box, matching the reference script's visual quality while
keeping this project's improved normalization behavior (comparable colors
across frames for instantaneous/cumulative, still-available vivid
per-frame mode for classic) rather than reverting to a naive per-frame
min-max stretch across the board.

### 5.4 Expected behavior vs. YOLO and the density models

P2PNet is expected to occupy a middle position on the occlusion-handling
spectrum discussed in §8: unlike YOLO, it doesn't need to draw a full,
separately-shaped box around each person (which is where a box-based
detector tends to fail on heavy overlap), so it should localize more of
a tightly-packed group correctly than YOLO does; but unlike CSRNet/
DM-Count's fully implicit density surface, it still commits to a
discrete point per head, so in the most extreme occlusion it may still
under-count relative to a pure density approach, which never has to make
that discrete a decision at all.

**[SCREENSHOT: P2PNet `combined` output on an occluded group, alongside YOLO's and a density model's output on the same frame]**

---

## 6. Heatmap Methodology

Heatmap construction differs by what kind of output a backend gives.

### 6.1 An option that was considered and rejected

A single flat-colored overlay derived only from the total person count
(no per-location information) was considered and rejected outright: it
cannot show *where* on the sidewalk people are concentrated, which is
exactly what the assignment explicitly asks for. Every mode described
below carries real spatial information.

### 6.2 Instance-based backends (YOLO, P2PNet): three modes

- **Instantaneous.** The density map is rebuilt from scratch every
  frame. Each detection contributes a 2D Gaussian centered at the
  **bottom-center of their bounding box** (their feet - for P2PNet's
  fixed-size pseudo-boxes, this is effectively the point itself), not the
  box center and not an assumed head position — the feet point is the
  only part of a full-body bounding box that actually lies *on* the
  sidewalk's ground plane. Gaussian radius (`sigma`) scales with box
  width for box-based backends (YOLO); for P2PNet, whose pseudo-box size
  is an arbitrary rendering marker rather than a real measurement, a
  fixed `heatmap.point_sigma` (default 25px) is used instead in every
  mode - see §5.3 for why. Intensity is normalized against a
  **slowly-adapting reference ceiling** rather than a naive per-frame
  min-max stretch (a real issue found in early draft scripts this
  project started from: it makes one lone pedestrian look exactly as
  "fully red" as a packed crowd), so color intensity stays comparable
  *across* frames, not just within one.

- **Cumulative.** Identical to instantaneous, except the density map
  persists across frames with exponential decay (`heatmap.decay`,
  default 0.97) instead of resetting — answers "which spots see
  sustained/repeated traffic over a longer window" (a dwell/hotspot map),
  distinct from but complementary to the instantaneous view.

- **Classic.** Reproduces, close to verbatim, the look of an earlier
  draft script that turned out to look visually convincing despite not
  being comparable frame-to-frame: box-**center**-anchored (not feet), no
  blur, plain per-frame min-max normalization. Always looks visually
  "full", at the cost of the exact caveat instantaneous mode avoids (a
  single person can look as saturated as a crowd). Kept as an explicit,
  labeled choice, not the default, specifically because it was requested
  for how it looks, with the trade-off understood rather than hidden.

### 6.3 Native-density backends (CSRNet, DM-Count): no mode choice

These networks already output a continuous density map directly - there
is no discrete per-person location to build a Gaussian from, so the
instantaneous/cumulative/classic choice doesn't apply and isn't offered.
Their raw output is rendered through the same adaptive-ceiling
normalization as instantaneous mode above
(`core/heatmap.NativeDensityRenderer`, sharing the `blend_density_map()`
helper with the instance-based path), so all four backends' heatmaps are
visually consistent with each other despite the very different
underlying math.

### 6.4 Output darkening

In any output stream that includes a heatmap overlay, the underlying
camera frame is darkened (`heatmap.background_darken`, default `0.6`,
~40% darker) *before* the heatmap is blended on top, so the JET color
scale reads clearly. The `detection`-only stream (YOLO/P2PNet) is left
at full brightness, since it has no heatmap to make room for.

---

## 7. Output Streams

Every run always produces everything the selected backend is capable of
- there is no "what to save" choice:

- **`combined`** *(YOLO, P2PNet)* — markers (boxes for YOLO, small
  circles for P2PNet) + count + heatmap overlay in one frame; also the
  live preview.
- **`detection`** *(YOLO, P2PNet)* — markers + count, full brightness, no
  heatmap. Isolates raw localization performance.
- **`heatmap`** *(all four backends)* — count + heatmap only, darkened
  background, no markers. The only stream CSRNet/DM-Count can produce,
  since they have no localization to draw markers from.

Each can be saved as a single `.mp4` video or a folder of numbered
frames, chosen once per run.

---

## 8. Model Comparison

**[SCREENSHOT: a frame with several visually-overlapping/occluded people, all four backends' output on the same frame, side by side]**

### 8.1 Occlusion handling (the TA's specific comparison request)

- **YOLO** must draw a distinct box around each person it recognizes as
  a separate object. When two or more people overlap heavily, the visual
  evidence for "this is a second, separate person" is partially hidden
  behind the first, and a bounding-box detector — even a fine-tuned one —
  tends to miss or merge overlapping individuals. This directly
  suppresses the count in dense clusters.
- **P2PNet** only has to commit to a point, not a full box, for each
  person, which is a strictly easier localization problem under partial
  occlusion than drawing a correctly-sized, correctly-shaped box — it is
  expected to hold onto more of a tightly-packed group than YOLO does,
  while still, unlike the two density models below, having to make a
  discrete "is there a separate person here" decision per point.
- **CSRNet and DM-Count** never make an instance-level decision at all —
  they regress a continuous density value from learned crowd-texture
  features. A partially-occluded pair of people still contributes visible
  head/shoulder area and characteristic local image statistics the
  network associates with "two heads' worth of density here," without
  ever explicitly localizing either person. This is the central reason
  purpose-built density models were developed in the first place, and is
  the behavior the TA's comparison task is specifically asking to
  demonstrate: **in scenes with significant overlap, YOLO is expected to
  undercount relative to the density models, with P2PNet in between.**

**[SCREENSHOT / DATA: table of YOLO / P2PNet / CSRNet / DM-Count count on the same 3-5 frames with varying crowd density/overlap, to quantify this]**

### 8.2 Generalization to unfamiliar camera footage

All three specialized models were trained on curated, dataset-specific
crowd photography (ShanghaiTech for CSRNet/P2PNet, UCF-QNRF for
DM-Count) and, as documented in §3.4, can produce clutter-triggered false
positives on footage that looks sufficiently different from that
training distribution — a fixed-camera domain-shift issue that YOLO,
fine-tuned directly on this project's own MOT16-derived data, does not
share to the same degree (YOLO's `person`-class filtering is explicit and
learned from labels, not an implicit texture correlation). This is an
important, honest asymmetry in the comparison: the specialized models'
theoretical advantage in occlusion handling (§8.1) comes with a practical
generalization cost that a fine-tuned general detector is less exposed
to.

**[SCREENSHOT / DATA: side-by-side of CSRNet, DM-Count, and P2PNet on the same problem frame from §3.4, to see whether the false-positive behavior is shared across all three or specific to CSRNet]**

### 8.3 Other differences

- **Localization and identity.** YOLO gives full boxes + track IDs;
  P2PNet gives points with no IDs; CSRNet/DM-Count give neither, only a
  scalar count and a density surface. Anything needing "where exactly is
  person #7 right now" requires YOLO specifically.
- **Training data mismatch across backends.** CSRNet and P2PNet share a
  training set (ShanghaiTech Part A), so a difference between them
  isolates architecture/loss effects; DM-Count's UCF-QNRF training data
  differs from both, so any DM-Count-specific behavior could stem from
  architecture *or* training-data differences — a limitation on how
  cleanly its results can be attributed (see §11).
- **Absolute error scale.** CSRNet/DM-Count's headline test-set MAE
  (≈70-72, §3.2) reflects performance on genuinely dense benchmark crowds
  (hundreds of people); on this project's much sparser sidewalk-scale
  footage (tens of people at most), absolute count error for all three
  specialized models should be substantially better than that number
  suggests, since error tends to scale with true crowd size.
- **Class filtering.** YOLO is class-filterable at both the data level
  (only `person` was ever in its training labels) and inference level
  (`classes=[0]`), directly satisfying the assignment's requirement to
  ignore vehicles/motorcycles. None of the three specialized models have
  a notion of object classes — they only ever respond to person-like
  crowd texture, with no explicit filtering step being necessary *or*
  possible, which is also precisely why §3.4/§8.2's clutter false-positive
  issue exists for them and not for YOLO.

---

## 9. Software Architecture

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
│   ├── heatmap.py                # instantaneous/cumulative/classic + native-density renderer
│   ├── counter.py                 # EMA count smoothing
│   ├── renderer.py                # box/marker/HUD drawing
│   ├── output_writer.py           # video / image-sequence export, run naming
│   └── cli.py                      # interactive terminal wizard
├── weights/
│   ├── yolo/, csrnet/, dmcount/, p2pnet/   # one subfolder per backend
├── config/settings.yaml           # default parameters, incl. external repo paths
├── videos/, images/, outputs/     # sample inputs / run outputs
├── deployment/crowd_monitor.service  # systemd template for the board
├── requirements.txt                # base requirements (YOLO backend)
└── requirements-specialized.txt     # extra requirements for the other three
```

**Two execution paths, four backends.** `main.py` dispatches to one of
two generalized loops rather than one per backend:

- `run_instance_based()` — used by YOLO and P2PNet. Both expose a
  `detect(frame) -> list[Detection]` interface (P2PNet wraps its points
  into pseudo-detections, §5.3), so the same heatmap-mode logic, marker
  drawing, and three-stream output apply to both without duplicating code.
- `run_native_density()` — used by CSRNet and DM-Count. Both expose an
  `estimate(frame) -> (count, density_map)` interface, so the same
  adaptive-normalization rendering and single-stream output apply to both.

**Why DM-Count and P2PNet aren't vendored like CSRNet.** CSRNet's
architecture is small and fully specified in the training notebook, so
copying it directly into `core/csrnet.py` was safe. DM-Count and
P2PNet's model code is not something safe to retype from memory — a
subtly wrong reproduction wouldn't necessarily crash, it would just
quietly compute wrong numbers against a checkpoint trained for the real
architecture. Both instead insert a local clone of their real upstream
repo onto `sys.path` and import the model-building function straight
from it, exactly as running their own scripts would.

**Shared post-processing lives in the right layer.** The peak-detection
mitigation (§3.4) lives inside `CSRNetEstimator.estimate()` itself, not
in `main.py` or the rendering layer - it changes what count and density
map the backend *reports*, which is the correct place for a
backend-specific correctness mitigation, as opposed to a purely cosmetic
display adjustment like `heatmap.noise_floor`, which stays in the shared
rendering path (`core/heatmap.blend_density_map()`) since it applies
uniformly across backends.

### Board readiness

- Camera access is abstracted behind `CameraSource`, which accepts a
  stream URL — a CCTV/RTSP camera on the board is the same code path as
  a laptop webcam.
- Every interactive run saves its exact settings as `run_config.json`.
  `main.py --config <file> --no-display` replays it with zero terminal
  interaction and never touches `cv2.imshow`.
- CSRNet's constructor (`load_weights=True`) deliberately avoids
  requiring an internet connection at inference time. DM-Count has no
  such flag to worry about. P2PNet's backbone-weight download (§5.2) is
  the one place a specialized backend still needs internet access (or a
  one-time offline pre-download) before it can run standalone on a board.

---

## 10. Installation and Usage Walkthrough

### 10.1 Install

```bash
pip install -r requirements.txt
# only if using a specialized backend:
pip install -r requirements-specialized.txt
```

Place model files under the matching subfolder in `weights/` (see
`weights/README.txt`). For DM-Count/P2PNet, also clone their upstream
repos somewhere on disk (paths are provided in the CLI, or pre-filled in
`config/settings.yaml`'s `external_repos:` section).

### 10.2 Run

```bash
python main.py
```

The terminal then walks through, in order:

1. **Select a backend** — YOLO, CSRNet, DM-Count, or P2PNet.
2. **Select a model file** — from that backend's `weights/<backend>/`
   subfolder.
3. *(DM-Count/P2PNet only)* **Confirm the cloned repo path** — defaults
   to whatever is set in `config/settings.yaml`.
4. **Select input source** — `Camera` (index `0` for a laptop webcam, or
   an `rtsp://...` URL for a CCTV camera later on the board), `Video
   file` (e.g. `videos/TownCentreXVID.mp4`), or `Image folder` (e.g.
   `images/MOT16-03/img1`).
5. **Select heatmap mode** *(YOLO/P2PNet only)* — `Instantaneous`,
   `Cumulative`, or `Classic`. Skipped automatically for CSRNet/DM-Count.
6. **Select output format** — `Video file (.mp4)` or `Image sequence`.
7. **Adjust advanced parameters?** — `conf_threshold` (YOLO/P2PNet),
   `smoothing` (all backends), `decay` (cumulative mode only),
   `count_method` (CSRNet only - §3.4).

The pipeline then runs with a live preview (the `combined` stream for
YOLO/P2PNet, the single `heatmap` stream for CSRNet/DM-Count) until the
source is exhausted or `q` is pressed, and prints where the output files
were saved (`outputs/run_<timestamp>/`).

For headless/unattended use later (e.g. on the board):

```bash
python main.py --config outputs/run_20260729_143210/run_config.json --no-display
```

---

## 11. Limitations and Future Work

- The clutter false-positive issue (§3.4, §8.2) has only been directly
  observed on CSRNet so far; whether DM-Count and P2PNet exhibit the
  same behavior on the same footage, and to what degree, is a concrete
  next test to run and report (screenshots/counts requested in §4.3 and
  §5.4).
- None of the three specialized models were fine-tuned on this project's
  own footage (MOT16/TownCentre) — all comparisons in §8 are of
  off-the-shelf checkpoints trained on different datasets, not a
  fully-controlled architecture-only comparison; a stronger version of
  this study would fine-tune all three on identical target-domain data.
- The `peak_detect` mitigation (§3.4) is a heuristic, validated so far
  only against a synthetic test case (a diffuse blob vs. sharp peaks) -
  not yet against real footage's actual clutter/false-positive patterns.
  It also only changes the numeric count, not the displayed heatmap,
  which still visually lights up over clutter regardless of count_method.
  Whether it noticeably improves real counts (vs. `density_sum`) on the
  problem footage from §3.4 is a concrete next test to run.
- The three heatmap modes for instance-based backends are all
  Gaussian-blob reconstructions from (pseudo-)bounding boxes; no attempt
  is made to model perspective (a person far from the camera occupies
  fewer pixels than one close to it).
- DM-Count and P2PNet's dependence on external cloned repos (rather than
  a single `pip install`) is a real, acknowledged deployment complexity
  for board setup — documented in `weights/README.txt` and the relevant
  modules' docstrings, but not eliminated.

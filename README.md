# Spatial Phase Query Fields (SPQF)

A compact implementation of phase-to-depth calibration for a fixed camera–projector system.
SPQF queries a shared MLP with pixel position and absolute phase. Depth is then converted to camera-space `(XC, YC, ZC)` using the calibrated camera ray.

![SPQF framework](docs/images/framework.png)

## Quick start

Edit the variables at the top of each script, then run it **without command-line arguments**:

```bash
git clone https://github.com/Wangh257/SPQF.git
cd SPQF
python -m pip install -r requirements.txt
bash tools/train.sh    # Train the MLP or fit a polynomial
bash tools/test.sh     # Evaluate calibration-board poses
bash tools/infer.sh    # Reconstruct a new object from fringe images
```

Use Python 3.10+. Run these commands from this `code` directory. The scripts set `PYTHONPATH` automatically.
`device: auto` selects CUDA, Apple MPS, or CPU, in that order. Training on a Mac is supported.
If memory is limited, reduce `batch_size`, `cache_planes`, or `inference_chunk_size` in the YAML configuration.
To select a different Python environment: `PYTHON_BIN=/path/to/python bash tools/train.sh`.

**Data and pretrained models are not included.** Supply your own data or set the script paths to existing directories.
The original project and its data are unchanged.

## Files

```text
tools/train.sh                 Training / polynomial calibration
tools/test.sh                  Batch evaluation on reference boards
tools/infer.sh                 New-object reconstruction
tools/extract_raw_phase.sh     Fringe images → raw absolute phase
tools/generate_phase.sh        CPC fitting / noise / phase-jump variants
tools/prepare_manifest.sh      Manifest for a new calibration dataset
codes/phase_field/             Python implementation, configuration and tests
legacy/matlab/                 Original MATLAB sources for reference
docs/images/                   Paper illustrations shown in this README
```

## 1. Prepare data

Recommended layout:

```text
data/raw_phase_no_mask/PSP_1.mat ... PSP_24.mat
data/raw_phase_cpc_mask/PSP_1.mat ... PSP_24.mat   # Optional processed phase
data/camera_numeric.mat                         # For preparing a new manifest
cal/mask/mask_1.bmp ... mask_24.bmp
cal/PSPImg/                                     # Optional raw calibration images
data/object_images/                            # Raw images of a new object
```

- Phase files should contain a numeric `phase` array of shape `H × W`, in **radians**, with absolute phase already unwrapped.
- `i_PSP_*.bmp` files are previews, not numerical phase inputs.
- Single-channel masks use nonzero pixels for reliable regions and zero for excluded regions.
- Image size, pixel coordinates, mask alignment and calibration-pose ordering must match. Do not resize/crop images while reusing unchanged camera geometry.
- Use ordinary numeric MAT files readable by SciPy, not MATLAB v7.3/HDF5 files.

### Calibration manifest

The JSON manifest stores camera intrinsics/distortion, image size, per-pose board planes, phase paths and dataset splits.
Reference depth comes from camera-ray/board-plane intersections, **not from polynomial reconstructions**.

The included `codes/phase_field/data/calibration_manifest_raw_no_mask.json` is for the original **24-pose, 1200 × 900** dataset only.
Poses **16, 17 and 19** are in `test`; the remaining 21 poses are used for training. The split is fixed, not resampled on each run.
Because this `test` split is also used for checkpoint selection, it is a **validation set**, not an independent final test set.

For a new system or new board poses, edit and run:

```bash
bash tools/prepare_manifest.sh
```

Set `camera_params`, `phase_dir` and `output_path`. The default split randomly holds out three poses with seed 42.
Set the resulting manifest path in **both** `train.sh` and `test.sh`. Do not reuse the old board planes for new data.
When only changing phase preprocessing for the same poses, keep the same manifest and change `phase_dir` instead.
The preparation script does not attach masks to training samples; evaluation masks are configured separately.

`camera_numeric.mat` contains numeric `K`, `R`, `T`, `WorldPoints`, `RadialDistortion` and `TangentialDistortion`.
The converter expects the MATLAB convention `P_camera = P_world * R + T`, with board coordinates and translation in mm.
If `cameraParams.mat` contains a MATLAB class object, export it using MATLAB Engine:

```bash
PYTHONPATH=codes python -m phase_field.export_camera_params \
  --input /path/to/cameraParams.mat --output data/camera_numeric.mat
```

This optional export requires MATLAB and its Python Engine. Normal training does not require MATLAB.

### Extract phase once

In `tools/extract_raw_phase.sh`, set `input_dir` and `output_dir`, then run:

```bash
bash tools/extract_raw_phase.sh
```

The default acquisition is **4 frequencies × 12 steps = 48 BMP images per pose**, with frequency ratios `4, 4, 4`.
Images must be in one directory, naturally numbered: consecutive phase shifts within each frequency, frequencies from low to high, then the next pose.
Subdirectories are not scanned. Keep white/reference images and masks separate; `blank_images_per_pose=0`.
Outputs include `PSP_*.mat`, previews and extraction metadata. Existing phase files are skipped by default, so use a new output directory when changing source data.

### Optional phase preprocessing

Edit `tools/generate_phase.sh`, then run `bash tools/generate_phase.sh`.

| `mode` | Operation | Main settings |
| --- | --- | --- |
| `cpc` | Fit board phase using reliable mask pixels | `mask_dir`, `cpc_order` |
| `gaussian` | Add Gaussian noise | `sigma`, `mean`, `seed` |
| `offset` | Add a constant regional offset | `value`, `region` |
| `fringe_jump_region` | Apply a regional phase-order jump | `orders`, `region` |
| `fringe_jump_blocks` | Random small blocks with positive/negative jumps | `block_count`, `block_size`, `blocks_region` |
| `fringe_jump_pixels` | Random isolated pixels with positive/negative jumps | `pixel_count`, `pixels_region` |

Leave `output_dir=""` for automatic naming or set a separate output directory. `region=(x y width height)`.
`blocks_region` is the placement area, not the size of an individual block. Set full-image regions to `(0 0 W H)`.
`cpc_order` controls phase-surface fitting; `degree` below controls phase-to-depth calibration. They are different settings.
The CPC tool consumes an existing mask; it does not generate a CPSD mask automatically.

## 2. Train

Edit `tools/train.sh`:

```bash
method="mlp"                       # mlp / poly / inverse_poly
degree="3"                         # Ignored for MLP
phase_dir="data/raw_phase_no_mask"
manifest="codes/phase_field/data/calibration_manifest_raw_no_mask.json"
output_dir="outputs/mlp_raw"
use_mask="true"                    # Evaluation only
mask_dir="cal/mask"
```

Then run `bash tools/train.sh`.

| Method | Mapping | Saved model |
| --- | --- | --- |
| `mlp` | Shared phase-query MLP → depth | `best.pt`, `last.pt` |
| `poly` | `ZC = sum(a_k q^k)` per pixel | `polynomial_degree_3.npz` |
| `inverse_poly` | `ZC = 1 / sum(b_k q^k)` per pixel | `inverse_polynomial_degree_3.npz` |

For degree 4 or 5, change `degree`; the saved filename changes accordingly. Polynomial calibration solves coefficients rather than running neural-network epochs.
Use a different `output_dir` for each method, degree and phase variant: existing model files can be overwritten.

**Training does not extract phase or perform mask+CPC fitting.** To train on processed phase, simply use:

```bash
phase_dir="data/raw_phase_cpc_mask"
output_dir="outputs/mlp_cpc"         # Or outputs/poly_cpc_d3, etc.
```

`use_mask=true` affects evaluation metrics and therefore best-checkpoint selection, not training phase or training sampling.
If masks are unavailable, set it to false and compare methods using consistent evaluation regions.

MLP settings are in `codes/phase_field/configs/direct_mlp_raw_no_mask.yaml`:

- 100 epochs, 50 batches per epoch, batch size 16384.
- AdamW, initial learning rate 0.001 with cosine decay, Smooth L1 loss.
- Evaluate every epoch; save the lowest-MAE checkpoint as `best.pt`, the latest as `last.pt`.
- Training history: `metrics.csv`; run configuration: `resolved_config.json`.
- For resuming, set `training.resume` to `last.pt` and `epochs` to the target total, keeping data/configuration consistent.

Physical query variables are `(u,v,Phi)`. The default implementation uses normalized coordinates/phase **plus derived `sin(Phi)` and `cos(Phi)` features** (`use_periodic_phase: true`).
Set this option to false and retrain for a three-channel input; do not change it when loading an existing model.
Normalization is estimated from training data and saved with the model for inference.

## 3. Test calibration boards

Edit `tools/test.sh`, then run `bash tools/test.sh`:

```bash
phase_dir="data/raw_phase_no_mask"
manifest="codes/phase_field/data/calibration_manifest_raw_no_mask.json"
split="test"                       # test / train / all
model_checkpoint="outputs/mlp_raw/best.pt"
output_dir="results/mlp_raw_test_masked"
use_mask="true"
mask_dir="cal/mask"
```

This evaluates every pose in the chosen split, including 16/17/19 with the included manifest.
For polynomial models, select the actual `.npz` file, for example `outputs/poly_raw_d3/polynomial_degree_3.npz`.
The model stores its degree and ordinary/reciprocal mapping; no test-time degree argument is needed.
Renaming a file to `.pt`, `.npy` or `.npz` does not convert its model type.

Results are saved as `metrics.json` and per-pose `PSP_*/` reconstruction folders.
**Printed MAE/RMSE measure ZC depth error in mm, not Euclidean 3D distance.** Aggregate metrics pool all valid pixels.
The evaluation mask excludes marker pixels from metrics but does not crop the saved reconstruction or refit phase.
Use different result directories when comparing `use_mask=true` and false.

For like-for-like calibration-board comparisons, use matching raw/CPC phase processing for training and evaluation.
Noise robustness experiments can intentionally use different test phases, but should be named and reported separately.

## 4. Reconstruct a new object

Edit `tools/infer.sh`, then run `bash tools/infer.sh`:

```bash
input_dir="data/object_images"
model_checkpoint="outputs/mlp_raw/best.pt"    # Or a polynomial .npz
save_dir="results/object_mlp_raw"
use_mask="false"
mask_path=""
```

This reads one or more complete 48-image sequences, extracts phase and saves:

```text
results/object_mlp_raw/extracted_phase/PSP_1.mat
results/object_mlp_raw/PSP_1/XC.npy, YC.npy, ZC.npy
results/object_mlp_raw/PSP_1/point_map.npy        # H × W × 3, mm
results/object_mlp_raw/PSP_1/point_cloud.ply
results/object_mlp_raw/PSP_1/valid_mask.npy
```

Use a new/empty `save_dir`; the script rejects nonempty directories to avoid stale phase reuse.
Normal object measurement does not require calibration-marker masks or board-plane phase fitting, even when the calibration model was trained using CPC phase.
Do not fit a general object's phase to a board surface: this can remove real shape.
An optional object mask can restrict output pixels. Without reference geometry, this step reconstructs shape but does not measure accuracy.

Already have object phase? Skip extraction:

```bash
PYTHONPATH=codes python -m phase_field.infer \
  --checkpoint outputs/mlp_raw/best.pt \
  --phase /path/to/object_phase.mat --output results/object_from_phase

PYTHONPATH=codes python -m phase_field.polynomial_infer \
  --model outputs/poly_raw_d3/polynomial_degree_3.npz \
  --phase /path/to/object_phase.mat --output results/object_poly_from_phase
```

## Illustrative comparisons

<img src="docs/images/marker_artifacts.png" width="360" alt="Object comparison: cubic calibration and SPQF">

Object reconstruction: **(a)** cubic polynomial calibration; **(b)** SPQF. Repeated calibration-marker artifacts are visible in (a), particularly on the background board.

![Out-of-range board comparison](docs/images/extrapolation.png)

Beyond the calibrated depth range: **(a)** raw-phase cubic calibration; **(b)** cubic calibration after masking and phase fitting; **(c)** SPQF.
The fitted-phase polynomial reconstruction in (b) exhibits visible bending. These supplied figures are qualitative illustrations, not outputs automatically regenerated by this code package or metric error maps.

## Optional tools and checks

- `tools/xyz_polynomial_compare.py`: independent XC/YC/ZC fitting (12 coefficients per pixel for degree 3). Edit its top variables, then run it directly. Its model format is separate from the standard depth-only test/inference pipeline; detailed coordinate and Euclidean metrics are in its JSON output.
- `phase_field.phase_sensitivity`: optional sensitivity analysis (`--help`).
- `phase_field.polynomial_calibrate_masked`: retained legacy mask+CPC entry point; not called by `train.sh`.
- `legacy/matlab/`: historical source only; original paths may need adaptation. Python training does not call these scripts.

```bash
PYTHONPATH=codes python -m pytest codes/phase_field/tests -q
```

The tests use small synthetic data, not a full reproduction of the paper. This package does not include a new real-data training run.
Only portable script defaults, a training `--manifest` override, and input/output safety checks were added; the original project remains intact.

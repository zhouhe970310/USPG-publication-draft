# DSP + SPC polyp segmentation — publication draft

This draft packages the supplied optimized experiment results and their source code.
The paper title, authors, license, exact environment, and final manuscript configuration
must be verified before release. This is not yet a published or fully verified reproduction.

## Scope

The supplied main experiment uses a ResNet-18 U-Net with training-time domain style
perturbation (DSP) and structure-preserving consistency (SPC), at 256 x 256 pixels.
Main models use seeds 42, 43, 44 and a fixed Kvasir split seed of 20260822.
The repository also retains other model implementations required by the shared model
factory; their presence does not mean that they produced the supplied main results.

## Contents

- `code/`: model definitions, losses, data loading, training, evaluation, and analysis.
- `code/splits/`: fixed train/validation/test image lists.
- `code/audit/`: dataset audit metadata, without image files.
- `evaluation/`: per-image metrics, aggregate metrics, and validation calibration.
- `robustness/`: corruption evaluation records.
- `results/`: supplied numerical summaries and non-image-data plots.

## Dependencies and data

From `code/`, install `python -m pip install -r requirements.txt` in an isolated environment.
Requirements are lower bounds from the supplied project, not an exact environment lock.
The original package reports Python 3.12 and a PyTorch 2.13 CPU build; this environment
claim has not been independently verified. Do not substitute the reviewer's environment
as the original experiment environment.

Obtain Kvasir-SEG and CVC-ColonDB from their official providers under their terms.
Images and masks are not redistributed here. Each dataset must contain `images/`
and `masks/` with paired filenames. Set `USPG_DATA_ROOT` to their parent directory.
The ResNet-18 implementation requires the local pretrained file
`resnet18-5c106cde.pth`; set `USPG_RESNET18_WEIGHTS` to its path.
Model checkpoints and training histories are absent from the supplied package.

## Supplied optimized training entry point

Run from `code/` in a Bash environment:

```bash
export KROOT=/path/to/datasets/Kvasir-SEG
export OUT=/path/to/experiment-runs
export WEIGHTS=/path/to/resnet18-5c106cde.pth
bash run_bspc_optimization.sh
```

The baseline trains for up to 8 epochs; DSP/SPC variants then fine-tune from the
corresponding baseline checkpoint for up to 4 further epochs. The supplied main
comparison therefore does not use equal total training budgets. This protocol must
be described accurately in the manuscript; it has not been changed in this draft.

## Evaluation

The main supplied records use validation-selected HFlip TTA and threshold 0.50.
Always pass `--image_size 256` explicitly: several shared tools default to 352.
For re-running source-validation calibration and then evaluation of the main models:

```bash
python tools/evaluate_calibrated_tta.py --runs_root /path/to/experiment-runs --kvasir_root /path/to/datasets/Kvasir-SEG --cvc_root /path/to/datasets/CVC-ColonDB --output_root /path/to/new-evaluation --image_size 256 --model_types resunet resunet_spc --seeds 42 43 44
```

Preserve the original `evaluation/` records separately from regenerated outputs.
CVC-Dedup is a sensitivity analysis of CVC-ColonDB, not an independent dataset.
Architecture comparisons and DSP/region-only ablations contain seed 42 only.
Robustness metadata should be consulted for its own image size and evaluation setup.

## Publication checks still required

- Confirm that this 256-pixel DSP+SPC experiment is the final manuscript experiment.
- Supply exact package versions, original training logs/configurations, and checkpoints
  if available; full numerical reproduction has not been run during packaging.
- Confirm source-code provenance and an author-approved license before adding LICENSE.
- Add the final paper citation and verified public repository URL after publication setup.
- Dataset-image panels are excluded pending review of redistribution terms.

Machine-specific checkpoint prefixes in CSV/JSON metadata were replaced with
`<EXPERIMENT_ROOT>/`; model identifiers, hashes and numerical results were retained.

# Julia parcel model: BMM-trained cloud activation surrogate

This repository uses the **Bin Microphysics Model (BMM)** to generate reference
parcel trajectories and Julia/Flux/SciML to train fast cloud-activation
surrogates.

The repository is deliberately organised so that the common workflow is driven
from the top-level `Makefile`. You should not normally need to remember the
platform-specific BMM build command or work from inside `julia/`.

> **Important:** `bmm/` is a git subtree imported from the separate
> Bin Microphysics Model repository. The parent project invokes it, but does
> not reorganise or maintain the internals of that subtree.

## Quick start

From the repository root:

```bash
make setup
make build
make smoke
```

Then the normal surrogate workflow is:

```bash
make generate
make analyse
make train
make compare
```

Run `make help` at any time to see the available commands and overrides.

## Recommended PowerEdge workflow

The synthetic BMM cases are independent. On a many-core Linux server, choose
how many BMM processes may run concurrently rather than simply using every
logical CPU:

```bash
make generate CASES=256 WORKERS=64
```

For a larger ensemble:

```bash
make generate CASES=5000 WORKERS=128
```

The generator uses a dynamic worker queue, so when one BMM case finishes that
worker immediately takes the next case. Progress is reported while the
ensemble is running, including completed, running and failed cases.

`WORKERS` is also used as the Julia thread count for generation. The effective
number of concurrent BMM jobs is capped by the number of cases and available
Julia threads.

## Repository layout

```text
julia-parcel-model/
├── README.md             this file
├── Makefile              user-facing build/workflow commands
├── .gitignore
├── LICENSE
├── bmm/                  BMM git subtree; maintained separately
├── julia/                Julia source + Project.toml/Manifest.toml
├── configs/              parent-project example configuration
├── scripts/              subtree/update helper scripts
├── docs/                 historical/development notes
├── data/                 generated datasets (gitignored)
├── models/               trained surrogate weights (gitignored)
├── results/              plots, splits and metrics (gitignored)
└── scratch/              temporary local work (gitignored)
```

Only `julia/` is a Julia package environment. There is intentionally no
second `Project.toml` at repository root.

## Build BMM

Use the same command on macOS and Ubuntu:

```bash
make build
```

or equivalently:

```bash
make bmm
```

The top-level Makefile detects the platform.

### macOS

The wrapper calls the BMM subtree's normal default build:

```bash
make -C bmm
```

so the existing BMM Makefile defaults are preserved.

### Ubuntu / Linux

The wrapper calls the BMM Makefile with the NetCDF/compiler settings currently
required on the Ubuntu development machine:

```bash
make -C bmm \
  NETCDFLIB=-L/usr/lib/x86_64-linux-gnu/ \
  NETCDFMOD=/usr/include/ \
  FFLAGS='-O2 -w -o' \
  FFLAGS2='-O2 -w -o' \
  FFLAGSOMP='-O2 -w -o'
```

These are command-line overrides only. **No Makefile or source file inside
`bmm/` is changed by the wrapper.**

## Prerequisites

You need:

- `make`, `gfortran` and standard GNU build tools;
- NetCDF C and NetCDF-Fortran development libraries;
- Julia 1.10, 1.11 or 1.12;
- the Julia packages recorded in `julia/Project.toml` and
  `julia/Manifest.toml`.

On Ubuntu/Debian the system dependencies are typically:

```bash
sudo apt update
sudo apt install -y build-essential gfortran \
    libnetcdf-dev libnetcdff-dev netcdf-bin git curl
```

Check the toolchain with:

```bash
gfortran --version
nc-config --version
nf-config --version
julia --version
make --version
```

Instantiate the Julia environment with:

```bash
make setup
```

## Smoke test

After building, run:

```bash
make smoke
```

This exercises the complete integration path:

```text
Julia -> BMM namelist -> BMM executable -> NetCDF -> Julia
```

A successful run prints `BMM/Julia multimode smoke test OK`.

## Generate a synthetic BMM dataset

The default pilot is 256 cases, up to four aerosol modes, using seed
`20260831`:

```bash
make generate
```

Equivalent explicit invocation:

```bash
make generate CASES=256 MAX_MODES=4 WORKERS=64 SURROGATE_SEED=20260831
```

The default output is:

```text
data/synthetic_bmm.nc
```

Failed BMM cases retain their diagnostic namelist/log/output files under a
sibling `*_failures` directory.

To use a different dataset filename:

```bash
make generate BMM_DATASET=data/pilot_001.nc
```

## Analyse the dataset

```bash
make analyse
```

or for another dataset:

```bash
make analyse BMM_DATASET=data/pilot_001.nc
```

## Train the surrogate

The current default is the v3 profile surrogate:

```bash
make train
```

To change the training length:

```bash
make train PROFILE_EPOCHS=250
```

Generated model weights are written to:

```text
models/profile_surrogate.bson
```

The train/validation/test split CSV is written under `results/`.

The NeuralODE experiment remains available explicitly:

```bash
make train-neuralode
```

but the profile surrogate is the current baseline.

## Evaluate held-out cases

```bash
make compare
```

This reads the trained profile model from `models/` and writes metrics and
figures under:

```text
results/surrogate_comparison/
```

## Current physical formulation

Synthetic cases contain one to four external lognormal aerosol modes. Each
mode contains number mixing ratio, dry median diameter, log-width and
hygroscopicity/composition information.

The current v3 surrogate predicts:

- liquid-water mixing ratio `ql`;
- bounded total activated fraction;
- extinction coefficient `beta_ext`.

Supersaturation is diagnosed from conserved total water rather than predicted
independently:

```text
qtot = ql + (1 + S) qvs(T,P)
```

so:

```text
S = (qtot - ql) / qvs(T,P) - 1
```

This keeps the surrogate thermodynamically constrained.

The next scientific development is expected to add per-original-aerosol-mode
activation diagnostics to BMM and train a mode-resolved activation head.

## BMM numerical configuration used by the synthetic generator

The current liquid-only training configuration fixes:

```text
ice_flag        = 0
bin_scheme_flag = 0
sce_flag        = 0
```

The synthetic generator currently samples broad thermodynamic, updraft and
modal aerosol conditions and runs a 500 m parcel ascent.

BMM aerosol number input is number mixing ratio per kg dry air. The synthetic
sampler defines intuitive concentrations in cm^-3 and converts them using the
dry-air density convention used by the project.

## Updating the BMM subtree

Because `bmm/` belongs to a separate repository, update it using the existing
subtree workflow rather than manually reorganising files inside it. See:

```text
scripts/bmm-subtree.sh
```

Changes intended for BMM itself should normally be made in the BMM repository
and then brought into this project as a subtree update.

## Direct Julia commands

The top-level Makefile is preferred, but scripts can still be run directly:

```bash
cd julia

BMM_EXE=../bmm/main.exe BMM_WORKERS=64 \
  julia --project=. -t 64 generate_synthetic_dataset.jl \
  256 ../data/synthetic_bmm.nc 4 20260831 64

julia --project=. analyse_dataset.jl ../data/synthetic_bmm.nc

MODEL_DIR=../models RESULTS_DIR=../results \
  julia --project=. train_surrogate.jl \
  ../data/synthetic_bmm.nc profile 150 20260831
```

## Useful Make variables

| Variable | Default | Purpose |
|---|---:|---|
| `CASES` | `256` | Number of BMM trajectories |
| `WORKERS` | `64` | Maximum concurrent BMM processes |
| `MAX_MODES` | `4` | Maximum aerosol modes per synthetic case |
| `SURROGATE_SEED` | `20260831` | Sampling and split seed |
| `PROFILE_EPOCHS` | `150` | Profile-surrogate epochs |
| `NEURALODE_EPOCHS` | `40` | NeuralODE experiment epochs |
| `BMM_DATASET` | `data/synthetic_bmm.nc` | Dataset path from repo root |
| `NPLOTS` | `8` | Number of selected held-out case plots |

For the current command summary:

```bash
make help
```

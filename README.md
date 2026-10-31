# Julia parcel model

This project uses the **Bin Microphysics Model (BMM)** to generate reference
cloud-parcel trajectories and Julia/Flux to develop fast learned
parameterisations of cloud activation and condensational growth.

The repository is intended to be operated from its top directory. The common
workflow is deliberately short:

```bash
make setup
make build
make smoke
make generate
make analyse
make train
make compare
```

Run `make help` whenever you need a reminder.

> `bmm/` is a **git subtree from the separate Bin Microphysics Model
> repository**. The parent project builds and runs it, but does not reorganise
> or maintain the internals of that subtree.

## 1. First-time setup

### Julia and system dependencies

You need Julia 1.10–1.12, `make`, `gfortran`, and the NetCDF C and Fortran
development libraries.

On Ubuntu/Debian:

```bash
sudo apt update
sudo apt install -y build-essential gfortran \
    libnetcdf-dev libnetcdff-dev netcdf-bin git curl
```

Check the installation:

```bash
julia --version
gfortran --version
nc-config --version
nf-config --version
```

Instantiate the Julia environment:

```bash
make setup
```

### NVIDIA GPU training

GPU support is optional. On an NVIDIA machine, install the Julia CUDA packages
once:

```bash
make gpu-setup
make gpu-check
```

`gpu-check` should report `CUDA functional = true`.

The normal CPU environment remains usable on machines without an NVIDIA GPU.

## 2. Build and test BMM

Use the same command on macOS and Ubuntu:

```bash
make build
```

The top-level Makefile detects the operating system.

On macOS it simply calls the BMM subtree's own default build. On Ubuntu it
passes the NetCDF/compiler settings currently required by the Linux
development machine:

```bash
make -C bmm \
  NETCDFLIB=-L/usr/lib/x86_64-linux-gnu/ \
  NETCDFMOD=/usr/include/ \
  FFLAGS='-O2 -w -o' \
  FFLAGS2='-O2 -w -o' \
  FFLAGSOMP='-O2 -w -o'
```

These are command-line overrides only. **Nothing inside `bmm/` is modified.**

Then run the end-to-end smoke test:

```bash
make smoke
```

A successful run exercises:

```text
Julia -> BMM namelist -> BMM executable -> NetCDF -> Julia
```

and prints `BMM/Julia multimode smoke test OK`.

## 3. Generate BMM training data

The default pilot dataset contains 256 cases with one to four aerosol modes:

```bash
make generate
```

On the PowerEdge, a useful explicit command is:

```bash
make generate CASES=256 WORKERS=64
```

For a larger ensemble later:

```bash
make generate CASES=5000 WORKERS=128
```

`WORKERS` controls the maximum number of concurrent BMM processes. Generation
uses a dynamic queue: when one BMM calculation completes, the next waiting case
is launched.

The default dataset is written to:

```text
data/synthetic_bmm.nc
```

To use another filename:

```bash
make generate BMM_DATASET=data/pilot_002.nc
```

Failed-case diagnostics are retained in a sibling `*_failures` directory.

## 4. Check the dataset

Always inspect a newly generated ensemble before training:

```bash
make analyse
```

or:

```bash
make analyse BMM_DATASET=data/pilot_002.nc
```

This reports the sampled thermodynamic/aerosol ranges and key BMM outputs.

## 5. Train a surrogate

### NVIDIA GPU

On the PowerEdge, after `make gpu-setup`, the normal command is simply:

```bash
make train
```

The default `DEVICE=auto` uses a functional NVIDIA/CUDA backend when one is
available and otherwise falls back to CPU.

To change the number of epochs:

```bash
make train DEVICE=gpu PROFILE_EPOCHS=250
```

The training script verifies that CUDA is functional and reports the selected
GPU at startup. The trained parameters are moved back to CPU before evaluation
and saving, so the resulting checkpoint is portable.

### CPU

To force CPU training, use:

```bash
make train DEVICE=cpu
```

### Current GPU implementation

GPU execution is supported for the current **profile surrogate**. The model is
still evaluated trajectory-by-trajectory internally, so GPU support should be
viewed as the first deployment step rather than the final performance design.
A later optimisation can batch many trajectories/grid points into larger GPU
operations without changing the user-facing `make train DEVICE=gpu` workflow.

The default model checkpoint is written under:

```text
models/profile_surrogate.bson
```

and the deterministic train/validation/test split is written under `results/`.

## 6. Evaluate held-out BMM cases

After training:

```bash
make compare
```

This reports held-out metrics and writes diagnostic plots/tables under:

```text
results/surrogate_comparison/
```

## Repository layout

```text
julia-parcel-model/
├── README.md              this guide
├── Makefile               user-facing workflow
├── .gitignore
├── bmm/                   separate BMM git subtree
├── julia/                 Julia source + Project.toml/Manifest.toml
├── configs/               example parent-project configuration
├── scripts/               repository/subtree helper scripts
├── docs/                  development notes (not extra READMEs)
├── data/                  generated datasets (gitignored)
├── models/                trained checkpoints (gitignored)
├── results/               metrics and plots (gitignored)
└── scratch/               temporary local work (gitignored)
```

There is only one parent-project `README.md`, here at the repository root.
Documentation that belongs to the independently maintained BMM subtree remains
part of that subtree.

There is also only one Julia environment for the parent project:

```text
julia/Project.toml
julia/Manifest.toml
```

## The model is expected to evolve

The **workflow and physical interface** are intended to be stable; the neural
network architecture is not.

At present the profile surrogate is the working research baseline. It uses a
permutation-invariant modal aerosol encoder and predicts liquid water,
activation and extinction-related quantities while diagnosing supersaturation
from total-water conservation.

That implementation will change as the parameterisation improves. Likely
future work includes mode-resolved activation, larger BMM training ensembles,
a local host-model timestep interface, stronger batching/vectorisation and
possibly different Flux/Lux/SciML architectures.

For that reason, scripts should use the public workflow (`make train`,
`make compare`) rather than depending unnecessarily on a particular internal
network layout or a version number such as "v3".

## Physical principle retained by the current surrogate

The current model does not independently predict supersaturation. Total water
is used to diagnose it:

```text
qtot = ql + (1 + S) qvs(T,P)
S    = (qtot - ql) / qvs(T,P) - 1
```

This is an example of the intended design philosophy: keep known physics and
hard constraints explicit, and use the learned component for the unresolved
microphysics.

## Useful Make variables

| Variable | Default | Purpose |
|---|---:|---|
| `CASES` | `256` | Number of synthetic BMM trajectories |
| `WORKERS` | `64` | Maximum concurrent BMM processes |
| `MAX_MODES` | `4` | Maximum aerosol modes per case |
| `SURROGATE_SEED` | `20260831` | Sampling and train/test split seed |
| `PROFILE_EPOCHS` | `150` | Profile-model training epochs |
| `DEVICE` | `auto` | `auto`, `cpu` or `gpu` for training |
| `BMM_DATASET` | `data/synthetic_bmm.nc` | Dataset path from repository root |
| `NPLOTS` | `8` | Number of selected comparison plots |

Examples:

```bash
make generate CASES=2000 WORKERS=128
make train DEVICE=gpu PROFILE_EPOCHS=200
make compare NPLOTS=12
```

## Direct Julia commands

The top-level Makefile is preferred, but the equivalent commands can be useful
for debugging:

```bash
cd julia

BMM_EXE=../bmm/main.exe BMM_WORKERS=64 \
  julia --project=. -t 64 generate_synthetic_dataset.jl \
  256 ../data/synthetic_bmm.nc 4 20260831 64

julia --project=. analyse_dataset.jl ../data/synthetic_bmm.nc

MODEL_DIR=../models RESULTS_DIR=../results DEVICE=gpu \
  julia --project=. train_surrogate.jl \
  ../data/synthetic_bmm.nc profile 150 20260831
```

## Updating the BMM subtree

Do not manually reorganise `bmm/` from this repository. Update it through the
git-subtree workflow. If a `bmm-upstream` remote has been configured:

```bash
git status
git fetch bmm-upstream
git subtree pull --prefix=bmm bmm-upstream main --squash
```

Changes to BMM itself should normally be developed in its own repository and
then pulled into this project as a subtree update.

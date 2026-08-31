# BMM-trained cloud activation surrogate

This repository uses the **Bin Microphysics Module (BMM)** as the reference
microphysics model and Julia/SciML for generation and training of a fast
surrogate.

The immediate workflow is:

1. sample synthetic cloud-base thermodynamic and multimode aerosol states;
2. run the BMM subtree as the reference model;
3. resample every trajectory onto height above cloud base;
4. store an ML-friendly NetCDF training ensemble;
5. train a surrogate on that ensemble.

## Fixed BMM numerical configuration

All reference/training runs use exactly:

```text
bin_scheme_flag = 0
sce_flag        = 0
```

These choices are hard-coded in the Julia namelist writer and are deliberately **not** exposed as `BMMCase` fields or synthetic-sampling parameters. This prevents a training dataset from silently mixing different BMM bin or collision/coalescence schemes.

## Multimode aerosol interface

A mode is represented in Julia by

```julia
AerosolMode(
    N       = 400e6,     # kg^-1 dry air (BMM n_aer1 input)
    Dm      = 80e-9,     # dry number-median diameter, m
    lnsig   = 0.4,       # ln geometric standard deviation
    kappa   = 0.6,
    nu      = 3.0,
    molw    = 132.14e-3, # kg mol^-1
    density = 1770.0     # kg m^-3
)
```

Each Julia `AerosolMode` is mapped to a separate **external BMM mode**. BMM's
`n_intern` means lognormal submodes *inside one external mode*; it is therefore
not the right mechanism for modes with different hygroscopicity. For this
surrogate dataset we use `n_intern=1`, `n_mode=length(modes)` and one unique
BMM component per external mode with an identity mass-fraction matrix.

A case can therefore contain any number of modes:

```julia
case = cloud_base_case(
    Tinit = 280.0,
    pinit = 85000.0,
    winit = 0.7,
    modes = [mode1, mode2, mode3]
)
```

(The actual keyword names are `tinit` and `pinit`.)

## Kappa vs classical Koehler

BMM supports a case-wide `kappa_flag`:

- `kappa_flag=1`: kappa-Koehler;
- `kappa_flag=0`: classical ideal-solution Koehler using van't Hoff factor,
  molecular weight and dry density.

Different modes can have different kappa values or different classical solute
properties. The flag is global to the BMM run, so a single run should use one
formulation consistently.

For BMM's ideal classical water-activity expression, a pure component has the
exact equivalent

```text
kappa_eff = nu * (rho_dry / rho_water) * (M_water / M_solute)
```

so the two formulations can later be mapped to a common ML hygroscopicity
feature if desired. The first synthetic ensemble uses kappa-Koehler because it
is the most direct way to span hygroscopicity without inventing correlated
molecular weights and densities.

## Installation and setup

The repository needs four things:

1. a GNU-style build environment (`make`, `gfortran`, `ar`);
2. the NetCDF C and NetCDF-Fortran development libraries (`nc-config` and
   `nf-config` must be available);
3. Julia 1.10--1.12;
4. the Julia packages listed in `julia/Project.toml`.

The BMM Makefiles are Unix/GNU Makefiles. macOS and Linux can build them
natively. On Windows, **WSL2 with Ubuntu is the recommended environment** for
this repository; it avoids maintaining a separate Windows Fortran/NetCDF build
and lets the same commands be used on all three operating systems.

### macOS

Install the Apple command-line tools if they are not already present:

```bash
xcode-select --install
```

Install [Homebrew](https://brew.sh/) if required, then install the Fortran and
NetCDF dependencies:

```bash
brew install gcc netcdf netcdf-fortran
```

Install Julia using the official `juliaup` installer:

```bash
curl -fsSL https://install.julialang.org | sh
```

Open a new Terminal after installation and verify:

```bash
gfortran --version
nc-config --version
nf-config --version
julia --version
make --version
```

Both Apple Silicon and Intel Macs are supported by the Homebrew packages. The
top-level Makefile obtains the NetCDF installation prefixes from `nc-config`
and `nf-config`, so Homebrew's architecture-specific prefix does not need to be
hard-coded.

### Linux

#### Ubuntu / Debian

Install the compiler, NetCDF development libraries and basic tools:

```bash
sudo apt update
sudo apt install -y \
    build-essential gfortran \
    libnetcdf-dev libnetcdff-dev netcdf-bin \
    curl git
```

Then install Julia using the official `juliaup` installer:

```bash
curl -fsSL https://install.julialang.org | sh
```

Open a new shell (or follow the PATH instruction printed by `juliaup`) and
verify:

```bash
gfortran --version
nc-config --version
nf-config --version
julia --version
make --version
```

#### Fedora / RHEL-family systems

The corresponding packages are normally:

```bash
sudo dnf install \
    gcc-gfortran make gcc gcc-c++ \
    netcdf-devel netcdf-fortran-devel \
    git curl
```

Then install Julia with:

```bash
curl -fsSL https://install.julialang.org | sh
```

Package names can differ on older enterprise distributions. The important
check is that both `nc-config` and `nf-config` are on `PATH`.

### Windows

#### Recommended: WSL2 + Ubuntu

The BMM build currently assumes GNU Make, `gfortran`, Unix paths and standard
Unix shell commands. The simplest Windows setup is therefore to run **both BMM
and Julia inside WSL2**.

From an Administrator PowerShell terminal, install WSL and Ubuntu if needed:

```powershell
wsl --install -d Ubuntu
```

Restart Windows if requested, launch Ubuntu, create the Linux username/password
when prompted, and then inside the Ubuntu terminal run:

```bash
sudo apt update
sudo apt install -y \
    build-essential gfortran \
    libnetcdf-dev libnetcdff-dev netcdf-bin \
    curl git unzip

curl -fsSL https://install.julialang.org | sh
```

Open a new WSL shell and verify:

```bash
gfortran --version
nc-config --version
nf-config --version
julia --version
make --version
```

Keep the repository inside the WSL Linux filesystem for best build and I/O
performance, for example:

```bash
mkdir -p ~/src
cd ~/src
unzip /mnt/c/Users/<your-user>/Downloads/BMM_Julia_Surrogate_Multimode_20260831_crossplatform.zip
cd BMM_Julia_Surrogate_Multimode_20260831_crossplatform
```

If the archive extracts directly into the current directory rather than a
named directory, simply `cd` to wherever `Makefile`, `bmm/` and `julia/` are
located.

Native Windows Julia is also available through the official Julia installer /
Microsoft Store, but **native Windows BMM compilation is not currently the
supported path for this repository**. It would require a separate MSYS2 or
similar NetCDF-Fortran toolchain and Makefile testing.

### Install the Julia project dependencies

Once the platform prerequisites above are available, the rest of the workflow
is identical on macOS, Linux and Windows/WSL. From the repository root:

```bash
make julia-instantiate
```

This runs Julia's package manager using `julia/Project.toml` and installs the
required packages (Flux, DifferentialEquations/SciML, NCDatasets, etc.). The
first run can spend some time precompiling packages.

To verify the Julia environment directly:

```bash
cd julia
julia --project=. -e 'using Flux, DifferentialEquations, NCDatasets; println("Julia packages OK")'
cd ..
```

### Build BMM

From the repository root:

```bash
make bmm
```

The top-level Makefile runs `nc-config --prefix` and `nf-config --prefix` and
passes those locations into BMM. A successful build produces:

```text
bmm/main.exe
```

Despite the `.exe` suffix this is also the normal executable name on macOS and
Linux; it does **not** imply a Windows binary.

### Smoke test the complete Julia -> BMM interface

Run:

```bash
make smoke
```

This performs a small multimode calculation and checks the complete path:

```text
Julia
  -> write a BMM namelist
  -> execute bmm/main.exe
  -> produce BMM NetCDF output
  -> read the NetCDF in Julia
  -> resample the trajectory in height
  -> check the cloud diagnostics
```

Only proceed to ensemble generation once this test succeeds.

### Quick setup checklist

All of these commands should work before generating training data:

```bash
gfortran --version
nc-config --version
nf-config --version
julia --version
make --version
ls -l bmm/main.exe
```

Then:

```bash
make julia-test
make smoke
```

### Common setup problems

**`julia: command not found`**  
Open a new terminal after installing `juliaup`, or follow the PATH instruction
printed by the installer.

**`nf-config: command not found`**  
The NetCDF-Fortran *development* package is missing. Installing only the NetCDF
C library is not sufficient for BMM.

**`cannot find -lnetcdff`**  
Check:

```bash
nf-config --prefix
nf-config --flibs
```

and confirm that the prefix contains the NetCDF-Fortran library. The repository
Makefile normally obtains this prefix automatically.

**Julia packages fail after changing Julia versions**  
From `julia/` run:

```bash
julia --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'
```

**Windows PowerShell commands do not match the README**  
After the one-time `wsl --install` command, run the repository commands in the
Ubuntu/WSL terminal, not PowerShell.


## Synthetic training ensemble

`julia/SyntheticDataset.jl` creates a broad but physically structured pilot
design. The defaults are deliberately configurable rather than hard-coded into
the surrogate.

The default pilot samples:

| quantity | synthetic range |
|---|---|
| number of modes | 1--4, balanced |
| constant updraft | 0.1--10 m s^-1, log sampled |
| cloud-base pressure | 60--102 kPa |
| cloud-base temperature | mostly pressure-correlated lower-tropospheric values, with 10% broad 250--303 K stress samples |
| nucleation Dm | 3--20 nm |
| Aitken Dm | 8--80 nm |
| accumulation Dm | 50--500 nm |
| coarse Dm | 0.5--4 micrometres |
| ln(sigma_g) | approximately 0.20--0.80 |
| kappa | 0--1.4, stratified so low- and high-hygroscopicity cases are represented |

Number concentration is sampled conditionally on size class, because using one
independent `N` range for a 5 nm mode and a 3 micrometre coarse mode produces
many physically unhelpful combinations. The synthetic ranges are specified in
**cm^-3 at cloud base** for physical readability, then converted using cloud-base
dry-air density to BMM's required `n_aer1` number mixing ratio in **# kg^-1 dry
air** before writing the namelist. The class name is stored only as
sampling metadata and **must not be given to the surrogate**.

The pilot runs to 500 m above cloud base, resampling BMM onto a 5 m height grid.
The BMM timestep is capped by both 5 s and 2 m of vertical travel per step.

**Important:** datasets generated by earlier versions that wrote sampled
cm^-3 values directly into `n_aer1` should be regenerated. Re-running only the
analysis script is not sufficient, because those BMM trajectories were run with
the wrong number-mixing-ratio input.

Generate 256 cases:

```bash
make bmm
make julia-instantiate
cd julia
BMM_EXE=../bmm/main.exe julia --project=. -t auto generate_synthetic_dataset.jl \
    256 synthetic_bmm.nc 4 20260831
```

or with environment variables:

```bash
NCASES=1000 MAX_MODES=4 BMM_DATASET=data/synthetic_1000.nc \
BMM_EXE=../bmm/main.exe julia --project=. -t auto generate_synthetic_dataset.jl
```

The output NetCDF contains:

### Inputs

- `n_modes`
- `T_cloud_base`, `P_cloud_base`, `w`
- `mode_N(case,mode)` in # kg^-1 dry air (the BMM `n_aer1` input)
- `mode_N_cloud_base(case,mode)` in m^-3 (diagnostic)
- `cloud_base_rhod(case)` in kg m^-3
- `mode_Dm(case,mode)`
- `mode_lnsig(case,mode)`
- `mode_kappa(case,mode)`
- classical-Koehler metadata `mode_nu`, `mode_molw`, `mode_density`

Unused mode slots are NaN. This fixed NetCDF storage shape is only a disk
format; a downstream Deep Sets / attention encoder can consume the valid modes
as a variable-length set.

### BMM targets and trajectory diagnostics

- parcel pressure `P(case,height)` in Pa
- parcel temperature `T(case,height)` in K
- relative humidity `rh(case,height)`
- dry-air density `rhod(case,height)` in kg m^-3
- supersaturation `S(case,height)`
- cloud-drop number `Nd(case,height)` in m^-3
- native cloud-drop number mixing ratio `Nd_kg(case,height)` in # kg^-1 dry air
- extinction `beta_ext(case,height)` in m^-1
- liquid-water mixing ratio `ql(case,height)`
- effective diameter `deff(case,height)`
- per-mode diagnostic `Dcrit_mode(case,height,mode)`

`ql` and `deff` are stored even though they are not required for the first
surrogate. They are likely to be useful when the work is extended to existing
cloud water / accelerated updrafts.

## Recommended generation sequence

Do **not** immediately launch tens of thousands of BMM runs. A better sequence
is:

1. 256-case pilot: check failures and distributions of `Nd`, `Smax` and
   extinction;
2. 1,000--2,000 cases: train the first set-based surrogate and examine error by
   aerosol regime;
3. add targeted samples where error is high, especially weak-updraft/high-N
   activation competition and giant-CCN cases;
4. finally mix in the actual climate-model aerosol fields when they are
   available.

Very weak updrafts below 0.1 m s^-1 are intentionally not numerous in the
initial set because a fixed 500 m trajectory becomes expensive. They should be
added later as a targeted tranche if the intended host model needs them.

## Height coordinate

The truth dataset is parameterised by **height above cloud base**, not elapsed
time. For the current constant-updraft experiments,

```text
runtime = height_top / w
```

so every case covers the same vertical extent. This also leaves a clean route
to a future `w(z)` forcing without redefining the target.

## Repository layout

```text
bmm/                              BMM upstream source at subtree prefix
julia/BMM.jl                      multimode BMM namelist/process interface
julia/SyntheticDataset.jl         synthetic design + NetCDF writer
julia/generate_synthetic_dataset.jl
julia/SurrogateData.jl            trajectory loader + leakage-safe split
julia/SetSurrogate.jl             variable-mode set encoder + profile/Neural ODE models
julia/analyse_dataset.jl          ensemble coverage diagnostics
julia/train_surrogate.jl          profile or Neural ODE training
julia/compare_surrogates.jl       held-out comparison on the same split
julia/smoke.jl                    two-mode BMM/Julia smoke test
scripts/bmm-subtree.sh            git-subtree helper
```

The old fixed-context single-mode surrogate has been removed so it cannot be
used accidentally with the multimode dataset. Both current models use the same
permutation-invariant variable-mode encoder.

## BMM subtree

The supplied archive has no parent `.git` history or upstream BMM URL, so an
actual subtree merge commit cannot be manufactured from the archive alone.
The helper is ready for the real repository:

```bash
scripts/bmm-subtree.sh add <BMM_GIT_URL> main
scripts/bmm-subtree.sh pull <BMM_GIT_URL> main
```

## Build and smoke test

```bash
make bmm
make julia-instantiate
make smoke
```

BMM is intentionally run as a fresh executable process per truth case. The
Fortran module owns process-global allocated state, so treating BMM as an
offline truth generator is safer than differentiating through or repeatedly
calling it in-process. The trained surrogate will be pure Julia.

## Diagnosing failed BMM ensemble cases

Synthetic truth generation is **liquid-only**, including for supercooled cloud
bases. The Julia wrapper always writes:

```text
ice_flag=0
bin_scheme_flag=0
sce_flag=0
```

If BMM exits non-zero, the terminal warning prints the complete sampled case:
cloud-base T/P, constant updraft, runtime/timestep, and every aerosol mode's
N, Dm, lnsig, kappa, density, van't Hoff factor, and molecular weight.

The original BMM inputs and logs are also retained. For an output named
`synthetic_bmm.nc`, failures are stored by default as:

```text
synthetic_bmm_failures/
  case_0131/
    case_summary.txt
    namelist.in
    stdout.log
    stderr.log
    output.nc          # only if BMM created one before failing
```

Override the directory with `BMM_FAILURE_DIR`, for example:

```bash
BMM_FAILURE_DIR=failed_cases \
BMM_EXE=../bmm/main.exe \
julia --project=. -t auto generate_synthetic_dataset.jl 256 synthetic_bmm.nc 4 20260831
```

A failed case can then be reproduced directly from the repository root with:

```bash
./bmm/main.exe julia/synthetic_bmm_failures/case_0131/namelist.in
```

(adjust the path if the dataset was written elsewhere). The saved namelist is
the exact input used in the failed ensemble run.


## Frozen BMM truth-model baseline for surrogate generation

The BMM source included in this revision contains the warm/ice DVODE recovery
policy established during the synthetic stress tests:

```text
particle liquid/ice mass ATOL = 1e-25 kg
DVODE MXSTEP per call          = 100
maximum recovery restarts      = 100
recover ISTATE -1 or -5        = restart from last accepted state with ISTATE=1
```

The same recovery rule is used for liquid condensation/evaporation and ice
deposition/sublimation. The physical growth equations are not changed by the
recovery policy. `fparcelwarm` also initializes the complete RHS (`ydot`) on
every call, and prescribed updraft type 3 uses `ydot(iw)` rather than modifying
DVODE's state vector from inside the RHS callback.

This is the truth-model baseline for the first surrogate dataset and should not
be changed while comparing surrogate architectures, otherwise the target model
would move during training experiments.

## First surrogate-training workflow

Once `synthetic_bmm.nc` has been generated successfully, freeze the BMM truth
configuration for this stage and work entirely in `julia/`.

### 1. Diagnose the synthetic ensemble

From `julia/`:

```bash
julia --project=. analyse_dataset.jl synthetic_bmm.nc
```

The analysis reports distributions of cloud-base conditions, total aerosol
number, maximum supersaturation, maximum/final cloud-drop number, activated
fraction, extinction, liquid water and effective diameter. Activated fraction is
computed exactly from native number mixing ratios,
`max(Nd_kg) / sum(mode_N)`, so it is independent of air-density changes with
height. The cloud-base values of `S` and activated fraction are also checked
because surrogate v2 imposes their known cloud-base boundary conditions.

### 2. Variable-number-of-modes representation (v2)

The model still uses a permutation-invariant number-weighted set encoder. Each
real mode is encoded from intensive properties and an activation-relevant derived
feature:

```text
log(Dm), ln(sigma), transformed kappa_eff, approximate median Scrit
```

The `Scrit` feature is the standard kappa-Koehler critical supersaturation at the
mode median dry diameter, evaluated at cloud-base temperature. It is only a
physics-informed input feature: BMM remains the training truth and the network is
not replaced by an activation parameterisation.

For mode `i`, the shared encoder produces `phi_i` and

```text
Ntot  = sum_i N_i
E_aer = sum_i (N_i/Ntot) phi_i
```

The pooled embedding is augmented by `log(Ntot)`, cloud-base `T`/`P`, and two
permutation/splitting-invariant lognormal bulk moments proportional to dry
surface area and dry volume. This preserves:

- invariance to aerosol-mode ordering;
- support for 1, 2, 3, ... modes without changing network size;
- exact invariance to splitting one mode into identical submodes whose numbers
  sum to the original mode.

### 3. Physically constrained targets

The first pilot showed that learning absolute `Nd` allowed unphysical predictions
above the available aerosol number and gave too little weight to very weak
activation. Surrogate v2 therefore learns

```text
S(z),  f_act(z) = Nd_kg(z) / sum(mode_N),  beta_ext(z)
```

where `f_act` is represented in a finite scaled-logit latent variable and is
converted back with an explicit `[0,1]` bound. Therefore the model cannot create
more cloud droplets than aerosol particles. `Nd_kg` is reconstructed as
`f_act * Ntot`; conversion to m^-3 uses dry-air density only as a unit conversion.
New datasets store the complete `rhod(z)` profile; older v1 datasets remain
readable and use cloud-base density as a fallback for this display conversion.

Known cloud-base conditions are imposed rather than learned:

```text
S(0)     = 0
f_act(0) = 0
```

Extinction is *not* forced to zero at cloud base because hydrated but
unactivated aerosol can already contribute extinction.

The profile loss remains primarily pointwise, with weak peak and final-state
terms added after the pilot showed that a pure pointwise MSE smoothed the narrow
supersaturation maximum and activation transition.

### 4. Train/test split

`train_surrogate.jl` splits by complete BMM trajectory, never by height point.
The default is approximately 70% training, 15% validation and 15% test,
stratified by aerosol mode count. `training_split.csv` records the original BMM
case number assigned to each split. Model initialization is also seeded for
reproducibility.

### 5. Direct profile control model

Train this first:

```bash
cd julia
julia --project=. train_surrogate.jl synthetic_bmm.nc profile 150 20260831
```

The v2 profile network uses a slightly larger shared encoder/head and three
continuous height coordinates (`h`, `sqrt(h)`, and a compressed near-base
coordinate) so it can resolve the sharp activation layer more efficiently.
Training includes gradient clipping, best-validation checkpoint restoration and
early-flat-validation early stopping. The saved file is
`profile_surrogate.bson`.

**Models trained by the earlier absolute-`Nd` architecture are format v1 and are
not loadable by the v2 code. Retrain after updating the code.**

### 6. Neural ODE

After the v2 profile model is satisfactory, run a short Neural ODE smoke test:

```bash
julia --project=. train_surrogate.jl synthetic_bmm.nc neuralode 5 20260831
```

The Neural ODE integrates in normalized height above cloud base. Its initial
supersaturation and activated-fraction states are fixed to the physical cloud-
base values; only the initial extinction latent is learned. The aerosol context
is carried as a zero-derivative augmented state so gradients still reach the
set encoder. If the smoke test is stable, use 40 epochs as the current default.

### 7. Compare held-out BMM and surrogate fields

After training one or both v2 models:

```bash
julia --project=. compare_surrogates.jl synthetic_bmm.nc 20260831 profile 8
julia --project=. compare_surrogates.jl synthetic_bmm.nc 20260831 both 8
```

The comparison script caches held-out predictions and writes
`surrogate_comparison/` containing:

- BMM versus surrogate vertical profiles of `S`, activated fraction, `Nd` and
  extinction, with BMM `ql` and `deff` for context;
- `comparison_metrics.csv`;
- `selected_cases.txt`, including each plotted mode's `N`, `Dm`, `lnsig`, kappa
  and median critical supersaturation;
- all-test 1:1 scatter plots for maximum supersaturation, maximum activation and
  maximum extinction;
- terminal metrics broken down by aerosol mode count and BMM activation regime.

The most useful v2 metrics are activated-fraction RMSE/MAE, native-`Nd_kg`
log-space RMSE, supersaturation RMSE and peak error, extinction log-space RMSE,
and the number of activation-bound violations (which should be exactly zero).

### Recommended progression

1. Retrain the v2 profile model on the existing 256-case pilot and compare the
   same held-out profiles that exposed the v1 weaknesses.
2. Only if those changes behave as intended, run the 5-epoch v2 Neural ODE
   smoke test.
3. Generate roughly 2,000 BMM trajectories and retrain both models on exactly
   the same split.
4. Use the mode-count/activation-regime diagnostics and all-test peak scatters to
   target additional BMM samples rather than increasing the ensemble blindly.

`ql`, `deff`, mode-resolved `Dcrit`, and the new `P/T/rh/rhod` trajectory fields
remain diagnostic rather than primary v2 learning targets. They are retained for
the later existing-cloud-water, variable-updraft and host-model extensions.

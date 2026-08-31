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

## Multimode aerosol interface

A mode is represented in Julia by

```julia
AerosolMode(
    N       = 400e6,     # m^-3
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
many physically unhelpful combinations. The class name is stored only as
sampling metadata and **must not be given to the surrogate**.

The pilot runs to 500 m above cloud base, resampling BMM onto a 5 m height grid.
The BMM timestep is capped by both 5 s and 2 m of vertical travel per step.

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
- `mode_N(case,mode)`
- `mode_Dm(case,mode)`
- `mode_lnsig(case,mode)`
- `mode_kappa(case,mode)`
- classical-Koehler metadata `mode_nu`, `mode_molw`, `mode_density`

Unused mode slots are NaN. This fixed NetCDF storage shape is only a disk
format; a downstream Deep Sets / attention encoder can consume the valid modes
as a variable-length set.

### BMM targets

- supersaturation `S(case,height)`
- cloud-drop number `Nd(case,height)` in m^-3
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
julia/NeuralODESurrogate.jl       current single-mode baseline surrogate
julia/example_generate_and_train.jl
julia/smoke.jl                    two-mode BMM/Julia smoke test
scripts/bmm-subtree.sh            git-subtree helper
```

The current `NeuralODESurrogate.jl` is retained as a single-mode baseline. The
next training change should replace its fixed aerosol context with a
permutation-invariant mode-set encoder before training on the multimode file.

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

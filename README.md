# BMM <-> Julia Neural ODE surrogate

Verified against your uploaded BMM tree: compiles cleanly with gfortran +
NetCDF, one adiabatic activation case (`configs/namelist.basic`) runs in
~0.25s and produces sane output (S peaks ~0.1-0.3%, Nd saturates fast,
`beta_ext` keeps climbing with LWC after activation).

## 1. Two things to fix in your subtree first

Your `configs/namelist.basic` has drifted from the current
`bin_microphysics_module.f90` namelist schema (probably an older example
kept around from before a refactor). Two fixes were needed to get a clean
run:

1. **`&chamber_options`** — the example file uses field names
   (`chamber_bl_temp_mode`, missing `chamber_bl_alpha_t`, etc.) that don't
   match the `chamber_options` namelist declared at
   `bin_microphysics_module.f90:426`. Since chamber options are optional
   (defaults are set right before the read, see line ~469-497), the
   simplest fix is to just **omit the whole `&chamber_options` block** for
   parcel-only (non-chamber) runs. `BMM.jl`'s namelist writer does this —
   it never emits a `&chamber_options` block.
2. **`vert_ent=.true.`** — the code now `error stop`s with *"Sanchez
   cloud-top entrainment has been removed; use lateral entrainment"* if
   this is true. `BMM.jl` defaults `vert_ent=false`.

A known-good example namelist is saved at
`configs/namelist.basic.example` if you want to diff it against yours.

## 2. Reentrancy: BMM is not safe to call twice in one process

Confirmed experimentally (see `fortran/bmm_capi.f90` docstring): calling
`bmm_driver` twice in the same OS process fails on the second call with
`STOP *** Not enough memory ***`, because
`bin_microphysics_module.f90`'s module-level arrays are `allocate`d on
every call without an `if (allocated(x)) deallocate(x)` guard first.

This is why the recommended Julia interface (`BMM.run_bmm` /
`BMM.run_bmm_batch` in `julia/BMM.jl`) spawns a **fresh `main.exe` process
per case** rather than `ccall`ing into a shared library in a loop. It's
simple, already-safe, and still fast (~0.25s/run, parallel across cores
via `Threads.@threads`).

An optional `ccall`-based path (`fortran/bmm_capi.f90` → `libbmm.so`,
wrapped by `BMM.run_bmm_lib`) is included for the single-call-per-process
case (e.g. one-off validation), with the reentrancy caveat documented
in-line. If you want true in-process speed for many calls, the real fix is
patching `bin_microphysics_module.f90` with `deallocate` guards — flagging
that as a decision for your subtree rather than doing it silently, since
it's a change to upstream BMM logic, not just a wrapper.

## 3. Build

```bash
# from your BMM checkout root (adjust NETCDF paths for your system if not Ubuntu/apt netCDF)
make cleanall
make NETCDF_FOR=/usr NETCDF_C=/usr FFLAGS="-O3 -fPIC -g -o" FFLAGS2="-g -O3 -fPIC -o"
# main.exe now exists, and all .o/.a were built with -fPIC (needed for libbmm.so below)

# optional: build the ccall shared library
gfortran -c fortran/bmm_capi.f90 -I. -Iosnf -Isce -Iopt -O3 -fPIC -o bmm_capi.o
gfortran -shared -fPIC -o fortran/libbmm.so bmm_capi.o \
    bin_microphysics_module.o b_micro_lib.a opt/optics.a \
    osnf/osnf_lib.a sce/sce_micro_lib.a sce/sce_module.o -lnetcdff
```

A working `libbmm.so` built this way (exports `bmm_run_c`, verified via a
single ctypes call) is included at `fortran/libbmm.so` for reference, but
rebuild it against your actual subtree — this one is linked against the
exact archive you uploaded.

## 4. Julia side

```bash
cd julia
julia --project=. -e 'using Pkg; Pkg.instantiate()'
export BMM_EXE=/path/to/your/bmm/main.exe
julia --project=. -t auto example_generate_and_train.jl
```

- `BMM.jl` — namelist writer + subprocess runner + NetCDF reader +
  `Dcrit` derivation (from the bin-resolved `nliq`/`nwat`/`mbinedges`
  fields, which are already in BMM's NetCDF output — no need to derive
  extinction or Nd separately, `beta_ext` and `ndrop` are direct outputs).
- `NeuralODESurrogate.jl` — Flux + DifferentialEquations.jl +
  SciMLSensitivity.jl Neural ODE: state `[S, log1p(Nd), log1p(beta_ext),
  Dcrit/1e-7]`, forced by `w(t)`, conditioned on a static context vector
  (aerosol N/Dg/sigma, kappa, T0, P0). Pure Neural ODE, not a UDE — see
  the module docstring for the natural UDE upgrade path (hardcode the
  known adiabatic-cooling term for `dS/dt` and let the NN only learn the
  condensation/activation correction) if you want tighter physics
  constraints once the baseline is working.
- `example_generate_and_train.jl` — template wiring the two together with
  a random parameter sweep. Swap `generate_case_data` for a loader over
  your own existing calibration dataset via
  `NeuralODESurrogate.load_training_cases` (currently a stub — point it at
  your on-disk format) and skip calling BMM at all if you don't need more
  data.

## 5. One physics note worth acting on

From the test run: Nd is set almost entirely by the peak-supersaturation
transient in the first ~1-2 minutes of ascent, then stays flat while
`beta_ext` keeps climbing with LWC for the rest of the run. If your
calibration set spans long updraft periods, consider whether Nd/Dcrit
really need the full ODE treatment (a plain feedforward net over context +
w might do as well and train far faster/more robustly) versus reserving
the Neural ODE specifically for `beta_ext(t)`, which is genuinely
time-evolving. Easy to test empirically once you have the pure Neural ODE
baseline running: compare its Nd/Dcrit accuracy against a cheap feedforward
baseline trained on the same data.

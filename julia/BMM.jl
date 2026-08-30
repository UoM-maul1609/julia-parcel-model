"""
    module BMM

Julia interface to the Fortran Bin Microphysics Module (BMM), for generating
adiabatic-parcel activation cases (droplet number Nd, extinction, critical
activation diameter Dcrit) to train/validate a Neural ODE surrogate.

Two backends are provided:

  * `run_bmm` (recommended, default): writes a namelist, spawns `main.exe` as
    a fresh OS process, reads the resulting NetCDF file. Safe to call any
    number of times, trivially parallel across processes/cores. This is the
    only backend that's currently safe for repeated calls, because BMM's
    Fortran internals are not reentrant within a single process (see the
    caveat in fortran/bmm_capi.f90 -- calling the driver twice in one process
    fails with "Not enough memory" because module-level arrays are allocated
    without being freed first).

  * `run_bmm_lib` (advanced, single call per process only): ccall's into
    fortran/libbmm.so directly, skipping the process-spawn overhead. Only
    use this if you spawn a fresh `julia -p` worker (or `julia --project
    script.jl`) per call, e.g. as a `julia -e '...'` subprocess itself, or if
    you've patched BMM's Fortran to be reentrant (add
    `if (allocated(x)) deallocate(x)` guards before every `allocate(x(...))`
    in bin_microphysics_module.f90).

Usage:

    using .BMM
    case = BMMCase(winit=1.5, tinit=282.0, pinit=90000.0, rhinit=0.95,
                    n_aer1=[100e6], d_aer1=[80e-9], sig_aer1=[0.4],
                    kappa_core1=[0.6])
    result = run_bmm(case; exe="/path/to/main.exe")
    # result.t, result.z, result.p, result.T, result.S, result.w,
    # result.ndrop, result.beta_ext, result.dcrit
"""
module BMM

using Printf
using NCDatasets

export BMMCase, run_bmm, run_bmm_lib, run_bmm_batch, derive_dcrit

# -----------------------------------------------------------------------
# Case definition
# -----------------------------------------------------------------------

"""
    BMMCase

All fields needed to build a BMM namelist for a single adiabatic warm-cloud
activation case (single external aerosol mode, `n_intern` lognormal
submodes, kappa-Koehler activation). Extend/adapt if you need multiple
external modes, ice, or chamber forcing -- see configs/namelist.basic in the
BMM repo for the full namelist schema; anything not exposed here just keeps
BMM's own defaults.

Fields mirror BMM's namelist variable names directly so `to_namelist` is a
near-literal transcription.
"""
Base.@kwdef struct BMMCase
    # &run_vars
    runtime::Float64 = 3000.0
    dt::Float64 = 10.0
    zinit::Float64 = 300.0
    tpert::Float64 = 0.0
    winit::Float64 = 0.6            # updraft velocity [m/s]
    tinit::Float64 = 282.0          # cloud-base temperature [K]
    pinit::Float64 = 90000.0        # cloud-base pressure [Pa]
    rhinit::Float64 = 0.95          # initial RH (fraction)
    radinit::Float64 = 500.0
    microphysics_flag::Int = 1
    ice_flag::Int = 0               # liquid only
    bin_scheme_flag::Int = 0        # BIN_FULL_MOVING (fastest; required for run_bmm_lib)
    sce_flag::Int = 0
    vent_flag::Int = 1
    kappa_flag::Int = 1             # kappa-Koehler (simplest input set)
    updraft_type::Int = 1           # constant updraft
    t_thresh::Float64 = 1.0e6
    adiabatic_prof::Bool = true
    vert_ent::Bool = false          # removed scheme upstream; keep false
    z_ctop::Float64 = -1.0
    ent_rate::Float64 = 1.0e-3
    alpha_therm::Float64 = 1.0
    alpha_cond::Float64 = 1.0
    alpha_therm_ice::Float64 = 1.0
    alpha_dep::Float64 = 1.0
    hm_flag::Bool = false
    break_flag::Int = 0
    mode1_flag::Bool = false
    mode2_flag::Bool = false

    # &aerosol_setup
    n_mode::Int = 1
    n_bins::Int = 60
    n_comps::Int = 1
    n_sv::Int = 0
    sv_flag::Int = 0

    # &aerosol_spec  (single external mode, n_intern lognormal submodes)
    n_aer1::Vector{Float64} = [400.0e6]        # number conc per submode [m^-3]
    d_aer1::Vector{Float64} = [80.0e-9]        # number-median dry diameter [m]
    sig_aer1::Vector{Float64} = [0.4]          # ln(sigma_g) per submode
    kappa_core1::Vector{Float64} = [0.6]       # hygroscopicity per component
    dmina::Float64 = 1.0e-9
    dmaxa::Float64 = 3.0e-6
    mass_frac_aer1::Vector{Float64} = [1.0]
    molw_core1::Vector{Float64} = [132.14e-3]
    density_core1::Vector{Float64} = [1770.0]
    nu_core1::Vector{Float64} = [3.0]
end

n_intern(case::BMMCase) = length(case.n_aer1)

# -----------------------------------------------------------------------
# Namelist rendering
# -----------------------------------------------------------------------

_fbool(b::Bool) = b ? ".true." : ".false."
_farr(v::AbstractVector) = join(@sprintf("%.8e", x) for x in v), ", "

function to_namelist(case::BMMCase, outputfile::AbstractString)
    ni = n_intern(case)
    io = IOBuffer()
    println(io, " &run_vars")
    println(io, "    outputfile = '", outputfile, "',")
    println(io, "    runtime=", case.runtime, ",")
    println(io, "    dt=", case.dt, ",")
    println(io, "    zinit=", case.zinit, ",")
    println(io, "    tpert=", case.tpert, ",")
    println(io, "    use_prof_for_tprh=.false.,")
    println(io, "    winit=", case.winit, ",")
    println(io, "    winit2=", case.winit, ",")
    println(io, "    amplitude2=100.,")
    println(io, "    tinit=", case.tinit, ",")
    println(io, "    pinit=", case.pinit, ",")
    println(io, "    rhinit=", case.rhinit, ",")
    println(io, "    radinit=", case.radinit, ",")
    println(io, "    bubble_flag=.false.,")
    println(io, "    microphysics_flag=", case.microphysics_flag, ",")
    println(io, "    ice_flag=", case.ice_flag, ",")
    println(io, "    bin_scheme_flag=", case.bin_scheme_flag, ",")
    println(io, "    sce_flag=", case.sce_flag, ",")
    println(io, "    vent_flag=", case.vent_flag, ",")
    println(io, "    kappa_flag=", case.kappa_flag, ",")
    println(io, "    updraft_type=", case.updraft_type, ",")
    println(io, "    t_thresh=", case.t_thresh, ",")
    println(io, "    adiabatic_prof=", _fbool(case.adiabatic_prof), ",")
    println(io, "    entrain_period=0,")
    println(io, "    thresh_to_start_hom_mix=-10.,")
    println(io, "    release_aerosol=.true.,")
    println(io, "    entrain_aerosol=.true.,")
    println(io, "    vert_ent=", _fbool(case.vert_ent), ",")
    println(io, "    z_ctop=", case.z_ctop, ",")
    println(io, "    ent_rate=", case.ent_rate, ",")
    println(io, "    n_levels_s = 2,")
    println(io, "    alpha_therm=", case.alpha_therm, ",")
    println(io, "    alpha_cond=", case.alpha_cond, ",")
    println(io, "    alpha_therm_ice=", case.alpha_therm_ice, ",")
    println(io, "    alpha_dep=", case.alpha_dep, ",")
    println(io, "    hm_flag=", _fbool(case.hm_flag), ",")
    println(io, "    break_flag=", case.break_flag, ",")
    println(io, "    mode1_flag=", _fbool(case.mode1_flag), ",")
    println(io, "    mode2_flag=", _fbool(case.mode2_flag))
    println(io, "    /")
    println(io, "&aerosol_setup")
    println(io, "    n_mode            = ", case.n_mode, ",")
    println(io, "    n_intern          = ", ni, ",")
    println(io, "    n_sv              = ", case.n_sv, ",")
    println(io, "    sv_flag           = ", case.sv_flag, ",")
    println(io, "    n_bins            = ", case.n_bins, ",")
    println(io, "    n_comps           = ", case.n_comps, "/")
    println(io, "&sounding_spec")
    # Minimal 2-level sounding; BMM initialises T/P/RH from tinit/pinit/rhinit
    # directly (use_prof_for_tprh=.false. above), but q_read/theta_read/
    # rh_read/z_read must still be present with n_levels_s entries.
    println(io, "    psurf=", case.pinit, ",")
    println(io, "    tsurf=", case.tinit, ",")
    println(io, "    q_read(1,1:2)   = 7.0e-3, 7.0e-3,")
    println(io, "    theta_read(1:2) = ", case.tinit, ", ", case.tinit, ",")
    println(io, "    rh_read(1:2)    = ", case.rhinit, ", ", case.rhinit, ",")
    println(io, "    z_read(1:2)     = 0.0, 10000.0/")
    println(io, "&aerosol_spec")
    println(io, "    n_aer1(1:", ni, ",1)   = ", join(case.n_aer1, ", "), ",")
    println(io, "    d_aer1(1:", ni, ",1)   = ", join(case.d_aer1, ", "), ",")
    println(io, "    sig_aer1(1:", ni, ",1) = ", join(case.sig_aer1, ", "), ",")
    println(io, "    dmina = ", case.dmina, ",")
    println(io, "    dmaxa = ", case.dmaxa, ",")
    println(io, "    mass_frac_aer1(1,1:", case.n_comps, ") = ", join(case.mass_frac_aer1, ", "), ",")
    println(io, "    molw_core1(1:", case.n_comps, ")    = ", join(case.molw_core1, ", "), ",")
    println(io, "    density_core1(1:", case.n_comps, ") = ", join(case.density_core1, ", "), ",")
    println(io, "    nu_core1(1:", case.n_comps, ")       = ", join(case.nu_core1, ", "), ",")
    println(io, "    kappa_core1(1:", case.n_comps, ")    = ", join(case.kappa_core1, ", "), "/")
    return String(take!(io))
end

# -----------------------------------------------------------------------
# Dcrit derivation from bin-resolved output
# -----------------------------------------------------------------------

"""
    derive_dcrit(nliq, nwat, mbinedges, density) -> Vector{Float64}

Critical DRY activation diameter [m] at each output time, derived from the
bin-resolved activated fraction nliq./nwat (nliq is the activated portion of
each bin's number, nwat the total). Finds the mass-bin edge where the
activated fraction crosses 0.5 and converts that edge from dry mass [kg] to
dry diameter [m] assuming spherical particles of the given material density
[kg/m^3]. `nliq`/`nwat` here are the (nmodes=1) 1-D bin arrays at one time
(size nbinst); loop over `times` to build the full series (see `run_bmm`).
Returns NaN for times where no activation has occurred yet (S < 0 for the
whole run) or all bins are activated.
"""
function derive_dcrit(nliq_t::AbstractVector, nwat_t::AbstractVector,
                       mbinedges::AbstractVector, density::Float64)
    frac = similar(nliq_t, Float64)
    @inbounds for i in eachindex(nliq_t)
        frac[i] = nwat_t[i] > 0 ? nliq_t[i] / nwat_t[i] : 0.0
    end
    idx = findfirst(>=(0.5), frac)
    if idx === nothing || idx == 1
        return NaN
    end
    m_edge = mbinedges[idx]  # dry mass at the lower edge of the first
                             # >=50%-activated bin [kg]
    d_edge = (6.0 * m_edge / (pi * density))^(1/3)
    return d_edge
end

# -----------------------------------------------------------------------
# Backend 1 (default): fresh subprocess per call
# -----------------------------------------------------------------------

"""
    run_bmm(case::BMMCase; exe, workdir=mktempdir(), density=case.density_core1[1])

Run one BMM case via a fresh `main.exe` subprocess and return a NamedTuple
of time series: t, z, p, T, S, w, ndrop [m^-3, converted from #/kg via rho_air],
beta_ext [m^-1], dcrit [m].

`exe` is the path to the compiled BMM executable (build with the repo's
Makefile; see README.md).
"""
function run_bmm(case::BMMCase; exe::AbstractString, workdir::AbstractString=mktempdir(),
                  density::Float64=case.density_core1[1])
    nmlpath = joinpath(workdir, "namelist.in")
    ncpath = joinpath(workdir, "output.nc")
    open(nmlpath, "w") do f
        write(f, to_namelist(case, ncpath))
    end

    proc = run(pipeline(`$exe $nmlpath`; stdout=joinpath(workdir, "stdout.log"),
                         stderr=joinpath(workdir, "stderr.log")); wait=false)
    wait(proc)
    if proc.exitcode != 0
        errlog = read(joinpath(workdir, "stderr.log"), String)
        error("BMM run failed (exit $(proc.exitcode)) for namelist $nmlpath:\n$errlog")
    end

    ds = NCDataset(ncpath)
    t = Array(ds["time"])
    z = Array(ds["z"])
    p = Array(ds["p"])
    T = Array(ds["t"])
    rh = Array(ds["rh"])
    w = Array(ds["w"])
    ndrop_kg = Array(ds["ndrop"])       # #/kg (dry air, per model convention)
    beta_ext = Array(ds["beta_ext"])
    nliq = Array(ds["nliq"])            # (times, nmodes, nbinst)
    nwat = Array(ds["nwat"])
    mbinedges = Array(ds["mbinedges"])  # (nmodes, nbinste)
    close(ds)

    ntimes = length(t)
    dcrit = Vector{Float64}(undef, ntimes)
    for i in 1:ntimes
        dcrit[i] = derive_dcrit(view(nliq, i, 1, :), view(nwat, i, 1, :),
                                 view(mbinedges, 1, :), density)
    end

    # rho_air from ideal gas law to convert ndrop from #/kg to #/m^3 (more
    # directly comparable across cases at different p, T)
    Rd = 287.05
    rho_air = p ./ (Rd .* T)
    ndrop = ndrop_kg .* rho_air

    S = rh .- 1.0  # supersaturation (fraction); rh here is already a fraction

    return (t=t, z=z, p=p, T=T, S=S, w=w, ndrop=ndrop, beta_ext=beta_ext, dcrit=dcrit,
            case=case)
end

"""
    run_bmm_batch(cases::Vector{BMMCase}; exe, ntasks=Sys.CPU_THREADS, kwargs...)

Run many cases in parallel OS processes (via Julia's `Threads` spawning
subprocesses -- each `run_bmm` call is I/O/subprocess bound, not CPU bound
in the Julia process itself, so `Threads.@spawn` over subprocess calls
parallelizes well without needing `Distributed`). Start Julia with
`julia -t auto` (or `-t N`) to actually use multiple threads for this.
Returns a Vector of the same NamedTuples as `run_bmm`, in input order.
Any case that errors is reported and skipped (returns `nothing` in its slot).
"""
function run_bmm_batch(cases::Vector{BMMCase}; exe::AbstractString, kwargs...)
    results = Vector{Any}(undef, length(cases))
    Threads.@threads for i in eachindex(cases)
        wd = mktempdir()
        try
            results[i] = run_bmm(cases[i]; exe=exe, workdir=wd, kwargs...)
        catch e
            @warn "BMM case $i failed" exception=e
            results[i] = nothing
        finally
            rm(wd; recursive=true, force=true)
        end
    end
    return results
end

# -----------------------------------------------------------------------
# Backend 2 (advanced): in-process ccall, SINGLE CALL PER PROCESS ONLY
# -----------------------------------------------------------------------

"""
    run_bmm_lib(case::BMMCase; libpath, workdir=mktempdir(), density=...)

ccall's into fortran/libbmm.so's `bmm_run_c(nmlfile::Cstring) -> Cint`.

*** Only call this ONCE per Julia process. *** See the module docstring and
fortran/bmm_capi.f90 for why: BMM's Fortran internals allocate module-level
arrays without freeing them, so a second call in the same process will error
out with a Fortran-level allocation failure that Julia cannot catch cleanly
(it may even crash the whole Julia process, since `error stop` inside the
Fortran aborts the process). If you need many in-process-fast calls, either
(a) patch bin_microphysics_module.f90 for reentrancy (add
`if (allocated(x)) deallocate(x)` before each `allocate`), or (b) just use
`run_bmm`/`run_bmm_batch` above, which sidestep the issue entirely by using
one OS process per call.
"""
function run_bmm_lib(case::BMMCase; libpath::AbstractString, workdir::AbstractString=mktempdir(),
                      density::Float64=case.density_core1[1])
    nmlpath = joinpath(workdir, "namelist.in")
    ncpath = joinpath(workdir, "output.nc")
    open(nmlpath, "w") do f
        write(f, to_namelist(case, ncpath))
    end

    rc = ccall((:bmm_run_c, libpath), Cint, (Cstring,), nmlpath)
    rc == 0 || error("bmm_run_c returned nonzero exit code $rc")

    ds = NCDataset(ncpath)
    t = Array(ds["time"]); z = Array(ds["z"]); p = Array(ds["p"]); T = Array(ds["t"])
    rh = Array(ds["rh"]); w = Array(ds["w"])
    ndrop_kg = Array(ds["ndrop"]); beta_ext = Array(ds["beta_ext"])
    nliq = Array(ds["nliq"]); nwat = Array(ds["nwat"]); mbinedges = Array(ds["mbinedges"])
    close(ds)

    ntimes = length(t)
    dcrit = [derive_dcrit(view(nliq, i, 1, :), view(nwat, i, 1, :), view(mbinedges, 1, :), density)
             for i in 1:ntimes]
    Rd = 287.05
    ndrop = ndrop_kg .* (p ./ (Rd .* T))
    S = rh .- 1.0
    return (t=t, z=z, p=p, T=T, S=S, w=w, ndrop=ndrop, beta_ext=beta_ext, dcrit=dcrit, case=case)
end

end # module BMM

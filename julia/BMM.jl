"""
Julia interface to the Bin Microphysics Module (BMM).

Each `AerosolMode` is mapped to one *external* BMM aerosol mode. This is
important: external modes may have distinct composition/hygroscopicity,
whereas `n_intern` describes lognormal submodes inside the same external mode.
For the surrogate dataset we use one lognormal submode per external mode.
"""
module BMM

using NCDatasets

export AerosolMode, BMMCase, cloud_base_case, effective_kappa,
       default_bmm_exe, run_bmm, run_bmm_batch, derive_dcrit,
       resample_profile, to_namelist, case_diagnostics

const MOLW_WATER = 18.01528e-3
const RHO_WATER = 1000.0

Base.@kwdef struct AerosolMode
    N::Float64 = 400.0e6       # m^-3
    Dm::Float64 = 80.0e-9      # dry number-median diameter, m
    lnsig::Float64 = 0.40      # ln(geometric standard deviation)

    # kappa-Koehler representation
    kappa::Float64 = 0.60

    # Classical ideal-solution Koehler representation. These are also useful
    # metadata for kappa cases, although density is not an independent CCN
    # control variable when Dm and kappa are specified.
    nu::Float64 = 3.0
    molw::Float64 = 132.14e-3  # kg mol^-1
    density::Float64 = 1770.0  # kg m^-3
end

"""Ideal-solution classical-Koehler parameters expressed as an equivalent kappa."""
effective_kappa(m::AerosolMode) = m.nu * (m.density / RHO_WATER) * (MOLW_WATER / m.molw)

Base.@kwdef struct BMMCase
    runtime::Float64 = 1800.0
    dt::Float64 = 2.0
    zinit::Float64 = 0.0
    tpert::Float64 = 0.0
    winit::Float64 = 0.6
    tinit::Float64 = 282.0
    pinit::Float64 = 90000.0
    rhinit::Float64 = 1.0
    radinit::Float64 = 500.0

    microphysics_flag::Int = 1
    vent_flag::Int = 1
    kappa_flag::Int = 1       # 1 = kappa-Koehler, 0 = classical ideal-solution Koehler
    updraft_type::Int = 1
    t_thresh::Float64 = 1.0e6
    adiabatic_prof::Bool = true
    vert_ent::Bool = false
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

    n_bins::Int = 60
    n_sv::Int = 0
    sv_flag::Int = 0
    dmina::Float64 = 1.0e-9
    dmaxa::Float64 = 10.0e-6
    modes::Vector{AerosolMode} = [AerosolMode()]
end

n_modes(c::BMMCase) = length(c.modes)

function _validate(m::AerosolMode)
    m.N >= 0 || throw(ArgumentError("mode N must be non-negative"))
    m.Dm > 0 || throw(ArgumentError("mode Dm must be positive"))
    m.lnsig > 0 || throw(ArgumentError("mode lnsig must be positive"))
    m.kappa >= 0 || throw(ArgumentError("mode kappa must be non-negative"))
    m.nu >= 0 || throw(ArgumentError("mode van't Hoff factor must be non-negative"))
    m.molw > 0 || throw(ArgumentError("mode molecular weight must be positive"))
    m.density > 0 || throw(ArgumentError("mode dry density must be positive"))
end

function _validate(c::BMMCase)
    # One component per external mode plus an identity mass-fraction matrix
    # lets each mode carry its own kappa or classical-Koehler chemistry.
    # This is more appropriate than putting multiple lognormals in n_intern.
    # Keep at least one mode because BMM requires n_mode >= 1.
    isempty(c.modes) && throw(ArgumentError("BMMCase requires at least one aerosol mode"))
    c.kappa_flag in (0, 1) || throw(ArgumentError("kappa_flag must be 0 or 1"))
    c.n_bins > 0 || throw(ArgumentError("n_bins must be positive"))
    c.dmina > 0 || throw(ArgumentError("dmina must be positive"))
    c.dmaxa > c.dmina || throw(ArgumentError("dmaxa must exceed dmina"))
    foreach(_validate, c.modes)
end

"""
    cloud_base_case(; winit, height_top=500, max_dt=5, max_dz=2, kwargs...)

Construct a saturated, constant-updraft BMM case that covers a fixed distance
above cloud base. `runtime = height_top / winit`. `max_dz` limits the vertical
distance travelled in one BMM step at strong updrafts.
"""
function cloud_base_case(; winit::Float64=0.6, height_top::Float64=500.0,
                          max_dt::Float64=5.0, max_dz::Float64=2.0,
                          zinit::Float64=0.0, rhinit::Float64=1.0, kwargs...)
    winit > 0 || throw(ArgumentError("winit must be positive"))
    height_top > 0 || throw(ArgumentError("height_top must be positive"))
    dt = min(max_dt, max_dz / winit)
    BMMCase(; runtime=height_top / winit, dt=dt, zinit=zinit,
            winit=winit, rhinit=rhinit, kwargs...)
end

default_bmm_exe() = normpath(joinpath(@__DIR__, "..", "bmm", "main.exe"))
_fbool(b::Bool) = b ? ".true." : ".false."

function to_namelist(c::BMMCase, outputfile::AbstractString)
    _validate(c)
    nm = n_modes(c)
    # One lognormal submode in each external BMM mode; one unique component per
    # external mode so chemistry/hygroscopicity can differ between modes.
    n_intern = 1
    n_comps = nm
    io = IOBuffer()

    println(io, " &run_vars")
    println(io, "    outputfile='", outputfile, "',")
    println(io, "    runtime=", c.runtime, ",")
    println(io, "    dt=", c.dt, ",")
    println(io, "    zinit=", c.zinit, ",")
    println(io, "    tpert=", c.tpert, ",")
    println(io, "    use_prof_for_tprh=.false.,")
    println(io, "    winit=", c.winit, ",")
    println(io, "    winit2=", c.winit, ",")
    println(io, "    amplitude2=100.,")
    println(io, "    tinit=", c.tinit, ",")
    println(io, "    pinit=", c.pinit, ",")
    println(io, "    rhinit=", c.rhinit, ",")
    println(io, "    radinit=", c.radinit, ",")
    println(io, "    bubble_flag=.false.,")
    println(io, "    microphysics_flag=", c.microphysics_flag, ",")
    # Liquid-only truth generation. Keep ice physics out of this surrogate
    # dataset even when cloud-base temperature is below 273.15 K.
    println(io, "    ice_flag=0,")
    # Fixed numerical configuration for all truth-generation cases.
    # These are intentionally not BMMCase inputs: a training dataset must not
    # silently mix different bin/advection or collision-coalescence schemes.
    println(io, "    bin_scheme_flag=0,")
    println(io, "    sce_flag=0,")
    println(io, "    vent_flag=", c.vent_flag, ",")
    println(io, "    kappa_flag=", c.kappa_flag, ",")
    println(io, "    updraft_type=", c.updraft_type, ",")
    println(io, "    t_thresh=", c.t_thresh, ",")
    println(io, "    adiabatic_prof=", _fbool(c.adiabatic_prof), ",")
    println(io, "    entrain_period=0,")
    println(io, "    thresh_to_start_hom_mix=-10.,")
    println(io, "    release_aerosol=.true.,")
    println(io, "    entrain_aerosol=.true.,")
    println(io, "    vert_ent=", _fbool(c.vert_ent), ",")
    println(io, "    z_ctop=", c.z_ctop, ",")
    println(io, "    ent_rate=", c.ent_rate, ",")
    println(io, "    n_levels_s=2,")
    println(io, "    alpha_therm=", c.alpha_therm, ",")
    println(io, "    alpha_cond=", c.alpha_cond, ",")
    println(io, "    alpha_therm_ice=", c.alpha_therm_ice, ",")
    println(io, "    alpha_dep=", c.alpha_dep, ",")
    println(io, "    hm_flag=", _fbool(c.hm_flag), ",")
    println(io, "    break_flag=", c.break_flag, ",")
    println(io, "    mode1_flag=", _fbool(c.mode1_flag), ",")
    println(io, "    mode2_flag=", _fbool(c.mode2_flag))
    println(io, "    /")

    println(io, "&aerosol_setup")
    println(io, "    n_mode=", nm, ",")
    println(io, "    n_intern=", n_intern, ",")
    println(io, "    n_sv=", c.n_sv, ",")
    println(io, "    sv_flag=", c.sv_flag, ",")
    println(io, "    n_bins=", c.n_bins, ",")
    println(io, "    n_comps=", n_comps, "/")

    # These profiles are required by the BMM namelist even when explicit
    # initial T/P/RH are supplied.
    println(io, "&sounding_spec")
    println(io, "    psurf=", c.pinit, ",")
    println(io, "    tsurf=", c.tinit, ",")
    println(io, "    q_read(1,1:2)=7.0e-3, 7.0e-3,")
    println(io, "    theta_read(1:2)=", c.tinit, ", ", c.tinit, ",")
    println(io, "    rh_read(1:2)=", c.rhinit, ", ", c.rhinit, ",")
    println(io, "    z_read(1:2)=0.0, 10000.0/")

    println(io, "&aerosol_spec")
    for (j, m) in enumerate(c.modes)
        println(io, "    n_aer1(1,", j, ")=", m.N, ",")
        println(io, "    d_aer1(1,", j, ")=", m.Dm, ",")
        println(io, "    sig_aer1(1,", j, ")=", m.lnsig, ",")
    end
    println(io, "    dmina=", c.dmina, ",")
    println(io, "    dmaxa=", c.dmaxa, ",")

    # Identity composition matrix: external mode j consists entirely of
    # component j. This permits distinct chemistry in every mode.
    for j in 1:nm, k in 1:nm
        println(io, "    mass_frac_aer1(", j, ",", k, ")=", j == k ? 1.0 : 0.0, ",")
    end
    println(io, "    molw_core1(1:", nm, ")=", join((m.molw for m in c.modes), ", "), ",")
    println(io, "    density_core1(1:", nm, ")=", join((m.density for m in c.modes), ", "), ",")
    println(io, "    nu_core1(1:", nm, ")=", join((m.nu for m in c.modes), ", "), ",")
    println(io, "    kappa_core1(1:", nm, ")=", join((m.kappa for m in c.modes), ", "), "/")

    String(take!(io))
end

function derive_dcrit(nliq_t::AbstractVector, nwat_t::AbstractVector,
                      mbinedges::AbstractVector, density::Float64)
    frac = similar(nliq_t, Float64)
    @inbounds for i in eachindex(nliq_t)
        frac[i] = nwat_t[i] > 0 ? clamp(nliq_t[i] / nwat_t[i], 0.0, 1.0) : 0.0
    end
    idx = findfirst(>=(0.5), frac)
    (idx === nothing || idx == 1) && return NaN
    m_edge = mbinedges[idx]
    m_edge > 0 || return NaN
    (6.0 * m_edge / (pi * density))^(1 / 3)
end

# Match BMM's output conversion from #/kg dry air to #/m3.
_svp_liq(T) = 100.0 * 6.1121 * exp((18.678 - (T - 273.15) / 234.5) *
                                    (T - 273.15) / (257.14 + (T - 273.15)))
function _dry_air_density(p, T, rh)
    Ra = 8.314 / 29e-3
    Rv = 8.314 / 18e-3
    eps = Ra / Rv
    es = _svp_liq(T)
    qv = eps * rh * es / (p - es)
    p / ((Ra + qv * Rv) * T)
end

function _read_output(ncpath::AbstractString, c::BMMCase)
    vals = NCDataset(ncpath) do ds
        (t=Array(ds["time"]), z=Array(ds["z"]), p=Array(ds["p"]),
         T=Array(ds["t"]), rh=Array(ds["rh"]), w=Array(ds["w"]),
         ndrop_kg=Array(ds["ndrop"]), beta_ext=Array(ds["beta_ext"]),
         ql=Array(ds["ql"]), deff=Array(ds["deff"]),
         nliq=Array(ds["nliq"]), nwat=Array(ds["nwat"]),
         mbinedges=Array(ds["mbinedges"]))
    end

    nt = length(vals.t)
    nm = n_modes(c)
    dcrit = fill(NaN, nt, nm)
    for it in 1:nt, im in 1:nm
        dcrit[it, im] = derive_dcrit(view(vals.nliq, :, im, it),
                                      view(vals.nwat, :, im, it),
                                      view(vals.mbinedges, :, im),
                                      c.modes[im].density)
    end

    rhod = [_dry_air_density(vals.p[i], vals.T[i], vals.rh[i]) for i in 1:nt]
    (t=vals.t,
     height=vals.z .- vals.z[1],
     z=vals.z,
     p=vals.p,
     T=vals.T,
     S=vals.rh .- 1.0,
     rh=vals.rh,
     w=vals.w,
     ndrop=vals.ndrop_kg .* rhod,
     ndrop_kg=vals.ndrop_kg,
     beta_ext=vals.beta_ext,
     ql=vals.ql,
     deff=vals.deff,
     dcrit=dcrit,
     case=c)
end

"""Return a compact, human-readable description of one BMM input case."""
function case_diagnostics(c::BMMCase; case_index=nothing)
    io = IOBuffer()
    case_index === nothing || println(io, "case: ", case_index)
    println(io, "  liquid-only: ice_flag=0; bin_scheme_flag=0; sce_flag=0")
    println(io, "  T_cloud_base = ", c.tinit, " K")
    println(io, "  P_cloud_base = ", c.pinit, " Pa (", c.pinit / 100.0, " hPa)")
    println(io, "  w = ", c.winit, " m s^-1")
    println(io, "  runtime = ", c.runtime, " s")
    println(io, "  outer dt = ", c.dt, " s")
    println(io, "  n_modes = ", length(c.modes), "; n_bins = ", c.n_bins)
    println(io, "  kappa_flag = ", c.kappa_flag)
    for (j, m) in enumerate(c.modes)
        println(io, "  mode ", j, ":")
        println(io, "    N       = ", m.N, " m^-3  (", m.N / 1e6, " cm^-3)")
        println(io, "    Dm      = ", m.Dm, " m  (", m.Dm * 1e9, " nm)")
        println(io, "    lnsig   = ", m.lnsig)
        println(io, "    kappa   = ", m.kappa)
        println(io, "    density = ", m.density, " kg m^-3")
        println(io, "    nu      = ", m.nu)
        println(io, "    molw    = ", m.molw, " kg mol^-1")
        println(io, "    kappa_eff(classical) = ", effective_kappa(m))
    end
    String(take!(io))
end

function _copy_failure_files(src::AbstractString, dst::AbstractString, c::BMMCase,
                             i::Integer, err)
    mkpath(dst)
    for name in ("namelist.in", "stdout.log", "stderr.log", "output.nc")
        f = joinpath(src, name)
        isfile(f) && cp(f, joinpath(dst, name); force=true)
    end
    open(joinpath(dst, "case_summary.txt"), "w") do io
        write(io, case_diagnostics(c; case_index=i))
        println(io, "\nJulia exception:")
        showerror(io, err)
        println(io)
    end
end

function run_bmm(c::BMMCase; exe::AbstractString=default_bmm_exe(),
                 workdir::AbstractString=mktempdir())
    isfile(exe) || error("BMM executable not found at $exe; run `make bmm`")
    mkpath(workdir)
    nmlpath = joinpath(workdir, "namelist.in")
    ncpath = joinpath(workdir, "output.nc")
    outlog = joinpath(workdir, "stdout.log")
    errlog = joinpath(workdir, "stderr.log")
    open(nmlpath, "w") do io
        write(io, to_namelist(c, ncpath))
    end
    proc = run(pipeline(Cmd([exe, nmlpath]); stdout=outlog, stderr=errlog); wait=false)
    wait(proc)
    if proc.exitcode != 0
        outtxt = isfile(outlog) ? read(outlog, String) : ""
        errtxt = isfile(errlog) ? read(errlog, String) : ""
        error("BMM failed (exit $(proc.exitcode))\nSTDOUT:\n$outtxt\nSTDERR:\n$errtxt")
    end
    _read_output(ncpath, c)
end

function run_bmm_batch(cases::Vector{BMMCase}; exe::AbstractString=default_bmm_exe(),
                       failure_dir::Union{Nothing,AbstractString}=nothing, kwargs...)
    results = Vector{Any}(undef, length(cases))
    Threads.@threads for i in eachindex(cases)
        wd = mktempdir()
        try
            results[i] = run_bmm(cases[i]; exe=exe, workdir=wd, kwargs...)
        catch err
            saved = nothing
            if failure_dir !== nothing
                saved = joinpath(abspath(failure_dir), "case_" * lpad(string(i), 4, '0'))
                try
                    _copy_failure_files(wd, saved, cases[i], i, err)
                catch save_err
                    @warn "Could not save diagnostics for failed BMM case $i" exception=(save_err, catch_backtrace())
                end
            end
            msg = "BMM case $i failed\n" * case_diagnostics(cases[i]; case_index=i)
            saved === nothing || (msg *= "  retained failure files: $saved\n")
            @warn msg exception=(err, catch_backtrace())
            results[i] = nothing
        finally
            rm(wd; recursive=true, force=true)
        end
    end
    results
end

function _linear_interp(x::Real, xs::AbstractVector, ys::AbstractVector)
    x <= xs[1] && return ys[1]
    x >= xs[end] && return ys[end]
    i = searchsortedlast(xs, x)
    f = (x - xs[i]) / (xs[i + 1] - xs[i])
    ys[i] + f * (ys[i + 1] - ys[i])
end

function _interp_dcrit(x, xs, ys)
    finite = findall(isfinite, ys)
    isempty(finite) && return NaN
    i0, i1 = first(finite), last(finite)
    x < xs[i0] && return NaN
    x >= xs[i1] && return ys[i1]
    _linear_interp(x, view(xs, i0:i1), view(ys, i0:i1))
end

function resample_profile(r; dz::Float64=5.0, height_top::Float64=r.height[end])
    # Constant-updraft runs should reach height_top exactly, but tolerate small
    # integration/output roundoff and clamp the final interpolation to the last
    # BMM point. A genuinely short trajectory remains visibly short.
    hmax = r.height[end] >= height_top - 0.25 * dz ? height_top : r.height[end]
    h = collect(0.0:dz:hmax)
    (isempty(h) || h[end] < hmax - 1e-8) && push!(h, hmax)
    interp(v) = [_linear_interp(x, r.height, v) for x in h]

    nm = size(r.dcrit, 2)
    dcrit = fill(NaN, length(h), nm)
    for im in 1:nm
        dcrit[:, im] .= [_interp_dcrit(x, r.height, view(r.dcrit, :, im)) for x in h]
    end

    (height=h,
     w=interp(r.w),
     S=interp(r.S),
     ndrop=interp(r.ndrop),
     beta_ext=interp(r.beta_ext),
     ql=interp(r.ql),
     deff=interp(r.deff),
     dcrit=dcrit,
     case=r.case)
end

end # module BMM

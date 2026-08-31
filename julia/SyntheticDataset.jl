"""
Synthetic design-of-experiments generator for BMM cloud-activation training.

The generator deliberately samples lognormal aerosol modes over broad,
climate-model-like ranges. The *class labels* below are only used to build a
physically sensible sampling distribution; they are not intended as ML inputs.
"""
module SyntheticDataset

using Random
using NCDatasets
using ..BMM

export SamplingConfig, sample_cases, summarise_cases, write_dataset,
       run_and_write_dataset

Base.@kwdef struct SamplingConfig
    n_cases::Int = 256
    max_modes::Int = 4
    seed::Int = 20260831
    height_top::Float64 = 500.0
    dz_output::Float64 = 5.0
    max_dt::Float64 = 5.0
    max_dz_step::Float64 = 2.0
    n_bins::Int = 60

    # Main pilot range. We deliberately start at 0.1 m/s because very weak
    # updrafts make fixed-height parcel trajectories expensive; they can be
    # added later as a targeted low-w tranche.
    w_min::Float64 = 0.1
    w_max::Float64 = 10.0
    p_min::Float64 = 60_000.0
    p_max::Float64 = 102_000.0
    t_box_min::Float64 = 250.0
    t_box_max::Float64 = 303.0

    # kappa-Koehler synthetic ensemble for v1.
    kappa_flag::Int = 1
    dmina::Float64 = 1.0e-9
    dmaxa::Float64 = 10.0e-6
end

const CLASS_NAMES = ("nucleation", "aitken", "accumulation", "coarse")
# Probabilities only control synthetic coverage. Dm itself is passed to the ML.
const CLASS_CDF = (0.12, 0.42, 0.88, 1.00)

_loguniform(rng, lo, hi) = exp(log(lo) + rand(rng) * (log(hi) - log(lo)))
_uniform(rng, lo, hi) = lo + rand(rng) * (hi - lo)

function _choose_class(rng)
    u = rand(rng)
    findfirst(x -> u <= x, CLASS_CDF)::Int
end

"""Return `(mode, class_id)` for one synthetic aerosol mode."""
function _sample_mode(rng)
    cls = _choose_class(rng)

    # N ranges are in cm^-3 here for readability and converted to m^-3 below.
    # Overlap in Dm between adjacent classes is intentional: real modal
    # decompositions overlap and the network should not rely on class identity.
    if cls == 1                 # nucleation / ultrafine
        Dm = _loguniform(rng, 3e-9, 20e-9)
        Ncm = _loguniform(rng, 10.0, 2.0e4)
        lnsig = _uniform(rng, 0.20, 0.65)
    elseif cls == 2             # Aitken
        Dm = _loguniform(rng, 8e-9, 80e-9)
        Ncm = _loguniform(rng, 5.0, 1.0e4)
        lnsig = _uniform(rng, 0.25, 0.70)
    elseif cls == 3             # accumulation
        Dm = _loguniform(rng, 50e-9, 500e-9)
        Ncm = _loguniform(rng, 0.1, 5.0e3)
        lnsig = _uniform(rng, 0.25, 0.80)
    else                        # coarse / giant CCN tail
        Dm = _loguniform(rng, 0.5e-6, 4.0e-6)
        Ncm = _loguniform(rng, 1.0e-3, 30.0)
        lnsig = _uniform(rng, 0.30, 0.80)
    end

    # Stratified kappa mixture: broad ambient range plus explicit coverage of
    # nearly hydrophobic particles and highly hygroscopic salts.
    u = rand(rng)
    kappa = if u < 0.10
        _uniform(rng, 0.0, 0.05)
    elseif u < 0.65
        _uniform(rng, 0.05, 0.60)
    elseif u < 0.90
        _uniform(rng, 0.60, 1.00)
    else
        _uniform(rng, 1.00, 1.40)
    end

    # Density cancels out of ideal kappa-Koehler activation for a supplied dry
    # diameter; keep a benign fixed value for BMM's dry mass bookkeeping.
    mode = AerosolMode(N=Ncm * 1e6, Dm=Dm, lnsig=lnsig, kappa=kappa,
                       nu=3.0, molw=132.14e-3, density=1770.0)
    mode, cls
end

function _sample_cloud_base(rng, cfg::SamplingConfig)
    p = _uniform(rng, cfg.p_min, cfg.p_max)

    # 90% of cases follow a loose lower-tropospheric T-P correlation. This
    # avoids wasting most samples in implausible corners of a rectangular T-P
    # box while still leaving 10% broad-box stress cases for coverage.
    if rand(rng) < 0.90
        z_approx = -8000.0 * log(p / 101325.0)
        t_mean = 288.15 - 6.5e-3 * z_approx
        T = clamp(t_mean + _uniform(rng, -9.0, 9.0),
                  cfg.t_box_min, cfg.t_box_max)
    else
        T = _uniform(rng, cfg.t_box_min, cfg.t_box_max)
    end
    T, p
end

"""
    sample_cases(cfg) -> (cases, class_ids)

Generate a deterministic, balanced-by-mode-count synthetic ensemble. Mode count
cycles through 1:max_modes then the cases are shuffled, so a finite pilot does
not accidentally contain mostly one-mode cases.
"""
function sample_cases(cfg::SamplingConfig)
    cfg.n_cases > 0 || throw(ArgumentError("n_cases must be positive"))
    cfg.max_modes > 0 || throw(ArgumentError("max_modes must be positive"))
    rng = MersenneTwister(cfg.seed)
    cases = BMMCase[]
    classes = Vector{Vector{Int}}()

    counts = [1 + mod(i - 1, cfg.max_modes) for i in 1:cfg.n_cases]
    shuffle!(rng, counts)

    for nm in counts
        modes = AerosolMode[]
        cls = Int[]
        for _ in 1:nm
            m, c = _sample_mode(rng)
            push!(modes, m)
            push!(cls, c)
        end

        # Sorting is only for deterministic storage/readability. A downstream
        # set encoder should still be permutation invariant.
        order = sortperm(getfield.(modes, :Dm))
        modes = modes[order]
        cls = cls[order]

        T0, P0 = _sample_cloud_base(rng, cfg)
        w = _loguniform(rng, cfg.w_min, cfg.w_max)
        push!(cases, cloud_base_case(
            winit=w,
            height_top=cfg.height_top,
            max_dt=cfg.max_dt,
            max_dz=cfg.max_dz_step,
            tinit=T0,
            pinit=P0,
            kappa_flag=cfg.kappa_flag,
            n_bins=cfg.n_bins,
            dmina=cfg.dmina,
            dmaxa=cfg.dmaxa,
            modes=modes))
        push!(classes, cls)
    end

    cases, classes
end

function summarise_cases(cases::Vector{BMMCase})
    nm = length.(getfield.(cases, :modes))
    ws = getfield.(cases, :winit)
    Ts = getfield.(cases, :tinit)
    Ps = getfield.(cases, :pinit)
    allmodes = reduce(vcat, getfield.(cases, :modes))
    Ns = getfield.(allmodes, :N) ./ 1e6
    Ds = getfield.(allmodes, :Dm) .* 1e9
    sig = getfield.(allmodes, :lnsig)
    kap = getfield.(allmodes, :kappa)
    println("Synthetic BMM ensemble")
    println("  cases: $(length(cases)); modes/case: $(minimum(nm))-$(maximum(nm))")
    println("  w: $(minimum(ws))-$(maximum(ws)) m s^-1")
    println("  cloud-base T: $(minimum(Ts))-$(maximum(Ts)) K")
    println("  cloud-base P: $(minimum(Ps)/100)-$(maximum(Ps)/100) hPa")
    println("  mode N: $(minimum(Ns))-$(maximum(Ns)) cm^-3")
    println("  mode Dm: $(minimum(Ds))-$(maximum(Ds)) nm")
    println("  mode lnsig: $(minimum(sig))-$(maximum(sig))")
    println("  mode kappa: $(minimum(kap))-$(maximum(kap))")
end

function _nearest_profile(q, hgrid, field::Symbol)
    vals = getfield(q, field)
    # q has already been resampled with the same requested dz/height_top. This
    # guard provides a clear error rather than silently writing shifted data.
    length(q.height) == length(hgrid) ||
        error("resampled height count $(length(q.height)) != expected $(length(hgrid))")
    maximum(abs.(q.height .- hgrid)) < 1e-6 ||
        error("resampled BMM height grid does not match dataset height grid")
    vals
end

"""Write cases and resampled BMM results to one ML-friendly NetCDF file."""
function write_dataset(path::AbstractString, cases::Vector{BMMCase}, classes,
                       raw_results; cfg::SamplingConfig)
    ncase = length(cases)
    length(raw_results) == ncase || throw(ArgumentError("result count mismatch"))
    maxm = cfg.max_modes
    hgrid = collect(0.0:cfg.dz_output:cfg.height_top)
    nh = length(hgrid)

    success = zeros(Int8, ncase)
    nmode = Int32[length(c.modes) for c in cases]
    T0 = [c.tinit for c in cases]
    P0 = [c.pinit for c in cases]
    w0 = [c.winit for c in cases]

    mode_N = fill(NaN, ncase, maxm)
    mode_Dm = fill(NaN, ncase, maxm)
    mode_lnsig = fill(NaN, ncase, maxm)
    mode_kappa = fill(NaN, ncase, maxm)
    mode_nu = fill(NaN, ncase, maxm)
    mode_molw = fill(NaN, ncase, maxm)
    mode_density = fill(NaN, ncase, maxm)
    mode_class = fill(Int16(0), ncase, maxm)

    S = fill(NaN, ncase, nh)
    Nd = fill(NaN, ncase, nh)
    beta = fill(NaN, ncase, nh)
    ql = fill(NaN, ncase, nh)
    deff = fill(NaN, ncase, nh)
    dcrit = fill(NaN, ncase, nh, maxm)

    for i in 1:ncase
        c = cases[i]
        for (j, m) in enumerate(c.modes)
            mode_N[i, j] = m.N
            mode_Dm[i, j] = m.Dm
            mode_lnsig[i, j] = m.lnsig
            mode_kappa[i, j] = m.kappa
            mode_nu[i, j] = m.nu
            mode_molw[i, j] = m.molw
            mode_density[i, j] = m.density
            mode_class[i, j] = classes[i][j]
        end

        r = raw_results[i]
        r === nothing && continue
        q = BMM.resample_profile(r; dz=cfg.dz_output, height_top=cfg.height_top)
        length(q.height) == nh || begin
            @warn "case $i ended before requested height; marking failed" final_height=q.height[end]
            continue
        end
        success[i] = 1
        S[i, :] .= _nearest_profile(q, hgrid, :S)
        Nd[i, :] .= _nearest_profile(q, hgrid, :ndrop)
        beta[i, :] .= _nearest_profile(q, hgrid, :beta_ext)
        ql[i, :] .= _nearest_profile(q, hgrid, :ql)
        deff[i, :] .= _nearest_profile(q, hgrid, :deff)
        dcrit[i, :, 1:length(c.modes)] .= q.dcrit
    end

    mkpath(dirname(abspath(path)))
    NCDataset(path, "c") do ds
        defDim(ds, "case", ncase)
        defDim(ds, "height", nh)
        defDim(ds, "mode", maxm)

        ds.attrib["title"] = "Synthetic BMM cloud-activation training ensemble"
        ds.attrib["seed"] = cfg.seed
        ds.attrib["height_top_m"] = cfg.height_top
        ds.attrib["dz_output_m"] = cfg.dz_output
        ds.attrib["kappa_flag"] = cfg.kappa_flag
        ds.attrib["mode_class_note"] = "class is sampling metadata only; do not use as a surrogate input"
        ds.attrib["mode_class_names"] = join(CLASS_NAMES, ",")

        defVar(ds, "height", Float64, ("height",))[:] = hgrid
        defVar(ds, "success", Int8, ("case",))[:] = success
        defVar(ds, "n_modes", Int32, ("case",))[:] = nmode
        defVar(ds, "T_cloud_base", Float64, ("case",))[:] = T0
        defVar(ds, "P_cloud_base", Float64, ("case",))[:] = P0
        defVar(ds, "w", Float64, ("case",))[:] = w0

        defVar(ds, "mode_N", Float64, ("case", "mode"))[:, :] = mode_N
        defVar(ds, "mode_Dm", Float64, ("case", "mode"))[:, :] = mode_Dm
        defVar(ds, "mode_lnsig", Float64, ("case", "mode"))[:, :] = mode_lnsig
        defVar(ds, "mode_kappa", Float64, ("case", "mode"))[:, :] = mode_kappa
        defVar(ds, "mode_nu", Float64, ("case", "mode"))[:, :] = mode_nu
        defVar(ds, "mode_molw", Float64, ("case", "mode"))[:, :] = mode_molw
        defVar(ds, "mode_density", Float64, ("case", "mode"))[:, :] = mode_density
        defVar(ds, "mode_class", Int16, ("case", "mode"))[:, :] = mode_class

        defVar(ds, "S", Float64, ("case", "height"))[:, :] = S
        defVar(ds, "Nd", Float64, ("case", "height"))[:, :] = Nd
        defVar(ds, "beta_ext", Float64, ("case", "height"))[:, :] = beta
        defVar(ds, "ql", Float64, ("case", "height"))[:, :] = ql
        defVar(ds, "deff", Float64, ("case", "height"))[:, :] = deff
        defVar(ds, "Dcrit_mode", Float64, ("case", "height", "mode"))[:, :, :] = dcrit

        ds["height"].attrib["units"] = "m above cloud base"
        ds["T_cloud_base"].attrib["units"] = "K"
        ds["P_cloud_base"].attrib["units"] = "Pa"
        ds["w"].attrib["units"] = "m s-1"
        ds["mode_N"].attrib["units"] = "m-3"
        ds["mode_Dm"].attrib["units"] = "m"
        ds["mode_lnsig"].attrib["units"] = "1"
        ds["mode_kappa"].attrib["units"] = "1"
        ds["mode_molw"].attrib["units"] = "kg mol-1"
        ds["mode_density"].attrib["units"] = "kg m-3"
        ds["S"].attrib["units"] = "1"
        ds["Nd"].attrib["units"] = "m-3"
        ds["beta_ext"].attrib["units"] = "m-1"
        ds["ql"].attrib["units"] = "kg kg-1"
        ds["deff"].attrib["units"] = "m"
        ds["Dcrit_mode"].attrib["units"] = "m"
    end

    println("wrote $path ($(sum(success))/$(length(success)) successful cases)")
    path
end

function run_and_write_dataset(path::AbstractString; cfg::SamplingConfig=SamplingConfig(),
                               exe::AbstractString=BMM.default_bmm_exe())
    cases, classes = sample_cases(cfg)
    summarise_cases(cases)
    println("Running BMM using $(Threads.nthreads()) Julia worker threads...")
    results = BMM.run_bmm_batch(cases; exe=exe)
    write_dataset(path, cases, classes, results; cfg=cfg)
end

end # module SyntheticDataset

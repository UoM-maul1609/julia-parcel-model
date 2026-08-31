"""
Variable-mode cloud-activation surrogates.

The aerosol is treated as a weighted set of modes. Version 3 predicts liquid
water mixing ratio, bounded activated fraction, and extinction. Supersaturation
is diagnosed from total-water conservation using the supplied parcel T(z), P(z)
thermodynamic trajectory:

    qtot = ql + (1 + S) qvs(T,P)

so S is physically consistent with ql, T and P by construction.
"""
module SetSurrogate

using Flux
using DifferentialEquations
using SciMLBase
using SciMLSensitivity
using Zygote
using Random
using Statistics
using BSON
using ..SurrogateData: MLCase, qvs_liq, initial_total_water

export ProfileSurrogate, NeuralODESurrogate, build_profile_model, build_neuralode_model,
       aerosol_context, profile_loss, neuralode_loss, train_profile!, train_neuralode!,
       predict_profile, predict_neuralode, transformed_target, physical_target,
       activation_fraction, fraction_latent, latent_fraction, critical_supersaturation,
       evaluate_profile, evaluate_neuralode, save_profile_model, save_neuralode_model,
       load_profile_model, load_neuralode_model, BETA_SCALE

const MODEL_VERSION = 3

# Target/input scales chosen to keep neural-network variables O(1).
const QL_SCALE = 1.0f-4               # kg kg^-1
const QT_SCALE = 1.0f-2               # kg kg^-1
const BETA_SCALE = 1.0f-3             # m^-1
const N_SCALE = 1.0f8                 # kg^-1 dry air
const DM_SCALE = 1.0f-7               # m
const M2_SCALE = 1.0f-5               # sum(N D^2), m^2 kg^-1
const M3_SCALE = 1.0f-12              # sum(N D^3), m^3 kg^-1
const W_FLOOR = 1.0f-3

# Activated-fraction transform. The small epsilon keeps exact 0/1 targets
# finite in latent space; the inverse is explicitly clamped to [0,1].
const FACT_EPS = 1.0f-4
const FACT_LOGIT_SCALE = 4.0f0

# Approximate kappa-Koehler critical supersaturation feature.
const SIGMA_WATER = 0.072f0
const MOLW_WATER = 18.01528f-3
const R_GAS = 8.314f0
const RHO_WATER = 1000.0f0

_logit(p) = log(p / (1f0 - p))

function fraction_latent(f::Real)
    fc = clamp(Float32(f), 0f0, 1f0)
    q = FACT_EPS + (1f0 - 2f0 * FACT_EPS) * fc
    Float32(_logit(q) / FACT_LOGIT_SCALE)
end

function latent_fraction(y::Real)
    x = clamp(Float32(y) * FACT_LOGIT_SCALE, -30f0, 30f0)
    q = 1f0 / (1f0 + exp(-x))
    clamp((q - FACT_EPS) / (1f0 - 2f0 * FACT_EPS), 0f0, 1f0)
end

const FACT_ZERO_LATENT = fraction_latent(0f0)

ql_latent(q::Real) = Float32(log1p(max(Float32(q), 0f0) / QL_SCALE))
latent_ql(y::Real) = max(0f0, Float32(expm1(Float32(y)) * QL_SCALE))

"""Approximate critical supersaturation (dimensionless) at a mode median dry diameter."""
function critical_supersaturation(Dm::Real, kappa::Real, T::Real)
    d = max(Float32(Dm), 1f-12)
    kap = max(Float32(kappa), 1f-6)
    temp = max(Float32(T), 180f0)
    A = 4f0 * SIGMA_WATER * MOLW_WATER / (R_GAS * temp * RHO_WATER)
    sc = sqrt(4f0 * A^3 / (27f0 * kap * d^3))
    clamp(sc, 1f-8, 1f0)
end

"""BMM activated fraction at one height, using native # kg^-1 units."""
function activation_fraction(c::MLCase, i::Int)
    ntot = sum(c.mode_N)
    ntot > 0 || return 0f0
    clamp(c.Nd_kg[i] / ntot, 0f0, 1f0)
end

# v3 target = [ql latent, activation-fraction latent, extinction latent].
transformed_target(ql::Real, fact::Real, beta::Real) = Float32[
    ql_latent(ql),
    fraction_latent(fact),
    log1p(max(Float32(beta), 0f0) / BETA_SCALE)
]
transformed_target(c::MLCase, i::Int) =
    transformed_target(c.ql[i], activation_fraction(c, i), c.beta_ext[i])

"""Convert one v3 latent output to physical quantities.

Total water is enforced exactly. ql is bounded to [0,qtot], which also ensures
non-negative vapour mixing ratio and S >= -1.
"""
function physical_target(y, ntot_kg::Real, rhod::Real,
                         qtot::Real, T::Real, P::Real)
    fact = latent_fraction(y[2])
    qt = max(Float64(qtot), 0.0)
    ql = clamp(Float64(latent_ql(y[1])), 0.0, qt)
    qvs = Float64(qvs_liq(T, P))
    qvs > 0 || throw(ArgumentError("qvs must be positive"))
    S = (qt - ql) / qvs - 1.0
    ndkg = fact * max(Float64(ntot_kg), 0.0)
    (
        S = S,
        ql = ql,
        activation_fraction = Float64(fact),
        Nd_kg = ndkg,
        Nd = ndkg * Float64(rhod),
        beta_ext = max(0.0, Float64(expm1(y[3]) * BETA_SCALE)),
    )
end

# The encoder excludes N_i. Number enters through pooling weights and N_total,
# preserving permutation invariance and exact invariance to splitting an
# identical physical mode into duplicate submodes.
function _mode_features(c::MLCase)
    nmode = length(c.mode_N)
    dm = Float32[log(max(c.mode_Dm[j], 1f-12) / DM_SCALE) for j in 1:nmode]
    sig = Float32[(c.mode_lnsig[j] - 0.5f0) / 0.25f0 for j in 1:nmode]
    kap = Float32[log1p(max(c.mode_kappa[j], 0f0)) / log1p(1.4f0) for j in 1:nmode]
    scrit = Float32[
        log10(critical_supersaturation(c.mode_Dm[j], c.mode_kappa[j], c.T0) / 1f-2) / 2f0
        for j in 1:nmode
    ]
    vcat(reshape(dm, 1, :), reshape(sig, 1, :),
         reshape(kap, 1, :), reshape(scrit, 1, :))
end

function _mode_weights(c::MLCase)
    ntot = sum(c.mode_N)
    ntot > 0 || throw(ArgumentError("aerosol number must be positive"))
    Float32.(c.mode_N ./ ntot)
end

function _bulk_features(c::MLCase)
    ntot = sum(c.mode_N)
    ntot > 0 || throw(ArgumentError("aerosol number must be positive"))

    m2 = sum(c.mode_N[j] * c.mode_Dm[j]^2 * exp(2f0 * c.mode_lnsig[j]^2)
             for j in eachindex(c.mode_N))
    m3 = sum(c.mode_N[j] * c.mode_Dm[j]^3 * exp(4.5f0 * c.mode_lnsig[j]^2)
             for j in eachindex(c.mode_N))
    qt0 = initial_total_water(c)

    Float32[
        log(ntot / N_SCALE),
        (c.T0 - 280f0) / 15f0,
        (c.P0 - 85_000f0) / 20_000f0,
        qt0 / QT_SCALE,
        log(max(m2, 1f-20) / M2_SCALE),
        log(max(m3, 1f-30) / M3_SCALE),
    ]
end

Zygote.@nograd _mode_features
Zygote.@nograd _mode_weights
Zygote.@nograd _bulk_features

function aerosol_context(encoder, c::MLCase)
    embeddings = encoder(_mode_features(c))
    pooled = embeddings * _mode_weights(c)
    vcat(pooled, _bulk_features(c))
end

forcing_w(c::MLCase, hhat) = Float32(log(max(c.w, W_FLOOR)))

function height_features(hhat::Real)
    h = max(Float32(hhat), 0f0)
    Float32[h, sqrt(h), log1p(20f0 * h) / log(21f0)]
end

thermo_features(T::Real, P::Real) = Float32[
    (Float32(T) - 280f0) / 15f0,
    (Float32(P) - 85_000f0) / 20_000f0,
]
thermo_features(c::MLCase, i::Int) = thermo_features(c.T[i], c.P[i])

function _linear_interp(x::Real, xs::AbstractVector, ys::AbstractVector)
    x <= xs[1] && return ys[1]
    x >= xs[end] && return ys[end]
    i = searchsortedlast(xs, x)
    f = (x - xs[i]) / (xs[i+1] - xs[i])
    ys[i] + f * (ys[i+1] - ys[i])
end

struct ProfileCore{E,H}
    encoder::E
    head::H
end
Flux.@layer ProfileCore

struct ODECore{E,I,R}
    encoder::E
    init_net::I
    rhs_net::R
end
Flux.@layer ODECore

struct ProfileSurrogate
    re
    embed_dim::Int
    hidden::Int
    hscale::Float32
end

struct NeuralODESurrogate
    re
    embed_dim::Int
    hidden::Int
    hscale::Float32
end

function build_profile_model(; embed_dim::Int=24, hidden::Int=96, hscale::Real=500.0)
    encoder = Chain(Dense(4, hidden, tanh), Dense(hidden, embed_dim, tanh))
    nctx = embed_dim + 6
    # context + w + local(T,P) + three height features
    head = Chain(Dense(nctx + 1 + 2 + 3, hidden, tanh),
                 Dense(hidden, hidden, tanh), Dense(hidden, 3))
    core = ProfileCore(encoder, head)
    p, re = Flux.destructure(core)
    p, ProfileSurrogate(re, embed_dim, hidden, Float32(hscale))
end

function build_neuralode_model(; embed_dim::Int=24, hidden::Int=96, hscale::Real=500.0)
    encoder = Chain(Dense(4, hidden, tanh), Dense(hidden, embed_dim, tanh))
    nctx = embed_dim + 6
    # Initial ql and beta are learned; activated fraction is exactly zero.
    init_net = Chain(Dense(nctx + 1 + 2, hidden, tanh), Dense(hidden, 2))
    rhs_net = Chain(Dense(3 + nctx + 1 + 2 + 3, hidden, tanh),
                    Dense(hidden, hidden, tanh), Dense(hidden, 3))
    core = ODECore(encoder, init_net, rhs_net)
    p, re = Flux.destructure(core)
    p, NeuralODESurrogate(re, embed_dim, hidden, Float32(hscale))
end

function _profile_point(core, ctx, c::MLCase, i::Int, h::Real, hscale::Real)
    hhat = Float32(h / hscale)
    raw = core.head(vcat(ctx, forcing_w(c, hhat), thermo_features(c, i),
                         height_features(hhat)))
    if abs(hhat) <= 1f-7
        # Initial hydrated aerosol liquid and extinction are learned. Activation
        # is exactly zero at the initial RH=0.95 parcel state.
        return Float32[raw[1], FACT_ZERO_LATENT, raw[3]]
    end
    raw
end

function _profile_transformed(m::ProfileSurrogate, p, c::MLCase)
    core = m.re(p)
    ctx = aerosol_context(core.encoder, c)
    reduce(hcat, (_profile_point(core, ctx, c, i, c.height[i], m.hscale)
                  for i in eachindex(c.height)))
end

function _truth_matrix(c::MLCase)
    reduce(hcat, (transformed_target(c, i) for i in eachindex(c.height)))
end

function _trajectory_loss(pred, truth)
    err = pred .- truth
    ql_loss   = mean(abs2, view(err, 1, :))
    fact_loss = mean(abs2, view(err, 2, :))
    beta_loss = mean(abs2, view(err, 3, :))
    point = ql_loss + 2.0f0 * fact_loss + beta_loss
    peak_ql = (maximum(view(pred,1,:)) - maximum(view(truth,1,:)))^2
    peak_fact = (maximum(view(pred,2,:)) - maximum(view(truth,2,:)))^2
    peak_beta = (maximum(view(pred,3,:)) - maximum(view(truth,3,:)))^2
    peak = (peak_ql + 2.0f0*peak_fact + peak_beta) / 4f0
    final = mean(abs2,view(pred,:,size(pred,2)) .- view(truth,:,size(truth,2)))
    point + 0.15f0*peak + 0.05f0*final
end

profile_loss(m::ProfileSurrogate, p, c::MLCase) =
    _trajectory_loss(_profile_transformed(m, p, c), _truth_matrix(c))

"""Solve the v3 Neural ODE in normalized height."""
function _solve_neuralode(m::NeuralODESurrogate, p, c::MLCase;
                          solver=Tsit5(), saveat=c.height)
    core0 = m.re(p)
    ctx0 = aerosol_context(core0.encoder, c)
    w0 = forcing_w(c, 0f0)
    init = core0.init_net(vcat(ctx0, w0, thermo_features(c, 1)))
    y0 = Float32[init[1], FACT_ZERO_LATENT, init[2]]
    u0 = vcat(y0, ctx0)
    nctx = length(ctx0)

    function rhs(u, p, hhat)
        core = m.re(p)
        y = view(u, 1:3)
        ctx = view(u, 4:3+nctx)
        h = Float32(hhat) * m.hscale
        Tloc = _linear_interp(h, c.height, c.T)
        Ploc = _linear_interp(h, c.height, c.P)
        dy = core.rhs_net(vcat(y, ctx, forcing_w(c, hhat),
                               thermo_features(Tloc, Ploc), height_features(hhat)))
        vcat(dy, zeros(eltype(dy), nctx))
    end

    hh = Float32.(c.height ./ m.hscale)
    savehh = Float32.(saveat ./ m.hscale)
    prob = ODEProblem(rhs, u0, (hh[1], hh[end]), p)
    solve(prob, solver; saveat=savehh,
          sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
          abstol=1f-5, reltol=1f-5)
end

function neuralode_loss(m::NeuralODESurrogate, p, c::MLCase)
    sol = _solve_neuralode(m, p, c)
    SciMLBase.successful_retcode(sol) || return 1.0f6
    pred = hcat((Float32.(view(u, 1:3)) for u in sol.u)...)
    _trajectory_loss(pred, _truth_matrix(c))
end

function _train_flat!(p, model, cases, lossfun; epochs::Int, lr::Real,
                      batchsize::Int, val_cases, rng, label::String,
                      grad_clip::Real=10.0, patience::Int=30,
                      min_delta::Real=1e-4)
    isempty(cases) && throw(ArgumentError("no training cases"))
    opt = Flux.setup(Adam(Float32(lr)), p)
    n = length(cases)
    best_val = Inf
    best_p = copy(p)
    stale = 0

    for ep in 1:epochs
        perm = randperm(rng, n)
        total = 0.0
        for s in 1:batchsize:n
            ix = perm[s:min(s + batchsize - 1, n)]
            batch = cases[ix]
            loss, grad = Flux.withgradient(p) do theta
                mean(lossfun(model, theta, c) for c in batch)
            end
            g = grad[1]
            gnorm = sqrt(sum(abs2, g))
            if isfinite(gnorm) && gnorm > grad_clip
                g = g .* Float32(grad_clip / gnorm)
            end
            Flux.update!(opt, p, g)
            total += Float64(loss) * length(ix)
        end

        trainloss = total / n
        valloss = (val_cases === nothing || isempty(val_cases)) ? NaN :
                  mean(lossfun(model, p, c) for c in val_cases)

        if isfinite(valloss)
            if valloss < best_val - min_delta
                best_val = valloss
                best_p .= p
                stale = 0
            else
                stale += 1
            end
        elseif isnan(valloss)
            best_p .= p
        end

        if ep == 1 || ep % 5 == 0 || ep == epochs
            println("$label epoch $ep/$epochs train=$(round(trainloss,sigdigits=6)) val=$(round(valloss,sigdigits=6))")
        end

        if isfinite(valloss) && patience > 0 && stale >= patience
            println("$label early stopping at epoch $ep; best validation loss=$(round(best_val,sigdigits=6))")
            break
        end
    end

    p .= best_p
    p
end

function train_profile!(p, m, cases; epochs=150, lr=7.5e-4, batchsize=16,
                        val_cases=nothing, rng=Random.default_rng(),
                        grad_clip=10.0, patience=30)
    _train_flat!(p, m, cases, profile_loss; epochs=epochs, lr=lr,
                 batchsize=batchsize, val_cases=val_cases, rng=rng,
                 label="profile-v3", grad_clip=grad_clip, patience=patience)
end

function train_neuralode!(p, m, cases; epochs=40, lr=4e-4, batchsize=4,
                          val_cases=nothing, rng=Random.default_rng(),
                          grad_clip=5.0, patience=20)
    _train_flat!(p, m, cases, neuralode_loss; epochs=epochs, lr=lr,
                 batchsize=batchsize, val_cases=val_cases, rng=rng,
                 label="neuralODE-v3", grad_clip=grad_clip, patience=patience)
end

function _physical_matrix(mat, c::MLCase; rhod=c.rhod, heights=c.height)
    ntot = sum(c.mode_N)
    qt = initial_total_water(c)
    nout = size(mat, 2)
    length(heights) == nout || throw(ArgumentError("heights must match output columns"))

    densities = if rhod isa Real
        fill(Float64(rhod), nout)
    elseif length(rhod) == nout && length(heights) == length(c.height) && all(heights .== c.height)
        Float64.(rhod)
    else
        Float64[_linear_interp(h, c.height, c.rhod) for h in heights]
    end
    Ts = Float64[_linear_interp(h, c.height, c.T) for h in heights]
    Ps = Float64[_linear_interp(h, c.height, c.P) for h in heights]

    vals = [physical_target(view(mat, :, i), ntot, densities[i], qt, Ts[i], Ps[i])
            for i in 1:nout]
    (
        S = [v.S for v in vals],
        ql = [v.ql for v in vals],
        activation_fraction = [v.activation_fraction for v in vals],
        Nd_kg = [v.Nd_kg for v in vals],
        Nd = [v.Nd for v in vals],
        beta_ext = [v.beta_ext for v in vals],
    )
end

function predict_profile(m::ProfileSurrogate, p, c::MLCase; rhod=c.rhod)
    q = _physical_matrix(_profile_transformed(m, p, c), c; rhod=rhod, heights=c.height)
    (height=Float64.(c.height), S=q.S, ql=q.ql,
     activation_fraction=q.activation_fraction,
     Nd_kg=q.Nd_kg, Nd=q.Nd, beta_ext=q.beta_ext)
end

function predict_neuralode(m::NeuralODESurrogate, p, c::MLCase; rhod=c.rhod)
    sol = _solve_neuralode(m, p, c)
    mat = hcat((Float32.(view(u, 1:3)) for u in sol.u)...)
    heights = Float64.(sol.t) .* m.hscale
    q = _physical_matrix(mat, c; rhod=rhod, heights=heights)
    (height=heights, S=q.S, ql=q.ql,
     activation_fraction=q.activation_fraction,
     Nd_kg=q.Nd_kg, Nd=q.Nd, beta_ext=q.beta_ext)
end

function _evaluate(predictfun, model, p, cases)
    isempty(cases) && return (
        n=0, S_rmse_percent=NaN, ql_rmse_gkg=NaN,
        fact_rmse=NaN, fact_mae=NaN, Ndkg_log10_rmse=NaN,
        beta_log10_rmse=NaN, Smax_rmse_percent=NaN,
        qlmax_rmse_gkg=NaN, factmax_rmse=NaN, bound_violations=0)

    seS=0.0; seQ=0.0; seF=0.0; aeF=0.0; seN=0.0; seB=0.0
    seSmax=0.0; seQmax=0.0; seFmax=0.0; npt=0; violations=0

    for c in cases
        pred = predictfun(model, p, c)
        ftrue = [Float64(activation_fraction(c, i)) for i in eachindex(c.height)]
        for i in eachindex(c.height)
            fp = pred.activation_fraction[i]
            seS += (100 * (pred.S[i] - c.S[i]))^2
            seQ += (1000 * (pred.ql[i] - c.ql[i]))^2
            seF += (fp - ftrue[i])^2
            aeF += abs(fp - ftrue[i])
            seN += (log10(1 + pred.Nd_kg[i]/1e6) - log10(1 + c.Nd_kg[i]/1e6))^2
            # Use the same reference scale as the beta target transform.
            seB += (log10(1 + pred.beta_ext[i]/BETA_SCALE) -
                    log10(1 + c.beta_ext[i]/BETA_SCALE))^2
            violations += (fp < -1e-7 || fp > 1 + 1e-7) ? 1 : 0
            npt += 1
        end
        seSmax += (100 * (maximum(pred.S) - maximum(c.S)))^2
        seQmax += (1000 * (maximum(pred.ql) - maximum(c.ql)))^2
        seFmax += (maximum(pred.activation_fraction) - maximum(ftrue))^2
    end

    (
        n=length(cases),
        S_rmse_percent=sqrt(seS/npt),
        ql_rmse_gkg=sqrt(seQ/npt),
        fact_rmse=sqrt(seF/npt),
        fact_mae=aeF/npt,
        Ndkg_log10_rmse=sqrt(seN/npt),
        beta_log10_rmse=sqrt(seB/npt),
        Smax_rmse_percent=sqrt(seSmax/length(cases)),
        qlmax_rmse_gkg=sqrt(seQmax/length(cases)),
        factmax_rmse=sqrt(seFmax/length(cases)),
        bound_violations=violations,
    )
end

evaluate_profile(m, p, cases) = _evaluate(predict_profile, m, p, cases)
evaluate_neuralode(m, p, cases) = _evaluate(predict_neuralode, m, p, cases)

function save_profile_model(path, p, m::ProfileSurrogate)
    model_version = MODEL_VERSION
    embed_dim = m.embed_dim; hidden = m.hidden; hscale = m.hscale
    BSON.@save path p embed_dim hidden hscale model_version
end

function save_neuralode_model(path, p, m::NeuralODESurrogate)
    model_version = MODEL_VERSION
    embed_dim = m.embed_dim; hidden = m.hidden; hscale = m.hscale
    BSON.@save path p embed_dim hidden hscale model_version
end

function _load_versioned(path)
    d = BSON.load(path)
    version = get(d, :model_version, 1)
    version == MODEL_VERSION || error(
        "model at $path is surrogate format v$version, but this code expects v$(MODEL_VERSION). " *
        "v3 predicts ql and diagnoses S from total-water conservation; retrain the model.")
    d
end

function load_profile_model(path)
    d = _load_versioned(path)
    p=d[:p]; embed_dim=d[:embed_dim]; hidden=d[:hidden]; hscale=d[:hscale]
    _, m = build_profile_model(embed_dim=embed_dim, hidden=hidden, hscale=hscale)
    p, m
end

function load_neuralode_model(path)
    d = _load_versioned(path)
    p=d[:p]; embed_dim=d[:embed_dim]; hidden=d[:hidden]; hscale=d[:hscale]
    _, m = build_neuralode_model(embed_dim=embed_dim, hidden=hidden, hscale=hscale)
    p, m
end

end # module SetSurrogate

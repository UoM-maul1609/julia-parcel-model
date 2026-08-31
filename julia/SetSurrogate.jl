"""
Variable-mode cloud-activation surrogates.

The aerosol is treated as a weighted *set* of modes. Each mode's intensive
properties pass through the same encoder. Embeddings are number-weighted and
pooled, while total number and physically useful bulk moments are supplied
separately. This preserves permutation invariance and exact invariance to
splitting one physical mode into identical submodes whose numbers sum to the
original mode.

Version 2 learns cloud-drop number as an *activated fraction* in native BMM
number-mixing-ratio units. The inverse transform is bounded to [0,1], so the
surrogate cannot create more cloud droplets than aerosol particles. Known
cloud-base boundary conditions S(0)=0 and f_act(0)=0 are imposed explicitly.
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
using ..SurrogateData: MLCase

export ProfileSurrogate, NeuralODESurrogate, build_profile_model, build_neuralode_model,
       aerosol_context, profile_loss, neuralode_loss, train_profile!, train_neuralode!,
       predict_profile, predict_neuralode, transformed_target, physical_target,
       activation_fraction, fraction_latent, latent_fraction, critical_supersaturation,
       evaluate_profile, evaluate_neuralode, save_profile_model, save_neuralode_model,
       load_profile_model, load_neuralode_model

const MODEL_VERSION = 2

# Target/input scales. They are chosen only to keep neural-network variables O(1).
const S_SCALE = 5.0f-3
const BETA_SCALE = 1.0f-3
const N_SCALE = 1.0f8                 # kg^-1 dry air
const DM_SCALE = 1.0f-7
const M2_SCALE = 1.0f-5               # sum(N D^2), m^2 kg^-1
const M3_SCALE = 1.0f-12              # sum(N D^3), m^3 kg^-1
const W_FLOOR = 1.0f-3
const S_INITIAL = -0.05f0
const S_INITIAL_LATENT = asinh(S_INITIAL / S_SCALE)


# Activated-fraction transform. The small epsilon keeps exact 0/1 targets
# finite in latent space; the inverse is explicitly clamped to [0,1].
const FACT_EPS = 1.0f-4
const FACT_LOGIT_SCALE = 4.0f0

# Approximate kappa-Koehler critical supersaturation feature.
const SIGMA_WATER = 0.072f0           # N m^-1
const MOLW_WATER = 18.01528f-3        # kg mol^-1
const R_GAS = 8.314f0                 # J mol^-1 K^-1
const RHO_WATER = 1000.0f0            # kg m^-3

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

"""Approximate critical supersaturation (dimensionless) at a mode's median dry diameter."""
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

# Robust target transforms. The network learns [supersaturation latent,
# activation-fraction latent, extinction latent].
transformed_target(S::Real, fact::Real, beta::Real) = Float32[
    asinh(Float32(S) / S_SCALE),
    fraction_latent(fact),
    log1p(max(Float32(beta), 0f0) / BETA_SCALE)
]
transformed_target(c::MLCase, i::Int) =
    transformed_target(c.S[i], activation_fraction(c, i), c.beta_ext[i])

function physical_target(y, ntot_kg::Real, rhod::Real)
    fact = latent_fraction(y[2])
    ndkg = fact * max(Float64(ntot_kg), 0.0)
    (
        S = Float64(sinh(y[1]) * S_SCALE),
        activation_fraction = Float64(fact),
        Nd_kg = ndkg,
        Nd = ndkg * Float64(rhod),
        beta_ext = max(0.0, Float64(expm1(y[3]) * BETA_SCALE)),
    )
end

# The encoder deliberately excludes N_i. Number enters through the pooling
# weights and N_total, making the representation invariant to splitting an
# otherwise identical lognormal mode into duplicate submodes.
function _mode_features(c::MLCase)
    nmode = length(c.mode_N)
    dm = Float32[
        log(max(c.mode_Dm[j], 1f-12) / DM_SCALE)
        for j in 1:nmode
    ]
    sig = Float32[
        (c.mode_lnsig[j] - 0.5f0) / 0.25f0
        for j in 1:nmode
    ]
    kap = Float32[
        log1p(max(c.mode_kappa[j], 0f0)) / log1p(1.4f0)
        for j in 1:nmode
    ]
    # A derived activation feature substantially reduces the burden on a small
    # pilot network: Dm, kappa and cloud-base T still remain available, so this
    # does not remove any information or add an empirical activation scheme.
    scrit = Float32[
        log10(critical_supersaturation(c.mode_Dm[j], c.mode_kappa[j], c.T0) / 1f-2) / 2f0
        for j in 1:nmode
    ]
    vcat(reshape(dm, 1, :),
         reshape(sig, 1, :),
         reshape(kap, 1, :),
         reshape(scrit, 1, :))
end

function _mode_weights(c::MLCase)
    ntot = sum(c.mode_N)
    ntot > 0 || throw(ArgumentError("aerosol number must be positive"))
    Float32.(c.mode_N ./ ntot)
end

function _bulk_features(c::MLCase)
    ntot = sum(c.mode_N)
    ntot > 0 || throw(ArgumentError("aerosol number must be positive"))

    # Lognormal raw moments E[D^k] = Dg^k exp(0.5 k^2 ln(sigma_g)^2).
    # M2 is a dry-surface/condensation-sink proxy; M3 is a dry-volume proxy.
    m2 = sum(c.mode_N[j] * c.mode_Dm[j]^2 * exp(2f0 * c.mode_lnsig[j]^2)
             for j in eachindex(c.mode_N))
    m3 = sum(c.mode_N[j] * c.mode_Dm[j]^3 * exp(4.5f0 * c.mode_lnsig[j]^2)
             for j in eachindex(c.mode_N))

    Float32[
        log(ntot / N_SCALE),
        (c.T0 - 280f0) / 15f0,
        (c.P0 - 85_000f0) / 20_000f0,
        log(max(m2, 1f-20) / M2_SCALE),
        log(max(m3, 1f-30) / M3_SCALE),
    ]
end

# These functions only construct fixed input data. Gradients are required with
# respect to encoder/network parameters, not with respect to an MLCase.
Zygote.@nograd _mode_features
Zygote.@nograd _mode_weights
Zygote.@nograd _bulk_features

function aerosol_context(encoder, c::MLCase)
    embeddings = encoder(_mode_features(c))              # embed x mode
    pooled = embeddings * _mode_weights(c)               # embed
    vcat(pooled, _bulk_features(c))
end

forcing_w(c::MLCase, hhat) = Float32(log(max(c.w, W_FLOOR)))

# Multiple height coordinates help the direct network resolve the sharp
# cloud-base activation layer without sacrificing a continuous-height input.
function height_features(hhat::Real)
    h = max(Float32(hhat), 0f0)
    Float32[h, sqrt(h), log1p(20f0 * h) / log(21f0)]
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
    nctx = embed_dim + 5
    head = Chain(Dense(nctx + 1 + 3, hidden, tanh),
                 Dense(hidden, hidden, tanh), Dense(hidden, 3))
    core = ProfileCore(encoder, head)
    p, re = Flux.destructure(core)
    p, ProfileSurrogate(re, embed_dim, hidden, Float32(hscale))
end

function build_neuralode_model(; embed_dim::Int=24, hidden::Int=96, hscale::Real=500.0)
    encoder = Chain(Dense(4, hidden, tanh), Dense(hidden, embed_dim, tanh))
    nctx = embed_dim + 5
    # S(0)=0 and f_act(0)=0 are known. Only cloud-base extinction is learned.
    init_net = Chain(Dense(nctx + 1, hidden, tanh), Dense(hidden, 1))
    rhs_net = Chain(Dense(3 + nctx + 1 + 3, hidden, tanh),
                    Dense(hidden, hidden, tanh), Dense(hidden, 3))
    core = ODECore(encoder, init_net, rhs_net)
    p, re = Flux.destructure(core)
    p, NeuralODESurrogate(re, embed_dim, hidden, Float32(hscale))
end

function _profile_point(core, ctx, c::MLCase, h::Real, hscale::Real)
    hhat = Float32(h / hscale)
    raw = core.head(vcat(ctx, forcing_w(c, hhat), height_features(hhat)))
    if abs(hhat) <= 1f-7
        # Exact cloud-base constraints. Extinction remains learned because
        # unactivated hydrated aerosol can already have finite extinction.
        return Float32[S_INITIAL_LATENT, FACT_ZERO_LATENT, raw[3]]
    end
    raw
end

function _profile_transformed(m::ProfileSurrogate, p, c::MLCase)
    core = m.re(p)
    ctx = aerosol_context(core.encoder, c)
    reduce(hcat, (_profile_point(core, ctx, c, h, m.hscale) for h in c.height))
end

function _truth_matrix(c::MLCase)
    reduce(hcat, (transformed_target(c, i) for i in eachindex(c.height)))
end

# Pointwise loss is supplemented by weak peak/final-state terms. The first
# pilot showed that a pure pointwise MSE could smooth the narrow S maximum and
# activation transition while still achieving a respectable global loss.
function _trajectory_loss(pred, truth)
    point = mean(abs2, pred .- truth)
    peak = mean((maximum(view(pred, j, :)) - maximum(view(truth, j, :)))^2
                for j in axes(pred, 1))
    final = mean(abs2, view(pred, :, size(pred, 2)) .- view(truth, :, size(truth, 2)))
    point + 0.15f0 * peak + 0.05f0 * final
end

function profile_loss(m::ProfileSurrogate, p, c::MLCase)
    _trajectory_loss(_profile_transformed(m, p, c), _truth_matrix(c))
end

"""
Solve in normalized height. Static aerosol/T/P context is carried as a zero-
derivative augmented ODE state so encoder gradients flow through the initial
condition without recomputing the set encoder at every RHS evaluation.
"""
function _solve_neuralode(m::NeuralODESurrogate, p, c::MLCase;
                          solver=Tsit5(), saveat=c.height)
    core0 = m.re(p)
    ctx0 = aerosol_context(core0.encoder, c)
    w0 = forcing_w(c, 0f0)
    beta0 = core0.init_net(vcat(ctx0, w0))[1]
    y0 = Float32[S_INITIAL_LATENT, FACT_ZERO_LATENT, beta0]
    u0 = vcat(y0, ctx0)
    nctx = length(ctx0)

    function rhs(u, p, hhat)
        core = m.re(p)
        y = view(u, 1:3)
        ctx = view(u, 4:3+nctx)
        dy = core.rhs_net(vcat(y, ctx, forcing_w(c, hhat), height_features(hhat)))
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
                 label="profile-v2", grad_clip=grad_clip, patience=patience)
end

function train_neuralode!(p, m, cases; epochs=40, lr=4e-4, batchsize=4,
                          val_cases=nothing, rng=Random.default_rng(),
                          grad_clip=5.0, patience=20)
    _train_flat!(p, m, cases, neuralode_loss; epochs=epochs, lr=lr,
                 batchsize=batchsize, val_cases=val_cases, rng=rng,
                 label="neuralODE-v2", grad_clip=grad_clip, patience=patience)
end

function _physical_matrix(mat, c::MLCase; rhod=c.rhod)
    ntot = sum(c.mode_N)
    nout = size(mat, 2)
    densities = if rhod isa Real
        fill(Float64(rhod), nout)
    else
        length(rhod) == nout || throw(ArgumentError("rhod must be a scalar or have one value per output height"))
        Float64.(rhod)
    end
    vals = [physical_target(view(mat, :, i), ntot, densities[i]) for i in 1:nout]
    (
        S = [v.S for v in vals],
        activation_fraction = [v.activation_fraction for v in vals],
        Nd_kg = [v.Nd_kg for v in vals],
        Nd = [v.Nd for v in vals],
        beta_ext = [v.beta_ext for v in vals],
    )
end

function predict_profile(m::ProfileSurrogate, p, c::MLCase; rhod=c.rhod)
    q = _physical_matrix(_profile_transformed(m, p, c), c; rhod=rhod)
    (height=Float64.(c.height), S=q.S,
     activation_fraction=q.activation_fraction,
     Nd_kg=q.Nd_kg, Nd=q.Nd, beta_ext=q.beta_ext)
end

function predict_neuralode(m::NeuralODESurrogate, p, c::MLCase; rhod=c.rhod)
    sol = _solve_neuralode(m, p, c)
    mat = hcat((Float32.(view(u, 1:3)) for u in sol.u)...)
    q = _physical_matrix(mat, c; rhod=rhod)
    (height=Float64.(sol.t) .* m.hscale, S=q.S,
     activation_fraction=q.activation_fraction,
     Nd_kg=q.Nd_kg, Nd=q.Nd, beta_ext=q.beta_ext)
end

function _evaluate(predictfun, model, p, cases)
    isempty(cases) && return (
        n=0, S_rmse_percent=NaN, fact_rmse=NaN, fact_mae=NaN,
        Ndkg_log10_rmse=NaN, beta_log10_rmse=NaN,
        Smax_rmse_percent=NaN, factmax_rmse=NaN, bound_violations=0)

    seS = 0.0; seF = 0.0; aeF = 0.0; seN = 0.0; seB = 0.0
    seSmax = 0.0; seFmax = 0.0; npt = 0; violations = 0

    for c in cases
        pred = predictfun(model, p, c)
        ftrue = [Float64(activation_fraction(c, i)) for i in eachindex(c.height)]
        for i in eachindex(c.height)
            fp = pred.activation_fraction[i]
            seS += (100 * (pred.S[i] - c.S[i]))^2
            seF += (fp - ftrue[i])^2
            aeF += abs(fp - ftrue[i])
            seN += (log10(1 + pred.Nd_kg[i] / 1e6) -
                    log10(1 + c.Nd_kg[i] / 1e6))^2
            seB += (log10(1 + pred.beta_ext[i] / 1e-8) -
                    log10(1 + c.beta_ext[i] / 1e-8))^2
            violations += (fp < -1e-7 || fp > 1 + 1e-7) ? 1 : 0
            npt += 1
        end
        seSmax += (100 * (maximum(pred.S) - maximum(c.S)))^2
        seFmax += (maximum(pred.activation_fraction) - maximum(ftrue))^2
    end

    (
        n=length(cases),
        S_rmse_percent=sqrt(seS / npt),
        fact_rmse=sqrt(seF / npt),
        fact_mae=aeF / npt,
        Ndkg_log10_rmse=sqrt(seN / npt),
        beta_log10_rmse=sqrt(seB / npt),
        Smax_rmse_percent=sqrt(seSmax / length(cases)),
        factmax_rmse=sqrt(seFmax / length(cases)),
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
        "The v2 activation-fraction target and feature set changed the architecture; retrain the model.")
    d
end

function load_profile_model(path)
    d = _load_versioned(path)
    p = d[:p]; embed_dim = d[:embed_dim]; hidden = d[:hidden]; hscale = d[:hscale]
    _, m = build_profile_model(embed_dim=embed_dim, hidden=hidden, hscale=hscale)
    p, m
end

function load_neuralode_model(path)
    d = _load_versioned(path)
    p = d[:p]; embed_dim = d[:embed_dim]; hidden = d[:hidden]; hscale = d[:hscale]
    _, m = build_neuralode_model(embed_dim=embed_dim, hidden=hidden, hscale=hscale)
    p, m
end

end # module

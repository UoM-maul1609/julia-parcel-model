"""
    module NeuralODESurrogate

Neural ODE surrogate for BMM adiabatic-activation trajectories, using
Flux.jl for the network, DifferentialEquations.jl for integration, and
SciMLSensitivity.jl for adjoint-based gradients through the solve.

STATE VECTOR (per trajectory), all in transformed/normalized units for
training stability:
    y = [S, log1p(Nd), log1p(beta_ext), Dhat]
      S           supersaturation (fraction, order 1e-3 -- already small,
                   left untransformed)
      log1p(Nd)   Nd in m^-3, log-transformed (spans orders of magnitude)
      log1p(bext) extinction coefficient, log-transformed
      Dhat        Dcrit normalized by an aerosol-scale diameter (so it's
                   O(1) instead of O(1e-7))

FORCING: updraft velocity w(t), linearly interpolated from the case's
sample points and fed to the NN alongside the state.

CONTEXT: static per-case parameters (aerosol mode N/Dg/sigma, kappa, T0, P0)
that don't change over the trajectory, concatenated to the NN input so one
network generalizes across cases instead of being retrained per case.

This is a pure Neural ODE (dy/dt = NN(y, w(t), context)), not a UDE with a
hardcoded physics term. If you want a UDE instead (recommended if you have
limited data, since it constrains the NN to only learn a correction), the
natural split is:
    dS/dt   = alpha(context) * w(t) - beta(context) * d(Nd)/dt   [known adiabatic form]
    d(...)  = NN(...)                                            [learned]
with alpha, beta computed from moist thermodynamics (latent heat, Cp, Lv/RT^2
etc. -- these are already implemented in bin_microphysics_module.f90, e.g.
search for `svp`, `eps`, `cp`, `lv` -- worth lifting those closed-form
expressions out if you go this route). Flagging this as the natural upgrade
path rather than building it blindly, since it changes the loss landscape a
lot and is easiest to add once the pure Neural ODE baseline is working.

Usage sketch:

    using .NeuralODESurrogate
    data = load_training_cases(...)   # Vector{CaseData}, see below -- adapt
                                       # this loader to however you already
                                       # have your calibration dataset stored
    model = build_model(ncontext=length(data[1].context))
    trained = train!(model, data; epochs=200, lr=1e-3)
    save_model(trained, "surrogate.bson")
"""
module NeuralODESurrogate

using Flux
using DifferentialEquations
using SciMLSensitivity
using Zygote
using BSON: @save, @load
using Statistics
using Random

export CaseData, SurrogateModel, build_model, train!, predict, save_model, load_model

# -----------------------------------------------------------------------
# Data container
# -----------------------------------------------------------------------

"""
    CaseData

One training trajectory, already extracted from a BMM run (or from your own
calibration dataset -- construct this struct however your data is stored;
the rest of this file only depends on this struct, not on BMM.jl).

    t        :: Vector{Float64}   time points [s]
    w        :: Vector{Float64}   updraft velocity at each t [m/s]
    S        :: Vector{Float64}   supersaturation (fraction)
    Nd       :: Vector{Float64}   droplet number [m^-3]
    beta_ext :: Vector{Float64}   extinction coefficient [m^-1]
    Dcrit    :: Vector{Float64}   critical dry activation diameter [m]
    context  :: Vector{Float64}   static case parameters (aerosol N/Dg/sigma,
                                   kappa, T0, P0, ... ), same length across
                                   all cases, ideally pre-normalized (e.g.
                                   log(N), log(Dg), kappa, (T0-273)/30, ...)
"""
struct CaseData
    t::Vector{Float64}
    w::Vector{Float64}
    S::Vector{Float64}
    Nd::Vector{Float64}
    beta_ext::Vector{Float64}
    Dcrit::Vector{Float64}
    context::Vector{Float64}
end

"""
    load_training_cases(path) -> Vector{CaseData}

STUB -- replace the body with a loader for however your calibration dataset
is actually stored (CSV per run, a single Parquet/Arrow table keyed by run
id, JLD2, etc). Left unimplemented deliberately since you're keeping the
dataset local; wire it up to your own files and the rest of this module
doesn't need to change.
"""
function load_training_cases(path)
    error("load_training_cases: implement this for your dataset's on-disk format")
end

# Normalization scale for Dcrit -> Dhat (O(1e-7) m is a typical aerosol
# scale; adjust to your dmina/dmaxa range).
const DCRIT_SCALE = 1.0e-7

to_state(c::CaseData, i::Int) = Float32[c.S[i], log1p(c.Nd[i]), log1p(c.beta_ext[i]),
                                          c.Dcrit[i] / DCRIT_SCALE]
from_state(y) = (S=y[1], Nd=expm1(y[2]), beta_ext=expm1(y[3]), Dcrit=y[4]*DCRIT_SCALE)

# -----------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------

"""
    SurrogateModel

Wraps the Flux network, its destructured parameter vector (needed for
DifferentialEquations.jl's `ODEProblem` interface), and normalization info.
"""
struct SurrogateModel
    net_restructure  # Flux.re from destructure -- rebuilds the Chain from params
    ncontext::Int
    nstate::Int
end

"""
    build_model(; ncontext, nstate=4, hidden=64) -> (params, model)

4-layer MLP: input = [state (nstate); w(t); context (ncontext)],
output = dstate/dt (nstate). Returns the initial flat parameter vector and
the SurrogateModel wrapper. tanh hidden activations (bounded, and their
derivatives are well-behaved for the adjoint solve); linear output layer.
"""
function build_model(; ncontext::Int, nstate::Int=4, hidden::Int=64)
    nin = nstate + 1 + ncontext   # +1 for w(t)
    chain = Chain(
        Dense(nin, hidden, tanh),
        Dense(hidden, hidden, tanh),
        Dense(hidden, nstate),
    )
    p, re = Flux.destructure(chain)
    return p, SurrogateModel(re, ncontext, nstate)
end

# -----------------------------------------------------------------------
# ODE right-hand side
# -----------------------------------------------------------------------

"""
    linear_interp(t, ts, vs) -> value

Minimal dependency-free linear interpolation of forcing w(t) at solver query
points that fall between the case's sampled time points. Swap for
DataInterpolations.jl's `LinearInterpolation` if you want boundary handling
options / extrapolation control.
"""
function linear_interp(t::Real, ts::AbstractVector, vs::AbstractVector)
    t <= ts[1] && return vs[1]
    t >= ts[end] && return vs[end]
    i = searchsortedlast(ts, t)
    frac = (t - ts[i]) / (ts[i+1] - ts[i])
    return vs[i] + frac * (vs[i+1] - vs[i])
end

"""
    make_dudt(model, context, ts, ws)

Returns a closure `dudt(u, p, t)` suitable for `ODEProblem`, capturing the
case's fixed context vector and forcing series (ts, ws). `p` is the flat NN
parameter vector (this is what SciMLSensitivity differentiates through).
"""
function make_dudt(model::SurrogateModel, context::Vector{Float32}, ts, ws)
    ctx = context
    tsf = Float32.(ts)
    wsf = Float32.(ws)
    function dudt(u, p, t)
        wt = linear_interp(t, tsf, wsf)
        net = model.net_restructure(p)
        x = vcat(u, wt, ctx)
        return net(x)
    end
    return dudt
end

# -----------------------------------------------------------------------
# Forward solve + loss
# -----------------------------------------------------------------------

"""
    solve_case(model, p, case; solver=Tsit5(), saveat=case.t)

Integrate the Neural ODE for one case, returning the solution states at
`saveat` (defaults to the case's own sample times, so the loss lines up
directly against BMM output).
"""
function solve_case(model::SurrogateModel, p, case::CaseData;
                     solver=Tsit5(), saveat=case.t)
    u0 = to_state(case, 1)
    ctx = Float32.(case.context)
    dudt = make_dudt(model, ctx, case.t, case.w)
    tspan = (Float32(case.t[1]), Float32(case.t[end]))
    prob = ODEProblem(dudt, u0, tspan, p)
    sol = solve(prob, solver; saveat=Float32.(saveat),
                sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),
                abstol=1f-6, reltol=1f-6)
    return sol
end

"""
    case_loss(model, p, case)

MSE between the Neural ODE solve and BMM ground truth, in the same
normalized state space used for integration (S untransformed, Nd/beta_ext
log1p, Dcrit scaled). Weight the four state components if one dominates the
loss in practice -- e.g. S is O(1e-3) while log1p(Nd) is O(1) to O(20), so
you likely want per-component weights; a starting point is included below,
tune against your own data.
"""
const STATE_WEIGHTS = Float32[50.0, 1.0, 1.0, 1.0]  # upweight S; rescale as needed

function case_loss(model::SurrogateModel, p, case::CaseData)
    sol = solve_case(model, p, case)
    if sol.retcode != ReturnCode.Success
        return 1f6  # heavily penalize failed/stiff solves rather than erroring
                    # the whole training loop; investigate cases that hit this
                    # a lot (may need a stiffer solver, e.g. Rodas5(), or
                    # tighter tolerances near the fast activation transient)
    end
    n = length(case.t)
    loss = 0.0f0
    for i in 1:n
        target = to_state(case, i)
        pred = sol.u[i]
        loss += sum(STATE_WEIGHTS .* (pred .- target) .^ 2)
    end
    return loss / n
end

# -----------------------------------------------------------------------
# Training loop
# -----------------------------------------------------------------------

"""
    train!(p, model, cases; epochs=200, lr=1e-3, batchsize=8, val_cases=nothing, verbose=true)

Minibatch Adam training over a list of CaseData. Each minibatch's loss is
the mean case_loss over the batch; gradients come from SciMLSensitivity's
adjoint through each case's solve (via Zygote). Returns the trained
parameter vector `p` (mutated in place is not possible for a Flux
destructure vector across restarts, so just use the returned value).

For large datasets, solving N trajectories per epoch is the dominant cost;
start with a subsample to validate the setup (few hundred cases, ~50
epochs) before scaling to your full calibration set.
"""
function train!(p, model::SurrogateModel, cases::Vector{CaseData};
                 epochs::Int=200, lr::Float64=1e-3, batchsize::Int=8,
                 val_cases::Union{Nothing,Vector{CaseData}}=nothing,
                 verbose::Bool=true, rng::AbstractRNG=Random.default_rng())
    opt_state = Flux.setup(Adam(lr), p)
    n = length(cases)

    for epoch in 1:epochs
        perm = randperm(rng, n)
        epoch_loss = 0.0
        for batch_start in 1:batchsize:n
            batch_idx = perm[batch_start:min(batch_start+batchsize-1, n)]
            batch = cases[batch_idx]

            loss, grads = Flux.withgradient(p) do p
                mean(case_loss(model, p, c) for c in batch)
            end
            Flux.update!(opt_state, p, grads[1])
            epoch_loss += loss * length(batch_idx)
        end
        epoch_loss /= n

        if verbose && (epoch % 10 == 0 || epoch == 1)
            msg = "epoch $epoch  train_loss=$(round(epoch_loss, sigdigits=4))"
            if val_cases !== nothing
                vloss = mean(case_loss(model, p, c) for c in val_cases)
                msg *= "  val_loss=$(round(vloss, sigdigits=4))"
            end
            println(msg)
        end
    end
    return p
end

"""
    predict(model, p, case) -> NamedTuple of vectors (S, Nd, beta_ext, Dcrit)

Run the trained surrogate on one case's (t, w, context) and return
predictions in physical units, for comparison against case.S/.Nd/etc.
"""
function predict(model::SurrogateModel, p, case::CaseData)
    sol = solve_case(model, p, case)
    n = length(sol.u)
    S = Vector{Float64}(undef, n); Nd = similar(S); bext = similar(S); Dcrit = similar(S)
    for i in 1:n
        phys = from_state(sol.u[i])
        S[i], Nd[i], bext[i], Dcrit[i] = phys.S, phys.Nd, phys.beta_ext, phys.Dcrit
    end
    return (t=sol.t, S=S, Nd=Nd, beta_ext=bext, Dcrit=Dcrit)
end

save_model(p, model::SurrogateModel, path) = @save path p ncontext=model.ncontext nstate=model.nstate

function load_model(path; hidden::Int=64)
    @load path p ncontext nstate
    _, model = build_model(; ncontext=ncontext, nstate=nstate, hidden=hidden)
    return p, model
end

end # module NeuralODESurrogate

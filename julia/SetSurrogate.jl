"""
Variable-mode cloud-activation surrogates.

The aerosol is treated as a weighted *set* of modes. Each mode's intensive
properties (Dm, ln sigma, hygroscopicity) pass through the same encoder.
Embeddings are number-weighted and pooled, while total number is supplied
separately. Thus splitting one physical mode into two identical half-number
modes gives the same aerosol representation, and input order is irrelevant.
"""
module SetSurrogate

using Flux
using DifferentialEquations
using SciMLBase
using SciMLSensitivity
using Zygote
using Random
using Statistics
using BSON: @save, @load
using ..SurrogateData: MLCase

export ProfileSurrogate, NeuralODESurrogate, build_profile_model, build_neuralode_model,
       aerosol_context, profile_loss, neuralode_loss, train_profile!, train_neuralode!,
       predict_profile, predict_neuralode, transformed_target, physical_target,
       evaluate_profile, evaluate_neuralode, save_profile_model, save_neuralode_model,
       load_profile_model, load_neuralode_model

const S_SCALE = 5.0f-3
const ND_SCALE = 1.0f8
const BETA_SCALE = 1.0f-3
const N_SCALE = 1.0f8
const DM_SCALE = 1.0f-7
const W_FLOOR = 1.0f-3

# Robust target transforms: the networks work in O(1) variables while physical
# outputs retain a very large dynamic range.
transformed_target(S, Nd, beta) = Float32[
    asinh(Float32(S) / S_SCALE),
    log1p(max(Float32(Nd), 0f0) / ND_SCALE),
    log1p(max(Float32(beta), 0f0) / BETA_SCALE)
]
transformed_target(c::MLCase, i::Int) = transformed_target(c.S[i], c.Nd[i], c.beta_ext[i])
physical_target(y) = (
    S = Float64(sinh(y[1]) * S_SCALE),
    Nd = max(0.0, Float64(expm1(y[2]) * ND_SCALE)),
    beta_ext = max(0.0, Float64(expm1(y[3]) * BETA_SCALE))
)

# The encoder deliberately excludes N_i. Number enters through the pooling
# weights and N_total, making the representation invariant to splitting an
# otherwise identical lognormal mode into duplicate submodes.
function _mode_features(c::MLCase)
    reduce(hcat, (Float32[
        log(max(c.mode_Dm[j], 1f-12) / DM_SCALE),
        (c.mode_lnsig[j] - 0.5f0) / 0.25f0,
        (c.mode_kappa[j] - 0.5f0) / 0.5f0
    ] for j in eachindex(c.mode_N)))
end

function aerosol_context(encoder, c::MLCase)
    ntot = sum(c.mode_N)
    ntot > 0 || throw(ArgumentError("aerosol number must be positive"))
    weights = c.mode_N ./ ntot
    embeddings = encoder(_mode_features(c))              # embed x mode
    pooled = embeddings * weights                        # embed
    vcat(pooled,
         Float32(log(ntot / N_SCALE)),
         Float32((c.T0 - 280f0) / 15f0),
         Float32((c.P0 - 85_000f0) / 20_000f0))
end

forcing_w(c::MLCase, hhat) = Float32(log(max(c.w, W_FLOOR)))

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

function build_profile_model(; embed_dim::Int=16, hidden::Int=64, hscale::Real=500.0)
    encoder = Chain(Dense(3, hidden, tanh), Dense(hidden, embed_dim, tanh))
    nctx = embed_dim + 3
    head = Chain(Dense(nctx + 2, hidden, tanh),
                 Dense(hidden, hidden, tanh), Dense(hidden, 3))
    core = ProfileCore(encoder, head)
    p, re = Flux.destructure(core)
    p, ProfileSurrogate(re, embed_dim, hidden, Float32(hscale))
end

function build_neuralode_model(; embed_dim::Int=16, hidden::Int=64, hscale::Real=500.0)
    encoder = Chain(Dense(3, hidden, tanh), Dense(hidden, embed_dim, tanh))
    nctx = embed_dim + 3
    init_net = Chain(Dense(nctx + 1, hidden, tanh), Dense(hidden, 3))
    rhs_net = Chain(Dense(3 + nctx + 2, hidden, tanh),
                    Dense(hidden, hidden, tanh), Dense(hidden, 3))
    core = ODECore(encoder, init_net, rhs_net)
    p, re = Flux.destructure(core)
    p, NeuralODESurrogate(re, embed_dim, hidden, Float32(hscale))
end

function _profile_transformed(m::ProfileSurrogate, p, c::MLCase)
    core = m.re(p)
    ctx = aerosol_context(core.encoder, c)
    reduce(hcat, (core.head(vcat(ctx, forcing_w(c,h/m.hscale), Float32(h/m.hscale)))
                  for h in c.height))
end

function profile_loss(m::ProfileSurrogate, p, c::MLCase)
    pred = _profile_transformed(m,p,c)
    truth = reduce(hcat, (transformed_target(c,i) for i in eachindex(c.height)))
    mean(abs2, pred .- truth)
end

"""
Solve in normalized height. Static aerosol/T/P context is carried as zero-
derivative augmented ODE state so encoder gradients flow through the initial
condition without recomputing the set encoder at every RHS evaluation.
"""
function _solve_neuralode(m::NeuralODESurrogate, p, c::MLCase;
                          solver=Tsit5(), saveat=c.height)
    core0 = m.re(p)
    ctx0 = aerosol_context(core0.encoder, c)
    w0 = forcing_w(c, 0f0)
    y0 = core0.init_net(vcat(ctx0, w0))
    u0 = vcat(y0, ctx0)
    nctx = length(ctx0)

    function rhs(u, p, hhat)
        core = m.re(p)
        y = view(u, 1:3)
        ctx = view(u, 4:3+nctx)
        dy = core.rhs_net(vcat(y, ctx, forcing_w(c,hhat), Float32(hhat)))
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
    sol = _solve_neuralode(m,p,c)
    SciMLBase.successful_retcode(sol) || return 1.0f6
    total = 0.0f0
    for i in eachindex(c.height)
        total += sum(abs2, view(sol.u[i],1:3) .- transformed_target(c,i))
    end
    total / (3 * length(c.height))
end

function _train_flat!(p, model, cases, lossfun; epochs::Int, lr::Real,
                      batchsize::Int, val_cases, rng, label::String)
    isempty(cases) && throw(ArgumentError("no training cases"))
    opt = Flux.setup(Adam(Float32(lr)), p)
    n = length(cases)
    best_val = Inf
    best_p = copy(p)
    for ep in 1:epochs
        perm = randperm(rng, n)
        total = 0.0
        for s in 1:batchsize:n
            ix = perm[s:min(s+batchsize-1,n)]
            batch = cases[ix]
            loss, grad = Flux.withgradient(p) do theta
                mean(lossfun(model,theta,c) for c in batch)
            end
            Flux.update!(opt, p, grad[1])
            total += Float64(loss) * length(ix)
        end
        trainloss = total/n
        valloss = (val_cases === nothing || isempty(val_cases)) ? NaN :
                  mean(lossfun(model,p,c) for c in val_cases)
        if isfinite(valloss) && valloss < best_val
            best_val = valloss
            best_p .= p
        elseif isnan(valloss)
            best_p .= p
        end
        if ep == 1 || ep % 5 == 0 || ep == epochs
            println("$label epoch $ep/$epochs train=$(round(trainloss,sigdigits=6)) val=$(round(valloss,sigdigits=6))")
        end
    end
    p .= best_p
    p
end

function train_profile!(p,m,cases;epochs=100,lr=1e-3,batchsize=16,val_cases=nothing,
                        rng=Random.default_rng())
    _train_flat!(p,m,cases,profile_loss;epochs=epochs,lr=lr,batchsize=batchsize,
                 val_cases=val_cases,rng=rng,label="profile")
end

function train_neuralode!(p,m,cases;epochs=30,lr=5e-4,batchsize=4,val_cases=nothing,
                          rng=Random.default_rng())
    _train_flat!(p,m,cases,neuralode_loss;epochs=epochs,lr=lr,batchsize=batchsize,
                 val_cases=val_cases,rng=rng,label="neuralODE")
end

function _physical_matrix(mat)
    vals = [physical_target(view(mat,:,i)) for i in axes(mat,2)]
    (S=[v.S for v in vals], Nd=[v.Nd for v in vals], beta_ext=[v.beta_ext for v in vals])
end

function predict_profile(m::ProfileSurrogate,p,c::MLCase)
    q = _physical_matrix(_profile_transformed(m,p,c))
    (height=Float64.(c.height), S=q.S, Nd=q.Nd, beta_ext=q.beta_ext)
end

function predict_neuralode(m::NeuralODESurrogate,p,c::MLCase)
    sol = _solve_neuralode(m,p,c)
    mat = hcat((Float32.(view(u,1:3)) for u in sol.u)...)
    q = _physical_matrix(mat)
    (height=Float64.(sol.t).*m.hscale, S=q.S, Nd=q.Nd, beta_ext=q.beta_ext)
end

function _evaluate(predictfun, model, p, cases)
    isempty(cases) && return (n=0, S_rmse_percent=NaN, Nd_log10_rmse=NaN, beta_log10_rmse=NaN)
    seS=0.0; seN=0.0; seB=0.0; npt=0
    for c in cases
        pred = predictfun(model,p,c)
        for i in eachindex(c.height)
            seS += (100*(pred.S[i]-c.S[i]))^2
            seN += (log10(1 + pred.Nd[i]/1e6) - log10(1 + c.Nd[i]/1e6))^2
            seB += (log10(1 + pred.beta_ext[i]/1e-8) - log10(1 + c.beta_ext[i]/1e-8))^2
            npt += 1
        end
    end
    (n=length(cases), S_rmse_percent=sqrt(seS/npt),
     Nd_log10_rmse=sqrt(seN/npt), beta_log10_rmse=sqrt(seB/npt))
end

evaluate_profile(m,p,cases) = _evaluate(predict_profile,m,p,cases)
evaluate_neuralode(m,p,cases) = _evaluate(predict_neuralode,m,p,cases)

function save_profile_model(path,p,m::ProfileSurrogate)
    embed_dim=m.embed_dim; hidden=m.hidden; hscale=m.hscale; @save path p embed_dim hidden hscale
end
function save_neuralode_model(path,p,m::NeuralODESurrogate)
    embed_dim=m.embed_dim; hidden=m.hidden; hscale=m.hscale; @save path p embed_dim hidden hscale
end
function load_profile_model(path)
    @load path p embed_dim hidden hscale
    _,m=build_profile_model(embed_dim=embed_dim,hidden=hidden,hscale=hscale); p,m
end
function load_neuralode_model(path)
    @load path p embed_dim hidden hscale
    _,m=build_neuralode_model(embed_dim=embed_dim,hidden=hidden,hscale=hscale); p,m
end

end # module

include("BMM.jl")
include("NeuralODESurrogate.jl")
using .BMM
using .NeuralODESurrogate
using Random

const EXE = get(ENV, "BMM_EXE", BMM.default_bmm_exe())
const HTOP = 600.0
const DZ = 5.0
loguniform(rng, a, b) = exp(log(a) + rand(rng) * (log(b) - log(a)))

function sample_cases(n; rng=Random.default_rng())
    out = BMMCase[]
    for _ in 1:n
        w = loguniform(rng, 0.1, 8.0)
        mode = AerosolMode(N=loguniform(rng, 50e6, 2000e6),
                           Dm=loguniform(rng, 30e-9, 200e-9),
                           lnsig=0.2 + 0.5 * rand(rng),
                           kappa=0.05 + 0.95 * rand(rng))
        push!(out, cloud_base_case(winit=w, height_top=HTOP,
              tinit=268 + 25 * rand(rng), pinit=65000 + 35000 * rand(rng), modes=[mode]))
    end
    out
end

function context(c)
    m = c.modes[1]
    Float64[log(m.N / 1e8), log(m.Dm / 1e-7), m.lnsig, m.kappa,
            (c.tinit - 273.15) / 20, log(c.pinit / 80000)]
end

rng = MersenneTwister(0)
raw = run_bmm_batch(sample_cases(40; rng=rng); exe=EXE)
data = CaseData[]
for r in raw
    r === nothing && continue
    q = resample_profile(r; dz=DZ, height_top=HTOP)
    push!(data, CaseData(q.height, q.w, q.S, q.ndrop, q.beta_ext,
                         vec(q.dcrit[:, 1]), context(q.case)))
end
shuffle!(rng, data)
ntr = clamp(round(Int, 0.8 * length(data)), 1, length(data) - 1)
tr, va = data[1:ntr], data[ntr+1:end]
p, model = build_model(ncontext=length(tr[1].context))
p = train!(p, model, tr; epochs=100, batchsize=8, val_cases=va, rng=rng)
save_model(p, model, "surrogate.bson")
pr = predict(model, p, va[1])
println("final held-out Nd BMM=$(va[1].Nd[end]) surrogate=$(pr.Nd[end])")

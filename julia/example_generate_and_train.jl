"""
Example: generate a small ensemble with BMM.jl and train the Neural ODE
surrogate on it. This is a template for wiring the two modules together --
swap `generate_case_data` for a loader over your own existing calibration
dataset (via NeuralODESurrogate.load_training_cases) and skip BMM entirely
if you don't need to generate more data.
"""
include("BMM.jl")
include("NeuralODESurrogate.jl")
using .BMM
using .NeuralODESurrogate
using Random

const BMM_EXE = get(ENV, "BMM_EXE", "/path/to/bmm-repo/main.exe")

# --- 1. Build a small parameter sweep of BMMCases -------------------------

function sample_cases(n::Int; rng=Random.default_rng())
    cases = BMMCase[]
    for _ in 1:n
        w = 10 .^ (rand(rng) * (log10(8.0) - log10(0.1)) + log10(0.1))   # 0.1-8 m/s
        N = 10 .^ (rand(rng) * (log10(2000e6) - log10(50e6)) + log10(50e6))  # 50-2000 cm^-3 in m^-3
        Dg = 10 .^ (rand(rng) * (log10(200e-9) - log10(30e-9)) + log10(30e-9))
        sig = 0.2 + rand(rng) * 0.4
        kappa = 0.1 + rand(rng) * 0.9
        T0 = 260.0 + rand(rng) * 30.0
        P0 = 60000.0 + rand(rng) * 40000.0
        push!(cases, BMMCase(winit=w, tinit=T0, pinit=P0, rhinit=0.95,
                              n_aer1=[N], d_aer1=[Dg], sig_aer1=[sig],
                              kappa_core1=[kappa], runtime=1800.0, dt=5.0))
    end
    return cases
end

function to_context(case::BMMCase)
    Float64[log(case.n_aer1[1]), log(case.d_aer1[1]), case.sig_aer1[1],
             case.kappa_core1[1], (case.tinit - 273.0) / 30.0,
             (case.pinit - 80000.0) / 20000.0]
end

function generate_case_data(cases::Vector{BMMCase}; exe=BMM_EXE)
    raw = run_bmm_batch(cases; exe=exe)
    data = CaseData[]
    for r in raw
        r === nothing && continue
        push!(data, CaseData(r.t, r.w, r.S, r.ndrop, r.beta_ext, r.dcrit, to_context(r.case)))
    end
    return data
end

# --- 2. Generate a small ensemble, split train/val -------------------------

rng = MersenneTwister(0)
cases = sample_cases(60; rng=rng)      # bump this up once the pipeline is validated
data = generate_case_data(cases)
shuffle!(rng, data)
ntrain = round(Int, 0.8 * length(data))
train_data, val_data = data[1:ntrain], data[ntrain+1:end]
println("generated $(length(data)) usable cases ($(ntrain) train / $(length(val_data)) val)")

# --- 3. Build and train the surrogate --------------------------------------

ncontext = length(train_data[1].context)
p, model = build_model(; ncontext=ncontext, hidden=64)
p = train!(p, model, train_data; epochs=100, lr=1e-3, batchsize=8, val_cases=val_data)

save_model(p, model, "surrogate.bson")
println("saved surrogate.bson")

# --- 4. Quick check on one held-out case ------------------------------------

if !isempty(val_data)
    pred = predict(model, p, val_data[1])
    println("val case 0: final BMM Nd=$(val_data[1].Nd[end])  surrogate Nd=$(pred.Nd[end])")
end

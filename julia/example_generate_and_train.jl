# Minimal example using an already-generated synthetic BMM dataset.
# For the full CLI workflow use train_surrogate.jl and compare_surrogates.jl.
include("SurrogateData.jl")
include("SetSurrogate.jl")
using .SurrogateData
using .SetSurrogate
using Random

path = length(ARGS) >= 1 ? ARGS[1] : "synthetic_bmm.nc"
cases = read_cases(path)
split = split_cases(cases; seed=20260831)
hscale = maximum(maximum(c.height) for c in cases)

Random.seed!(20260831)
p, model = build_profile_model(hscale=hscale)
train_profile!(p, model, split.train; epochs=10, val_cases=split.val,
               rng=MersenneTwister(20260831))
println("10-epoch example TEST metrics: ", evaluate_profile(model, p, split.test))

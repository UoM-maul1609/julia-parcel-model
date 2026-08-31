include("SurrogateData.jl")
include("SetSurrogate.jl")
using .SurrogateData
using .SetSurrogate
using Random

# Usage:
#   julia --project=. train_surrogate.jl synthetic_bmm.nc profile [epochs] [seed]
#   julia --project=. train_surrogate.jl synthetic_bmm.nc neuralode [epochs] [seed]
path = length(ARGS) >= 1 ? ARGS[1] : "synthetic_bmm.nc"
kind = lowercase(length(ARGS) >= 2 ? ARGS[2] : "profile")
default_epochs = kind == "neuralode" ? 40 : 150
epochs = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : default_epochs
seed = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 20260831

cases = read_cases(path)
length(cases) >= 20 || error("need at least 20 successful trajectories; found $(length(cases))")
split = split_cases(cases; seed=seed)
write_split_csv("training_split.csv", split)
println("trajectory split: $(length(split.train)) train / $(length(split.val)) validation / $(length(split.test)) test")
for n in sort(unique(length.(getfield.(cases, :mode_N))))
    println("  $n modes: train=$(count(c -> length(c.mode_N) == n, split.train)) val=$(count(c -> length(c.mode_N) == n, split.val)) test=$(count(c -> length(c.mode_N) == n, split.test))")
end
println("surrogate v2 target: S, bounded activated fraction Nd_kg/sum(mode_N), beta_ext")
println("hard constraints: S(z=0)=0 and activated_fraction(z=0)=0; 0 <= activated_fraction <= 1")

hscale = maximum(maximum(c.height) for c in cases)
rng = MersenneTwister(seed)
Random.seed!(seed)  # deterministic Flux parameter initialization as well as split/batching

if kind == "profile"
    p, m = build_profile_model(hscale=hscale)
    train_profile!(p, m, split.train; epochs=epochs, val_cases=split.val, rng=rng)
    println("train metrics: ", evaluate_profile(m, p, split.train))
    println("validation metrics: ", evaluate_profile(m, p, split.val))
    println("TEST metrics: ", evaluate_profile(m, p, split.test))
    save_profile_model("profile_surrogate.bson", p, m)
    println("saved profile_surrogate.bson (surrogate format v2)")
elseif kind == "neuralode"
    p, m = build_neuralode_model(hscale=hscale)
    train_neuralode!(p, m, split.train; epochs=epochs, val_cases=split.val, rng=rng)
    println("train metrics: ", evaluate_neuralode(m, p, split.train))
    println("validation metrics: ", evaluate_neuralode(m, p, split.val))
    println("TEST metrics: ", evaluate_neuralode(m, p, split.test))
    save_neuralode_model("neuralode_surrogate.bson", p, m)
    println("saved neuralode_surrogate.bson (surrogate format v2)")
else
    error("model kind must be 'profile' or 'neuralode'")
end

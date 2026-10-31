include("SurrogateData.jl")
include("SetSurrogate.jl")
using .SurrogateData
using .SetSurrogate
using Random
using Flux

# Usage:
#   julia --project=. train_surrogate.jl synthetic_bmm.nc profile [epochs] [seed]
#   julia --project=. train_surrogate.jl synthetic_bmm.nc neuralode [epochs] [seed]
path = length(ARGS) >= 1 ? ARGS[1] : "synthetic_bmm.nc"
kind = lowercase(length(ARGS) >= 2 ? ARGS[2] : "profile")
default_epochs = kind == "neuralode" ? 40 : 150
epochs = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : default_epochs
seed = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 20260831

# Training device. CUDA is a declared dependency of this project, so load it
# normally at top level. DEVICE=auto uses the NVIDIA GPU when CUDA is functional,
# DEVICE=gpu forces it, and DEVICE=cpu forces CPU execution.
using CUDA

device_request = lowercase(get(ENV, "DEVICE", "auto"))
device_request in ("auto", "cpu", "gpu") ||
    error("DEVICE must be 'auto', 'cpu' or 'gpu'; got '$device_request'")

cuda_ok = CUDA.functional()

if device_request == "gpu" && !cuda_ok
    error("DEVICE=gpu requested but CUDA is not functional. Run `make gpu-check` for diagnostics.")
end

use_gpu = device_request == "gpu" || (device_request == "auto" && cuda_ok)

if use_gpu
    kind == "profile" || error("GPU training is currently supported for the profile surrogate only")
    CUDA.allowscalar(false)
    to_device = CUDA.cu
    println("training device: GPU ($(CUDA.name(CUDA.device())))")
else
    to_device = identity
    println("training device: CPU")
end

batch_request = lowercase(get(ENV, "BATCHSIZE", "auto"))
batchsize = batch_request == "auto" ? (use_gpu ? 64 : 16) : parse(Int, batch_request)
batchsize > 0 || error("BATCHSIZE must be positive")
kind == "profile" && println("profile minibatch size: $batchsize trajectories")

model_dir = get(ENV, "MODEL_DIR", ".")
results_dir = get(ENV, "RESULTS_DIR", ".")
mkpath(model_dir)
mkpath(results_dir)

cases = read_cases(path)
length(cases) >= 20 || error("need at least 20 successful trajectories; found $(length(cases))")
split = split_cases(cases; seed=seed)
write_split_csv(joinpath(results_dir, "training_split.csv"), split)
println("trajectory split: $(length(split.train)) train / $(length(split.val)) validation / $(length(split.test)) test")
for n in sort(unique(length.(getfield.(cases, :mode_N))))
    println("  $n modes: train=$(count(c -> length(c.mode_N) == n, split.train)) val=$(count(c -> length(c.mode_N) == n, split.val)) test=$(count(c -> length(c.mode_N) == n, split.test))")
end
println("surrogate v3 target: ql, bounded activated fraction Nd_kg/sum(mode_N), beta_ext")
println("physics: S diagnosed from qtot = ql + (1+S) qvs(T,P)")
println("inputs include qtot_initial and local T(z), P(z); activated_fraction(z=0)=0 exactly")

hscale = maximum(maximum(c.height) for c in cases)
rng = MersenneTwister(seed)
Random.seed!(seed)

if kind == "profile"
    p, m = build_profile_model(hscale=hscale)
    p = to_device(p)
    train_profile!(p, m, split.train; epochs=epochs, batchsize=batchsize, val_cases=split.val, rng=rng, device=to_device)
    p = use_gpu ? Array(p) : p
    println("train metrics: ", evaluate_profile(m, p, split.train))
    println("validation metrics: ", evaluate_profile(m, p, split.val))
    println("TEST metrics: ", evaluate_profile(m, p, split.test))
    model_path = joinpath(model_dir, "profile_surrogate.bson")
    save_profile_model(model_path, p, m)
    println("saved $model_path (surrogate format v3)")
elseif kind == "neuralode"
    p, m = build_neuralode_model(hscale=hscale)
    train_neuralode!(p, m, split.train; epochs=epochs, val_cases=split.val, rng=rng, device=to_device)
    println("train metrics: ", evaluate_neuralode(m, p, split.train))
    println("validation metrics: ", evaluate_neuralode(m, p, split.val))
    println("TEST metrics: ", evaluate_neuralode(m, p, split.test))
    model_path = joinpath(model_dir, "neuralode_surrogate.bson")
    save_neuralode_model(model_path, p, m)
    println("saved $model_path (surrogate format v3)")
else
    error("model kind must be 'profile' or 'neuralode'")
end

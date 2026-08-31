include("SurrogateData.jl")
include("SetSurrogate.jl")
using .SurrogateData
using .SetSurrogate
using Printf
using Statistics

try
    @eval using Plots
catch err
    error("""
Plots.jl is required for profile comparison figures.

Install it once from julia/ with:
    julia --project=. -e 'using Pkg; Pkg.add("Plots")'

Original error:
$(sprint(showerror, err))
""")
end

# Usage:
#   julia --project=. compare_surrogates.jl synthetic_bmm.nc [seed] [profile|neuralode|both] [nplots]
#
# Examples:
#   julia --project=. compare_surrogates.jl synthetic_bmm.nc 20260831 profile 8
#   julia --project=. compare_surrogates.jl synthetic_bmm.nc 20260831 neuralode 8
#   julia --project=. compare_surrogates.jl synthetic_bmm.nc 20260831 both 8
#
# Figures and a per-case metric table are written to surrogate_comparison/.

path = length(ARGS) >= 1 ? ARGS[1] : "synthetic_bmm.nc"
seed = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 20260831
kind = lowercase(length(ARGS) >= 3 ? ARGS[3] : "both")
nplots = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 8
kind in ("profile", "neuralode", "both") ||
    error("model kind must be 'profile', 'neuralode', or 'both'")

cases = read_cases(path)
split = split_cases(cases; seed=seed)
test = split.test
isempty(test) && error("test split is empty")

profile_model = nothing
neuralode_model = nothing

if kind in ("profile", "both")
    isfile("profile_surrogate.bson") ||
        error("profile_surrogate.bson not found in $(pwd())")
    p, m = load_profile_model("profile_surrogate.bson")
    profile_model = (p=p, m=m)
    println("profile validation: ", evaluate_profile(m, p, split.val))
    println("profile TEST:       ", evaluate_profile(m, p, split.test))
end

if kind in ("neuralode", "both")
    isfile("neuralode_surrogate.bson") ||
        error("neuralode_surrogate.bson not found in $(pwd())")
    p, m = load_neuralode_model("neuralode_surrogate.bson")
    neuralode_model = (p=p, m=m)
    println("neuralODE validation: ", evaluate_neuralode(m, p, split.val))
    println("neuralODE TEST:       ", evaluate_neuralode(m, p, split.test))
end

activation_fraction(c::MLCase) = sum(c.mode_N) > 0 ? maximum(c.Nd_kg) / sum(c.mode_N) : 0.0
ntot_cm3_cb(c::MLCase) = sum(c.mode_N) * dry_air_density(c.P0, c.T0, 1.0) / 1e6

function _closest_to_median_activation(g)
    a = activation_fraction.(g)
    target = median(a)
    g[argmin(abs.(a .- target))]
end

"""
Choose deterministic, informative held-out cases:
  * one median-activation case for each aerosol mode count
  * lowest/highest activation fraction
  * lowest/highest updraft
Then de-duplicate and fill from the test split if nplots asks for more.
"""
function select_cases(test, nplots)
    selected = MLCase[]
    for nm in sort(unique(length(c.mode_N) for c in test))
        g = [c for c in test if length(c.mode_N) == nm]
        isempty(g) || push!(selected, _closest_to_median_activation(g))
    end

    push!(selected, test[argmin(activation_fraction.(test))])
    push!(selected, test[argmax(activation_fraction.(test))])
    push!(selected, test[argmin(getfield.(test, :w))])
    push!(selected, test[argmax(getfield.(test, :w))])

    # Stable de-duplication by case id.
    unique_selected = MLCase[]
    seen = Set{Int}()
    for c in selected
        if !(c.id in seen)
            push!(unique_selected, c)
            push!(seen, c.id)
        end
    end

    # Fill deterministically with remaining test cases if requested.
    for c in sort(test; by=c -> c.id)
        length(unique_selected) >= nplots && break
        if !(c.id in seen)
            push!(unique_selected, c)
            push!(seen, c.id)
        end
    end

    unique_selected[1:min(nplots, length(unique_selected))]
end

selected = select_cases(test, nplots)
outdir = "surrogate_comparison"
mkpath(outdir)

function summary_metrics(c, pred)
    (
        Smax_ref = maximum(c.S) * 100,
        Smax_pred = maximum(pred.S) * 100,
        Ndmax_ref = maximum(c.Nd) / 1e6,
        Ndmax_pred = maximum(pred.Nd) / 1e6,
        Ndfinal_ref = c.Nd[end] / 1e6,
        Ndfinal_pred = pred.Nd[end] / 1e6,
        betamax_ref = maximum(c.beta_ext),
        betamax_pred = maximum(pred.beta_ext),
    )
end

csvpath = joinpath(outdir, "comparison_metrics.csv")
open(csvpath, "w") do io
    println(io,
        "case,model,n_modes,w_m_s,T_K,P_hPa,Ntot_cm3_cb,activation_fraction," *
        "Smax_BMM_percent,Smax_pred_percent," *
        "Ndmax_BMM_cm3,Ndmax_pred_cm3,Ndfinal_BMM_cm3,Ndfinal_pred_cm3," *
        "betamax_BMM_m-1,betamax_pred_m-1")

    for c in selected
        predictions = Pair{String,Any}[]
        if profile_model !== nothing
            push!(predictions, "profile" =>
                predict_profile(profile_model.m, profile_model.p, c))
        end
        if neuralode_model !== nothing
            push!(predictions, "neuralode" =>
                predict_neuralode(neuralode_model.m, neuralode_model.p, c))
        end

        for (name, pred) in predictions
            q = summary_metrics(c, pred)
            println(io, join((
                c.id, name, length(c.mode_N), c.w, c.T0, c.P0 / 100,
                ntot_cm3_cb(c), activation_fraction(c),
                q.Smax_ref, q.Smax_pred,
                q.Ndmax_ref, q.Ndmax_pred,
                q.Ndfinal_ref, q.Ndfinal_pred,
                q.betamax_ref, q.betamax_pred
            ), ","))
        end

        z = Float64.(c.height)
        titletext = @sprintf(
            "case %d | %d modes | w=%.3g m s⁻¹ | Ncb=%.3g cm⁻³ | fact=%.3f",
            c.id, length(c.mode_N), c.w, ntot_cm3_cb(c), activation_fraction(c)
        )

        pS = plot(100 .* Float64.(c.S), z;
            label="BMM", xlabel="S (%)", ylabel="height above cloud base (m)",
            title=titletext, linewidth=2, legend=:best)

        pN = plot(Float64.(c.Nd) ./ 1e6, z;
            label="BMM", xlabel="Nd (cm⁻³)", ylabel="height (m)",
            linewidth=2, legend=:best)

        pB = plot(Float64.(c.beta_ext), z;
            label="BMM", xlabel="βext (m⁻¹)", ylabel="height (m)",
            linewidth=2, legend=:best)

        # Context-only BMM fields: not currently predicted by the surrogate.
        pQ = plot(1000 .* Float64.(c.ql), z;
            label="BMM", xlabel="ql (g kg⁻¹)", ylabel="height (m)",
            linewidth=2, legend=false)

        pD = plot(1e6 .* Float64.(c.deff), z;
            label="BMM", xlabel="deff (µm)", ylabel="height (m)",
            linewidth=2, legend=false)

        if profile_model !== nothing
            pred = predict_profile(profile_model.m, profile_model.p, c)
            plot!(pS, 100 .* pred.S, pred.height; label="profile", linestyle=:dash, linewidth=2)
            plot!(pN, pred.Nd ./ 1e6, pred.height; label="profile", linestyle=:dash, linewidth=2)
            plot!(pB, pred.beta_ext, pred.height; label="profile", linestyle=:dash, linewidth=2)
        end

        if neuralode_model !== nothing
            pred = predict_neuralode(neuralode_model.m, neuralode_model.p, c)
            plot!(pS, 100 .* pred.S, pred.height; label="neuralODE", linestyle=:dot, linewidth=2)
            plot!(pN, pred.Nd ./ 1e6, pred.height; label="neuralODE", linestyle=:dot, linewidth=2)
            plot!(pB, pred.beta_ext, pred.height; label="neuralODE", linestyle=:dot, linewidth=2)
        end

        fig = plot(pS, pN, pB, pQ, pD;
            layout=(1,5), size=(1900,520), margin=4 * Plots.mm)

        pngpath = joinpath(outdir, @sprintf("case_%04d.png", c.id))
        savefig(fig, pngpath)
        println("wrote ", pngpath)
    end
end

println("wrote ", csvpath)
println("selected test cases:")
for c in selected
    @printf("  case %4d: %d modes, w=%7.4f m/s, Ncb=%9.3g cm^-3, activation=%6.3f\n",
            c.id, length(c.mode_N), c.w, ntot_cm3_cb(c), activation_fraction(c))
end

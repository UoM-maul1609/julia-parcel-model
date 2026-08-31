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

activation_profile(c::MLCase) = Float64[activation_fraction(c, i) for i in eachindex(c.height)]
activation_max(c::MLCase) = maximum(activation_profile(c))
ntot_cm3_ref(c::MLCase) = sum(c.mode_N) * dry_air_density(c.P0, c.T0, 1.0) / 1e6

_fmt_array(v; sigdigits=4) =
    "[" * join((string(round(Float64(x), sigdigits=sigdigits)) for x in v), ", ") * "]"

function case_plot_metadata(c::MLCase)
    rhod0 = dry_air_density(c.P0, c.T0, 1.0)
    Naer_cm3 = Float64.(c.mode_N) .* rhod0 ./ 1e6
    Daer_um = Float64.(c.mode_Dm) .* 1e6
    logsig = Float64.(c.mode_lnsig)
    kappa = Float64.(c.mode_kappa)
    Scrit_percent = Float64[
        100 * critical_supersaturation(c.mode_Dm[j], c.mode_kappa[j], c.T0)
        for j in eachindex(c.mode_N)
    ]
    qt0 = initial_total_water(c) * 1000

    line1 = @sprintf(
        "case %d | %d modes | w = %.4g m s⁻¹ | T₀ = %.2f K | P₀ = %.1f hPa | qtot₀ = %.4g g kg⁻¹ | Ntot = %.4g cm⁻³ | fact,max = %.3f",
        c.id, length(c.mode_N), c.w, c.T0, c.P0 / 100, qt0, sum(Naer_cm3), activation_max(c)
    )
    line2 = "N_aer = $(_fmt_array(Naer_cm3)) cm⁻³    d_aer = $(_fmt_array(Daer_um)) µm"
    line3 = "log_sig = $(_fmt_array(logsig))    κ = $(_fmt_array(kappa))    Scrit = $(_fmt_array(Scrit_percent)) %"
    string(line1, "\n", line2, "\n", line3)
end

function _closest_to_median_activation(g)
    a = activation_max.(g)
    target = median(a)
    g[argmin(abs.(a .- target))]
end

function select_cases(test, nplots)
    selected = MLCase[]
    for nm in sort(unique(length(c.mode_N) for c in test))
        g = [c for c in test if length(c.mode_N) == nm]
        isempty(g) || push!(selected, _closest_to_median_activation(g))
    end
    push!(selected, test[argmin(activation_max.(test))])
    push!(selected, test[argmax(activation_max.(test))])
    push!(selected, test[argmin(getfield.(test, :w))])
    push!(selected, test[argmax(getfield.(test, :w))])

    unique_selected = MLCase[]
    seen = Set{Int}()
    for c in selected
        if !(c.id in seen)
            push!(unique_selected, c); push!(seen, c.id)
        end
    end
    for c in sort(test; by=c -> c.id)
        length(unique_selected) >= nplots && break
        if !(c.id in seen)
            push!(unique_selected, c); push!(seen, c.id)
        end
    end
    unique_selected[1:min(nplots, length(unique_selected))]
end

models = Dict{String,Any}()
if kind in ("profile", "both")
    isfile("profile_surrogate.bson") || error("profile_surrogate.bson not found")
    p, m = load_profile_model("profile_surrogate.bson")
    models["profile"] = (p=p, m=m, predict=(c -> predict_profile(m, p, c)))
end
if kind in ("neuralode", "both")
    isfile("neuralode_surrogate.bson") || error("neuralode_surrogate.bson not found")
    p, m = load_neuralode_model("neuralode_surrogate.bson")
    models["neuralODE"] = (p=p, m=m, predict=(c -> predict_neuralode(m, p, c)))
end

cache = Dict{String,Dict{Int,Any}}()
for (name, obj) in models
    println("predicting $(length(test)) held-out trajectories with $name ...")
    pc = Dict{Int,Any}()
    for (k, c) in enumerate(test)
        pc[c.id] = obj.predict(c)
        (k == 1 || k % 10 == 0 || k == length(test)) && println("  $name: $k/$(length(test))")
    end
    cache[name] = pc
end

function metrics(subset, pcache)
    isempty(subset) && return nothing
    seS=0.0; seQ=0.0; seF=0.0; aeF=0.0; seN=0.0; seB=0.0
    seSmax=0.0; seQmax=0.0; seFmax=0.0; npt=0; violations=0
    for c in subset
        pred = pcache[c.id]
        ft = activation_profile(c)
        for i in eachindex(c.height)
            fp = pred.activation_fraction[i]
            seS += (100 * (pred.S[i] - c.S[i]))^2
            seQ += (1000 * (pred.ql[i] - c.ql[i]))^2
            seF += (fp - ft[i])^2
            aeF += abs(fp - ft[i])
            seN += (log10(1 + pred.Nd_kg[i]/1e6) - log10(1 + c.Nd_kg[i]/1e6))^2
            seB += (log10(1 + pred.beta_ext[i]/BETA_SCALE) - log10(1 + c.beta_ext[i]/BETA_SCALE))^2
            violations += (fp < -1e-7 || fp > 1 + 1e-7) ? 1 : 0
            npt += 1
        end
        seSmax += (100 * (maximum(pred.S) - maximum(c.S)))^2
        seQmax += (1000 * (maximum(pred.ql) - maximum(c.ql)))^2
        seFmax += (maximum(pred.activation_fraction) - maximum(ft))^2
    end
    (n=length(subset), S_rmse_percent=sqrt(seS/npt), ql_rmse_gkg=sqrt(seQ/npt),
     fact_rmse=sqrt(seF/npt), fact_mae=aeF/npt,
     Ndkg_log10_rmse=sqrt(seN/npt), beta_log10_rmse=sqrt(seB/npt),
     Smax_rmse_percent=sqrt(seSmax/length(subset)),
     qlmax_rmse_gkg=sqrt(seQmax/length(subset)),
     factmax_rmse=sqrt(seFmax/length(subset)), bound_violations=violations)
end

function print_breakdown(name, pcache)
    println("\n$name TEST metrics: ", metrics(test, pcache))
    println("$name by aerosol mode count:")
    for nm in sort(unique(length(c.mode_N) for c in test))
        g = [c for c in test if length(c.mode_N) == nm]
        println("  $nm modes: ", metrics(g, pcache))
    end
    regimes = [
        ("<1%", f -> f < 0.01), ("1-10%", f -> 0.01 <= f < 0.10),
        ("10-50%", f -> 0.10 <= f < 0.50), ("50-90%", f -> 0.50 <= f < 0.90),
        (">=90%", f -> f >= 0.90),
    ]
    println("$name by BMM maximum activated fraction:")
    for (label, accept) in regimes
        g = [c for c in test if accept(activation_max(c))]
        isempty(g) || println("  $(rpad(label,7)): ", metrics(g, pcache))
    end
end

for (name, pc) in cache
    print_breakdown(name, pc)
end

selected = select_cases(test, nplots)
outdir = "surrogate_comparison"
mkpath(outdir)

open(joinpath(outdir, "selected_cases.txt"), "w") do io
    for c in selected
        @printf(io, "case %d | %d modes | w=%.6g m/s | T=%.4f K | P=%.4f hPa | qtot0=%.6g g/kg | Nref=%.6g cm^-3 | factmax=%.6g\n",
                c.id, length(c.mode_N), c.w, c.T0, c.P0/100, 1000*initial_total_water(c), ntot_cm3_ref(c), activation_max(c))
        rhodref = dry_air_density(c.P0, c.T0, 1.0)
        for j in eachindex(c.mode_N)
            @printf(io, "  mode %d: Nref=%.6g cm^-3 Dm=%.6g nm lnsig=%.6g kappa=%.6g Scrit_med=%.6g %%\n",
                    j, c.mode_N[j]*rhodref/1e6, c.mode_Dm[j]*1e9, c.mode_lnsig[j],
                    c.mode_kappa[j], 100*critical_supersaturation(c.mode_Dm[j], c.mode_kappa[j], c.T0))
        end
        println(io)
    end
end

csvpath = joinpath(outdir, "comparison_metrics.csv")
open(csvpath, "w") do io
    println(io,
        "case,model,n_modes,w_m_s,T_K,P_hPa,qtot0_gkg,Ntot_cm3_ref,factmax_BMM," *
        "Smax_BMM_percent,Smax_pred_percent,qlmax_BMM_gkg,qlmax_pred_gkg,factmax_pred," *
        "Ndmax_BMM_cm3,Ndmax_pred_cm3,Ndfinal_BMM_cm3,Ndfinal_pred_cm3," *
        "betamax_BMM_m-1,betamax_pred_m-1")

    for c in selected
        z = Float64.(c.height)
        fref = activation_profile(c)
        metadata = case_plot_metadata(c)

        pS = plot(100 .* Float64.(c.S), z; label="BMM", xlabel="S (%)",
                  ylabel="height above initial level (m)", linewidth=2, legend=:best)
        pF = plot(fref, z; label="BMM", xlabel="activated fraction",
                  ylabel="height (m)", linewidth=2, legend=:best, xlims=(-0.02, 1.02))
        pN = plot(Float64.(c.Nd) ./ 1e6, z; label="BMM", xlabel="Nd (cm⁻³)",
                  ylabel="height (m)", linewidth=2, legend=:best)
        pB = plot(Float64.(c.beta_ext), z; label="BMM", xlabel="βext (m⁻¹)",
                  ylabel="height (m)", linewidth=2, legend=:best)
        pQ = plot(1000 .* Float64.(c.ql), z; label="BMM", xlabel="ql (g kg⁻¹)",
                  ylabel="height (m)", linewidth=2, legend=:best)
        pD = plot(1e6 .* Float64.(c.deff), z; label="BMM", xlabel="deff (µm)",
                  ylabel="height (m)", linewidth=2, legend=false)

        for name in sort(collect(keys(models)))
            pred = cache[name][c.id]
            style = name == "profile" ? :dash : :dot
            plot!(pS, 100 .* pred.S, pred.height; label=name, linestyle=style, linewidth=2)
            plot!(pF, pred.activation_fraction, pred.height; label=name, linestyle=style, linewidth=2)
            plot!(pN, pred.Nd ./ 1e6, pred.height; label=name, linestyle=style, linewidth=2)
            plot!(pB, pred.beta_ext, pred.height; label=name, linestyle=style, linewidth=2)
            plot!(pQ, 1000 .* pred.ql, pred.height; label=name, linestyle=style, linewidth=2)

            println(io, join((
                c.id, name, length(c.mode_N), c.w, c.T0, c.P0/100,
                1000*initial_total_water(c), ntot_cm3_ref(c), maximum(fref),
                100*maximum(c.S), 100*maximum(pred.S),
                1000*maximum(c.ql), 1000*maximum(pred.ql), maximum(pred.activation_fraction),
                maximum(c.Nd)/1e6, maximum(pred.Nd)/1e6,
                c.Nd[end]/1e6, pred.Nd[end]/1e6,
                maximum(c.beta_ext), maximum(pred.beta_ext)
            ), ","))
        end

        fig = plot(pS, pF, pN, pB, pQ, pD;
                   layout=(2,3), size=(1450,980), plot_title=metadata,
                   plot_titlefontsize=10, left_margin=5 * Plots.mm,
                   right_margin=5 * Plots.mm, top_margin=5 * Plots.mm,
                   bottom_margin=10 * Plots.mm, guidefontsize=10, tickfontsize=9)
        pngpath = joinpath(outdir, @sprintf("case_%04d.png", c.id))
        savefig(fig, pngpath)
        println("wrote ", pngpath)
    end
end

for name in sort(collect(keys(models)))
    pc = cache[name]
    sref = [100*maximum(c.S) for c in test]
    spred = [100*maximum(pc[c.id].S) for c in test]
    qref = [1000*maximum(c.ql) for c in test]
    qpred = [1000*maximum(pc[c.id].ql) for c in test]
    fref = [activation_max(c) for c in test]
    fpred = [maximum(pc[c.id].activation_fraction) for c in test]
    bref = [maximum(c.beta_ext) for c in test]
    bpred = [maximum(pc[c.id].beta_ext) for c in test]

    function scatter_one(x, y, xlabeltext, ylabeltext, ttl)
        lo = min(minimum(x), minimum(y)); hi = max(maximum(x), maximum(y))
        q = scatter(x, y; xlabel=xlabeltext, ylabel=ylabeltext, title=ttl,
                    label="test cases", markersize=4)
        plot!(q, [lo, hi], [lo, hi]; label="1:1", linestyle=:dash)
        q
    end

    a = scatter_one(sref, spred, "BMM Smax (%)", "$name Smax (%)", "maximum supersaturation")
    b = scatter_one(qref, qpred, "BMM ql,max (g kg⁻¹)", "$name ql,max (g kg⁻¹)", "maximum liquid water")
    cplot = scatter_one(fref, fpred, "BMM max activated fraction", "$name max activated fraction", "maximum activation")
    d = scatter_one(bref, bpred, "BMM max βext (m⁻¹)", "$name max βext (m⁻¹)", "maximum extinction")
    fig = plot(a, b, cplot, d; layout=(2,2), size=(1100,850), margin=4 * Plots.mm)
    savefig(fig, joinpath(outdir, "summary_scatter_$(lowercase(name)).png"))
end

println("\nwrote ", csvpath)
println("wrote ", joinpath(outdir, "selected_cases.txt"))
println("selected test cases:")
for c in selected
    @printf("  case %4d: %d modes, w=%7.4f m/s, Nref=%9.3g cm^-3, activation=%6.3f\n",
            c.id, length(c.mode_N), c.w, ntot_cm3_ref(c), activation_max(c))
end

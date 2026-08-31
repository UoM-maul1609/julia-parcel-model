"""
Read BMM training NetCDF files into trajectory-level objects for ML.

The ML representation is deliberately variable-length in aerosol mode count.
No padding is presented to the neural network: each case stores only its real
modes, and the set encoder handles 1, 2, 3, ... modes with shared weights.
"""
module SurrogateData

using NCDatasets
using Random

export MLCase, read_cases, split_cases, write_split_csv, effective_kappa, dry_air_density

# Same saturated/dry-air density convention used by the Julia BMM wrapper.
_svp_liq(T) = 100.0 * 6.1121 * exp((18.678 - (T - 273.15) / 234.5) *
                                    (T - 273.15) / (257.14 + (T - 273.15)))
function dry_air_density(p::Real, T::Real, rh::Real=1.0)
    Ra = 8.314 / 29e-3
    Rv = 8.314 / 18e-3
    eps = Ra / Rv
    es = _svp_liq(T)
    qv = eps * rh * es / (p - es)
    p / ((Ra + qv * Rv) * T)
end

struct MLCase
    id::Int
    height::Vector{Float32}
    mode_N::Vector{Float32}
    mode_Dm::Vector{Float32}
    mode_lnsig::Vector{Float32}
    mode_kappa::Vector{Float32}
    mode_nu::Vector{Float32}
    mode_molw::Vector{Float32}
    mode_density::Vector{Float32}
    T0::Float32
    P0::Float32
    w::Float32
    S::Vector{Float32}
    Nd::Vector{Float32}        # m^-3
    Nd_kg::Vector{Float32}     # kg^-1 dry air
    beta_ext::Vector{Float32}
    ql::Vector{Float32}
    deff::Vector{Float32}
    Dcrit::Matrix{Float32}   # height x real aerosol mode
end

function effective_kappa(nu::Real, molw::Real, density::Real; rho_water=1000.0, mw_water=18.01528e-3)
    Float32(nu * (density / rho_water) * (mw_water / molw))
end

function _finite_kappa(kappa, nu, molw, density)
    isfinite(kappa) ? Float32(kappa) : effective_kappa(nu, molw, density)
end

"""Read successful complete trajectories from a synthetic BMM NetCDF dataset."""
function read_cases(path::AbstractString)
    NCDataset(path) do ds
        height = Float32.(Array(ds["height"]))
        kappa_flag = haskey(ds.attrib, "kappa_flag") ? Int(ds.attrib["kappa_flag"]) : 1
        success = Int.(Array(ds["success"]))
        nmode = Int.(Array(ds["n_modes"]))
        T0 = Array(ds["T_cloud_base"])
        P0 = Array(ds["P_cloud_base"])
        w0 = Array(ds["w"])
        mode_N = Array(ds["mode_N"])
        mode_Dm = Array(ds["mode_Dm"])
        mode_lnsig = Array(ds["mode_lnsig"])
        mode_kappa = Array(ds["mode_kappa"])
        mode_nu = Array(ds["mode_nu"])
        mode_molw = Array(ds["mode_molw"])
        mode_density = Array(ds["mode_density"])
        S = Array(ds["S"])
        Nd = Array(ds["Nd"])
        Nd_kg = haskey(ds, "Nd_kg") ? Array(ds["Nd_kg"]) : nothing
        beta = Array(ds["beta_ext"])
        ql = Array(ds["ql"])
        deff = Array(ds["deff"])
        Dcrit = Array(ds["Dcrit_mode"])

        cases = MLCase[]
        for i in eachindex(success)
            success[i] == 1 || continue
            nm = nmode[i]
            nm > 0 || continue
            kappas = Float32[
                kappa_flag == 1 ? _finite_kappa(mode_kappa[i,j], mode_nu[i,j], mode_molw[i,j], mode_density[i,j]) :
                                  effective_kappa(mode_nu[i,j], mode_molw[i,j], mode_density[i,j])
                for j in 1:nm
            ]
            push!(cases, MLCase(
                i,
                copy(height),
                Float32.(vec(mode_N[i,1:nm])),
                Float32.(vec(mode_Dm[i,1:nm])),
                Float32.(vec(mode_lnsig[i,1:nm])),
                kappas,
                Float32.(vec(mode_nu[i,1:nm])),
                Float32.(vec(mode_molw[i,1:nm])),
                Float32.(vec(mode_density[i,1:nm])),
                Float32(T0[i]), Float32(P0[i]), Float32(w0[i]),
                Float32.(vec(S[i,:])),
                Float32.(vec(Nd[i,:])),
                Nd_kg === nothing ? Float32.(vec(Nd[i,:])) ./ Float32(dry_air_density(P0[i], T0[i], 1.0)) : Float32.(vec(Nd_kg[i,:])),
                Float32.(vec(beta[i,:])),
                Float32.(vec(ql[i,:])),
                Float32.(vec(deff[i,:])),
                Float32.(Dcrit[i,:,1:nm])
            ))
        end
        cases
    end
end

"""
Split by complete BMM trajectory, stratified by number of aerosol modes.
No height from a trajectory can leak into a different split.
"""
function split_cases(cases::Vector{MLCase}; seed::Int=20260831,
                     train_fraction::Float64=0.70, val_fraction::Float64=0.15)
    0 < train_fraction < 1 || throw(ArgumentError("train_fraction must be in (0,1)"))
    0 <= val_fraction < 1 || throw(ArgumentError("val_fraction must be in [0,1)"))
    train_fraction + val_fraction < 1 || throw(ArgumentError("train + validation fractions must be < 1"))
    rng = MersenneTwister(seed)
    groups = Dict{Int,Vector{MLCase}}()
    for c in cases
        push!(get!(groups, length(c.mode_N), MLCase[]), c)
    end

    train = MLCase[]; val = MLCase[]; test = MLCase[]
    for nm in sort(collect(keys(groups)))
        g = copy(groups[nm]); shuffle!(rng, g); n = length(g)
        if n == 1
            append!(train, g); continue
        end
        ntr = clamp(round(Int, train_fraction*n), 1, max(1,n-1))
        nval = n >= 3 ? clamp(round(Int, val_fraction*n), 1, n-ntr) : 0
        ntr + nval > n && (nval = max(0, n-ntr))
        append!(train, g[1:ntr])
        nval > 0 && append!(val, g[ntr+1:ntr+nval])
        ntr+nval < n && append!(test, g[ntr+nval+1:end])
    end
    shuffle!(rng, train); shuffle!(rng, val); shuffle!(rng, test)
    (train=train, val=val, test=test)
end

function write_split_csv(path::AbstractString, split)
    open(path, "w") do io
        println(io, "case,split,n_modes")
        for (name, cs) in (("train", split.train), ("validation", split.val), ("test", split.test))
            for c in cs
                println(io, "$(c.id),$name,$(length(c.mode_N))")
            end
        end
    end
    path
end

end # module

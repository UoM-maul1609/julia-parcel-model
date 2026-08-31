# write_case_namelist.jl

include("BMM.jl")
include("SurrogateData.jl")
using .BMM
using .SurrogateData
using NCDatasets
using Printf

path = length(ARGS) >= 1 ? ARGS[1] : "synthetic_bmm.nc"
case_id = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : error("need case index")
outroot = length(ARGS) >= 3 ? ARGS[3] : "/tmp/mccikpc2"

function reconstruct_bmm_case(path, case_id)
    NCDataset(path) do ds
        success = Int.(Array(ds["success"]))
        success[case_id] == 1 || error("case $case_id was not successful")

        nm = Int(Array(ds["n_modes"])[case_id])

        T0 = Float64(Array(ds["T_cloud_base"])[case_id])
        P0 = Float64(Array(ds["P_cloud_base"])[case_id])
        w0 = Float64(Array(ds["w"])[case_id])

        mode_N = Array(ds["mode_N"])
        mode_Dm = Array(ds["mode_Dm"])
        mode_lnsig = Array(ds["mode_lnsig"])
        mode_kappa = Array(ds["mode_kappa"])
        mode_nu = Array(ds["mode_nu"])
        mode_molw = Array(ds["mode_molw"])
        mode_density = Array(ds["mode_density"])

        modes = AerosolMode[
            AerosolMode(
                N       = Float64(mode_N[case_id,j]),
                Dm      = Float64(mode_Dm[case_id,j]),
                lnsig   = Float64(mode_lnsig[case_id,j]),
                kappa   = Float64(mode_kappa[case_id,j]),
                nu      = Float64(mode_nu[case_id,j]),
                molw    = Float64(mode_molw[case_id,j]),
                density = Float64(mode_density[case_id,j]),
            )
            for j in 1:nm
        ]

        height_top = haskey(ds.attrib, "height_top_m") ?
                     Float64(ds.attrib["height_top_m"]) : 500.0

        return cloud_base_case(
            winit = w0,
            height_top = height_top,
            tinit = T0,
            pinit = P0,
            rhinit = 0.95,
            modes = modes,
        )
    end
end

c = reconstruct_bmm_case(path, case_id)

casedir = joinpath(outroot, @sprintf("case_%04d", case_id))
mkpath(casedir)

namelist_path = joinpath(casedir, "namelist.in")
output_path = joinpath(casedir, "output.nc")

open(namelist_path, "w") do io
    write(io, to_namelist(c, output_path))
end

println(case_diagnostics(c; case_index=case_id))
println("wrote: $namelist_path")
println("BMM output will be: $output_path")


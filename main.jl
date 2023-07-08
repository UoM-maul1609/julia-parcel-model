#=
    Paul J. Connolly, The University of Manchester
    Julia Bin Microphysics Model (JBMM)
    Bin cloud parcel model based on BMM
=#
import YAML

# Read the namelist data
#namelist_data = YAML.load_file(ARGS[1])
namelist_data = YAML.load_file("namelist.yml")

# Import the driver
include("bin_microphysics_module.jl")
# use the module
using .bmm


#=
    Call the driver to solve
=#
initialise_bmm_arrays(namelist_data)


#=
    Call the driver to solve
=#
bmm_driver(namelist_data)


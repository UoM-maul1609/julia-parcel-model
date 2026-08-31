include("BMM.jl")
include("SyntheticDataset.jl")
using .BMM
using .SyntheticDataset

# Minimal CLI/environment interface so this is convenient on a workstation or HPC node.
# Positional args: [n_cases] [output.nc] [max_modes] [seed]
ncases = length(ARGS) >= 1 ? parse(Int, ARGS[1]) : parse(Int, get(ENV, "NCASES", "256"))
outfile = length(ARGS) >= 2 ? ARGS[2] : get(ENV, "BMM_DATASET", "synthetic_bmm.nc")
maxm = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : parse(Int, get(ENV, "MAX_MODES", "4"))
seed = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : parse(Int, get(ENV, "SEED", "20260831"))
exe = get(ENV, "BMM_EXE", BMM.default_bmm_exe())

cfg = SamplingConfig(n_cases=ncases, max_modes=maxm, seed=seed)
run_and_write_dataset(outfile; cfg=cfg, exe=exe)

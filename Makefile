BMM_DIR ?= bmm
JULIA ?= julia
NETCDF_FOR ?= $(shell nf-config --prefix 2>/dev/null)
NETCDF_C ?= $(shell nc-config --prefix 2>/dev/null)
BMM_DATASET ?= synthetic_bmm.nc
SURROGATE_SEED ?= 20260831
PROFILE_EPOCHS ?= 150
NEURALODE_EPOCHS ?= 40

.PHONY: bmm clean-bmm julia-instantiate julia-test smoke synthetic-pilot \
        analyse-dataset train-profile train-neuralode compare-surrogates

bmm:
	$(MAKE) -C $(BMM_DIR) NETCDF_FOR="$(NETCDF_FOR)" NETCDF_C="$(NETCDF_C)" main.exe

clean-bmm:
	$(MAKE) -C $(BMM_DIR) cleanall

julia-instantiate:
	cd julia && $(JULIA) --project=. -e 'using Pkg; Pkg.resolve(); Pkg.instantiate()'

julia-test:
	cd julia && $(JULIA) --project=. test/runtests.jl

smoke: bmm julia-instantiate
	cd julia && BMM_EXE=../$(BMM_DIR)/main.exe $(JULIA) --project=. smoke.jl

synthetic-pilot: bmm julia-instantiate
	cd julia && BMM_EXE=../$(BMM_DIR)/main.exe $(JULIA) --project=. -t auto generate_synthetic_dataset.jl 256 $(BMM_DATASET) 4 $(SURROGATE_SEED)

analyse-dataset: julia-instantiate
	cd julia && $(JULIA) --project=. analyse_dataset.jl $(BMM_DATASET)

train-profile: julia-instantiate
	cd julia && $(JULIA) --project=. train_surrogate.jl $(BMM_DATASET) profile $(PROFILE_EPOCHS) $(SURROGATE_SEED)

train-neuralode: julia-instantiate
	cd julia && $(JULIA) --project=. train_surrogate.jl $(BMM_DATASET) neuralode $(NEURALODE_EPOCHS) $(SURROGATE_SEED)

compare-surrogates: julia-instantiate
	cd julia && $(JULIA) --project=. compare_surrogates.jl $(BMM_DATASET) $(SURROGATE_SEED) both 8

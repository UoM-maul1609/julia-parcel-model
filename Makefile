BMM_DIR ?= bmm
JULIA ?= julia
NETCDF_FOR ?= $(shell nf-config --prefix 2>/dev/null)
NETCDF_C ?= $(shell nc-config --prefix 2>/dev/null)

.PHONY: bmm clean-bmm julia-instantiate julia-test smoke synthetic-pilot

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
	cd julia && BMM_EXE=../$(BMM_DIR)/main.exe $(JULIA) --project=. -t auto generate_synthetic_dataset.jl 256 synthetic_bmm.nc 4 20260831

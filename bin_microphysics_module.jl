module bmm
    using SpecialFunctions
    using Roots
    using Optim
    using NetCDF
    using OrdinaryDiffEq
    using ForwardDiff
    
    export bmm_driver, initialise_bmm_arrays
    
    const r_gas=8.314
    const molw_a=29.e-3
    const molw_water=18.e-3
    const cp1=1005.0
    const cpv=1870.0
    const cpw=4.27e3
    const cpi=2104.6 
    const grav=9.81 
    const lv=2.5e6
    const ls1=2.837e6
    const lf=ls1-lv
    const ttr=273.15
    const joules_in_an_erg=1.0e-7
    const joules_in_a_cal=4.187e0 
    const rhow=1000.
    const ra=r_gas/molw_a
    const rv=r_gas/molw_water  
    const eps1=ra/rv
    const rhoice=910.0
    const oneThird=1.0/3.0
    const oneSixth=1.0/6.0
    
    function bmm_driver(data)
        println("Running the driver...")
        
        # number of time-steps
        nt::Int16 = ceil(data["run_vars"]["runtime"] / bmm.dt)
        for i=1:nt
            # output to file
            output!(new_file,outputfile)
        
            #println("Time-step $i of $nt time in seconds is $(i * bmm.dt)")
            bin_microphysics()
        end
        println("done")
    end
    
    function initialise_bmm_arrays(data)
    
        #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            set variables and allocate arrays for parcel
           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        =#
        global z            =data["run_vars"]["zinit"]
        global p            =data["run_vars"]["pinit"]
        global t            =data["run_vars"]["tinit"]
        global w            =data["run_vars"]["winit"]
        global rh           =data["run_vars"]["rhinit"]
        global dt           =data["run_vars"]["dt"]
        global n_bins       = data["aerosol_setup"]["n_bins"]
        global n_modes      = data["aerosol_setup"]["n_mode"]
        global n_comps      = data["aerosol_setup"]["n_comps"]
        global n_intern     = data["aerosol_setup"]["n_intern"]
        global ice_flag     = data["run_vars"]["ice_flag"]
        global n_bin_modew  = n_bins * n_modes
        global n_bin_mode1  = (n_bins+1) * n_modes
        global n_bin_mode   = n_bins * n_modes*(1+ice_flag)
        global imoms        = ice_flag*5
        global dmina        = data["aerosol_spec"]["dmina"]
        global dmaxa        = data["aerosol_spec"]["dmaxa"]
        global density_core1= data["aerosol_spec"]["density_core1"]
        global nu_core1     = data["aerosol_spec"]["nu_core1"]
        global molw_core1   = data["aerosol_spec"]["molw_core1"]
        global kappa_core1  = data["aerosol_spec"]["kappa_core1"]
        global kappa_flag   = data["run_vars"]["kappa_flag"]
        global vent_flag    = data["run_vars"]["vent_flag"]
        global new_file     = [true]
        global outputfile   = data["run_vars"]["outputfile"]
        global alpha_cond   = data["run_vars"]["alpha_cond"]
        global alpha_therm  = data["run_vars"]["alpha_therm"]
        
        global d            = zeros(Float64,n_bin_mode1)
        global mbinedges    = zeros(Float64,n_bins+1,n_modes)
        global maer         = zeros(Float64,n_bin_modew)
        global npart        = zeros(Float64,n_bin_modew)
        global npartall     = zeros(Float64,n_bin_mode)
        global mbin         = zeros(Float64,n_bin_modew,n_comps+1)
        global mbinall      = zeros(Float64,n_bin_mode,n_comps+1)
        global rho_core     = zeros(Float64,n_modes)

        global momtemp      = zeros(Float64,n_bin_mode)
        global moments      = zeros(Float64,n_bin_mode,n_comps+imoms)
        global momenttype   = zeros(Int16,n_comps+imoms)

        global rhobin   = zeros(Float64,n_bin_modew,n_comps)
        global nubin    = zeros(Float64,n_bin_modew,n_comps)
        global molwbin  = zeros(Float64,n_bin_modew,n_comps)
        global kappabin = zeros(Float64,n_bin_modew,n_comps)

        global rh_eq    = zeros(Float64,n_bin_modew)
        global rhoat    = zeros(Float64,n_bin_modew)
        global dw       = zeros(Float64,n_bin_modew)
        global da_dt    = zeros(Float64,n_bin_modew)
        global ndrop    = zeros(Float64,n_bin_modew)

        global ecoal    = zeros(Float64,n_bin_mode,n_bin_mode)
        global ecoll    = zeros(Float64,n_bin_mode,n_bin_mode)
        global indexc   = zeros(Int32,n_bin_mode,n_bin_mode)
        global vel      = zeros(Float64,n_bin_mode)
        global nre      = zeros(Float64,n_bin_mode)
        global cd1      = zeros(Float64,n_bin_mode)
        global fv       = zeros(Float64,n_bin_modew)
        global fh       = zeros(Float64,n_bin_modew)
        
        global iz       = n_bin_modew+1
        global ipr      = n_bin_modew+2
        global itr      = n_bin_modew+3
        global irh      = n_bin_modew+4
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




        #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            calculate the density of aerosol particles within a mode
           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        =#
        r=size(data["aerosol_spec"]["mass_frac_aer1"])[1]
        c=size(data["aerosol_spec"]["mass_frac_aer1"][1])[1]
        global mass_frac_aer1=reshape(
            reduce(vcat,data["aerosol_spec"]["mass_frac_aer1"]),(c,r))
        for i=1:n_modes
            var1=sum(mass_frac_aer1[i,:] ./ 
                data["aerosol_spec"]["density_core1"])
            rho_core[i] = 1.0 ./ var1
        end
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            set-up size distribution
           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        =#
        r=size(data["aerosol_spec"]["n_aer1"])[1]
        c=size(data["aerosol_spec"]["n_aer1"][1])[1]
        global n_aer1=reshape(
            reduce(vcat,data["aerosol_spec"]["n_aer1"]),(c,r))
        global d_aer1=reshape(
            reduce(vcat,data["aerosol_spec"]["d_aer1"]),(c,r))
        global sig_aer1=reshape(
            reduce(vcat,data["aerosol_spec"]["sig_aer1"]),(c,r))
        for k=1:n_modes
            global idum=k
            num1=lognormal_n_between_limits(n_aer1[:,k],d_aer1[:,k],sig_aer1[:,k],
                n_intern,dmina,dmaxa)
            number_per_bin = num1 / n_bins
            npart[1+(k-1)*n_bins:k*n_bins] .= number_per_bin
            d[1+(k-1)*(n_bins+1)]=dmina # min diameter for this mode
            
            for i=1:n_bins
                global d_dummy = d[i+(k-1)*(n_bins+1)]
                global n_dummy = number_per_bin*(1.0-1.e-5)
                d[i+1+(k-1)*(n_bins+1)]=find_zero(find_upper_diameter, 
                    d[i+(k-1)*(n_bins+1)] )
            end
            
            d[k*(n_bins+1)]=dmaxa # max diameter for this mode
        end
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            aerosol mass - total
           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        =#
        for k=1:n_modes
            for j=1:n_bins
                i=j+(k-1)*(n_bins+1)
                maer[j+(k-1)*(n_bins)] = pi/6.0*(0.5*(d[i+1]+d[i]))^3*rho_core[k]
            end
        end
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            calculate the mass of each component in a bin, including water
           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        =#
        for i=1:n_bins
            for j=1:n_modes
                for k=1:n_comps
                    mbin[i+(j-1)*n_bins,k]=maer[i+(j-1)*n_bins]*mass_frac_aer1[j,k]
                    # density in each bin
                    rhobin[i+(j-1)*n_bins,k]=density_core1[k]
                    # nu in each bin
                    nubin[i+(j-1)*n_bins,k]=nu_core1[k]
                    # molw in each bin
                    molwbin[i+(j-1)*n_bins,k]=molw_core1[k]                    
                    # kappa in each bin
                    kappabin[i+(j-1)*n_bins,k]=kappa_core1[k]
                end
            end
        end
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        global qinit = rh*eps1*svp_liq(t)/(p-svp_liq(t))
        
        #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        put water on bin, using koehler equation
        note that julia lacks a switch statement
        =# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        global n_sel = 1
        global rh_act=0.0
        global mult=-1.0
        if(kappa_flag == 0) 
            for i=1:n_bin_modew
                n_sel = i;rh_act=0.0;mult=-1.0
                
                # find the peak of koehler curve: optimisation
                a=optimize(koehler02, [0.0],Optim.Options(x_abstol=1e0,f_abstol=1e-30))
                test1=Optim.minimizer(a)[1] #xmin
                
                rh_act=min(rh,0.999)
                mult=1.0
                # now optimise / root find
                d_dummy = find_zero(koehler02, (1.e-50,test1))*molw_water

                mbin[i,n_comps+1] = d_dummy 
            end
        elseif(kappa_flag == 1)
            for i=1:n_bin_modew
                n_sel = i;rh_act=0.0;mult=-1.0
                
                # find the peak of koehler curve: optimisation
                a=optimize(kkoehler02, [0.0],Optim.Options(x_abstol=1e0,f_abstol=1e-30))
                test1=Optim.minimizer(a)[1] #xmin
                
                rh_act=min(rh,0.999)
                mult=1.0
                # now optimise / root find
                d_dummy = find_zero(kkoehler02, (1.e-50,test1))*molw_water

                mbin[i,n_comps+1] = d_dummy 
            end
        
        else
            exit()
        end
        moments[:] .= 0.0
        for j=1:n_comps
            for i=1:n_bin_modew
                moments[i,j]=npart[i]*mbin[i,j] 
            end
        end
        momenttype[1:n_comps] .= 1
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!



        #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        liquid bins and variables
        =# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        global u0 = zeros(n_bin_modew+4)
        u0[1:n_bin_modew]=mbin[:,n_comps+1]
        u0[n_bin_modew+1:n_bin_modew+4]= [z,p,t,rh]
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

        
        #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        ice stuff now
        =# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        if ice_flag == 1
            global dice         = zeros(Float64,n_bin_mode1)
            global maerice      = zeros(Float64,n_bin_modew)
            global npartice     = zeros(Float64,n_bin_modew)
            global mbinice      = zeros(Float64,n_bin_modew,n_comps+1)
            global rho_coreice  = zeros(Float64,n_modes)

            global rhobinice    = zeros(Float64,n_bin_modew,n_comps)
            global nubinice     = zeros(Float64,n_bin_modew,n_comps)
            global molwbinice   = zeros(Float64,n_bin_modew,n_comps)
            global kappabinice  = zeros(Float64,n_bin_modew,n_comps)

            global rh_eqice     = zeros(Float64,n_bin_modew)
            global rhoatice     = zeros(Float64,n_bin_modew)
            global dwice        = zeros(Float64,n_bin_modew)
            global da_dtice     = zeros(Float64,n_bin_modew)
            global nice         = zeros(Float64,n_bin_modew)

            global phi          = zeros(Float64,n_bin_modew)
            global rhoi         = zeros(Float64,n_bin_modew)
            global nump         = zeros(Float64,n_bin_modew)
            global rime         = zeros(Float64,n_bin_modew)
            
            phi[:]  .= 1.0
            rhoi[:] .= rhoice
            nump[:] .= 1.0
            rime[:] .= 0.0
            
            rho_coreice[:]  = rho_core[:]
            npartice[:]     .= 0.0
            dice[:]         = d
            maerice[:]      = maer[:]
            mbinice[:,:]    = mbin[:,:]
            rhobinice[:,:]  = rhobin[:,:]
            nubinice[:,:]   = nubin[:,:]
            molwbinice[:,:] = molwbin[:,:]
            kappabinice[:,:]= kappabin[:,:]
            
            for j=1:n_comps
                for i=1:n_bin_modew
                    moments[i+n_bin_modew,j]=npartice[i]*mbinice[i,j] 
                end
            end
            
            # extra ice moments
            for i=1:n_bin_modew
                # ice moments: phi, nmon, vol, rim, unfr
                # phi: 1*n
                moments[i+n_bin_modew,n_comps+1]    = npartice[i]
                # nmon: 1*n
                moments[i+n_bin_modew,n_comps+2]    = npartice[i]
                # vol: mass/rho
                moments[i+n_bin_modew,n_comps+3]    = npartice[i]
                # rim: mass
                moments[i+n_bin_modew,n_comps+4]    = npart[i] * mbin[i,n_comps+1]
                # unf: mass
                moments[i+n_bin_modew,n_comps+5]    = npart[i] * mbin[i,n_comps+1]
            end
            momenttype[n_comps+1:n_comps+imoms]     = [2,2,1,1,1]
        end
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    end
    
    
    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        surface tension of water - pruppacher and klett
        in: t - temperature
        return: surface tension of water
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function surface_tension(t)	

        tc=t-ttr
        tc = max(tc,-40.)

        # pruppacher and klett pg 130 
        surface_tension = 75.93 + 0.115 * tc + 6.818e-2 * tc^2 + 
                          6.511e-3 * tc^3 + 2.933e-4 * tc^4 + 
                          6.283e-6 * tc^5 + 5.285e-8 * tc^6
        if(tc>=0.) 
            surface_tension = 76.1 - 0.155*tc
        end 
    
        surface_tension = surface_tension*joules_in_an_erg # convert to j/cm2 
        surface_tension = surface_tension*1.e4 # convert to j/m2 

    end 
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        koehler equation: equilibrium humidity over a particle
        in: t,mwat,mbin,rhobin,nubin,molwbin,,
        out: RH, rhoat, dw
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function koehler01!(t,mwat,mbin,rhobin,nubin,molwbin,rhoat,dw)
        
        nw      = mwat ./ molw_water
        rhoat[:]   = mwat ./ rhow .+ sum(mbin[:,1:n_comps] ./ rhobin[:,1:n_comps],dims=2)
        rhoat[:]   = (mwat .+ sum(mbin[:,1:n_comps],dims=2))./rhoat
        dw[:]      = ((mwat .+sum(mbin[:,1:n_comps],dims=2)) .* 6.0 ./(pi.*rhoat)).^(oneThird)
        
        # calculate surface tension of water
        sigma   = surface_tension(t)
        
        # equilibrium rh over particle - nb rh_act set to zero if not root-finding
        
        return exp.(4.0.*molw_water.*sigma./r_gas./t./rhoat./dw).* 
               (nw)./(nw.+sum(mbin[:,1:n_comps] ./ 
               molwbin[:,1:n_comps] .* 
               nubin[:,1:n_comps],dims=2) )
    end
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        kappa-koehler equation: equilibrium humidity over a particle
        in: t,mwat,mbin,rhobin,nubin,molwbin,,
        out: RH, rhoat, dw
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function kkoehler01!(t,mwat,mbin,rhobin,nubin,molwbin,rhoat,dw)
        
        nw      = mwat ./ molw_water
        rhoat[:]   = mwat ./ rhow .+ sum(mbin[:,1:n_comps] ./ rhobin[:,1:n_comps],dims=2)
        rhoat[:]   = (mwat .+ sum(mbin[:,1:n_comps],dims=2))./rhoat
        dw[:]      = ((mwat .+sum(mbin[:,1:n_comps],dims=2)) .* 6.0 ./(pi.*rhoat)).^(oneThird)
        
        # calculate surface tension of water
        sigma   = surface_tension(t)
        
        dd=(sum(mbin[:,1:n_comps] ./ rhobin[:,1:n_comps],dims=2).* 
          6.0./(pi)).^oneThird # dry diameter
                                  # needed for eqn 6, petters and kreidenweis (2007)

        kappa=sum(mbin[:,1:n_comps] ./ rhobin[:,1:n_comps] .*  
                kappabin[:,1:n_comps],dims=2) ./ 
                sum(mbin[:,1:n_comps] ./ rhobin[:,1:n_comps],dims=2)
               # equation 7, petters and kreidenweis (2007)

        # equilibrium rh over particle - nb rh_act set to zero if not root-finding
        return exp.(4.0 .* molw_water .* sigma ./ r_gas ./ t ./ rhoat ./dw) .* 
           (dw.^3-dd.^3) ./ (dw.^3-dd.^3 .* (1.0 .-kappa))
           # eq 6 petters and kreidenweis (acp, 2007)
    end
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    
    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        koehler equation: this is coded so it can be called with a root-finder, 
        to find the inverse
        in: moles of water
        out: RH
        note there are some dummy variables set within the module
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function koehler02(nw)
        
        massw   = nw[1]*molw_water
        rhoat   = massw/rhow + sum(mbin[n_sel,1:n_comps] / rhobin[n_sel,1:n_comps])
        rhoat   = (massw+maer[n_sel])/rhoat
        dw      = ((massw+maer[n_sel])*6.0/(pi*rhoat))^(oneThird)
        
        # calculate surface tension of water
        sigma   = surface_tension(t)
        
        # equilibrium rh over particle - nb rh_act set to zero if not root-finding
        
        return mult*(exp(4.0*molw_water*sigma/r_gas/t/rhoat/dw)* 
               (nw[1])/(nw[1]+sum(mbin[n_sel,1:n_comps]/ 
               molwbin[n_sel,1:n_comps] * 
               nubin[n_sel,1:n_comps]) ))-rh_act
    end
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        kappa-koehler equation: this is coded so it can be called with a root-finder, 
        to find the inverse
        in: moles of water
        out: RH
        note there are some dummy variables set within the module
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function kkoehler02(nw)
        
        massw   = nw[1]*molw_water
        rhoat   = massw/rhow + sum(mbin[n_sel,1:n_comps] / rhobin[n_sel,1:n_comps])
        rhoat   = (massw+maer[n_sel])/rhoat
        dw      = ((massw+maer[n_sel])*6.0/(pi*rhoat))^(oneThird)
        
        # calculate surface tension of water
        sigma   = surface_tension(t)
        
        dd=(sum(mbin[n_sel,1:n_comps] ./ rhobin[n_sel,1:n_comps])* 
          6.0/(pi))^oneThird # dry diameter
                                  # needed for eqn 6, petters and kreidenweis (2007)

        kappa=sum(mbin[n_sel,1:n_comps] ./ rhobin[n_sel,1:n_comps] .*  
                kappabin[n_sel,1:n_comps]) / 
                sum(mbin[n_sel,1:n_comps] ./ rhobin[n_sel,1:n_comps])
               # equation 7, petters and kreidenweis (2007)

        # equilibrium rh over particle - nb rh_act set to zero if not root-finding
        return mult*(exp(4.0*molw_water*sigma/r_gas/t/rhoat/dw)* 
           (dw^3-dd^3)/(dw^3-dd^3*(1.0-kappa)))-rh_act
           # eq 6 petters and kreidenweis (acp, 2007)
    end
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        koehler equation: this is coded so it can be called with a root-finder, 
        to find the inverse
        in: moles of water
        out: RH - but called via root-finder so mbin is returned
        note there are some dummy variables set within the module
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function koehler03(mbin)
        
        nw   = d_dummy*molw_water
        rhoat   = d_dummy/rhow + mbin[1]*sum(mass_frac_aer1[n_sel,1:n_comps] / 
                density_core1[1:n_comps])
        rhoat   = (d_dummy+mbin[1])/rhoat
        dw      = ((d_dummy+mbin[1])*6.0/(pi*rhoat))^(oneThird)
        
        # calculate surface tension of water
        sigma   = surface_tension(t)
        
        # equilibrium rh over particle - nb rh_act set to zero if not root-finding
        
        return mult*(exp(4.0*molw_water*sigma/r_gas/t/rhoat/dw)* 
               (nw)/(nw+mbin[1]*sum(mass_frac_aer1[n_sel,1:n_comps]/ 
               molw_core1[1:n_comps] * 
               nu_core1[1:n_comps]) ))-rh_act
    end
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        kappa-koehler equation: this is coded so it can be called with a root-finder, 
        to find the inverse
        in: moles of water
        out: RH - but called via root-finder so mbin is returned
        note there are some dummy variables set within the module
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function kkoehler03(mbin)
        
        nw   = d_dummy*molw_water
        rhoat   = d_dummy/rhow + mbin[1]*sum(mass_frac_aer1[n_sel,1:n_comps] / 
                density_core1[1:n_comps])
        rhoat   = (d_dummy+mbin[1])/rhoat
        dw      = ((d_dummy+mbin[1])*6.0/(pi*rhoat))^(oneThird)
        
        # calculate surface tension of water
        sigma   = surface_tension(t)
        
        dd=(sum(mbin[1]*mass_frac_aer1[n_sel,1:n_comps] ./ density_core1[n_sel,1:n_comps])* 
          6.0/(pi))^oneThird # dry diameter
                                  # needed for eqn 6, petters and kreidenweis (2007)

        kappa=sum(mbin[1]*mass_frac_aer1[n_sel,1:n_comps] ./ density_core1[1:n_comps] .*  
                kappa_core1[1:n_comps]) / 
                sum(mbin[1]*mass_frac_aer1[n_sel,1:n_comps] ./ density_core1[1:n_comps])
               # equation 7, petters and kreidenweis (2007)

        # equilibrium rh over particle - nb rh_act set to zero if not root-finding
        return mult*(exp(4.0*molw_water*sigma/r_gas/t/rhoat/dw)* 
           (dw^3-dd^3)/(dw^3-dd^3*(1.0-kappa)))-rh_act
           # eq 6 petters and kreidenweis (acp, 2007)
    end
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    
    
    
    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        integrate lognormal
        finds the number of aerosol particles between two limits
        n_aer1,d_aer1,sig_aer1 - the aerosol parameters
        n_intern - the number of modes - of the same composition
        dmin, dmax - the limits
        returns the number between limits
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function lognormal_n_between_limits(n_aer1,d_aer1,sig_aer1,n_intern,dmin,dmax)
        num1=0.0
        for i=1:n_intern
            num1=num1+n_aer1[i]*(0.5*erfc(-log(dmax/d_aer1[i])/sqrt(2.0)/sig_aer1[i] ) -
                0.5*erfc(-log(dmin/d_aer1[i])/sqrt(2.0)/sig_aer1[i] ) )
        end
        return num1
    end
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    
    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       calculate the upper diameter of the bin-edge
       in: x - dmax guess 
       return num_calc-num = 0 when root round
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function find_upper_diameter(x)
        num1=lognormal_n_between_limits(n_aer1[:,idum],d_aer1[:,idum],sig_aer1[:,idum],
            n_intern,d_dummy,x)

        return num1-n_dummy
    end
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       saturation vapour pressure over liquid
       in: t - temperature 
       return saturation vapour pressure over liquid water
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function svp_liq(t)

        return 611.21*exp((18.678-(t-ttr)/234.5)*(t-ttr)/(257.14+(t-ttr)))
    end
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       saturation vapour pressure over ice
       in: t - temperature 
       return saturation vapour pressure over ice
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function svp_ice(t)

        return 611.15*exp((23.036-(t-ttr)/333.7)*(t-ttr)/(279.82+(t-ttr)))
    end
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       diffusivity of water vapour in air
       in: t, p
       out: dd - diffusivity of water vapour in air
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function dd1(t,p)
        t1 = max(t,200.0)
        return 2.11e-5.*(t1./ttr).^1.94.*(101325.0/p)
    end
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       thermal conductivity of air
       in: t
       out: ka - thermal conductivity of air
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function ka(t)
        t1=max(t,200.0)
        return (5.69+0.017.*(t1-ttr)).*1.e-3 .*joules_in_a_cal
    end
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    
    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       viscosity of air
       in: t
       out: viscosity_air - viscosity of air
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function viscosity_air(t)
        tc = t-ttr
        tc = max(tc,-200.0)

        if( tc >= 0.0) 
            return (1.718+0.0049*tc) * 1e-5 # the 1d-5 converts from poise to si units
        else
            return (1.718+0.0049*tc-1.2e-5*tc^2) * 1e-5
        end 
    end 
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       calculate the wet diameter
       in: mwat,mbin
       inout: rhobin,dw
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function wetdiam!(mwat,mbin,rhobin,dw)
        
        # calculate the diameter and radius
        rhoat[:] = mwat[:] ./ rhow .+ sum(mbin[:,1:n_comps] ./ rhobin[:,1:n_comps],dims=2)
        rhoat[:] = (mwat[:] .+ sum(mbin[:,1:n_comps],dims=2)) ./ rhoat[:]
        
        # wet diameter
        dw[:] = ((mwat[:] .+ sum(mbin[:,1:n_comps],dims=2)) .*6.0 ./(pi .*rhoat[:])).^oneThird
    end 
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       calculate the terminal velocity of cloud drops
       see pruppacher and klett
       inout: vel, nre, cd: terminal velocity, reynolds number and drag coefficeint
       in: diam, rhoat, t, p: diameter, density, temperature and pressure
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function terminal01!(vel,diam,rhoat,t,p,nre,cd1)
        tc=t-ttr
        vel[:] .= 0.0 # zero array
        rhoa    = p / (ra * t) # density of air
        diam2   =diam # temporary array that can be changed

        eta1    = viscosity_air(t)

        nre[:] .= 0.0 # zero array
        ind     = findall(diam2 .> 7000.e-6)
        diam2[ind] .= 7000.e-6
        mass    =pi/6.0.*diam2 .^3 .* rhoat
    
        sigma = surface_tension(t)
        
        # regime 3:  eqns 5-12, 10-146 & 10-148 from p & k 
        physnum = (sigma^3.0) * (rhoa^2) / ((eta1^4.0) * grav * (rhow - rhoa))		
        phys6 = physnum^(oneSixth)
        ind=findall(diam2 .> 1070.e-6) 
        bondnum = (4.0 .* oneThird) .* grav .* (rhow - rhoa) .* (diam2[ind].^2) ./ sigma

        x = log.(bondnum .* phys6)
        y = -5.00015 .+ 5.23778 .* x .- 2.04914 .* x .* x .+ 0.475294 .* (x .^ 3) 
            .- 0.542819e-1 .* (x.^4.0) .+ 0.238449e-2 .* (x.^5)

        nre[ind] = phys6 .* exp.(y)

        vel[ind] = eta1 * (nre[ind])/ (rhoa * diam2[ind])

        cd1[ind] = 8.0 .* mass[ind] .* grav .* rhoa./(pi .* ((diam2[ind] .* 0.5).* eta1).^2)
        cd1[ind] = cd1[ind]	./ (nre[ind].^2) 





        # regime 2:  eqns 10-142, 10-145 & 10-146 from p & k 
        ind=findall((diam2 .<= 1070.e-6) .& (diam2 .> 20.e-6))
        bestnm = 32.0 .* ((diam2[ind] .*0.5).^3) .* (rhow - rhoa) .* rhoa * 
                  grav ./ (3.0 .* eta1.^2)
        x = log.(bestnm)
        y = -3.18657 .+ 0.992696 .* x .- 0.153193e-2 .* x .* x 
            .- 0.987059e-3 .* (x.^3) .- 0.578878e-3 .* (x.^4) 
            .+ 0.855176e-4 .* (x.^5) .- 0.327815e-5 .* (x.^6)
        nre[ind] =  exp.(y)
        vel[ind] = eta1 .* nre[ind] ./ (2.0 .* rhoa .* (diam2[ind] .* 0.5))
        cd1[ind] = bestnm ./(nre[ind].^2)



        # regime 1:  eqns 10-138, 10-139 & 10-140 from p & k 
        mfpath = 6.6e-8 * (101325.0 / p) * (t / 293.15)
        ind=findall(diam2 .<= 20.e-6) 
        vel[ind] = 2.0 .* ((diam2[ind] .*0.5).^2) .* grav .* (rhow - rhoa) ./ (9.0 .* eta1)
        vel[ind] = vel[ind] .* (1.0 .+ 1.26 .* mfpath ./ (diam2[ind] .* 0.5))
        nre[ind] = vel[ind] .* rhoa .* diam2[ind] ./ eta1

        cd1[ind] = 8.0 .* mass[ind] .* grav .* rhoa./(pi .* ((diam2[ind] ./ 2.0).* eta1).^2)
        cd1[ind] = cd1[ind]	./ (nre[ind].^2) 


        ind=findall(isnan.(vel))
        vel[ind] .=0.0
    end 
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       ventilation factor for water drops
       inout: fv,fh: ventilation factors for vapour and heat
       inout: vel,nre,cd1: vel, reynolds, drag coeffient
       in: diam, rhoat, t, p: diameter, density, temperature and pressure
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function ventilation01!(vel,nre,cd1,diam,rhoat,t,p,fv,fh)
        # density of air
        rhoa = p/ra/t
        # diffusivity of water vapour in air
        d1 = dd1(t,p)
        # conductivity of air
        k1 = ka(t)
        # viscosity of air
        eta1=viscosity_air(t)
        # kinematic viscosity
        nu = eta1 / rhoa
        # schmitt numbers:
        nsc1 = nu / d1
        nsc2 = nu / k1

        # terminal velocity of water drops
        terminal01!(vel,diam,rhoat, t,p,nre,cd1)

        # mass ventilation - use dv+++++++++
        calc = (nsc1.^(oneThird)) .* sqrt.(nre)
        ind=findall(calc .> 51.4)
        calc[ind] .= 51.4

        ind=findall(calc .< 1.4)
        fv[ind]=1.00 .+0.108.*calc[ind].^2
        ind=findall(calc .>= 1.4)
        fv[ind]=0.78 .+0.308.*calc[ind]
        #-----------------------------------

        # heat ventilation - use ka---------
        calc = (nsc2.^(oneThird)) .* sqrt.(nre)
        ind=findall(calc .> 51.4)
        calc[ind] .=51.4

        ind=findall(calc .< 1.4)
        fh[ind]=1.00 .+0.108 .*calc[ind].^2
        fh[ind]=0.78 .+0.308 .*calc[ind]
        #-----------------------------------
    end 
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       calculate growth rate of a cloud droplet by diffusional growth
       inout: vel, nre, cd1, fv, fh
       in: t,p, rh, rh_eq
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function dropgrowthrate01!(vel,nre,cd1,diam,rhoat,t,p,fv,fh,rh,rh_eq)
        rad=diam .* 0.5
        # density of air
        rhoa=p/ra/t
        # diffusivity of water vapour in air
        d1=dd1(t,p)
        # thermal conductivity of air
        k1=ka(t)
        # ventilation coefficient
        fv[:] .= 1.0
        fh[:] .= 1.0
        if(vent_flag == 1)
            ventilation01!(vel,nre,cd1,diam,rhoat,t,p,fv,fh)
        end

        # modify diffusivity and conductivity
        dstar =d1.*fv./(rad./(rad.+0.7*8.e-8).+d1.*fv./rad./alpha_cond.*sqrt(2.0*pi/rv/t))
        kstar =k1.*fh./(rad./(rad.+2.16e-7).+
            k1.*fh./rad./alpha_therm./cp1./rhoa.*sqrt(2.0*pi/ra/t))

        # 455 jacobson and 511 pruppacher and klett
        dropgrowthrate01=dstar.*lv.*rh_eq.*svp_liq(t).* 
                       rhoat./kstar./t.*(lv.*molw_water./t./r_gas.-1.0) 
        dropgrowthrate01=dropgrowthrate01.+rhoat*r_gas*t/molw_water  
        dropgrowthrate01=dstar.*(rh.-rh_eq).*svp_liq(t)./rad./dropgrowthrate01

        return dropgrowthrate01
    end 
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    
    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       fparcelwarm
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function fparcelwarm!(ydot,y,p1,t)
        # local variables
        wv=0.0;wl=0.0;wi=0.0;drv=0.0;dri=0.0;dri2=0.0;

        # initialise and assign vars
        ydot[:] .= 0.0
        p = y[ipr]
        t = y[itr]
        rh=y[irh]
        
        # check there are no negative values
        ind=findall(y[1:n_bin_modew] .<= 0.0)
        y[ind] .= 1.e-22
        
        # calculate mixing ratios from rh, etc
        sl=svp_liq(t)*rh/(p-svp_liq(t)) # saturation ratio
        sl=(sl*p/(1.0+sl))/svp_liq(t)
        wv=eps1*rh*svp_liq(t) / (p-svp_liq(t))  # vapour mixing ratio
        wl=sum(npart .* y[1:n_bin_modew])       # liquid mixing ratio
        
        # calculate the moist gas constants and specific heats
        rm1 = ra +wv*rv
        cpm=cp1+wv*cpv+wl*cpw+wi*cpi
        
        # now calculate derivatives
        # adiabatic parcel model
        ydot[iz] = w                        # vertical wind
        ydot[ipr] = -p/rm1/t*grav*ydot[iz]  # hydrostatic equation

        # calculate the equilibrium rhs
        if kappa_flag==0
            rh_eq[:]=koehler01!(t,y[1:n_bin_modew],mbin,rhobin,nubin,molwbin,rhoat,dw)
        elseif kappa_flag==1
            rh_eq[:]=kkoehler01!(t,y[1:n_bin_modew],mbin,rhobin,nubin,molwbin,rhoat,dw)
        else
            println("Whoops!")
            exit()
        end
        
        # particle growth rate - radius growth rate
        da_dt=dropgrowthrate01!(vel[1:n_bin_modew],nre[1:n_bin_modew],cd1[1:n_bin_modew],
            dw[1:n_bin_modew],rhoat[1:n_bin_modew],t,p,fv,fh,rh,rh_eq)
        # do not bother if number concentration too small
        ind=findall( isnan.(da_dt) .| (npart[:] .<= 1.e-9))    
        da_dt[ind] .= 0.0
        
        # mass growth rate
        ydot[1:n_bin_modew] = pi .* rhoat .* dw.^2 .*da_dt
        # change in vapour content
        drv = -sum(ydot[1:n_bin_modew] .* npart[:])
        
        #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            change in temperature of the parcel
           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        =#
        ydot[itr] = rm1/p*ydot[ipr]*t/cpm  # temperature change: expansion
        ydot[itr] = ydot[itr]-lv/cpm*drv   # temperature change: condensation
        #  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
        #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            change in rh of the parcel
           !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        =#
        ydot[irh] = (p-svp_liq(t))*svp_liq(t)*drv
        ydot[irh] = ydot[irh]+svp_liq(t)*wv*ydot[ipr]
        ydot[irh] = ydot[irh]-wv*p*ydot[itr]*ForwardDiff.derivative(svp_liq,t)
        ydot[irh] = ydot[irh] / (eps1*svp_liq(t)^2)
        #  !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
    end 
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    
    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       bin_microphysics model, solves the ode over one time-step
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function bin_microphysics()

        tspan=(0,dt)
        prob = ODEProblem(fparcelwarm!, u0, tspan)
        sol = solve(prob, FBDF(autodiff=false), abstol=1.e-2,reltol=1.e-15)
        u0[:]=sol.u[end]
        
        global p = u0[ipr]
        global t = u0[itr]
        global rh= u0[irh]
        global z = u0[iz]
        
    end
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    
    
    
    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
       output 1 time-step of the model
       in: new_file, outputfile
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function output!(new_file,outputfile)

        if new_file[1]
            if isfile(outputfile)
                rm(outputfile)
            end
            new_file[1] = false


            tdim = NcDim("times",0,unlimited=true)
            xdim = NcDim("nbins",n_bins)
            ydim = NcDim("nmodes",n_modes)
            zdim = NcDim("ncomps",n_comps)

            #uvar = NcVar("u",[xdim,ydim,tdim],t=Float32)
            # define variable: time
            tvar = NcVar("time",tdim,t=Float32)
            tvar.atts = Dict("units" => "seconds")
            
            # define variable: z
            zvar = NcVar("z",tdim,t=Float32)
            zvar.atts = Dict("units" => "m")
            
            # define variable: p
            pvar = NcVar("p",tdim,t=Float32)
            pvar.atts = Dict("units" => "Pa")
            
            # define variable: t
            ttvar = NcVar("t",tdim,t=Float32)
            ttvar.atts = Dict("units" => "K")
            
            # define variable: rh
            rhvar = NcVar("rh",tdim,t=Float32)
            rhvar.atts = Dict("units" => "n/a")
            
            # define variable: w
            wvar = NcVar("w",tdim,t=Float32)
            wvar.atts = Dict("units" => "m s-1")

            # fn=tempname()
            global ncu = NetCDF.create(outputfile,
                [tvar,zvar,pvar,ttvar,rhvar,wvar],mode=NC_NETCDF4)

            #NetCDF.sync(ncu)
            global io_count=1
        end

        NetCDF.putvar(ncu,"time",[bmm.dt],start=[io_count]) #Write the time
        NetCDF.putvar(ncu,"z",[bmm.z],start=[io_count])
        NetCDF.putvar(ncu,"p",[bmm.p],start=[io_count])
        NetCDF.putvar(ncu,"t",[bmm.t],start=[io_count])
        NetCDF.putvar(ncu,"rh",[bmm.rh],start=[io_count])
        NetCDF.putvar(ncu,"w",[bmm.w],start=[io_count])
        # NetCDF.sync(ncu)
        io_count += 1
    end
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        
end

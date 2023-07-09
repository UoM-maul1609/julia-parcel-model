module bmm
    using SpecialFunctions
    using Roots
    using Optim

    
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
    
    function bmm_driver(data)
        println("Running the driver...")
        
        # number of time-steps
        nt::Int16 = ceil(data["run_vars"]["runtime"] / bmm.dt)
        for i=1:nt
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
        global momenttype   = zeros(Float64,n_comps+imoms)

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
        global indexc   = zeros(Float64,n_bin_mode,n_bin_mode)
        global vel      = zeros(Float64,n_bin_mode,n_bin_mode)
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
    function koehler01(t,mwat,mbin,rhobin,nubin,molwbin)
        
        nw      = mwat ./ molw_water
        rhoat   = mwat ./ rhow .+ sum(mbin[:,1:n_comps] ./ rhobin[:,1:n_comps],2)
        rhoat   = (mwat .+ sum(mbin[:,1:n_comps],2))./rhoat
        dw      = ((mwat .+sum(mbin[:,1:n_comps],2)) .* 6.0 ./(pi.*rhoat)).^(oneThird)
        
        # calculate surface tension of water
        sigma   = surface_tension(t)
        
        # equilibrium rh over particle - nb rh_act set to zero if not root-finding
        
        return exp(4.0*molw_water*sigma/r_gas/t/rhoat/dw)* 
               (nw)/(nw+sum(mbin[:,1:n_comps]/ 
               molwbin[:,1:n_comps] * 
               nubin[:,1:n_comps],2) )
    end
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    #= !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        kappa-koehler equation: equilibrium humidity over a particle
        in: t,mwat,mbin,rhobin,nubin,molwbin,,
        out: RH, rhoat, dw
       !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    =#
    function kkoehler01(t,mwat,mbin,rhobin,nubin,molwbin)
        
        nw      = mwat ./ molw_water
        rhoat   = mwat ./ rhow .+ sum(mbin[:,1:n_comps] ./ rhobin[:,1:n_comps],2)
        rhoat   = (mwat .+ sum(mbin[:,1:n_comps],2))./rhoat
        dw      = ((mwat .+sum(mbin[:,1:n_comps],2)) .* 6.0 ./(pi.*rhoat)).^(oneThird)
        
        # calculate surface tension of water
        sigma   = surface_tension(t)
        
        dd=(sum(mbin[:,1:n_comps] ./ rhobin[:,1:n_comps],2)* 
          6.0/(pi))^oneThird # dry diameter
                                  # needed for eqn 6, petters and kreidenweis (2007)

        kappa=sum(mbin[:,1:n_comps] ./ rhobin[:,1:n_comps] .*  
                kappabin[:,1:n_comps],2) / 
                sum(mbin[:,1:n_comps] ./ rhobin[:,1:n_comps],2)
               # equation 7, petters and kreidenweis (2007)

        # equilibrium rh over particle - nb rh_act set to zero if not root-finding
        return exp(4.0 .* molw_water .* sigma ./ r_gas ./ t ./ rhoat ./dw) .* 
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
    function kkoehler03(nw)
        
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
    
    
    function bin_microphysics()
        #println(myvariable)
        #println(dt)
    end
    
end

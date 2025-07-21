import numpy as np
from scipy import optimize
from scipy import integrate

#########################################################################
#    SMSlightcurve_CSMdisk class                                        #
#########################################################################
from matplotlib import pyplot as plt
def dbprint(var):
    print(f'{var = }') # prints name of variable an its value directly
    #print(var + "=" + repr(eval(var)))

class SMSlightcurve_CSMdisk:
    # member variables:
    debug = False
    # physical light curve model parameters:
    Eexp = 0. # explosion energy [erg/s]
    M0 = 0. # envelope mass [gram]
    R0 = 0. # initial radius [cm]
    kappa = 0.34 # opacity [cm^2/gram]
    Tion = 6000. # ionization temperature [Kevin]
    t_ion = 1e20 # ionization time [seconds]
    t_plateau = 0. # length of the plateau phase [seconds]
    t_plateau_peak = 0. # length of the phase where luminosity is within 10% of peak plateau luminosity
    f_Omega = 0.1 # fraction that the CSM disk takes up of the full solid angle (4pi)
    heating_model = "none" # can be "const", "CSM_disk"
    const_heating_rate = 0.
    Nickel_mass = 0.0 # in gram
    CSM_velocity = -1. # in cm/s
    CSM_accretion_rate = 0.0 # in gram/s

    # natural constants:
    sigma = 5.670374419e-5 # Stefan Boltzmann constant in cgs units [erg/s / cm^2 / K^4]
    c0 = 2.998e10 # speed of light [cm/s]
    m_Hydrogen = 1.6738e-24 # mean molecular wheight of atomic hydrogen [gram]
    G = 6.674e-8 # gravitational constant [cm^3/g/s^2]
    day = 60*60*24. # seconds in a day
    year = 365.25*day

    # containers for outputs over time
    time_arr = np.array(0.)
    Lbol_arr = np.array(0.)
    Temp_eff_arr = np.array(0.)
    Renvelope_arr = np.array(0.)
    Rphotosphere_arr = np.array(0.)
    vexp_arr = np.array(0.)
    Eint_arr = np.array(0.)
    Ekin_arr = np.array(0.)
    Heating_arr = np.array(0.)
    Lforwardshock_arr = np.array(0.)
    Lreverseshock_arr = np.array(0.)
    Rshock_arr = np.array(0.)
    vshock_arr = np.array(0.)
    Mshock_arr = np.array(0.)
    
    # integrator setings
    max_integration_time = 1000.* day
    include_acceleration_phase_initial = 1 # turns on/off acceleration due to radiation pressure
    include_acceleration_phase_plateau = 1 # turns on/off acceleration due to radiation pressure during the plateau phase (after t_ion)

    # constructor:
    def __init__(self, Eexp_in, M0_in, R0_in, Tion_in, kappa_in, include_acceleration_phase_initial_in, include_acceleration_phase_plateau_in, heating_model_in, const_heating_rate_in=0.0, Nickel_mass_in = 0.0, CSM_velocity_in = -1., CSM_accretion_rate_in = 0.0):
        self.Eexp = Eexp_in
        self.M0 = M0_in
        self.R0 = R0_in
        self.Tion = Tion_in
        self.kappa = kappa_in
        self.heating_model = heating_model_in
        self.const_heating_rate = const_heating_rate_in
        self.Nickel_mass = Nickel_mass_in
        self.include_acceleration_phase_initial = include_acceleration_phase_initial_in
        self.include_acceleration_phase_plateau = include_acceleration_phase_plateau_in
        self.CSM_velocity = CSM_velocity_in
        self.CSM_accretion_rate = CSM_accretion_rate_in
    
    #---------------------------------------------------------
    # physical model function for the integrator:
    def heating_function(self, t, y):
        if t < self.t_ion:
            R = y[0]; vexp = y[1]; Eint = y[2]; Rsh = y[3]; vsh = y[4]; Msh = y[5]; Rion = R
        else:
            R = y[0]; vexp = y[1]; Rion = y[2]; Rsh = y[3]; vsh = y[4]; Msh = y[5]
        ###
        if self.heating_model == "const":
            return self.const_heating_rate, 0, 0 # heating rate in erg/s
        elif self.heating_model == "nuclear_decay":
            eps_Ni = 3.9e10; eps_Co = 6.8e9; t_Ni = 8.8*self.day; t_Co = 113.3*self.day; gammaray_opacity_term =1.0; kappa_gammaray = 0.03
            if t > self.t_ion:
                tau = kappa_gammaray* 3.*self.M0 / (4.*np.pi* R**3)
                gammaray_opacity_term = 1. - np.exp(-Rion*tau)
            return (  self.Nickel_mass * ( (eps_Ni-eps_Co)*np.exp(-t/t_Ni) + eps_Co*np.exp(-t/t_Co) ) * gammaray_opacity_term  ), 0, 0
        elif (self.heating_model == "CSM_disk_Matsumoto") or (self.heating_model == "CSM_disk_Larson"):
            gamma = 5./3.; rho_SN = 3.*self.M0 / (4.*np.pi* R**3); rho_CSM, vCSM = self.get_CSM_Properies(t, Rsh)
            L_ForwardShock = 8.*np.pi*self.f_Omega* Rsh**2 * max(vsh - vCSM, 0.)**3 * rho_CSM / (gamma*gamma -1.)
            L_ReverseShock = 8.*np.pi*self.f_Omega* Rsh**2 * max(vexp*(Rsh/R) - vsh, 0.)**3 * rho_SN / (gamma*gamma -1.)
            #L_ForwardShock = self.CSM_accretion_rate * vsh**3 / abs(vCSM)
            #L_ForwardShock = 4.*np.pi*self.f_Omega* Rsh**2 * (vsh)**3 * rho_CSM / abs(vCSM)
            if (Rsh > Rion): return 0.,L_ForwardShock, L_ReverseShock # no heating if shock is outside the envelope, but still save the shock luminosities for later analysis
            return (L_ForwardShock), L_ForwardShock, L_ReverseShock #(L_ForwardShock + L_ReverseShock), L_ForwardShock, L_ReverseShock
        else:
            print('No valid heating model was chosen! Valid models are: "const", "nuclear_decay", "CSM_disk_Matsumoto", "CSM_disk_Larson".\nFor no heating, use heating_model="const" and set const_heating_rate=0.0')
            exit()
    
    def get_CSM_Properies(self, t, r):
        # Matsumotos CSM description:
        if (self.heating_model == "CSM_disk_Matsumoto"):
            density = self.CSM_accretion_rate / (4.*np.pi*self.f_Omega *abs(self.CSM_velocity) * r*r)
            velocity = self.CSM_velocity
            return density, velocity
        elif (self.heating_model == "CSM_disk_Larson"):
            # Larsons isothermal collapse solution, see https://inspirehep.net/literature/57090 :
            T_CSM = 8000. # Kelvin
            R_gas = 4.733e7 # specificgas constant for primordial gas in erg/g/K
            eta0 = 8.86; xi0 = 3.28 # numerical factors from Larson
            density = eta0 * R_gas*T_CSM / (4.*np.pi* self.f_Omega * self.G * r*r)
            velocity = - np.sqrt(R_gas*T_CSM) * xi0
            return density, velocity
        else:
            return 0.0, 0.0 # no CSM heating model

    #---------------------------------------------------------
    # integrator dy_dt ODE function derivatives:
    @staticmethod # use this so that we can use solveivp integrator. Note we just pass 'self' as an extra argument
    def dy_dt_phase_initial(t, y, self):
        # for better readability of the equations
        R = y[0]; vexp = y[1]; Eint = y[2]; Rsh = y[3]; vsh = y[4]; Msh = y[5]
        # ODEs dydt:
        dR = vexp
        #
        dvexp = 5.*Eint / (3. * self.M0 * R) * self.include_acceleration_phase_initial # option to toggle accceleration
        #
        L = 4.*np.pi* self.c0 *R*Eint / (3.* self.kappa * self.M0)
        H,_,_ = self.heating_function(t, y)
        dEint = -(dR / R) *Eint - L + H
        #
        dRsh = vsh
        #
        rho_SN = 3.* self.M0 / (4.*np.pi* R**3)
        vRsh = vexp * (Rsh / R)
        rho_CSM, vCSM = self.get_CSM_Properies(t, Rsh)
        #max(vRsh - vsh, 0.)
        dvsh = 4.*np.pi* self.f_Omega * Rsh**2 * rho_SN * max(vRsh - vsh, 0.)**2 / Msh  -  4.*np.pi* self.f_Omega * Rsh**2 * rho_CSM * max(vsh - vCSM, 0.)**2 / Msh
        #
        dMsh = 4.*np.pi* self.f_Omega * Rsh**2 * rho_SN * max(vRsh - vsh, 0.)  +  4.*np.pi* self.f_Omega * Rsh**2 * rho_CSM * max(vsh - vCSM, 0.)
        #
        #print(R, vexp, Eint, Rsh, vsh, Msh)
        return [dR, dvexp, dEint, dRsh, dvsh, dMsh]


    @staticmethod
    def dy_dt_phase_plateau(t, y, self):
        # for better readability of the equations
        R = y[0]; vexp = y[1]; Rion = y[2]; Rsh = y[3]; vsh = y[4]; Msh = y[5]
        # ODEs dydt:
        dR = vexp #vRion * (R / Rion) # dR == vexp
        #
        dvexp = 5.*( (self.sigma*self.kappa* self.Tion**4/self.c0)* Rion**4 / (R**4) ) * self.include_acceleration_phase_plateau # !!! assumig acceleration of the whole envelope with internal energy only from r<Rion: dvexp = 5.*Eint / (3. * self.M0 * R)
        #dvexp = 5.*( (self.sigma*self.kappa* self.Tion**4/self.c0/dR)* Rion**3 / (R**3) * dRion) * self.include_acceleration_phase_plateau # !!! modle from Daichi Tsuna 
        #
        L = 4.*np.pi* self.sigma * Rion**2 * self.Tion**4
        H,_,_ = self.heating_function(t, y)
        dRion = 3./5. * vexp*(Rion/R) + (self.c0 / (15.*self.kappa*self.sigma* self.Tion**4) ) * ( R**3 / (self.M0 * Rion**3)) * (H - L)
        #
        #dvRion = 5. * (dRion / vRion) * self.sigma * self.kappa * self.Tion**4 / self.c0 * self.include_acceleration_phase_plateau - 3./2. * (vRion / Rion) * (dRion - vRion)
        #dvRion = dR / R *(dRion - Rion/R * dR) # !!!!no accelleration case!!!!
        #dvRion = dvexp * (Rion/R)  + dR / R *(dRion - Rion/R * dR) # !!!!generic accelleration case!!!!
        #
        dRsh = vsh
        #
        rho_SN = 3.* self.M0 / (4.*np.pi* R**3)
        vRsh = vexp * (Rsh / R)
        rho_CSM, vCSM = self.get_CSM_Properies(t, Rsh)
        #max(vRsh - vsh, 0.)
        dvsh = 4.*np.pi* self.f_Omega * Rsh**2 * rho_SN * max(vRsh - vsh, 0.)**2 / Msh  -  4.*np.pi* self.f_Omega * Rsh**2 * rho_CSM * max(vsh - vCSM, 0.)**2 / Msh
        #
        dMsh = 4.*np.pi* self.f_Omega * Rsh**2 * rho_SN * max(vRsh - vsh, 0.)  +  4.*np.pi* self.f_Omega * Rsh**2 * rho_CSM * max(vsh - vCSM, 0.)
        #
        return [dR, dvexp, dRion, dRsh, dvsh, dMsh]

    
    #---------------------------------------------------------
    # integrator events:
    @staticmethod
    def event_reach_Tion(t, y, self): # use in light-curve initial phase
        R = y[0]; vexp = y[1]; Eint = y[2]; Rsh = y[3]; vsh = y[4]; Msh = [5]
        Lbol = 4.*np.pi* self.c0 *R*Eint / (3.* self.kappa * self.M0)
        return ( np.power( Lbol / (4.*np.pi* self.sigma * R**2 ) , 1./4.) - self.Tion ) # the effective temperature reaches the ionization temperature and then we can stop the integration
    
    @staticmethod
    def event_zero_Rion(t, y, self): # use in light-curve plateau phase
        R = y[0]; vexp = y[1]; Rion = y[2]; Rsh = y[3]; vsh = y[4]; Msh = y[5]
        return ( Rion - 1e-5 ) # the ionization radius is very small and then we can stop the integration (1e5cm =1km)


    #---------------------------------------------------------
    # integrator initial conditions:
    def get_init_conditions(self, init_stepsize):
        R_init = self.R0; vexp_init = np.sqrt(5.* self.Eexp / 3./ self.M0); Eint_init = self.Eexp / 2.; Rsh_init = self.R0; vsh_init= 0.
        # take one time step to get an initial shock mass for the integration:
        vRsh = vexp_init
        rho_SN = 3.*self.M0 / (4.*np.pi * R_init**3)
        rho_CSM, vCSM = self.get_CSM_Properies(0., R_init)
        dMsh = 4.*np.pi* self.f_Omega * R_init**2 * rho_SN * vRsh +  4.*np.pi* self.f_Omega * R_init**2 * rho_CSM * (-vCSM)
        Msh_init = dMsh * init_stepsize # integrate mass gain over one time step
        #Msh_init = 1e-2*self.M0
        return [R_init, vexp_init, Eint_init, Rsh_init, vsh_init, Msh_init]
    
    def get_init_conditions_t_ion(self, result_from_phase_initial):
        R_t_ion = result_from_phase_initial.y[0][-1] # == R(t_ion)
        Rion_t_ion = result_from_phase_initial.y[0][-1] # == R(t_ion)
        vexp_t_ion = result_from_phase_initial.y[1][-1] # == vexp(t_ion)
        #
        Rsh_t_ion = result_from_phase_initial.y[3][-1] # == Rsh(t_ion)
        vsh_t_ion = result_from_phase_initial.y[4][-1] # == vsh(t_ion)
        Msh_t_ion = result_from_phase_initial.y[5][-1] # == Msh(t_ion)
        return [R_t_ion, vexp_t_ion, Rion_t_ion, Rsh_t_ion, vsh_t_ion, Msh_t_ion]

    #---------------------------------------------------------
    # main integrator function
    def integrate_model(self):
        # set initial parameters
        timespan = [0., self.max_integration_time]
        init_stepsize = 1e-20 #seconds
        y_init = self.get_init_conditions(init_stepsize)
        self.event_reach_Tion.terminal = True

        result_arrs = integrate.solve_ivp(self.dy_dt_phase_initial, timespan, y_init, args=(self,), events=(self.event_reach_Tion), max_step=0.05*self.day, method="RK45")#, rtol=1e-7, atol=1e-10)

        self.fill_results_phase_initial(result_arrs)
        # debug:
        if self.debug:
            print(result_arrs.t)
            print(result_arrs.y[0])
            print(result_arrs.y[1])
            print(result_arrs.y[2])
            print(result_arrs.y[3])
            print(result_arrs.y[4])
            print(result_arrs.y[5])
        
        # integrate the second phase of the light curve (the plateau phase)
        #timespan = [self.t_ion, self.max_integration_time]
        print("timespans:", self.t_ion, " ", self.max_integration_time)
        if self.max_integration_time < self.t_ion: return # failsafe in case max integration time happens before t_ion
        timespan_plateau_phase = [self.t_ion, self.max_integration_time]
        y_init = self.get_init_conditions_t_ion(result_arrs)
        self.event_zero_Rion.terminal = True

        result_arrs_plateau = integrate.solve_ivp(self.dy_dt_phase_plateau, timespan_plateau_phase, y_init, args=(self,), events=(self.event_zero_Rion), max_step=0.05*self.day, method="RK45")

        self.fill_results_phase_plateau(result_arrs_plateau)
        # debug:
        
        if self.debug:
            print("x_ion plateau phase:")
            print(result_arrs_plateau.y[2]/result_arrs_plateau.y[0])
            print("plateau phase:")
            print(result_arrs_plateau.t)
            print(result_arrs_plateau.y[0])
            print(result_arrs_plateau.y[1])
            print(result_arrs_plateau.y[2])
            print(result_arrs_plateau.y[3])
            print(result_arrs_plateau.y[4])
            print(result_arrs_plateau.y[5])
            # print vRion:
            print(result_arrs_plateau.y[1] * (result_arrs_plateau.y[2]/result_arrs_plateau.y[0]))
            print(result_arrs_plateau.y[:,-2])
        #print(self.Lforwardshock_arr)
        #print(self.Lreverseshock_arr)

    
    #---------------------------------------------------------
    # integrator post-processing
    def fill_results_phase_initial(self, results_array):
        # fill the class arrays with physical values using the integrator results from the plateau phase
        time = results_array.t; R = results_array.y[0]; vexp = results_array.y[1]; Eint = results_array.y[2]; Rsh = results_array.y[3]; vsh = results_array.y[4]; Msh = results_array.y[5]
        #
        self.time_arr = time
        self.t_ion = time[-1] # last timestep of the initial phase is equal to the ionization time
        self.Lbol_arr = 4.*np.pi*self.c0* R* Eint / (3.* self.kappa*self.M0)
        self.Temp_eff_arr = np.power( self.Lbol_arr / (4.*np.pi* self.sigma * R**2 ) , 1./4.)
        self.Renvelope_arr = R
        self.Rphotosphere_arr = R
        self.vexp_arr = vexp
        self.Eint_arr = Eint
        self.Ekin_arr = 3./10. * self.M0 * vexp*vexp # is computed from integral over the whole region
        # heating:
        self.Heating_arr = np.zeros(len(time))
        self.Lforwardshock_arr = np.zeros(len(time))
        self.Lreverseshock_arr = np.zeros(len(time))
        for k in range(len(time)):
            self.Heating_arr[k], self.Lforwardshock_arr[k], self.Lreverseshock_arr[k] = self.heating_function(time[k], results_array.y[:,k])
            #
        self.Rshock_arr = Rsh
        self.vshock_arr = vsh
        self.Mshock_arr = Msh

    
    def fill_results_phase_plateau(self, results_array):
        # fill the class arrays with physical values using the integrator results from the plateau phase
        time = results_array.t; R = results_array.y[0]; vexp = results_array.y[1]; Rion = results_array.y[2]; Rsh = results_array.y[3]; vsh = results_array.y[4]; Msh = results_array.y[5]
        # after filling the values with the quantities from the lateau phase, append everything to the class result arrays:
        self.time_arr = np.append(self.time_arr, time)
        self.t_plateau = time[-1] - self.t_ion  # duration of the plateau phase is equal to (final_time - t_ion)
        tmp_Lbol_arr = 4.*np.pi*self.sigma * Rion**2 * self.Tion**4
        self.Lbol_arr = np.append(self.Lbol_arr, tmp_Lbol_arr)
        self.Temp_eff_arr = np.append(self.Temp_eff_arr, np.ones(len(time))*self.Tion ) #self.Tion
        self.Renvelope_arr = np.append(self.Renvelope_arr, R)
        self.Rphotosphere_arr = np.append(self.Rphotosphere_arr, Rion)
        self.vexp_arr = np.append(self.vexp_arr, vexp)
        tmp_Eint_arr = (3.*self.sigma*self.kappa* self.Tion**4 * self.M0/self.c0) * Rion**4 / (R**3)
        self.Eint_arr = np.append(self.Eint_arr, tmp_Eint_arr)
        tmp_Ekin_arr = 3./10. * self.M0 * vexp*vexp
        self.Ekin_arr = np.append(self.Ekin_arr, tmp_Ekin_arr) #3./10. * self.M0 * vexp*vexp # is computed from integral over the whole region
        # heating:
        tmp_Heating_arr = np.zeros(len(time))
        tmp_Lforwardshock_arr = np.zeros(len(time))
        tmp_Lreverseshock_arr = np.zeros(len(time))
        for k in range(len(time)):
            tmp_Heating_arr[k], tmp_Lforwardshock_arr[k], tmp_Lreverseshock_arr[k] = self.heating_function(time[k], results_array.y[:,k])
            #
        self.Heating_arr = np.append(self.Heating_arr, tmp_Heating_arr)
        self.Lforwardshock_arr = np.append(self.Lforwardshock_arr, tmp_Lforwardshock_arr)
        self.Lreverseshock_arr = np.append(self.Lreverseshock_arr, tmp_Lreverseshock_arr)

        self.Rshock_arr = np.append(self.Rshock_arr, Rsh)
        self.vshock_arr = np.append(self.vshock_arr, vsh)
        self.Mshock_arr = np.append(self.Mshock_arr, Msh)

        # compute peak plateau phase time:
        Lpeak_plateau = 0.0
        #peak_index = 0
        for i in range(len(time)):
            if tmp_Lbol_arr[i] > Lpeak_plateau:
                Lpeak_plateau = tmp_Lbol_arr[i]
                #peak_index = i
        # after finding plateau peak luminsity, scan plateau phase for applicable points:
        t_first = 0.0
        t_last = 1e20
        for l in range(len(time)):
            if tmp_Lbol_arr[l] > 0.9*Lpeak_plateau:
                if t_first < 1.0:
                    t_first = time[l]
                t_last = time[l]
        self.t_plateau_peak = t_last - t_first # earliest point where Lbol is bright enough vs last point where it is similar in brightness to peak luminosity
    
 
#########################################################################
#    SMSlightcurve_sphericalCSMshock                                    #
#########################################################################

class SMSlightcurve_sphericalCSMshock:
    # member variables:
    debug = False
    debug_Suzuki = False
    # natural constants:
    sigma = 5.670374419e-5 # Stefan Boltzmann constant in cgs units [erg/s / cm^2 / K^4]
    c0 = 2.998e10 # speed of light [cm/s]
    kB = 1.38065e-16 # in erg/K
    m_Hydrogen = 1.6738e-24 # mean molecular wheight of atomic hydrogen [gram]
    G = 6.674e-8 # gravitational constant [cm^3/g/s^2]
    M_sun_gram = 1.988e33
    day = 60*60*24. # seconds in a day
    year = 365.25*day
    arad = 4.*sigma/c0
    parsec = 3.0856775814671913e18 # in cm

    # physical light curve model parameters:
    Eexp = 0. # explosion energy [erg/s]
    M0 = 0. # envelope mass [gram]
    R0 = 0. # initial radius [cm]
    kappa = 0.34 # opacity [cm^2/gram]
    Tion = 6000. # ionization temperature [Kevin]
    efficiency = 0.1 # conversion efficiency of shock luminosity to photon energy

    E_rel = 1. # energy of relativistic part of the ejecta
    gamma_adiab = 4./3. # adiabatic index for radiation dominated gas = 4/3
    n_density = 5. # density exponent
    rho0 = 1. # normalization density for relativitic density distribution
    t0 = 10 # seconds
    Aconst_star = 1.
    Gamma_max = 5.
    scale_Energy = 1e51/c0**2 # erg
    scale_Mass = M_sun_gram #1e-5*M_sun_gram
    scale_momentum = 1.
    # CSM properties:
    use_custom_CSM_powerlaw = False
    CSM_power_law_exponent = 2 # all solutions match at the match radius
    CSM_match_radius_codeunit = 0.1*parsec / c0
    CSM_density_prefactor = 1.0 # multiply CSM density with a factor
    #
    t_ion = 1e20 # ionization time [seconds]
    t_plateau = 0. # length of the plateau phase [seconds]
    t_shock_transparent = 1.
    t_shock_transparent_index = 1
    t_plateau_peak = 0. # length of the phase where luminosity is within 10% of peak plateau luminosity
    f_Omega = 0.1 # fraction that the CSM disk takes up of the full solid angle (4pi)
    heating_model = "none" # can be "const", "CSM_disk"
    const_heating_rate = 0.
    Nickel_mass = 0.0 # in gram
    CSM_velocity = -1. # in cm/s
    CSM_accretion_rate = 0.0 # in gram/s
    
    # global output quantities for optically thick shock phase:
    shock_opt_thick_Lbol_peak_time = 0. # time when diffusion luminosity is maximal
    shock_opt_thick_Lbol_peak_luminosity = 1. # erg/s value when diffusion luminosity is maximal
    shock_opt_thick_Lbol_peak_variability_timescale = 0. # time max diffusion luminosity changes only within some percentage amount
    shock_opt_thick_Lbol_local_min_after_t0_index = 0
    shock_opt_thick_maximum_eta = 1e-10 # useful to check if at any point in the light curve there may be non-equilibrium (non-thermalization) effects
    # global output quantities for transparant shock phase:
    shock_transparent_Lbol_initial_luminosity = 10. # erg/s value when shock just became transparent
    

    # containers for outputs over time
    time_arr = np.array(0.)
    Lbol_arr = np.array(0.)
    Temp_eff_arr = np.array(0.) # photosphere temperature, of electron temperature
    Temp_BB_surface = np.array(0.) # black body temperature used to compute the colour temperature

    Renvelope_arr = np.array(0.)
    Rphotosphere_arr = np.array(0.)
    vexp_arr = np.array(0.)
    Eint_arr = np.array(0.)
    Eint_shock_arr = np.array(0.)
    Ekin_shock_arr = np.array(0.)
    Eint_envelope_arr = np.array(0.)
    Ekin_envelope_arr = np.array(0.)
    Heating_arr = np.array(0.)
    Cooling_work_arr = np.array(0.)
    Lforwardshock_arr = np.array(0.)
    Lreverseshock_arr = np.array(0.)

    Rforwardshock_arr = np.array(0.)
    Rreverseshock_arr = np.array(0.)
    Rshock_shell_arr = np.array(0.)
    Pres_forwardshock_arr = np.array(0.)
    Pres_reverseshock_arr = np.array(0.)
    Pres_shock_radiation_arr = np.array(0.)
    momentum_shock_arr = np.array(0.)
    v_forwardshock_arr = np.array(0.)
    v_reverseshock_arr = np.array(0.)
    v_shock_shell_arr = np.array(0.)
    Lfac_shock_shell_arr = np.array(0.)
    Mshock_arr = np.array(0.)
    optical_depth_shock_arr = np.array(0.)
    shock_average_density_arr = np.array(0.)
    shock_Temp_internal_arr = np.array(0.)
    shock_Temp_internal_electron_arr = np.array(0.)
    # quantities related to non-thermal effects in the spectrum: see Nakar&Sari 2010: https://iopscience.iop.org/article/10.1088/0004-637X/725/1/904/pdf
    eta_factor_arr = np.array(0.) # ratio between needed Photons in BlackBody vs produced photons. spectrum is modified if >1
    y_max_arr = np.array(0.) # related to minimum frequency for Comptonization
    
    # integrator setings
    max_integration_time = 1000.* day
    max_timestep = 4.0*day
    stop_after_opt_thick_phase = False
    last_timestep = 1.
    last_Ekin_env = 1.
    force_turn_off_non_thermal_effects = False #set eta_factor <1 to always tunrn off non-thermal Comptonization effects
    use_smooth_opacity_decrease = False
    use_steep_opacity_decrease = True

    # constructor:
    def __init__(self, E_rel_in, M0_in, R0_in, Tion_in, kappa_in, t0_in=10, Aconst_star_in=100., Gamma_max_in=5., n_density_in=0):
        self.Eexp = E_rel_in / (self.c0**2)
        self.M0 = M0_in
        self.R0 = R0_in / self.c0
        self.Tion = Tion_in
        self.kappa = kappa_in / (self.c0**2) # convert from ggs to units of c=1 (code units for the integrator)
        self.t0 = t0_in
        self.Aconst_star = Aconst_star_in
        self.E_rel = E_rel_in / (self.c0**2)
        self.Gamma_max = Gamma_max_in
        self.n_density = n_density_in
    
    #---------------------------------------------------------
    # physical model function for the integrator:
    def heating_function(self, t, y):
        return 0.
    
    def get_CSM_Properies(self, t, r):
        # Larsons isothermal collapse solution, see https://inspirehep.net/literature/57090 :
        T_CSM = 8000. # Kelvin
        R_gas = 4.733e7 # specificgas constant for primordial gas in erg/g/K
        eta0 = 8.86; xi0 = 3.28 # numerical factors from Larson
        density = eta0 * R_gas*T_CSM / (4.*np.pi* self.G * r*r) * self.c0 # in unts of c=1
        velocity = - np.sqrt(R_gas*T_CSM) * xi0 / self.c0 # in units of c=1
        if self.use_custom_CSM_powerlaw:
            # compute density and velocity at matching radius:
            density_mr = self.CSM_density_prefactor * density * (r/self.CSM_match_radius_codeunit)**2
            velocity_mr = velocity / self.CSM_density_prefactor
            # compute density from there on:
            density = density_mr * (self.CSM_match_radius_codeunit/r)**(self.CSM_power_law_exponent)
            velocity = velocity_mr * (self.CSM_match_radius_codeunit/r)**(2-self.CSM_power_law_exponent)
        # to reproduce suzuki paper:
        if self.debug_Suzuki:
            Aconst = self.Aconst_star * 5e11 * self.c0 # in g/cm to g/ls
            density = Aconst /r/r
        return density, velocity
    
    def get_ejecta_density(self, t, r):
        GbrBbr = 0.1
        GmaxBmax = self.Gamma_max * np.sqrt(1.- 1./(self.Gamma_max**2) ) # 5.* np.sqrt(1. - 1./ 25.)
        beta = r/t
        Lfac = self.lorentz_fac(beta)
        if self.debug_Suzuki:
            density = self.rho0 * (self.t0/t)**3
            if (Lfac*beta < GbrBbr):
                density = self.rho0 * (self.t0/t)**3 * (GbrBbr/GmaxBmax)**(-self.n_density)
            elif ((Lfac*beta > GbrBbr) and (Lfac*beta < GmaxBmax)):
                density = self.rho0 * (self.t0/t)**3 * (Lfac*beta/GmaxBmax)**(-self.n_density)
            else: density = self.rho0* 1e-20
            return density
        #
        density = self.rho0 * (self.t0/t)**3 * (Lfac*beta)**(-self.n_density)
        return density
    
    def analytic_powlaw_density_integral_I(self, beta_max):
        if self.n_density == 0:
            return 0.5*beta_max*np.sqrt(1.-beta_max**2) - beta_max - 0.5*np.arcsin(beta_max) + np.arctanh(beta_max)
        elif self.n_density == 1:
            return 0.5*(np.sqrt(1.-beta_max**2) - 1.)**2
        elif self.n_density == 2:
            return -0.5*beta_max*np.sqrt(1.-beta_max**2) + beta_max - 0.5*np.arcsin(beta_max)
        else:
            print("For power law, only density distributions with n_density = 0,1,2 are implemented")
            exit()
    
    def analytic_powlaw_density_integral_J(self,beta_max):
        if self.n_density == 0:
            return 0.5*(np.arcsin(beta_max) - beta_max*np.sqrt(1.-beta_max**2) )
        elif self.n_density == 1:
            return 0.5*beta_max**2
        elif self.n_density == 2:
            return 0.5*(np.arcsin(beta_max) + beta_max*np.sqrt(1.-beta_max**2) )
        else:
            print("For power law, only density distributions with n_density = 0,1,2 are implemented")
            exit()
    
    def analytic_powlaw_density_derivative_I(self, beta_max, dbeta):
        lfac_b = self.lorentz_fac(beta_max)
        if self.n_density < 4:
            return (lfac_b-1.) *lfac_b* beta_max**2 * (lfac_b* beta_max)**(-self.n_density) * dbeta
        else:
            print("For power law, only density distributions with n_density = 0,1,2 are implemented")
            exit()
    
    def analytic_powlaw_density_derivative_J(self,beta_max, dbeta):
        lfac_b = self.lorentz_fac(beta_max)
        if self.n_density < 4:
            return lfac_b* beta_max**2 * (lfac_b* beta_max)**(-self.n_density) * dbeta
        else:
            print("For power law, only density distributions with n_density = 0,1,2 are implemented")
            exit()
    
    def lorentz_fac(self, beta): # beta=v/c
        return 1. / np.sqrt(1. - beta*beta)
    
    def beta_shock(self, beta_u, beta_d): # downstream + upstream beta (beta=v/c)
        Lfac_u = self.lorentz_fac(beta_u)
        Lfac_d = self.lorentz_fac(beta_d)
        return ( self.gamma_adiab*Lfac_u*Lfac_d**2 * (beta_u-beta_d)*beta_d - (self.gamma_adiab-1.)*(Lfac_u-Lfac_d) ) / ( self.gamma_adiab*Lfac_u*Lfac_d**2 * (beta_u-beta_d) - (self.gamma_adiab-1.)*(Lfac_u*beta_u-Lfac_d*beta_d) )

    #---------------------------------------------------------
    # helper functions for integrator events, integration and fill_results:
    def compute_T_electron(self, rho_average_cgs, eta_factor, T_BB):
        if eta_factor<1.: return T_BB # then electron temp and BB temp are the same
        def T_ph_finder(T_in, rho_av, eta_fac, T_bb):
            kB = 1.38065e-16 # in erg/K
            y_max = 3.* (rho_av/1e-9)**(-0.5) * (kB*T_in / 1.6021766e-10)**(9./4.)
            xi_factor = max(1, 0.5*np.log(y_max)*(1.6+np.log(y_max)) )
            if y_max<1.: xi_factor=1.
            return T_in/T_bb - eta_fac**2 / xi_factor**2
        #
        T_electron = T_BB * eta_factor**2 # first estimate of T_electron
        y_max = 3.* (rho_average_cgs/1e-9)**(-0.5) * (1.38065e-16*T_electron / 1.6021766e-10)**(9./4.)
        if y_max < 2.2814419583330876: # y_max<2.28... will always lead to xi(T_electron)=1  analytical result y_max < e^(1/5 (-4 + sqrt(66)) )
            T_electron = T_BB * eta_factor**2
        else:
            try:
                result = optimize.root_scalar(T_ph_finder, args=(rho_average_cgs, eta_factor, T_BB,), method="bisect", bracket=[T_BB, T_BB*eta_factor**2+1.], xtol= 1e-5, rtol=1e-6, maxiter=100)
                T_electron = result.root
            except:
                T_electron = T_BB * eta_factor**2
                y_max = 3.* (rho_average_cgs/1e-9)**(-0.5) * (1.38065e-16*T_electron / 1.6021766e-10)**(9./4.)
                print("exception in 'compute_T_electron()' when root-finding T_electron at ", rho_average_cgs, eta_factor, T_BB, y_max)
        return T_electron

    def compute_T_electron_from_TBB_simplified(self, eta_factor, xi_factor, T_BB):
        if eta_factor<1.: return T_BB # then electron temp and BB temp are the same
        T_e_simple = T_BB * (eta_factor/xi_factor)**2
        return T_e_simple

    def compute_T_BB_from_Telectron(self, rho_average_cgs, eta_factor, T_e):
        if eta_factor<1.: return T_e
        y_max = 3.* (rho_average_cgs/1e-9)**(-0.5) * (self.kB*T_e / 1.6e-10)**(9./4.)
        xi_factor = max(1, 0.5*np.log(y_max)*(1.6+np.log(y_max)) )
        if y_max<1.: xi_factor=1.
        T_BB = T_e * xi_factor**2 / eta_factor**2
        return T_BB

    def compute_T_BB_from_Telectron_simplified(self, eta_factor, xi_factor, T_e):
        if eta_factor<1.: return T_e
        T_BB_simple = T_e * (xi_factor / eta_factor)**2
        return T_BB_simple

    def compute_eta_simplified(self, time, rho_average_cgs, T_BB, Ms, Rs, dR, beta_s):
        tau_s = self.kappa* Ms / Rs**2 / (4.*np.pi)
        beta_diff = (1. - beta_s**2) / ( (3.+ beta_s**2) * tau_s  + 2.*beta_s )
        if (beta_diff > 1. - beta_s): beta_diff = 1. - beta_s
        t_diff = dR / beta_diff
        eta_factor = ( 7e5/np.minimum(t_diff,time) ) * (rho_average_cgs/1e-10)**(-2) * (self.kB* T_BB / 1.6e-10)**(7./2) # cgs units
        if self.force_turn_off_non_thermal_effects: eta_factor = 0.01 # force smaller eta to turn off non-thermla effects
        return eta_factor # should produce a slight under-estimation of eta in the early time

    def compute_xi_from_Telectron(self, rho_average_cgs, T_e):
        kB = 1.38065e-16 # in erg/K
        y_max = 3.* (rho_average_cgs/1e-9)**(-0.5) * (kB*T_e / 1.6021766e-10)**(9./4.)
        xi_factor = max(1, 0.5*np.log(y_max)*(1.6+np.log(y_max)) )
        if y_max<1.: xi_factor=1.
        return xi_factor

    def compute_optical_depth(self, t, Sr, Msh, Rs, Rfs, Rrs, Eint_sh):
        beta_s = Sr / np.sqrt(Msh**2 + Sr**2)
        Lfac_s = self.lorentz_fac(beta_s)
        Vsh = 4./3. * np.pi *(Rfs**3 - Rrs**3)
        Vsh_cgs = Vsh * self.c0**3
        rho_average_cgs = Msh / (Lfac_s * Vsh_cgs )
        #----
        tau_s = self.kappa* Msh / (4.*np.pi* Rs*Rs)
        beta_diff = (1. - beta_s**2) / ( (3.+ beta_s**2) * tau_s  + 2.*beta_s )
        if (beta_diff > 1. - beta_s): beta_diff = 1. - beta_s
        # compute eta-factor to see if corrections to the spectrum and electron temperature are needed:
        Eint_sh_cgs = Eint_sh * self.c0**2
        T_BB_bulk = ((Eint_sh_cgs/Vsh_cgs) / self.arad)**(0.25) # result in Kelvin
        eta_factor = self.compute_eta_simplified(t, rho_average_cgs, T_BB_bulk, Msh, Rs, (Rfs-Rrs), beta_s)
        T_e_bulk = self.compute_T_electron(rho_average_cgs, eta_factor, T_BB_bulk)
        xi_factor = self.compute_xi_from_Telectron(rho_average_cgs, T_e_bulk)
        # estimate surface temperature assuming black body and using the eta-factor obtianed from the bulk quantities:
        T_eff_BB_surf = (beta_diff * (Eint_sh/Vsh) / self.sigma)**(0.25) # result in Kelvin
        '''T_e_surf = self.compute_T_electron(rho_average_cgs, eta_factor, T_eff_BB_surf)'''
        T_e_surf = self.compute_T_electron_from_TBB_simplified(eta_factor, xi_factor, T_eff_BB_surf)
        # based on T_e_surf we check if the shock is fully ionized or not:
        if ((T_e_surf < self.Tion) and (not self.debug_Suzuki) ): # here we are in the plateau phase:
            '''T_BB_if = self.compute_T_BB_from_Telectron(rho_average_cgs, eta_factor, self.Tion)'''
            T_BB_if = self.compute_T_BB_from_Telectron_simplified(eta_factor, xi_factor, self.Tion)
            beta_diff_Tion = self.sigma*T_BB_if**4 / (Eint_sh/Vsh)
            tau_s = ((1. - beta_s**2)/beta_diff_Tion - 2.*beta_s) / (3. + beta_s**2)  # lower tau_s value, taking into account ionization front
            #print("tau_s= ", tau_s, T_BB_if, Eint_sh)
        else:
            tau_s = self.kappa* Msh / (4.*np.pi* Rs*Rs)
        #
        if self.use_smooth_opacity_decrease: # option to smoothly force transparency at low bulk electron temperature:
            tau_s = tau_s* np.minimum(1., (T_e_bulk/ self.Tion)**11) # simulate effective change in kappa, see https://arxiv.org/pdf/1905.00037
        elif self.use_steep_opacity_decrease:
            if T_e_bulk < self.Tion:
                tau_s = 1e-10 # tau becomes essentially zero below Tion
        #print("T_e_bulk= ", T_e_bulk)
        return tau_s

    def compute_diffusion_luminosity(self, t, Sr, Msh, Rs, Rfs, Rrs, Eint_sh): # compute dEdiff
        beta_s = Sr / np.sqrt(Msh**2 + Sr**2)
        Lfac_s = self.lorentz_fac(beta_s)
        Vsh = 4./3. * np.pi *(Rfs**3 - Rrs**3)
        Vsh_cgs = Vsh * self.c0**3
        rho_average_cgs = Msh / (Lfac_s * Vsh_cgs )
        #----
        tau_s = self.kappa* Msh / (4.*np.pi* Rs*Rs)
        beta_diff = (1. - beta_s**2) / ( (3.+ beta_s**2) * tau_s  + 2.*beta_s )
        if (beta_diff > 1. - beta_s): beta_diff = 1. - beta_s
        # compute eta-factor to see if corrections to the spectrum/Luminosity and electron temperature are needed:
        Eint_sh_cgs = Eint_sh * self.c0**2
        T_BB_bulk = ((Eint_sh_cgs/Vsh_cgs) / self.arad)**(0.25) # result in Kelvin
        eta_factor = self.compute_eta_simplified(t, rho_average_cgs, T_BB_bulk, Msh, Rs, (Rfs-Rrs), beta_s)
        T_e_bulk = self.compute_T_electron(rho_average_cgs, eta_factor, T_BB_bulk)
        xi_factor = self.compute_xi_from_Telectron(rho_average_cgs, T_e_bulk)
        # estimate surface temperature assuming black body and using the eta-factor obtianed from the bulk quantities:
        T_eff_BB_surf = (beta_diff * (Eint_sh/Vsh) / self.sigma)**(0.25) # result in Kelvin
        '''T_e_surf = self.compute_T_electron(rho_average_cgs, eta_factor, T_eff_BB_surf)'''
        T_e_surf = self.compute_T_electron_from_TBB_simplified(eta_factor, xi_factor, T_eff_BB_surf)
        # based on T_e_surf we check if the shock is fully ionized or not:
        if ((T_e_surf < self.Tion) and (not self.debug_Suzuki) ): # here we are in the plateau phase:
            # taking into account ionization front BB temperature (is used to compute luminosity) at constant ion temperature = Tion
            # T_BB_if is blackbody temperature at the ionization front
            '''T_BB_if = self.compute_T_BB_from_Telectron(rho_average_cgs, eta_factor, self.Tion)''' # will be larger than T_eff_surf and thus lead to higher luminosity
            T_BB_if = self.compute_T_BB_from_Telectron_simplified(eta_factor, xi_factor, self.Tion)
            dEdiff = 4.*np.pi*Rs**2 * self.sigma * T_BB_if**4
        else:
            dEdiff = 4.*np.pi*Rs**2 * beta_diff * (Eint_sh/Vsh) # shock is fully ionized and thus have usual diffusion radiation loss
        return dEdiff
    
    #---------------------------------------------------------
    # integrator dy_dt ODE function derivatives:
    # equations taken from https://iopscience.iop.org/article/10.3847/1538-4357/834/1/32
    @staticmethod # use this so that we can use solveivp integrator. Note we just pass 'self' as an extra argument
    def dy_dt_phase_opt_thick_shock(t, y, self):
        # for better readability of the equations
        Sr = y[0]*self.scale_momentum; Msh = y[1]*self.scale_Mass; Rs = y[2]; Rfs = y[3]; Rrs = y[4]; Eint_sh = y[5]*self.scale_Energy
        #
        # compute all auxillarry variables:
        #Lfac_s = np.sqrt(1. + (Sr/Msh)**2 )
        beta_s = Sr / np.sqrt(Msh**2 + Sr**2)
        #beta_s = np.sqrt(1. - 1./Lfac_s**2)
        if self.debug: print("betas " + str(beta_s))
        Lfac_s = self.lorentz_fac(beta_s)
        if self.debug: print("betas, Lfac_s" + str(beta_s) + ", " + str(Lfac_s))
        # forward shock:
        rho_a_fs,_ = self.get_CSM_Properies(t, Rfs)
        beta_fs = self.beta_shock(0., beta_s)
        Pfs = rho_a_fs * beta_s* beta_fs / (1. - beta_s*beta_fs)
        # reverse shock:
        beta_ej_rs = Rrs/t
        rho_ej_rs = self.get_ejecta_density(t, Rrs)
        beta_rs = self.beta_shock(beta_ej_rs, beta_s)
        Lfac_ej_rs = self.lorentz_fac(beta_ej_rs)
        Prs = rho_ej_rs* Lfac_ej_rs**2 * (beta_ej_rs - beta_rs )*(beta_ej_rs - beta_s) / (1. - beta_s*beta_rs)
        # ODEs dydt:
        Frs = rho_ej_rs* Lfac_ej_rs**2 * beta_ej_rs * (beta_ej_rs - beta_rs)#max(beta_ej_rs - beta_rs,0.)
        dSr = 4.*np.pi*Rrs*Rrs*Prs - 4.*np.pi*Rfs*Rfs*Pfs + 4.*np.pi*Rrs*Rrs*Frs     # momentum of shell, NOTE: Typo in eq. (15) in Suzuki et al.
        #
        Gfs = - rho_a_fs*beta_fs
        Grs = rho_ej_rs * Lfac_ej_rs * (beta_ej_rs - beta_rs)# max(beta_ej_rs - beta_rs,0.)
        dMsh = 4.*np.pi*Rrs*Rrs*Grs - 4.*np.pi*Rfs*Rfs*Gfs  # mass of shell
        #
        dRs = beta_s # velocity of shell
        #
        dRfs = beta_fs # velocity of forward shock
        #
        dRrs = beta_rs # velocity of reverse shock
        #
        dEfs = 4.*np.pi*Rfs**2 * self.gamma_adiab*Pfs/(self.gamma_adiab-1.)* Lfac_s**2 * (beta_fs - beta_s)#max(beta_fs - beta_s, 0.)
        dErs = 4.*np.pi*Rrs**2 * self.gamma_adiab*Prs/(self.gamma_adiab-1.)* Lfac_s**2 * (beta_s - beta_rs)#max(beta_s - beta_rs, 0.)
        Vsh = 4./3. * np.pi *(Rfs**3 - Rrs**3)
        #if (Vsh < 1e-10): Vsh=1e-10 # set a volume floor
        dVdt = 4.*np.pi * (Rfs**2 * dRfs - Rrs**2 * dRrs)
        tau_s = self.kappa* Msh / (4.*np.pi* Rs*Rs)
        beta_diff = (1. - beta_s**2) / ( (3.+ beta_s**2) * tau_s  + 2.*beta_s )
        if (beta_diff > 1. - beta_s): beta_diff = 1. - beta_s
        dEdiff = 4.*np.pi*Rs**2 * beta_diff * Eint_sh/Vsh
        if ((beta_diff * Eint_sh/Vsh / self.sigma < self.Tion**4) and (not self.debug_Suzuki)): dEdiff = 4.*np.pi*Rs**2 * self.sigma * self.Tion**4 # effective temperature is below Tion
        dEint_sh = dEfs + dErs - Eint_sh/(3.*Vsh)*dVdt - dEdiff # internal energy of shell, NOTE: Typo in eq. (57) in Suzuki et al.
        #dEint_sh = dEfs + dErs - dEdiff # internal energy of shell, NOTE: Typo in eq. (57) in Suzuki et al.
        
        #
        # debug print:
        if self.debug: print("rho_fs/rho_rs" + str(rho_a_fs/rho_ej_rs))
        #print("Msh " + str(Msh))
        #print("betas " + str(beta_s))
        #print("betafs " + str(beta_fs))
        #print("betars " + str(beta_rs))
        
        return [dSr/self.scale_momentum, dMsh/self.scale_Mass, dRs, dRfs, dRrs, dEint_sh/self.scale_Energy]


    @staticmethod
    def dy_dt_phase_transparent_shock(t, y, self):
        # for better readability of the equations
        Sr = y[0]*self.scale_momentum; Msh = y[1]*self.scale_Mass; Rs = y[2]; Rfs = y[3]; Rrs = y[4]; Eint_sh = y[5]*self.scale_Energy
        #
        # compute all auxillarry variables:
        beta_s = Sr / np.sqrt(Msh**2 + Sr**2)
        Lfac_s = self.lorentz_fac(beta_s)
        if self.debug: print("betas " + str(beta_s))
        if self.debug: print("betas, Lfac_s" + str(beta_s) + ", " + str(Lfac_s))
        # forward shock:
        rho_a_fs,_ = self.get_CSM_Properies(t, Rfs)
        beta_fs = self.beta_shock(0., beta_s)
        Pfs = rho_a_fs * beta_s* beta_fs / (1. - beta_s*beta_fs)
        # reverse shock:
        beta_ej_rs = Rrs/t
        rho_ej_rs = self.get_ejecta_density(t, Rrs)
        beta_rs = self.beta_shock(beta_ej_rs, beta_s)
        Lfac_ej_rs = self.lorentz_fac(beta_ej_rs)
        Prs = rho_ej_rs* Lfac_ej_rs**2 * (beta_ej_rs - beta_rs )*(beta_ej_rs - beta_s) / (1. - beta_s*beta_rs)
        # ODEs dydt:
        Frs = rho_ej_rs* Lfac_ej_rs**2 * beta_ej_rs * (beta_ej_rs - beta_rs)#max(beta_ej_rs - beta_rs,0.)
        dSr = 4.*np.pi*Rrs*Rrs*Prs - 4.*np.pi*Rfs*Rfs*Pfs + 4.*np.pi*Rrs*Rrs*Frs     # momentum of shell, NOTE: Typo in eq. (15) in Suzuki et al.
        #
        Gfs = - rho_a_fs*beta_fs
        Grs = rho_ej_rs * Lfac_ej_rs * (beta_ej_rs - beta_rs)# max(beta_ej_rs - beta_rs,0.)
        dMsh = 4.*np.pi*Rrs*Rrs*Grs - 4.*np.pi*Rfs*Rfs*Gfs  # mass of shell
        #
        dRs = beta_s # velocity of shell
        #
        dRfs = beta_fs # velocity of forward shock
        #
        dRrs = beta_rs # velocity of reverse shock
        #
        Vsh = 4./3. * np.pi *(Rfs**3 - Rrs**3)
        dVdt = 4.*np.pi * (Rfs**2 * dRfs - Rrs**2 * dRrs)
        dEint_sh = - Eint_sh/(3.*Vsh)*dVdt # internal energy of shell, in transparent phase, there is no diffusion luminosity and no heating any more. This prescription here might not conserve the total energy. but is not important because we here ignore the contributions from the shock anyways
        
        return [dSr/self.scale_momentum, dMsh/self.scale_Mass, dRs, dRfs, dRrs, dEint_sh/self.scale_Energy]

    
    #---------------------------------------------------------
    # integrator events:
    @staticmethod
    def event_one_optical_depth(t, y, self): # use to detect when shock becomes transparent
        Sr = y[0]*self.scale_momentum; Msh = y[1]*self.scale_Mass; Rs = y[2]; Rfs = y[3]; Rrs = y[4]; Eint_sh = y[5]*self.scale_Energy
        # compute tau!!
        tau_s = self.compute_optical_depth(t, Sr, Msh, Rs, Rfs, Rrs, Eint_sh)
        return ( tau_s - 1. ) # opacity of shock region drops to one and then we can stop the integration

    @staticmethod
    def event_negative_Eint_shock(t, y, self): # use in light-curve initial phase
        Sr = y[0]*self.scale_momentum; Msh = y[1]*self.scale_Mass; Rs = y[2]; Rfs = y[3]; Rrs = y[4]; Eint_sh = y[5]*self.scale_Energy
        return (Eint_sh)#starts positive due to initial conditions, is negative when shock is transparent

    #---------------------------------------------------------
    # integrator initial conditions:
    def get_init_conditions(self, init_stepsize):
        Sr = 0.; Msh = 0.; Rs = self.R0; Rfs = self.R0*(1.+1e-10); Rrs = self.R0*(1.-1e-10); Eint_sh = 0.
        # compute some important initial parameters:
        if self.debug_Suzuki:
            # compute rho0 from Erel:
            GmaxBmax = self.Gamma_max * np.sqrt(1.- 1./(self.Gamma_max**2) )
            def beta_integrand(beta, self): # == F_nu_obs / nu_obs
                Gamma = self.lorentz_fac(beta)
                return ( pow(Gamma*beta/GmaxBmax, -self.n_density) * Gamma*(Gamma-1.)*beta*beta )
            beta_min = 1/np.sqrt(2.); beta_max = np.sqrt(1.- 1./(self.Gamma_max**2) ) # maximum Lfac
            beta_integral, err1 = integrate.quad( beta_integrand, beta_min, beta_max, args=(self,), epsabs=1e-13, epsrel=1e-13)
            
            self.rho0 = self.E_rel / (4.*np.pi* self.t0**3 * beta_integral )
        else:
            # compute density using the total kinetic energy and total ejecta mass:
            # for homologous expansion with uniform density distribution:
            # estimate velocity using Newtonian expressions:
            vexp = 1e-5
            if self.n_density < 4: # for power-laaw density, the solution exists only up to power v^-3
                vexp = np.sqrt( 2*(5-self.n_density)/(3-self.n_density) * self.E_rel/self.M0)
                #vexp = np.sqrt(10.*self.E_rel / (3.* self.M0))
                beta_max = vexp
                self.t0 = self.R0 / beta_max
                #self.rho0 = 3.*self.M0 / (4.*np.pi* self.R0**3)
                self.rho0 = (3-self.n_density)* beta_max**(self.n_density)* self.M0 / (4.*np.pi* self.R0**3)
            else:
                print("For non-broken power law, only density distributions with n_density = 0,1,2 are iplemented")
                exit()
            if self.debug:
                print("t0_1, rho0_1: ", self.t0, self.rho0)
                print("beta_max: ", beta_max)
            ########## realtivistic expressions for initial values
            ########## assuming uniform density:
            if (vexp > 1e-4): # use relativistic formula for high velocities (for too low velocities, there can be numerical problems when using the relativistiv formula)
                '''
                def beta_integral1(bet): 
                    return (0.5*bet*(np.sqrt(1.-bet**2) - 2.) - 0.5*np.arcsin(bet) + np.arctanh(bet))
                def beta_integral2(bet): 
                    return (-0.5*bet*np.sqrt(1.-bet**2) + 0.5*np.arcsin(bet))
                # obtain rho0 and t0 from the relativistic versions:
                # need to do root-finding to get t0:
                def root_function(t_root, self):
                    return (beta_integral1(self.R0/t_root)/beta_integral2(self.R0/t_root) - self.E_rel/self.M0)
                # root finding to find t0:
                results = optimize.root_scalar(root_function, args=(self,), method="bisect", bracket=[self.R0*1.001, self.R0/0.001 * 10], xtol= 1e-15, rtol=1e-15, maxiter=200)
                # assign the found to and rho0:
                self.t0 = results.root
                self.rho0 = self.M0 / (4.*np.pi*self.t0**3 * beta_integral2(self.R0/self.t0))
                '''
                ### new way to compute:
                def root_function_beta(beta_max_root, self):
                    return (self.analytic_powlaw_density_integral_I(beta_max_root)/self.analytic_powlaw_density_integral_J(beta_max_root) - self.E_rel/self.M0)
                # root finding to find beta_max:
                results = optimize.root_scalar(root_function_beta, args=(self,), method="bisect", bracket=[0.99e-4, 0.999], xtol= 1e-15, rtol=1e-15, maxiter=200)
                # compute t0:
                self.t0 = self.R0/results.root
                #
                self.rho0 = self.M0 / (4.*np.pi*self.t0**3 * self.analytic_powlaw_density_integral_J(beta_max))
                
                if self.debug:
                    #print("beta_integral1: ", beta_integral1(0.2), beta_integral1(0.5), beta_integral1(0.8))
                    #print("beta_integral2: ", beta_integral2(0.2), beta_integral2(0.5), beta_integral2(0.8))
                    #print("root funcition: ", root_function(self.R0/beta_max*0.5, self), root_function(2.*self.R0, self), root_function(5./4.*self.R0, self))
                    print("self.E_rel/self.M0 ", self.E_rel/self.M0)
                    print("beta_max: ", self.R0/self.t0)
                    print("results.root ", results.root)
                    print("t0, rho0: ", self.t0, self.rho0)

        # initial conditions for the non-relativistic case:
        # initial shock velocity: (equivalent to solving quadratic equation)
        #self.t0 = self.R0 / beta_max 
        rho_a_fs,_ = self.get_CSM_Properies(self.t0, Rfs)
        rho_ej_rs = self.get_ejecta_density(self.t0, Rrs)
        beta_ej_rs = Rrs/self.t0
        a = 1. + rho_a_fs/rho_ej_rs * ( beta_ej_rs*beta_ej_rs - 1.)
        b = - 2. * beta_ej_rs
        c = beta_ej_rs*beta_ej_rs
        p = b/a
        q = c/a
        beta_s = - p/2. - np.sqrt((p/2.)**2 - q) # take negative root because otherwise velocity could be superluminal...
        if self.debug:
            print("beta+, beta- : " + str(- p/2. + np.sqrt((p/2.)**2 - q)) + ", " + str(- p/2. - np.sqrt((p/2.)**2 - q)))

        # take one time step to get an initial shock mass for the integration:
        Gfs = - rho_a_fs*beta_s
        #dMsh = - 4.*np.pi*Rfs*Rfs*Gfs  # mass of shell
        Lfac_ej_rs = self.lorentz_fac(beta_ej_rs)
        Grs = rho_ej_rs * Lfac_ej_rs * (beta_ej_rs - beta_s)#max(beta_ej_rs - beta_s,0.)
        dMsh = 4.*np.pi*Rrs*Rrs*Grs - 4.*np.pi*Rfs*Rfs*Gfs  # mass of shell
        Msh_init = dMsh * init_stepsize # integrate mass gain over one time step
        # initial momentum:
        Lfac_s = self.lorentz_fac(beta_s)
        Sr_init = Msh_init * Lfac_s * beta_s # initial relativistic momentum
        # internal energy:
        Eint_sh_init = 1e5 / self.c0**2 # erg/c^2
        # initial radii:
        Rs_init = self.R0
        Rfs_init = self.R0 *(1.+1e-10)  # add small difference for numerical stability reasons
        Rrs_init = self.R0 *(1.-1e-10)
        #
        if self.debug:
            print("beta_s ", str(beta_s))

        self.last_timestep = self.t0
        self.last_Ekin_env = self.E_rel
        self.scale_momentum = Sr_init
        return [Sr_init/self.scale_momentum, Msh_init/self.scale_Mass, Rs_init, Rfs_init, Rrs_init, Eint_sh_init/self.scale_Energy]
    
    def get_init_conditions_for_next_phase(self, result_from_previous_phase):
        Sr_init = result_from_previous_phase.y[0][-1] # == Sr(t_transparent_shock)
        Msh_init = result_from_previous_phase.y[1][-1] # == Msh(t_transparent_shock)
        #
        Rs_init = result_from_previous_phase.y[2][-1] # == Rs(t_transparent_shock)
        Rfs_init = result_from_previous_phase.y[3][-1] # == Rfs(t_transparent_shock)
        Rrs_init = result_from_previous_phase.y[4][-1] # == Rrs(t_transparent_shock)
        Eint_sh_init = result_from_previous_phase.y[5][-1] # == Eint_sh(t_transparent_shock)
        # new variables:
        #Rph_init = Rs_init
        return [Sr_init, Msh_init, Rs_init, Rfs_init, Rrs_init, Eint_sh_init] # here no re-scaling because result_from_phase_initial is already scaled data

    #---------------------------------------------------------
    # main integrator function
    def integrate_model(self):
        # set initial parameters
        # for testing purposes, first use this mode here (later we compute t0 from a known initial radius, this is better)
        if self.debug_Suzuki:
            beta_max = np.sqrt(1.- 1./(self.Gamma_max**2) )
            self.R0 = self.t0 *beta_max
        #
        init_stepsize = 1e-10 # 1e-10 seconds
        y_init = self.get_init_conditions(init_stepsize)
        if self.debug: print(y_init)
        timespan = [self.t0, self.max_integration_time]
        

        self.event_one_optical_depth.terminal = True # stop the integration when reaching event: True
        self.event_one_optical_depth.direction = -1 # only trigger when going from tau>1 to tau<1
        #
        self.event_negative_Eint_shock.terminal = True
        self.event_negative_Eint_shock.direction = -1

        # failsafe algorithm for csaes where shock is transparent initially: small test integration, as the spuious cases only fail in the 2nd time step
        result_arrs = integrate.solve_ivp(self.dy_dt_phase_opt_thick_shock,  [self.t0, self.t0*1.0001], y_init, args=(self,), max_step=self.max_timestep, method="LSODA")#, rtol=1e-7, atol=1e-10)
        if result_arrs.y[5][-1] < 0.: # internal energy is negative
            result_arrs = integrate.solve_ivp(self.dy_dt_phase_opt_thick_shock,  [self.t0, self.t0*1.0001], y_init, args=(self,), max_step=self.max_timestep, method="LSODA")#, rtol=1e-7, atol=1e-10)
        else:
            result_arrs = integrate.solve_ivp(self.dy_dt_phase_opt_thick_shock, timespan, y_init, args=(self,), events=(self.event_one_optical_depth,self.event_negative_Eint_shock), max_step=self.max_timestep, method="LSODA")#, rtol=1e-7, atol=1e-10)
        #print(result_arrs.status, result_arrs.message)
        self.fill_results_phase_opt_thick_shock(result_arrs)
        # debug:
        if self.debug:
            print(result_arrs.t)
            print(result_arrs.y[0])
            print(result_arrs.y[1])
            print(result_arrs.y[2])
            print(result_arrs.y[3])
            print(result_arrs.y[4])
            print(result_arrs.y[5])
        
        # integrate the second phase of the light curve (the plateau phase)
        #timespan = [self.t_ion, self.max_integration_time]
        print("timespans:", self.t_shock_transparent, " ", self.max_integration_time)
        if self.max_integration_time < self.t_shock_transparent: return # failsafe in case max integration time happens before t_ion
        if self.stop_after_opt_thick_phase: self.max_integration_time = self.t_shock_transparent + 10*self.day
        timespan_plateau_phase = [self.t_shock_transparent, self.max_integration_time]
        y_init = self.get_init_conditions_for_next_phase(result_arrs)

        result_arrs_plateau = integrate.solve_ivp(self.dy_dt_phase_transparent_shock, timespan_plateau_phase, y_init, args=(self,), events=(), max_step=4.0*self.max_timestep, method="RK45")

        self.fill_results_phase_transparent_shock(result_arrs_plateau)
        # debug:
        if self.debug:
            print(result_arrs.t)
            print(result_arrs.y[0])
            print(result_arrs.y[1])
            print(result_arrs.y[2])
            print(result_arrs.y[3])
            print(result_arrs.y[4])
            print(result_arrs.y[5])
        

    
    #---------------------------------------------------------
    # integrator post-processing
    def fill_results_phase_opt_thick_shock(self, results_array):
        # fill the class arrays with physical values using the integrator results from the plateau phase
        time = results_array.t; Sr = results_array.y[0]*self.scale_momentum; Msh = results_array.y[1]*self.scale_Mass; Rs = results_array.y[2]; Rfs = results_array.y[3]; Rrs = results_array.y[4]; Eint_sh = results_array.y[5]*self.scale_Energy
        
        # save basic quantities and convert back to cgs units:
        self.time_arr = time
        self.momentum_shock_arr = Sr * self.c0
        self.Mshock_arr = Msh
        self.Rshock_shell_arr = Rs * self.c0
        self.Rforwardshock_arr = Rfs * self.c0
        self.Rreverseshock_arr = Rrs * self.c0
        self.Eint_shock_arr = Eint_sh * self.c0**2
        self.Rphotosphere_arr = Rs * self.c0
        
        # save auxillary quantities:
        self.t_shock_transparent = time[-1]
        self.t_shock_transparent_index = len(time)-1
        #
        beta_s = Sr / np.sqrt(Msh**2 + Sr**2)
        self.Lfac_shock_shell_arr = self.lorentz_fac(beta_s)
        self.v_shock_shell_arr = beta_s * self.c0
        ###self.optical_depth_shock_arr = self.kappa* Msh / (4.*np.pi* Rs*Rs) # in units of c=1 but does not matter since tau is dimensionless
        Vsh = 4./3. * np.pi *(Rfs**3 - Rrs**3)
        Vsh_cgs = Vsh * self.c0**3
        self.shock_average_density_arr = Msh / (self.Lfac_shock_shell_arr * Vsh_cgs)

        # compute Lbol, tau_shock, and different temperatures of bulk/surface/blackbody+color temperatures:
        Lbol_i = np.zeros(len(time))
        tau_sh_i = np.zeros(len(time))
        ##self.y_max = 3.* np.sqrt(dens /1e-9) * (self.kB * T_photosphere / 1.6e-10)**(9./4.)
        for i in range(len(time)):
            # luminosity:
            Lbol_i[i] = self.compute_diffusion_luminosity(time[i], Sr[i], Msh[i], Rs[i], Rfs[i], Rrs[i], Eint_sh[i])
            tau_sh_i[i] = self.compute_optical_depth(time[i], Sr[i], Msh[i], Rs[i], Rfs[i], Rrs[i], Eint_sh[i])
        # fill results in array:
        self.Lbol_arr = Lbol_i * (self.c0)**2 # convert to cgs units
        self.optical_depth_shock_arr = tau_sh_i
        
        # temperatures related to the shock including non-equilibrium quantities (i.e. non-thermal correction factors):
        eta_factor_i = np.zeros(len(time))
        y_max_i = np.zeros(len(time))
        T_BB_bulk_i = np.zeros(len(time))
        T_BB_surface_i = np.zeros(len(time))
        T_e_bulk_i = np.zeros(len(time))
        T_e_surface_i = np.zeros(len(time))
        for i in range(len(time)):
            # bulk shock:
            T_BB_bulk_i[i] = ( (self.Eint_shock_arr[i]/Vsh_cgs[i]) / self.arad )**(0.25)
            eta_factor_i[i] = self.compute_eta_simplified(time[i], self.shock_average_density_arr[i], T_BB_bulk_i[i], Msh[i], Rs[i], (Rfs[i]-Rrs[i]), beta_s[i])
            T_e_bulk_i[i] = self.compute_T_electron(self.shock_average_density_arr[i], eta_factor_i[i], T_BB_bulk_i[i])
            xi_factor = self.compute_xi_from_Telectron(self.shock_average_density_arr[i], T_e_bulk_i[i])
            y_max_i[i] = 3.* (self.shock_average_density_arr[i]/1e-9)**(-0.5) * (self.kB*T_e_bulk_i[i] / 1.6e-10)**(9./4.)
            # surface region:
            T_BB_surface_i[i] = ( self.Lbol_arr[i] / (4.*np.pi*self.sigma* self.Rshock_shell_arr[i]**2) )**(0.25)
            '''T_e_surface_i[i] = self.compute_T_electron(self.shock_average_density_arr[i], eta_factor_i[i], T_BB_surface_i[i])'''
            T_e_surface_i[i] = self.compute_T_electron_from_TBB_simplified(eta_factor_i[i], xi_factor, T_BB_surface_i[i])
        # fill results in array:
        # bulk temperatures
        self.shock_Temp_internal_arr = T_BB_bulk_i # average bulk BB temperature
        self.shock_Temp_internal_electron_arr = T_e_bulk_i
        # effective and surface temperatures:
        self.Temp_BB_surface = T_BB_surface_i
        self.Temp_eff_arr = T_e_surface_i # equal to color temperature , equal to observed effective temperature
        # fill non-thermla values into arrays:
        self.eta_factor_arr = eta_factor_i
        self.y_max_arr = y_max_i


        ''' old/obsolete code:
        # luminosity:
        beta_diff = (1. - beta_s**2) / ( (3.+ beta_s**2) * self.optical_depth_shock_arr  + 2.*beta_s )
        for i in range(len(beta_diff)):
            if (beta_diff[i] > 1. - beta_s[i]): beta_diff[i] = 1. - beta_s[i]
        self.Lbol_arr = 4.*np.pi*Rs**2 * beta_diff * Eint_sh/Vsh * (self.c0)**2 # in cgs units
        for k in range(len(self.Lbol_arr)):
            if (beta_diff[k] * Eint_sh[k]/Vsh[k] / self.sigma < self.Tion**4): self.Lbol_arr[k] = 4.*np.pi*Rs[k]**2 * self.sigma * self.Tion**4 * (self.c0)**2
        
        # correction for the calculation for optical depth if T<Tion:
        for k in range(len(self.optical_depth_shock_arr)):
            if (beta_diff[k] * Eint_sh[k]/Vsh[k] / self.sigma < self.Tion**4):
                beta_diff_Tion = self.sigma*self.Tion**4 / (Eint_sh[k]/Vsh[k])
                self.optical_depth_shock_arr[k] = ((1. - beta_s[k]**2)/beta_diff_Tion - 2.*beta_s[k]) / (3. + beta_s[k]**2)
        
        a_rad = 4.*self.sigma / self.c0 # radiation constant in cgs units
        self.shock_Temp_internal_arr = np.power( self.Eint_shock_arr / a_rad / Vsh_cgs , 1./4.)
        self.Pres_shock_radiation_arr = self.Eint_shock_arr / Vsh_cgs / 3.
        # effective temperature:
        self.Temp_eff_arr = np.power( self.Lbol_arr / (4.*np.pi*self.sigma* self.Rshock_shell_arr**2), 1./4.)
        '''
        # pressures:
        # first compute densities:
        rho_a_fs = np.zeros(len(time))
        rho_ej_rs = np.zeros(len(time))
        for i in range(len(time)):
            rho_a_fs[i],_ = self.get_CSM_Properies(time[i], Rfs[i])
            rho_ej_rs[i] = self.get_ejecta_density(time[i], Rrs[i])
        # forward shock:
        beta_fs = self.beta_shock(0., beta_s)
        self.Pres_forwardshock_arr = rho_a_fs * beta_s* beta_fs / (1. - beta_s*beta_fs) * (self.c0)**(-1)
        # reverse shock:
        beta_ej_rs = Rrs/time
        beta_rs = self.beta_shock(beta_ej_rs, beta_s)
        Lfac_ej_rs = self.lorentz_fac(beta_ej_rs)
        self.Pres_reverseshock_arr = rho_ej_rs* Lfac_ej_rs**2 * (beta_ej_rs - beta_rs )*(beta_ej_rs - beta_s) / (1. - beta_s*beta_rs) * (self.c0)**(-1)
        # bulk radiation pressure:
        self.Pres_shock_radiation_arr = (self.Eint_shock_arr/Vsh_cgs) / 3. # assuming radiation dominated gas, which is well fulfilled
        # velocities:
        self.v_forwardshock_arr = beta_fs * self.c0
        self.v_reverseshock_arr = beta_rs * self.c0

        # energy balance:
        # relativistic expression of the kinetic+internal energy of the envelope for a power-law density distibution
        if self.debug_Suzuki:
            self.Ekin_envelope_arr = np.ones(len(time))
            def beta_integrand(beta, self, time): # == F_nu_obs / nu_obs
                    Gamma = self.lorentz_fac(beta)
                    return ( self.get_ejecta_density(time, beta*time) * Gamma*(Gamma-1.)*beta*beta )
            for l in range(len(time)):
                res_beta_integral, err1 = integrate.quad( beta_integrand, 1e-5, beta_ej_rs[l], args=(self,time[l]), epsabs=1e-11, epsrel=1e-11)
                self.Ekin_envelope_arr[l] = res_beta_integral * 4.*np.pi*(self.c0)**2 * time[l]**3
        else:
            self.Ekin_envelope_arr = 4.*np.pi*(self.c0)**2 * self.t0**3 * self.rho0* self.analytic_powlaw_density_integral_I(beta_ej_rs)
        self.Eint_envelope_arr = self.Eexp * (self.R0 / Rrs) *(self.c0)**2 # assuming only adiabatic losses. only a rough estimate
        # kinetic+internal energy of thin shock:
        #self.Eint_shock_arr = Eint_sh * (self.c0)**2 # is already computed above
        self.Ekin_shock_arr = ( self.Lfac_shock_shell_arr- np.ones(len(time))) * self.Mshock_arr *(self.c0)**2 # rel. expression, valid for shin-shell limit
        #self.Ekin_shock_arr = 1./2. * self.Mshock_arr * beta_s*beta_s*(self.c0)**2 # non-rel limit
        #self.Ekin_shock_arr = 4.*np.pi*(self.c0)**5 * time**3 * self.shock_average_density_arr* ( beta_integral(beta_fs) - beta_integral(beta_rs) ) # rel. expression for thick shock

        # heating and cooling sources:
        self.Lforwardshock_arr = 4.*np.pi*(self.Rforwardshock_arr)**2 * self.gamma_adiab*(self.Pres_forwardshock_arr)/(self.gamma_adiab-1.)* (self.Lfac_shock_shell_arr)**2 * (beta_fs - beta_s) * self.c0
        self.Lreverseshock_arr = 4.*np.pi*(self.Rreverseshock_arr)**2 * self.gamma_adiab*(self.Pres_reverseshock_arr)/(self.gamma_adiab-1.)* (self.Lfac_shock_shell_arr)**2 * (beta_s - beta_rs) * self.c0
        self.Heating_arr = self.Lforwardshock_arr + self.Lreverseshock_arr
        #
        dVdt = 4.*np.pi * (Rfs**2 * beta_fs - Rrs**2 * beta_rs)
        self.Cooling_work_arr = Eint_sh/(3.*Vsh)*dVdt * (self.c0)**2

        ####
        t_dynamical = time[0] # dynamical time is just initial time
        # compute index of the first local minimum after the initial peak at t0=t_dynamical:
        local_min_index = 0
        min_Lbol = 1e100
        for m in range(len(time)-1,5,-1):
            if time[m] < 5.*t_dynamical:
                if self.Lbol_arr[m] < min_Lbol: # find minimum:
                    min_Lbol = self.Lbol_arr[m]
                    local_min_index = m
                else:
                    break # break once we leave the local minimum
        self.shock_opt_thick_Lbol_local_min_after_t0_index = local_min_index

        # compute the maximum value of eta_factor in the optically thick phase:
        max_eta = self.eta_factor_arr[-1]
        for n in range(len(time)-2,local_min_index,-1):
            if self.eta_factor_arr[n] > max_eta: max_eta = self.eta_factor_arr[n]
            if self.eta_factor_arr[n]<1:
                if self.eta_factor_arr[n-1] < self.eta_factor_arr[n+1]: break
            #else:
            #    break # break once we leave the local maximum
        self.shock_opt_thick_maximum_eta = max_eta
        #print(local_min_index, len(time))

        # compute peak luminosity, excluding the initial spurious peak at t< t_dynamical
        
        max_Lbol = 0.0
        max_Lbol_t = 0.
        max_index = 1
        if time[-1] < 1.4*t_dynamical: # edge case of extremely short optically thick shock phase
            max_Lbol = self.Lbol_arr[-1]; max_index = len(time)-1
        for k in range(len(time)-1,0,-1):
            #if ((self.Temp_eff_arr[k] < self.Tion+0.1) and (self.eta_factor_arr[k] > 1.)): continue # this should exclude the very sharp short peak that happens in some models
            if ((self.Temp_eff_arr[k] < self.Tion+0.1) and (max_eta > 1.) and (time[k] > time[-1]-1.*self.year)): continue # because previous line did not work, try ths one also
            if (time[k] > 1.4*t_dynamical):
                if max_Lbol < self.Lbol_arr[k]:
                    max_Lbol = self.Lbol_arr[k]; max_Lbol_t = time[k]; max_index=k
        self.shock_opt_thick_Lbol_peak_luminosity = max_Lbol
        self.shock_opt_thick_Lbol_peak_time = max_Lbol_t

        # compute variability timescale, for this scan left and right of max luminosity position:
        t_Lbol_max_later = time[max_index]
        t_Lbol_max_before = time[max_index]
        dt_Lbol_max = 0.
        variability_factor = 1.2 # corresponds to (+/-)0.2 mag variability
        for l in range(max_index,len(time)-1,1):
            if (self.Lbol_arr[l]< max_Lbol/variability_factor) or (self.Lbol_arr[l]> max_Lbol*variability_factor):
                # linearly interpolate to find better estimate fot t:
                L_star = max_Lbol/variability_factor
                t_star = time[l-1] + (L_star-self.Lbol_arr[l-1])/(self.Lbol_arr[l]-self.Lbol_arr[l-1]) * (time[l]-time[l-1])
                t_Lbol_max_later = t_star; break
        for l in range(max_index-1,1,-1):
            if (self.Lbol_arr[l]< max_Lbol/variability_factor) or (self.Lbol_arr[l]> max_Lbol*variability_factor):
                # linearly interpolate to find better estimate fot t:
                L_star = max_Lbol/variability_factor
                t_star = time[l] + (L_star-self.Lbol_arr[l])/(self.Lbol_arr[l+1]-self.Lbol_arr[l]) * (time[l+1]-time[l])
                t_Lbol_max_before = t_star; break
        dt_Lbol_max = t_Lbol_max_later-t_Lbol_max_before
        self.shock_opt_thick_Lbol_peak_variability_timescale = dt_Lbol_max

        #exit(s)

    
    def fill_results_phase_transparent_shock(self, results_array):
        # fill the class arrays with physical values using the integrator results from the plateau phase
        time = results_array.t; Sr = results_array.y[0]*self.scale_momentum; Msh = results_array.y[1]*self.scale_Mass; Rs = results_array.y[2]; Rfs = results_array.y[3]; Rrs = results_array.y[4]; Eint_sh = results_array.y[5]*self.scale_Energy
        
        # save basic quantities and convert back to cgs units:
        self.time_arr = np.append(self.time_arr, time)
        self.momentum_shock_arr = np.append(self.momentum_shock_arr, Sr * self.c0)
        self.Mshock_arr = np.append(self.Mshock_arr, Msh)
        self.Rshock_shell_arr = np.append(self.Rshock_shell_arr, Rs * self.c0)
        self.Rforwardshock_arr = np.append(self.Rforwardshock_arr, Rfs * self.c0)
        self.Rreverseshock_arr = np.append(self.Rreverseshock_arr, Rrs * self.c0)
        self.Eint_shock_arr = np.append(self.Eint_shock_arr, Eint_sh * self.c0**2)
        self.Rphotosphere_arr = np.append(self.Rphotosphere_arr, Rs * self.c0)

        # save auxillary variables:
        beta_s = Sr / np.sqrt(Msh**2 + Sr**2)
        Lfac_shock = self.lorentz_fac(beta_s)
        self.Lfac_shock_shell_arr = np.append(self.Lfac_shock_shell_arr, Lfac_shock)

        self.v_shock_shell_arr = np.append(self.v_shock_shell_arr,beta_s * self.c0)
        ###optical_depth_shock_arr = self.kappa* Msh / (4.*np.pi* Rs*Rs) # in units of c=1 but does not matter since tau is dimensionless
        Vsh = 4./3. * np.pi *(Rfs**3 - Rrs**3)
        Vsh_cgs = Vsh * self.c0**3
        shock_average_density_arr = Msh / (Lfac_shock * Vsh_cgs)
        self.shock_average_density_arr = np.append(self.shock_average_density_arr, shock_average_density_arr)

        # after shock is transparent, optical depth of the shock is below one.
        # just make some simple prescription for tau_s here. is not accurate anyways
        optical_depth_shock_arr = np.zeros(len(time))
        last_tau = self.optical_depth_shock_arr[-1]
        #print("last tau", last_tau)
        #last_T_e_bulk = self.shock_Temp_internal_electron_arr[-1]
        last_T_BB_bulk = self.shock_Temp_internal_arr[-1]
        #
        eta_factor_i = np.zeros(len(time))
        y_max_i = np.zeros(len(time))
        T_e_bulk_i = np.zeros(len(time))
        T_BB_bulk_i = np.zeros(len(time))
        #
        for i in range(len(time)):
            # get bulk temperature and ion temperature
            T_BB_bulk_i[i] = ( (Eint_sh[i]*self.c0**2/Vsh_cgs[i]) / self.arad )**(0.25)
            if T_BB_bulk_i[i] < 1: T_BB_bulk_i[i]=1
            eta_factor_i[i] = self.compute_eta_simplified(time[i], shock_average_density_arr[i], T_BB_bulk_i[i], Msh[i], Rs[i], (Rfs[i]-Rrs[i]), beta_s[i])
            T_e_bulk_i[i] = self.compute_T_electron(shock_average_density_arr[i], eta_factor_i[i], T_BB_bulk_i[i])
            y_max_i[i] = 3.* (shock_average_density_arr[i]/1e-9)**(-0.5) * (self.kB*T_e_bulk_i[i] / 1.6e-10)**(9./4.)
            # now get tau=
            optical_depth_shock_arr[i] = last_tau * (T_BB_bulk_i[i] / last_T_BB_bulk)**11 # to model quick recombination and tau dropoff
            if (optical_depth_shock_arr[i] < 1e-12): optical_depth_shock_arr[i] = 1e-12
        # fill arrays:
        self.optical_depth_shock_arr = np.append(self.optical_depth_shock_arr, optical_depth_shock_arr)
        #
        self.eta_factor_arr = np.append(self.eta_factor_arr, eta_factor_i)
        self.y_max_arr = np.append(self.y_max_arr, y_max_i)
        # shock bulk tempertures (even though the shock is now transparent we need something to fill the array):
        self.shock_Temp_internal_electron_arr = np.append(self.shock_Temp_internal_electron_arr, T_e_bulk_i)
        self.shock_Temp_internal_arr = np.append(self.shock_Temp_internal_arr, T_BB_bulk_i)


        ''' old/obsolete code:
        # correct calculation for optical depth if T<Tion:
        beta_diff = (1. - beta_s**2) / ( (3.+ beta_s**2) * optical_depth_shock_arr  + 2.*beta_s )
        for k in range(len(optical_depth_shock_arr)):
            if (beta_diff[k] * Eint_sh[k]/Vsh[k] / self.sigma < self.Tion**4):
                beta_diff_Tion = self.sigma*self.Tion**4 / (Eint_sh[k]/Vsh[k])
                optical_depth_shock_arr[k] = ((1. - beta_s[k]**2)/beta_diff_Tion - 2.*beta_s[k]) / (3. + beta_s[k]**2)
            if (optical_depth_shock_arr[k] < 1e-10): optical_depth_shock_arr[k] = 1e-10
        self.optical_depth_shock_arr = np.append(self.optical_depth_shock_arr, optical_depth_shock_arr)
        '''

        # shock region average temperature and pressure:
        #self.shock_Temp_internal_arr = np.append(self.shock_Temp_internal_arr, np.power( Eint_sh*self.c0**2 / self.arad / Vsh_cgs , 1./4.) )
        self.Pres_shock_radiation_arr = np.append(self.Pres_shock_radiation_arr, Eint_sh*self.c0**2 / Vsh_cgs / 3.)
        
        # pressures:
        # first compute densities:
        rho_a_fs = np.zeros(len(time))
        rho_ej_rs = np.zeros(len(time))
        for i in range(len(time)):
            rho_a_fs[i],_ = self.get_CSM_Properies(time[i], Rfs[i])
            rho_ej_rs[i] = self.get_ejecta_density(time[i], Rrs[i])
        # forward shock:
        beta_fs = self.beta_shock(0., beta_s)
        self.Pres_forwardshock_arr = np.append(self.Pres_forwardshock_arr, rho_a_fs * beta_s* beta_fs / (1. - beta_s*beta_fs) * (self.c0)**(-1))
        # reverse shock:
        beta_ej_rs = Rrs/time
        beta_rs = self.beta_shock(beta_ej_rs, beta_s)
        Lfac_ej_rs = self.lorentz_fac(beta_ej_rs)
        self.Pres_reverseshock_arr = np.append(self.Pres_reverseshock_arr, rho_ej_rs* Lfac_ej_rs**2 * (beta_ej_rs - beta_rs )*(beta_ej_rs - beta_s) / (1. - beta_s*beta_rs) * (self.c0)**(-1))
        # velocities:
        self.v_forwardshock_arr = np.append(self.v_forwardshock_arr, beta_fs * self.c0)
        self.v_reverseshock_arr = np.append(self.v_reverseshock_arr, beta_rs * self.c0)

        # energy balance:
        # relativistic expression of the kinetic#internal energy of the envelope for a power-law density distibution
        if self.debug_Suzuki:
            Ekin_envelope_arr = np.ones(len(time))
            def beta_integrand(beta, self, time): # == F_nu_obs / nu_obs
                    Gamma = self.lorentz_fac(beta)
                    return ( self.get_ejecta_density(time, beta*time) * Gamma*(Gamma-1.)*beta*beta )
            for l in range(len(time)):
                res_beta_integral, err1 = integrate.quad( beta_integrand, 1e-5, beta_ej_rs[l], args=(self,time[l]), epsabs=1e-11, epsrel=1e-11)
                Ekin_envelope_arr[l] = res_beta_integral * 4.*np.pi*(self.c0)**2 * time[l]**3
        else:
            Ekin_envelope_arr = 4.*np.pi*(self.c0)**2 * self.t0**3 * self.rho0* self.analytic_powlaw_density_integral_I(beta_ej_rs)
        self.Ekin_envelope_arr = np.append(self.Ekin_envelope_arr, Ekin_envelope_arr)
        # internal energy of envelope:
        ueje = (time)**(-4)*(self.t0**4) * 1. * Lfac_ej_rs*(Lfac_ej_rs-np.ones(len(time))) * self.rho0/self.c0 # converted to cgs units
        Eint_env = ueje* 4./3.*np.pi*(Rs*self.c0)**3 # only a rough estimate, not accurate
        self.Eint_envelope_arr = np.append(self.Eint_envelope_arr ,Eint_env) # assuming only adiabatic losses
        # kinetic energy of the shock (Eint is already known):
        Ekin_shock_arr = ( Lfac_shock- np.ones(len(time))) * Msh *(self.c0)**2
        self.Ekin_shock_arr = np.append(self.Ekin_shock_arr, Ekin_shock_arr) # rel expression

        # heating and cooling sources:
        self.Lforwardshock_arr = np.append(self.Lforwardshock_arr,np.zeros(len(time)))
        self.Lreverseshock_arr = np.append(self.Lreverseshock_arr,np.zeros(len(time)))
        self.Heating_arr = self.Lforwardshock_arr + self.Lreverseshock_arr # there is no heating any more in the transparent-shock phase
        #
        dVdt = 4.*np.pi * (Rfs**2 * beta_fs - Rrs**2 * beta_rs)
        Cooling_work_arr = Eint_sh/(3.*Vsh)*dVdt * (self.c0)**2
        self.Cooling_work_arr = np.append(self.Cooling_work_arr, Cooling_work_arr)

        # test to see how large T of the envelope is at this point after the shock becomes transparent:
        #Lbol_env = Eint_env/time # should be diffusion time but using 'time' we can get a rough upper estimate
        # (alternative way to) estimate approximately Lbol of envelope using derivative:
        #dEint_env = 4.*np.pi/6. * self.t0**4 * (self.rho0/self.c0**3) * (5.*(Rrs*self.c0)**4 * beta_rs*self.c0/time**6 - 6.*(Rrs*self.c0)**5 / time**7) 
        ####

        
        # final total bolometric luminosity = -dEkin_env
        # compute dEkin_env:
        dbeta_ej_rs = beta_rs/time - Rrs/time/time
        dEkin_env = 4.*np.pi * self.t0**3 * self.rho0*(Lfac_ej_rs**3 * beta_ej_rs**2 * dbeta_ej_rs* (beta_ej_rs**2 + np.sqrt(1.-beta_ej_rs**2) - 1.))
        ''' old/obsolete method:
        dEkin_sh = np.ones(len(time))
        for k in range(1,len(time)-1):
            #dEkin_sh[k] = (Ekin_shock_arr[k+1] - Ekin_shock_arr[k-1]) / 2./(time[k+1] - time[k-1]) # 2nd order derivative
            dEkin_sh[k] = (Ekin_shock_arr[k+1] - Ekin_shock_arr[k]) / (time[k+1] - time[k])
        dEkin_sh[0] = dEkin_sh[1]
        dEkin_sh[-1] = dEkin_sh[-2]
        # final shock luminosity is then:
        #print("dEkin_env=", dEkin_env* self.c0**2)
        #print("dEkin_sh", dEkin_sh)
        '''
        Lbol_arr = -self.efficiency * (dEkin_env)* self.c0**2 #- self.efficiency* (dEkin_sh)
        self.Lbol_arr = np.append(self.Lbol_arr, Lbol_arr)
        # effective temperature, not really well defined any more becaus eshock is transparent, better to go by photon color temperature?:
        Temp_eff_arr = np.power( Lbol_arr / (4.*np.pi*self.sigma* (Rs*self.c0)**2), 1./4.)
        ### This is technically wrong, because shock luminosity is in the non-thermal regime!! thus, Boltzmann law does not appply:
        self.Temp_eff_arr = np.append(self.Temp_eff_arr, Temp_eff_arr)
        self.Temp_BB_surface = np.append(self.Temp_BB_surface, Temp_eff_arr)
        #
        self.shock_transparent_Lbol_initial_luminosity = Lbol_arr[0] # initial shock luminosity is jut bolometric luminosity (after multiplying effieciency factor)
        
        # compute average energy of photons bases on luminosity:
        dt = np.zeros(len(time))
        for j in range(len(time)-1):
            dt[j] = time[j+1]-time[j]
        dt[-1] = dt[-2]
        E_radiated_bol = Lbol_arr* dt # radiated shock energy within one time step
        photon_energy_density = E_radiated_bol / Vsh_cgs # average amout of photon energy density within the shock region
        T_BB_surface = (photon_energy_density / self.arad)**(0.25)
        #print("TBB wo ways:")
        #print(Temp_eff_arr)
        #print(T_BB_surface)

        '''
        ####play around with modifying Lbol_total
        # interpolare during the time when 0<tau<1:
        print(optical_depth_shock_arr)
        Lbol_tion = 4.*np.pi*Rs**2 * self.sigma * self.Tion**4 * (self.c0)**2
        Ltot_modified = Lbol_arr*(1.-optical_depth_shock_arr) + optical_depth_shock_arr*(4.*np.pi*Rs**2 * self.sigma * self.Tion**4 * (self.c0)**2) #(self.Lbol_arr[-len(time)-1]) #

        print("Eint_env ",Eint_env)
        print("Eint_env/t_shock ",Eint_env/time)
        #
        #
        
        plt.plot(time/self.year,-0.1*dEkin_env*(self.c0**2), label="Lshock") # shock luminosity
        plt.plot(time/self.year, Eint_env/time[-1], label="Eint/t_transparent") # envelope luminosity estimate
        plt.plot(time/self.year, -dEint_env, label="dEint_env/dt") # envelope luminosity estimate
        plt.plot(self.time_arr/self.year, self.Lbol_arr, label="Lbol(t)", ls="--")
        plt.plot(time/self.year, Ltot_modified, label="Lbol_modified(t)", ls="--")
        plt.plot(time/self.year, Lbol_tion, label="Lbol_modified(t)", ls="--")
        plt.loglog()
        plt.grid(which="major", lw=0.7, ls="--")
        plt.xlabel("time/yr")
        #plt.ylabel("L_bol / erg/s")
        plt.ylabel("T_eff / K")
        #plt.plot(time/self.year,4.*np.pi*(self.Rshock_shell_arr*(tion/time)**(2))**2 *self.sigma*self.Tion**4  )
        plt.legend()
        plt.show()
        plt.close()
        #
        plt.plot(self.time_arr/self.year, self.Temp_eff_arr, label="Teff")
        plt.loglog()
        plt.grid(which="major", lw=0.7, ls="--")
        plt.xlabel("time/yr")
        #plt.ylabel("L_bol / erg/s")
        plt.ylabel("T_eff / K")
        #plt.plot(time/self.year,4.*np.pi*(self.Rshock_shell_arr*(tion/time)**(2))**2 *self.sigma*self.Tion**4  )
        plt.legend()
        plt.show()
        plt.close()
        exit()
        '''
        
    
    def write_data_into_file(self, filename):

        all_results = [self.time_arr, self.Mshock_arr, self.Eint_shock_arr, self.Rshock_shell_arr, self.Rforwardshock_arr, self.Rreverseshock_arr, self.Rphotosphere_arr, self.shock_average_density_arr, self.Pres_reverseshock_arr, self.Pres_forwardshock_arr, self.Pres_shock_radiation_arr, self.Lbol_arr, self.Temp_eff_arr]
        all_results_header = "time  Mshock  E_int_shock  R_shock  R_forward_shock  R_reverse_shock  R_photosphere  rho_shock_average  Pressure_reverse_shock  Pressure_forward_shock  Pressure_radiation_shock  Lbol  T_eff_surf"
        try:
            with open(filename, 'w') as file:

                file.write( "# " + all_results_header + ' [cgs units]\n') # header

                for i in range(len(self.time_arr)):
                    for element in all_results:
                        file.write(str(element[i]) + ' ')
                    file.write('\n') # line break
            print(f"Array has been successfully written to {filename}")
        except IOError:
            print(f"An error occurred while writing to the file {filename}")
    

#########################################################################
#    SMSlightcurve_sphericalCSMshock                                    #
#########################################################################

class SMSlightcurve_sphericalCSMshock_modified(SMSlightcurve_sphericalCSMshock):
    # this class has modified evolution for the kin+internal energy so that total energy is manifestly conserved
    # constrructor
    def __init__(self, E_rel_in, M0_in, R0_in, Tion_in, kappa_in, t0_in=10, Aconst_star_in=100., Gamma_max_in=5., n_density_in=0):
        super().__init__(E_rel_in, M0_in, R0_in, Tion_in, kappa_in, t0_in, Aconst_star_in, Gamma_max_in, n_density_in)

    @staticmethod # use this so that we can use solveivp integrator. Note we just pass 'self' as an extra argument
    def dy_dt_phase_opt_thick_shock(t, y, self):
        # for better readability of the equations
        Sr = y[0]*self.scale_momentum; Msh = y[1]*self.scale_Mass; Rs = y[2]; Rfs = y[3]; Rrs = y[4]; Eint_sh = y[5]*self.scale_Energy
        #
        # compute all auxillarry variables:
        beta_s = Sr / np.sqrt(Msh**2 + Sr**2)
        if self.debug: print("betas " + str(beta_s))
        Lfac_s = self.lorentz_fac(beta_s)
        if self.debug: print("betas, Lfac_s" + str(beta_s) + ", " + str(Lfac_s))
        # forward shock:
        rho_a_fs,_ = self.get_CSM_Properies(t, Rfs)
        beta_fs = self.beta_shock(0., beta_s)
        Pfs = rho_a_fs * beta_s* beta_fs / (1. - beta_s*beta_fs)
        # reverse shock:
        beta_ej_rs = Rrs/t
        rho_ej_rs = self.get_ejecta_density(t, Rrs)
        beta_rs = self.beta_shock(beta_ej_rs, beta_s)
        Lfac_ej_rs = self.lorentz_fac(beta_ej_rs)
        Prs = rho_ej_rs* Lfac_ej_rs**2 * (beta_ej_rs - beta_rs )*(beta_ej_rs - beta_s) / (1. - beta_s*beta_rs)
        # ODEs dydt:
        Frs = rho_ej_rs* Lfac_ej_rs**2 * beta_ej_rs * (beta_ej_rs - beta_rs)#max(beta_ej_rs - beta_rs,0.)
        dSr = 4.*np.pi*Rrs*Rrs*Prs - 4.*np.pi*Rfs*Rfs*Pfs + 4.*np.pi*Rrs*Rrs*Frs     # momentum of shell, NOTE: Typo in eq. (15) in Suzuki et al.
        #
        Gfs = - rho_a_fs*beta_fs
        Grs = rho_ej_rs * Lfac_ej_rs * (beta_ej_rs - beta_rs)# max(beta_ej_rs - beta_rs,0.)
        dMsh = 4.*np.pi*Rrs*Rrs*Grs - 4.*np.pi*Rfs*Rfs*Gfs  # mass of shell
        #
        dRs = beta_s # velocity of shell
        #
        dRfs = beta_fs # velocity of forward shock
        #
        dRrs = beta_rs # velocity of reverse shock
        #
        dEdiff = self.compute_diffusion_luminosity(t, Sr, Msh, Rs, Rfs, Rrs, Eint_sh)

        # compute the modified energy balance:
        dbeta_ej_rs = dRrs/t - Rrs/t/t
        dEkin_env = 4.*np.pi * self.t0**3 * self.rho0* self.analytic_powlaw_density_derivative_I(beta_ej_rs, dbeta_ej_rs)
        dEkin_sh = (Lfac_s - Lfac_s*beta_s**2 - 1.)*dMsh + beta_s*dSr # relativistic expression
        dEint_sh = - dEkin_env - dEkin_sh - dEdiff# manifestly energy conserving form
        
        return [dSr/self.scale_momentum, dMsh/self.scale_Mass, dRs, dRfs, dRrs, dEint_sh/self.scale_Energy]
    '''
    @staticmethod
    def dy_dt_phase_plateau(t, y, self):
        # for better readability of the equations
        Sr = y[0]*self.scale_momentum; Msh = y[1]*self.scale_Mass; Rs = y[2]; Rfs = y[3]; Rrs = y[4]; Eint_sh = y[5]*self.scale_Energy
        #
        # compute all auxillarry variables:
        beta_s = Sr / np.sqrt(Msh**2 + Sr**2)
        Lfac_s = self.lorentz_fac(beta_s)
        if self.debug: print("betas " + str(beta_s))
        if self.debug: print("betas, Lfac_s" + str(beta_s) + ", " + str(Lfac_s))
        # forward shock:
        rho_a_fs,_ = self.get_CSM_Properies(t, Rfs)
        beta_fs = self.beta_shock(0., beta_s)
        Pfs = rho_a_fs * beta_s* beta_fs / (1. - beta_s*beta_fs)
        # reverse shock:
        beta_ej_rs = Rrs/t
        rho_ej_rs = self.get_ejecta_density(t, Rrs)
        beta_rs = self.beta_shock(beta_ej_rs, beta_s)
        Lfac_ej_rs = self.lorentz_fac(beta_ej_rs)
        Prs = rho_ej_rs* Lfac_ej_rs**2 * (beta_ej_rs - beta_rs )*(beta_ej_rs - beta_s) / (1. - beta_s*beta_rs)
        # ODEs dydt:
        Frs = rho_ej_rs* Lfac_ej_rs**2 * beta_ej_rs * (beta_ej_rs - beta_rs)#max(beta_ej_rs - beta_rs,0.)
        dSr = 4.*np.pi*Rrs*Rrs*Prs - 4.*np.pi*Rfs*Rfs*Pfs + 4.*np.pi*Rrs*Rrs*Frs     # momentum of shell, NOTE: Typo in eq. (15) in Suzuki et al.
        #
        Gfs = - rho_a_fs*beta_fs
        Grs = rho_ej_rs * Lfac_ej_rs * (beta_ej_rs - beta_rs)# max(beta_ej_rs - beta_rs,0.)
        dMsh = 4.*np.pi*Rrs*Rrs*Grs - 4.*np.pi*Rfs*Rfs*Gfs  # mass of shell
        #
        dRs = beta_s # velocity of shell
        #
        dRfs = beta_fs # velocity of forward shock
        #
        dRrs = beta_rs # velocity of reverse shock
        #
        #Vsh = 4./3. * np.pi *(Rfs**3 - Rrs**3)
        #dVdt = 4.*np.pi * (Rfs**2 * dRfs - Rrs**2 * dRrs)
        dbeta_ej_rs = beta_rs/t - Rrs/t/t
        dEkin_env = 4.*np.pi* self.rho0*(Lfac_ej_rs**3 * beta_ej_rs**2 * dbeta_ej_rs* (beta_ej_rs**2 + np.sqrt(1.-beta_ej_rs**2) - 1.))
        dEkin_sh = (Lfac_s - Lfac_s*beta_s**2 - 1.)*dMsh + beta_s*dSr
        dEint_sh = -(1.-self.efficiency)*dEkin_env - dEkin_sh # internal energy of shell, in transparent phase, there is no diffusion luminosity any more
        
        return [dSr/self.scale_momentum, dMsh/self.scale_Mass, dRs, dRfs, dRrs, dEint_sh/self.scale_Energy]
    '''


#########################################################################
#    Telescope_filter class                                             #
#########################################################################

# filter implemented as a simple plateau-step-function
class Telescope_filter:
    # member variables:
    debug = False

    # physical model parameters:
    nu_min = 0.
    nu_max = 0.
    filter_transmissivity = 1. # amount of flux that is transmitted through the filter relative to total flux
    filter_name = "none"
    filter_magnitude_bound = 0. # limiting magnitude until a source can be observed

    # constructor:
    def __init__(self, filter_name_in, nu_min_in = 1e10, nu_max_in = 1e15, filter_transmissivity_in = 1.0):
        self.nu_min = nu_min_in
        self.nu_max = nu_max_in
        self.filter_transmissivity = filter_transmissivity_in
        self.filter_name = filter_name_in
        self.check_for_implemented_filters() # will overwrite any previous input if a pre-implemented filter exists
    
    def check_for_implemented_filters(self):
        ##### JWST NIRCam filters filters:
        print(self.filter_name)
        if self.filter_name == "JWST_NIRCam_F150W2": # JWST NIRCAM short wavelength filter
            self.nu_min = 1.2596e14; self.nu_max = 2.9771e14 # (1.007 - 2.38) micometers converted to Hz
            self.filter_transmissivity = 0.489
            self.filter_magnitude_bound = 29.8
            return
        if self.filter_name == "JWST_NIRCam_F322W2": # JWST NIRCAM long wavelength filter
            self.nu_min = 7.4705e13; self.nu_max = 1.2327e14 # (2.432 - 4.013) micometers converted to Hz
            self.filter_transmissivity = 0.499
            self.filter_magnitude_bound = 29.1
            return
        if self.filter_name == "JWST_NIRCam_F070W": # JWST NIRCAM long wavelength filter
            self.nu_min = 3.839e14; self.nu_max = 4.812e14 # (0.623 - 0.781) micometers converted to Hz
            self.filter_transmissivity = 0.235
            self.filter_magnitude_bound = 28.5
            return
        if self.filter_name == "JWST_NIRCam_F090W": # JWST NIRCAM long wavelength filter
            self.nu_min = 2.983e14; self.nu_max = 3.771e14 # (0.795 - 1.005) micometers converted to Hz
            self.filter_transmissivity = 0.306
            self.filter_magnitude_bound = 28.7
            return
        if self.filter_name == "JWST_NIRCam_F115W": # JWST NIRCAM long wavelength filter
            self.nu_min = 2.338e14; self.nu_max = 2.959e14 # (1.013 - 1.282) micometers converted to Hz
            self.filter_transmissivity = 0.328
            self.filter_magnitude_bound = 28.8
            return
        if self.filter_name == "JWST_NIRCam_F150W": # JWST NIRCAM long wavelength filter
            self.nu_min = 1.797e14; self.nu_max = 2.252e14 # (1.331 - 1.668) micometers converted to Hz
            self.filter_transmissivity = 0.457
            self.filter_magnitude_bound = 29.0
            return
        if self.filter_name == "JWST_NIRCam_F200W": # JWST NIRCAM long wavelength filter
            self.nu_min = 1.346e14; self.nu_max = 1.708e14 # (1.755 - 2.228) micometers converted to Hz
            self.filter_transmissivity = 0.506
            self.filter_magnitude_bound = 29.1
            return
        if self.filter_name == "JWST_NIRCam_F277W": # JWST NIRCAM long wavelength filter
            self.nu_min = 9.575e13; self.nu_max = 1.238e14 # (2.422 - 3.131) micometers converted to Hz
            self.filter_transmissivity = 0.439
            self.filter_magnitude_bound = 28.7
            return
        if self.filter_name == "JWST_NIRCam_F356W": # JWST NIRCAM long wavelength filter
            self.nu_min = 7.531e13; self.nu_max = 9.56e13 # (3.136 - 3.981) micometers converted to Hz
            self.filter_transmissivity = 0.539
            self.filter_magnitude_bound = 28.7
            return
        if self.filter_name == "JWST_NIRCam_F444W": # JWST NIRCAM long wavelength filter
            self.nu_min = 6.018e13; self.nu_max = 7.725e13 # (3.881 - 4.982) micometers converted to Hz
            self.filter_transmissivity = 0.532
            self.filter_magnitude_bound = 28.3
            return
        ##### EUCLID NISP camera filters:
        if self.filter_name == "EUCLID_NISP_Hband": # EUCLID H-band filter
            self.nu_min = 1.4831e14; self.nu_max = 1.9704e14 # (1521.5 - 2021.4) nanometers converted to Hz (50% cut-on and cut-off)
            self.filter_transmissivity = 0.782
            self.filter_magnitude_bound = 26.5 # Euclid deep field; euclidwide field: 24.5
            return
        if self.filter_name == "EUCLID_NISP_Jband": # EUCLID J-band filter
            self.nu_min = 1.9132e14; self.nu_max = 2.5676e14 # (1167.6 - 1567.0) nanometers converted to Hz (50% cut-on and cut-off)
            self.filter_transmissivity = 0.790
            self.filter_magnitude_bound = 26.5 # Euclid deep field; euclidwide field: 24.5
            return
        if self.filter_name == "EUCLID_NISP_Yband": # EUCLID Y-band filter
            self.nu_min = 2.4729e14; self.nu_max = 3.15704e14 # (949.6 - 1212.3) nanometers converted to Hz (50% cut-on and cut-off)
            self.filter_transmissivity = 0.772
            self.filter_magnitude_bound = 26.5 # Euclid deep field; euclidwide field: 24.5
            return
        ##### RST WFI camera filters: (Roman space telescope)
        if self.filter_name == "ROMAN_WFI_Jband": # RST J-band filter
            self.nu_min = 2.0618e14; self.nu_max = 2.6507e14 # (1.131-1.454) micrometers converted to Hz (50% cut-on and cut-off)
            self.filter_transmissivity = 1.0
            self.filter_magnitude_bound = 29.3 # RST SN medium/deep survey: mag J= 27.6 / J = 29.3
            return
        if self.filter_name == "ROMAN_WFI_Hband": # EUCLID H-band filter
            self.nu_min = 1.6899e14; self.nu_max = 2.1724e14 # (1.380-1.774) micrometers converted to Hz (50% cut-on and cut-off)
            self.filter_transmissivity = 1.0
            self.filter_magnitude_bound = 29.4 # RST SN medium/deep survey: mag H = 28.1 / H = 29.4
            return
        print("Non-standard filter. Proceed with caution. Pre-implemented filters are:")
        print('"JWST_NIRCam_F150W2", "JWST_NIRCam_F322W2", "JWST_NIRCam_F444W", "JWST_NIRCam_F090W", "JWST_NIRCam_F200W", "EUCLID_NISP_Hband", "EUCLID_NISP_Jband", "EUCLID_NISP_Yband"')


    def filter_function(self, nu_in):
        if self.debug: print("nu_in: ", nu_in, "; filter: ", self.filter_name, ", nu_min: ", self.nu_min, ", nu_max: ", self.nu_max, ", transmssivity: ", self.filter_transmissivity,  ", magnitude_bound: ", self.filter_magnitude_bound)
        if (nu_in > self.nu_min) and (nu_in < self.nu_max): # transmission range of (nu_min - nu_max) in Hz
            return self.filter_transmissivity
        else:
            return 0.0
    
    # reference of telescope filter fuctions:
    '''
    # JWST NIRCam filters:
    # for filter transmission range/data: https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-instrumentation/nircam-filters
    # for sensitivity/magnitude bound data: https://jwst-docs.stsci.edu/jwst-near-infrared-camera/nircam-performance/nircam-sensitivity

    # JWST MIRI filters:
    # ?

    # EUCLID telescope 
    # for filter data: https://arxiv.org/abs/2203.01650 figure 1 and Table 3

    # Roman space telescope:
    # filter data: https://arxiv.org/pdf/1305.5422 Table 2-1
    '''


#########################################################################
#    ABmagnitude_lightcurve class                                       #
#########################################################################

class ABmagnitude_lightcurve:
    # member variables:
    debug = False

    # physical model parameters:
    z_redshift= 1.
    use_GP_through = True # (should stay on per default) whether to include Gunn-Peterson through: https://en.wikipedia.org/wiki/Gunn%E2%80%93Peterson_trough
    use_Balmer_attenuation = False
    telescope_filter = None # Telescope_filter object
    d_lum = 1. # luminosity distance in cm
    # cosmology:
    H_0 = 67.4 # for Planck 2018. # for Planck 2015: 67.8 # Hubble parameter in km/s/Mpc
    Omega_r = 0. # present radiation density
    Omega_m = 0.315 # for Planck 2018. # for Planck 2015: 0.308 # present matter density
    Omega_k = 0.	# curvature
    Omega_Lambda = 0.685 # for Planck 2018. # for Planck 2015: 0.692 # dark energy density

    # natural constants:
    sigma = 5.670374419e-5 # Stefan Boltzmann constant in cgs units [erg/s / cm^2 / K^4]
    c0 = 2.998e10 # speed of light [cm/s]
    m_Hydrogen = 1.6738e-24 # mean molecular wheight of atomic hydrogen [gram]
    day = 60*60*24. # seconds in a day
    year = 365.25*day
    Jansky = 1e-23 # one Jansky in cgs units: erg/s/Hz/cm^2

    # global output quantities for otically thick shock phase:
    mAB_peak_time = 0. # time when diffusion luminosity is maximal
    mAB_peak_magnitude = 0. # erg/s value when diffusion luminosity is maximal
    mAB_peak_variability_timescale = 0. # time max diffusion luminosity changes only within some small amount

    # containers for outputs/inputs over time
    time_arr = np.array(0.)
    ABmag_arr = np.array(0.)
    Lbol_arr = np.array(0.)
    Temp_surf_color_arr = np.array(0.)
    Temp_surf_BB_arr = np.array(0.)
    Rphotosphere_arr = np.array(0.)

    # constructor
    def __init__(self, SMS_lightcurve, z_redshift_in, telescope_filter_in):
        self.z_redshift = z_redshift_in
        self.telescope_filter = telescope_filter_in # filterfunction class being assigned as a member
        # take the relevant values from the SMS lightcurve:
        self.time_arr = SMS_lightcurve.time_arr
        self.Lbol_arr = SMS_lightcurve.Lbol_arr
        self.Temp_surf_color_arr = SMS_lightcurve.Temp_eff_arr
        self.Temp_surf_BB_arr = SMS_lightcurve.Temp_BB_surface
        self.Rphotosphere_arr = SMS_lightcurve.Rphotosphere_arr
    
    #---------------------------------------------------------
    # functions to compute radiation related quantities:
    @staticmethod
    def spectral_radiance_B_nu(nu, T):
        c0 = 2.998e10 # speed of light in cm/s
        h = 6.626176e-27 # in erg*s
        kB = 1.38065e-16 # in erg/K
        return ( 2.*h*pow(nu,3) / pow(c0,2) / (np.exp(h*nu/ (kB*T)) -1. ))
    
    @staticmethod
    def modified_blackBody_B_nu(nu, T):
        c0 = 2.998e10 # speed of light in cm/s
        h = 6.626176e-27 # in erg*s
        kB = 1.38065e-16 # in erg/K
        kappa_es = 0.35 # electron scattering opacity i.e. Thompson opacity
        kappa_ff = 1. # free-free opacity, i.e. Bremsstrahlung
        return 2.*( 2.*h*pow(nu,3) / pow(c0,2) / (np.exp(h*nu/ (kB*T)) -1. )) / (1.+ np.sqrt( 1. + kappa_es / kappa_ff ))

    @staticmethod
    def spectral_radiance_B_lambda(wavelength, T):
        c0 = 2.998e10 # speed of light in cm/s
        h = 6.626176e-27 # in erg*s
        kB = 1.38065e-16 # in erg/K
        return ( 2.*h*pow(c0,2) / pow(wavelength,5) / (np.exp(h*c0/ (wavelength*kB*T)) -1. ))

    def Gunn_Peterson_through(self, nu_obs): # absorption due to intergalactic unionized Hydrogen
        # blocks all frequencies larger than Lyman lines at redshift higher than 5 or 6
        if self.use_GP_through:
            nu_Lymanalpha = 2.466067546e15 # Lyman alpha line in the rest frame of the emitter in Hz
            # check if nu_obs is larger than Lyman-alpha and then exclude all frequencies where this is the case
            if nu_obs*(1. + self.z_redshift) > nu_Lymanalpha:
                return 0.
            else:
                return 1.
        else:
            return 1.
    #---------------------------------------------------------
    # function to compute luminosity distance for a given redshift:
    # see: https://en.wikipedia.org/wiki/Distance_measure and https://en.wikipedia.org/wiki/Lambda-CDM_model for the formulas and parameters
    def get_luminosity_distance_at_redshift(self):
        # get cosmology:
        # present radiation density; present matter density; curvature; dark energy density
        Omega_r = self.Omega_r; Omega_m = self.Omega_m; Omega_k = self.Omega_k; Omega_Lambda = self.Omega_Lambda
        # integrate:
        def E(z_in): # dimensionless Hubble parameter as function of redshift:
            return 1. / np.sqrt(Omega_r*pow(1.+z_in,4.) + Omega_m*pow(1.+z_in,3.) + Omega_k*pow(1.+z_in,2.) + Omega_Lambda)
        # compute comoving distance d_C and the luminosity distance d_L = (1+z)*d_C:
        res, err = integrate.quad(E, 0., self.z_redshift, epsabs=1e-10)
        d_H = self.c0 / (self.H_0 * 3.2407793e-20) # Hubble distance c/H_0 in cm
        d_C = d_H*res # compute comoving distance
        # for Omega_k=0, we can compute the luminosity distance like this:
        self.d_lum  = (1+self.z_redshift)*d_C
        if self.debug:
            print("res, d_C, d_lum, z:")
            cm_to_Gly = 9.461e26
            print(res, d_C/cm_to_Gly, self.d_lum/cm_to_Gly, self.z_redshift)
    
    #---------------------------------------------------------
    # function to use in the integrators:


    #---------------------------------------------------------
    # function to compute the full light curve in AB magnitude:
    def compute_AB_magnitude(self):
        # first compute the redshift-dependant luminosity distance to the object
        self.get_luminosity_distance_at_redshift()

        # set integration bounds to telescope filter bounds (everything outside this range will be zero anyways)
        nu_min = self.telescope_filter.nu_min #lower integral bound in the frequency
        nu_max = self.telescope_filter.nu_max #upper integral bound in the frequency
        nu_Lymanalpha_obs = 2.466067546e15/(1. + self.z_redshift) # Lyman alpha frequency in the restframe. We use this to set point sof interest to make the integrator more stable
        nu_Balmer_obs = 5.803e14/(1. + self.z_redshift) # Balmer line in observer frame
        #for debugging:
        flux_integral_arr = [0.0]*len(self.time_arr)

        # compute the flux as F_nu = pi* I_nu * (R/d_lum)^2 , where: I_nu = spectral radiance B_nu
	    # need to do this for every time step and then compute the AB magnitude
	    # flux at a specific time as a function of frequency nu (in the observer frame!):
		# full integrand: F_nu_obs = np.pi*pow(R_obj/d_lum,2) * spectral_radiance_B_nu(nu_obs*(1.+z), T_eff_in) / pow(1.+z,3)
        # so we dont need to do many multiplications withing the integrator, the frequency-independent terms have been taken out of the interand
        # TODO: for now, only step-function filter functions are implemented. one could also implement arbitrary filter functions at a later date 
        flux_integral = 1.
        def flux_integrand(nu_in, T_eff_in, k, self): # == F_nu_obs / nu_obs
            attenuation = 1.0
            if self.use_Balmer_attenuation: # include possible effects of almer absorption:
                f21 = 1e-7 # fraction of unionized hydrogen atoms with electron in n=2 exited state
                sigma_bf = 9.5e-18* ( nu_Balmer_obs/nu_in )**3 # bound-free cross-section
                if nu_in < nu_Balmer_obs: sigma_bf = 0.0 #no Balmer absorption for lower-frequency photons
                tau_Balmer = sigma_bf * f21 * (1.904e42) / self.Rphotosphere_arr[k] # Balmer opacity
                attenuation = np.exp(-tau_Balmer)
            return ( attenuation*self.spectral_radiance_B_nu(nu_in*(1.+self.z_redshift), T_eff_in) * self.Gunn_Peterson_through(nu_in) / nu_in )

        normalization_integral = 1.
        def normalization_integrand(nu_in):
            return ( 3631.*self.Jansky / nu_in )
        normalization_integral, err1 = integrate.quad(normalization_integrand, nu_min, nu_max, epsabs=1e-13, epsrel=1e-13)
        
        mag_AB = np.zeros(len(self.time_arr))
        for k in range(len(self.time_arr)):
            # compute AB magnitude for every time step k:
            flux_integral, err2 = integrate.quad( flux_integrand, nu_min, nu_max, args=(self.Temp_surf_color_arr[k], k, self,), epsabs=1e-13, epsrel=1e-13, points=(nu_Lymanalpha_obs, nu_Balmer_obs))
            # apply correction for re-scaled blackbody in case of non-equilibrium radiation effects: rescaledBB is B_nu(T_color)*(T_BB/T_color)^4 = B_nu(T_color)*L_bol/(4pi*R^2*sigma*T_color^4):
            flux_integral = flux_integral* min(1., (self.Temp_surf_BB_arr[k]/self.Temp_surf_color_arr[k])**4)
            if flux_integral < 1e-40: flux_integral=1e-40
            mag_AB[k] = -2.5*np.log10(flux_integral/normalization_integral) - 2.5*np.log10(np.pi*pow(1.+self.z_redshift,1)) -5.0*np.log10(self.Rphotosphere_arr[k]) + 5.0*np.log10(self.d_lum)
            if mag_AB[k] > 100: mag_AB[k] = 100. # clamp magnitude value to avoid spurious reuslts
            flux_integral_arr[k] = flux_integral
        # finally assign the AB magnitude reults to the results array:
        self.ABmag_arr = mag_AB
        if self.debug:
            print(normalization_integral)
            #print(flux_integral_arr)
            #print(self.ABmag_arr)
            print(self.Lbol_arr)


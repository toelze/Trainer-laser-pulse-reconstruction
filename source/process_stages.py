# %%
from typing import Tuple
from source.class_collection import (Pulse_Properties,
                              Pulse_Data,
                              Streaked_Data,
                              Raw_Data,
                              Energy_eV_Data,
                              InputMissingError,
                              Exp_Env)
import numpy as np
import cupy as cp



# %%


def measurement(streaked_data: Streaked_Data) -> Raw_Data:
    raw_data = Raw_Data(streaked_data.get_augmented_spectra(), 
                        streaked_data.pulse_props,
                        streaked_data.exp_env)
        
    return raw_data 

def to_eVs(raw_data: Raw_Data) -> Energy_eV_Data:
    

    measurement_obj = Energy_eV_Data(raw_data.augment_vls(), 
                                    raw_data.augment_tof(),
                                    raw_data.exp_env,
                                    raw_data.pulse_props) 
    
    return measurement_obj

def to_eVs_from_file(number: int, 
                     exp_env: Exp_Env,
                     path : str = './resources/raw_mathematica/up/') -> Energy_eV_Data:
    
    tof0_data = np.fromfile(path+"TOF1-"+str(number)+".dat","float32")
    tof1_data = np.fromfile(path+"TOF2-"+str(number)+".dat","float32")
    tof_times = np.fromfile('./resources/raw_mathematica/'+'TOF_times.dat',"float32") # must be the same as defined  in exp_env
    
    vls_data = np.fromfile(path+"VLS-"+str(number)+".dat","float32")
    
    
    measurement_obj = Energy_eV_Data(vls_data, 
                                    (tof0_data, tof1_data),
                                    exp_env,
                                    None)     
    return measurement_obj
    

def get_exp_env() -> Exp_Env:
    
    
    def get_THz_parameters() -> Tuple[float,
                                      cp.ndarray,
                                      cp.ndarray,
                                      cp.ndarray,
                                      cp.ndarray]:
        
        # import precomputed components of phi_el
        dt = np.dtype([('up', np.float32), ('down', np.float32)])

        p_times_A_vals = np.fromfile('./resources/m_paval_32.dat', dtype=dt)
        A_square_vals = np.fromfile('./resources/m_aquadval_32.dat', dtype=dt)       
        
        # # p*A_THz und A_THz^2 have been sampled at 2 zerocrossings ('up' and 'down') 
        # of A with p0 of 1 and E0 of 1
        
        # to calculate these contributions for arbitrary values, 
        # the base values are multiplied by E0 / E0^2 and p0 / 1
        
        h = 4.135667662 # planck constant in eV*fs
        
        pA_up = 1/h*cp.asarray(p_times_A_vals['up'])
        pA_down = 1/h*cp.asarray(p_times_A_vals['down'])
        
        Asq_up = 1/h*cp.asarray(A_square_vals['up'])
        Asq_down = 1/h*cp.asarray(A_square_vals['down'])
    

        return h, pA_up, pA_down, Asq_up, Asq_down
    
    
    
    
    

    (h, pA_up, pA_down, Asq_up, Asq_down) = get_THz_parameters()
    
    
    
    tof_params = np.asarray([-755.6928301474567, 187.2222222222, -39.8])
    tof_times = np.asarray([(i-1)/3600e6 for i in np.arange(2500)+1]) 
    tof_zeroindx = 674
    
    tof_response = np.fromfile("./resources/TOF_response.dat", dtype="float64")
    tof_response = tof_response/np.sum(tof_response)    
    
    ionization_potential = 21.55 #Neon 2p 
    vls_params = np.asarray([1239.84, 0.0032, 11.41])
    vls_pixels = np.arange(1024) + 1 # from Mathematica + indexcorrection
    vls_energies = vls_params[0]/(vls_pixels*vls_params[1] + vls_params[2])  # VLS pix 2 nm calibration 
    vls_energies -= ionization_potential
    
    sim_energies = np.linspace(40, 110, 1401)
    sim_energies_gpu = cp.asarray(sim_energies)

    
    reconstruction_time = np.linspace(-250, 250, 512)
    reconstruction_energies = np.arange(45,110.1,0.2)
    
    exp_env = Exp_Env(
                      h = h,
                      THz_p_times_A_vals_up = pA_up,
                      THz_p_times_A_vals_down = pA_down,
                      THz_A_square_vals_up = Asq_up,
                      THz_A_square_vals_down = Asq_down,
                      tof_params = tof_params,
                      tof_times = tof_times,
                      tof_zeroindx = tof_zeroindx,
                      tof_response = tof_response,
                      ionization_potential = ionization_potential,
                      vls_params = vls_params,
                      vls_pixels = vls_pixels,
                      vls_energies = vls_energies,
                      sim_energies = sim_energies,
                      sim_energies_gpu = sim_energies_gpu,
                      reconstruction_energies = reconstruction_energies,
                      reconstruction_time = reconstruction_time)

    
    return exp_env

# %%

def streaking(streakspeed : float,
              exp_env: Exp_Env,
              pulse_props: Pulse_Properties = None, 
              pulse_data: Pulse_Data = None
              ) -> Streaked_Data:    
                
    ''' Creates a StreakedData object aka employs Light-Field-Streaking on the provided pulse.
    
        
        If pulse_props is provided pulse_data is calculated from pulse_props using GetSASE.
                    
        If pulse_props is not provided it is calculated from pulse_data. '''
    
    from streaking_cal.statistics import weighted_avg_and_std

    
    
    def calc_PulseData_via_GetSASE(pulse_props: Pulse_Properties) -> Pulse_Data:
        
        ''' Calculates a synthetic SASE pulse from provided properties pulse_props 
            using a GPU optimized GetSASE function. 
            
            The original version of GetSASE function was used in the following publication: 
            https://doi.org/10.1364/OL.35.003441
            
            It can be accessed via: 
            https://confluence.desy.de/display/FLASHUSER/Partial+Coherence+Simulation
            
            
            '''
            
        from streaking_cal.GetSASE import GetSASE_gpu as GS

            
            
        (enAxis, enOutput, tAxis, tOutput) = GS(CentralEnergy=pulse_props.centralE,
                                                dE_FWHM=2.355*pulse_props.dE*2**0.5,
                                                dt_FWHM=2.355*pulse_props.dT*2**0.5,
                                                onlyT=False)
        pulse_data = Pulse_Data(enAxis= enAxis, 
                            enOutput= enOutput, 
                            tAxis= tAxis,
                            tOutput= tOutput)
        
        return pulse_data
    
    
    
    
    def calc_PulseProperties(pulse_data: Pulse_Data) -> Pulse_Properties:
        from streaking_cal.statistics import weighted_avg_and_std

        # keep in mind: pulse must be centered around 0 fs
        pass
    
    
    
    
    if pulse_props is not None:
        pulse_data = calc_PulseData_via_GetSASE(pulse_props = pulse_props)
        
        # saving temporal profile in pulse_props as reference for training
        t_square = cp.square(cp.abs(pulse_data.tOutput))
        t_mean, _ = weighted_avg_and_std(pulse_data.tAxis.get(), t_square.get())
        
        # 
        pulse_props.tOutput = cp.interp(cp.asarray(exp_env.reconstruction_time), 
                                        pulse_data.tAxis-t_mean, 
                                        t_square).get()
        pulse_props.tOutput = pulse_props.tOutput/cp.sum(pulse_props.tOutput)
        pulse_props.tAxis = exp_env.reconstruction_time
            
        
        
    elif pulse_data is not None:
        pulse_props = calc_PulseProperties(pulse_data = pulse_data)
        
    else:
        raise(InputMissingError('At least one input for pulse_data or pulse_props is expected.')) 
    
        
    return Streaked_Data(pulse_props, pulse_data, exp_env, streakspeed)
# %%
from source.class_collection import (PulseProperties,
                              PulseData,
                              StreakedData,
                              Raw_Data,
                              Energy_eV_Data,
                              InputMissingError)
import numpy as np
import cupy as cp


standard_full_time = np.linspace(-250, 250, 512) # TODO das muss noch untergebracht werden
tof_ens = np.linspace(40, 110, 1401)




def streaking(streakspeed : float,
              pulse_props: PulseProperties = None, 
              pulse_data: PulseData = None
              ) -> StreakedData:    
                
    ''' Creates a StreakedData object aka employs Light-Field-Streaking on the provided pulse.
    
        
        If pulse_props is provided pulse_data is calculated from pulse_props using GetSASE.
                    
        If pulse_props is not provided it is calculated from pulse_data. '''
    
    from streaking_cal.statistics import weighted_avg_and_std

    
    
    def calc_PulseData_via_GetSASE(pulse_props: PulseProperties) -> PulseData:
        
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
        pulse_data = PulseData(enAxis= enAxis, 
                            enOutput= enOutput, 
                            tAxis= tAxis,
                            tOutput= tOutput)
        
        return pulse_data
    
    
    
    
    def calc_PulseProperties(pulse_data: PulseData) -> PulseProperties:
        from streaking_cal.statistics import weighted_avg_and_std

        # keep in mind: pulse must be centered around 0 fs
        pass
    
    
    
    
    if pulse_props is not None:
        pulse_data = calc_PulseData_via_GetSASE(pulse_props = pulse_props)
        
        # saving temporal profile in pulse_props as reference for training
        t_square = cp.square(cp.abs(pulse_data.tOutput))
        t_mean, _ = weighted_avg_and_std(pulse_data.tAxis.get(), t_square.get())
        
        # 
        pulse_props.tOutput = cp.interp(cp.asarray(standard_full_time), 
                                        pulse_data.tAxis-t_mean, 
                                        t_square).get()
        pulse_props.tOutput = pulse_props.tOutput/cp.sum(pulse_props.tOutput)
        pulse_props.tAxis = standard_full_time
            
        
        
    elif pulse_data is not None:
        pulse_props = calc_PulseProperties(pulse_data = pulse_data)
        
    else:
        raise(InputMissingError('At least one input for pulse_data or pulse_props is expected.')) 
    
    return StreakedData(pulse_props, pulse_data, streakspeed)
# %%


def measurement(streaked_data: StreakedData) -> Raw_Data:
    raw_data = Raw_Data(streaked_data.get_augmented_spectra(), 
                        tof_ens, 
                        streaked_data.pulse_props)
        
    return raw_data 

def to_eVs(raw_data: Raw_Data) -> Energy_eV_Data:
    

    measurement_obj = Energy_eV_Data(raw_data.augment_vls(), 
                                    raw_data.augment_tof(),
                                    raw_data.tof_times, 
                                    raw_data.pulse_props) 
    
    return measurement_obj
    
    

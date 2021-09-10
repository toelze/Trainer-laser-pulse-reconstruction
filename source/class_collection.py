# %%
from dataclasses import dataclass

import cupy as cp
from cupy import interp

import numpy as np
from numba import njit
from tensorflow.keras.utils import Sequence

from typing import Union, List


# %%
class InputMissingError(Exception):
    """Exception is raised if non sufficient input is provided to a method.
    
    
    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message

       
        
        
# %%

@dataclass(frozen=True)  
class Exp_Env:    
    
    h: float     # planck constant in eV*fs
    
    THz_p_times_A_vals_up : cp.ndarray 
    THz_p_times_A_vals_down : cp.ndarray
    THz_A_square_vals_up : cp.ndarray
    THz_A_square_vals_down : cp.ndarray
    
    tof_params: np.ndarray
    tof_times: np.ndarray
    tof_zeroindx: int
    tof_response: np.ndarray
        
    ionization_potential: float
    vls_params: np.ndarray
    vls_pixels: np.ndarray
    vls_energies: np.ndarray
    
    sim_energies:np.ndarray
    sim_energies_gpu:cp.ndarray
    
    reconstruction_energies: np.ndarray
    reconstruction_time: np.ndarray
    
    
    @staticmethod
    def fs_in_au(t): return 41.3414*t  # from fs to a.u.
    
    @staticmethod
    def eV_in_au(e): return 0.271106*np.sqrt(e)  # from eV to a.u.



# %%
@dataclass
class Pulse_Data:
    enAxis: cp.ndarray
    enOutput: cp.ndarray 
    tAxis: cp.ndarray
    tOutput: cp.ndarray
    
    def to_file(self): # TODO implement saving to file
        pass
    
    @classmethod
    def from_file(cls): # TODO implement reading from file
        # read data from files
        # call PulseData() with read data
        # return object
        pass



class Pulse_Properties:
    
    def __init__(self, fwhm_t: float, 
                 fwhm_E: float, 
                 num_electrons0: int, 
                 num_electrons1: int, 
                 centralE: float):
        # from streaking_cal.GetSASE import GetSASE_gpu as GS # TODO noGPU solution missing

        self.dE = fwhm_E/2.355 
        self.dT = fwhm_t/2.355
        self.fwhm_t = fwhm_t
        self.fwhm_E = fwhm_E
        self.num_electrons0 = num_electrons0
        self.num_electrons1 = num_electrons1
        self.centralE = centralE
        self.p0 = 0.271106*np.sqrt(centralE)
        self.temp_offset = 0



# %%
class Energy_eV_Data():
 
    def __init__(self, vls_data: np.ndarray, 
                 tof_data: np.ndarray, 
                 exp_env: Exp_Env,
                 pulse_props: Union[Pulse_Properties, None]):
        
        self.vls_in_data = vls_data # measured data
        self.vls_in_len = len(self.vls_in_data)

        self.pulse_props = pulse_props
        self.exp_env = exp_env

        self.tof_in_data = tof_data
        # self.tof_in_times = tof_times

        self.tof_eVs = self.exp_env.tof_params[0]**2/(self.exp_env.tof_times - self.exp_env.tof_params[1])**2 + self.exp_env.tof_params[2]

        self.tof_energies = np.array(list(map(self.tof_to_eV,self.exp_env.tof_times[self.exp_env.tof_zeroindx + 1:]*1e9)))  # OK

        self.spectra = np.asarray([self.vls_pix_to_eVenergies(self.vls_in_data, self.exp_env.reconstruction_energies),
                                   self.tof_to_eVenergies(self.tof_in_data[0], self.exp_env.reconstruction_energies),
                                   self.tof_to_eVenergies(self.tof_in_data[1], self.exp_env.reconstruction_energies) ])
        
        self.spectra[0]= self.spectra[0]/sum(self.spectra[0])
        self.spectra[1]= self.spectra[1]/sum(self.spectra[1])
        self.spectra[2]= self.spectra[2]/sum(self.spectra[2])



    def vls_pix_to_eVenergies(self,vls,energies = None):
        '''interpolation to calculate a spectrum from a VLS signal'''
        if energies is None:
            energies = self.exp_env.reconstruction_energies
        spec =  np.interp(energies,self.exp_env.vls_energies[::-1],vls[::-1],0,0)
        return spec

    def tof_to_eVenergies(self,tof,energies = None):
        '''interpolation and intensity correction to calculate a spectrum from a TOF signal'''
        if energies is None: 
            energies = self.exp_env.reconstruction_energies

        spec = -self.correctints(tof)
        spec = np.interp(energies,self.tof_energies[::-1],spec[::-1],0,0)
        return spec

    def correctints(self,spec):
        '''from TOF times to eV'''
        return 0.5 * self.exp_env.tof_params[0] * spec[self.exp_env.tof_zeroindx + 1:]/(self.tof_energies + self.exp_env.tof_params[2])**1.5

    def tof_to_eV(self,t):

        return self.exp_env.tof_params[0]**2/(t - self.exp_env.tof_params[1])**2 - self.exp_env.tof_params[2]


# %%
class Raw_Data():

    def __init__(self, spectra: np.ndarray, 
                 pulse_props: Pulse_Properties,
                 exp_env: Exp_Env):
        
        self.pulse_props = pulse_props
        self.exp_env = exp_env

        self.spectra = spectra
        self.tof_response = self.get_random_response_curve()

        self.tof_energies = np.array(list(map(self.tof_to_eV,self.exp_env.tof_times[self.exp_env.tof_zeroindx + 1:]*1e9)))  # OK

        self.calc_vls_spectrum()

        self.num_tof_noise0=int(0+np.random.rand()*35) # num of stray electrons in spectra
        self.num_tof_noise1=int(0+np.random.rand()*35)

    def tof_to_eV(self,t):

        return self.exp_env.tof_params[0]**2/(t - self.exp_env.tof_params[1])**2 - self.exp_env.tof_params[2]


    def get_random_response_curve(self):
        response=np.abs(self.exp_env.tof_response-0.015+0.03*np.random.rand(58))

        response= response/np.sum(response)
        return response

    def uncorrectints(self,cspec):
        '''from eV to TOF times'''
        spec = cspec *(self.tof_energies + self.exp_env.tof_params[[2]])**1.5/(0.5*self.exp_env.tof_params[[0]])
        spec = np.pad(spec, (self.exp_env.tof_zeroindx + 1,0),'constant',constant_values=(0, 0))
        return spec

    def eVenergies_to_tof(self,spec, energies = None):
        '''interpolation and intensity correction to calculate a TOF signal from a spectrum'''
        if energies is None: 
            energies = self.exp_env.sim_energies

        tof = np.interp(self.tof_energies,energies,spec,0,0)
        tof = -self.uncorrectints(tof)
        return tof

    def calc_tof_traces(self):

        tof_traces = self.spectra[1:]
        tof_traces = np.asarray([self.eVenergies_to_tof(tof_traces[0]),
                                 self.eVenergies_to_tof(tof_traces[1])])


        tof_traces[0] = self.discretized_spectrum(
            tof_traces[0], self.pulse_props.num_electrons0)

        tof_traces[1] = self.discretized_spectrum(
            tof_traces[1], self.pulse_props.num_electrons1)
   
        return tof_traces

    # eV to VLS pixel
    def eVenergies_to_vls_pix(self,spec,energies = None):
        '''interpolation to calculate a VLS signal from a spectrum'''
        if energies is None:
            energies = self.exp_env.sim_energies
        vls = np.interp(self.exp_env.vls_energies,energies,spec,0,0)
        return vls
    
    # VLS pixel to eV
    def vls_pix_to_eVenergies(self,vls,energies = None):
        '''interpolation to calculate a spectrum from a VLS signal'''
        if energies is None:
            energies = self.exp_env.sim_energies
        spec =  np.interp(energies,self.exp_env.vls_energies[::-1],vls[::-1],0,0)
        return spec


    def calc_vls_spectrum(self):
        self.vls_signal = self.eVenergies_to_vls_pix(self.spectra[0])


    def vls_finite_resolution(self,spectrum):
        from scipy import signal
        spectrum = np.convolve(spectrum,signal.gaussian(21, std=2),'same') # TODO is this (21+-2) correct?
        return spectrum

    def augment_vls(self):

        aug_vls = self.vls_finite_resolution(self.vls_signal)
        aug_vls = self.add_tof_noise_hf(aug_vls,0.00009,0.00013) # real measured noise = 0.00011

        return aug_vls

    def augment_tof(self):

        aug_tof0, aug_tof1 = self.calc_tof_traces()


        aug_tof0 = self.add_tof_noise(aug_tof0,self.num_tof_noise0)       
        aug_tof0 = np.convolve(aug_tof0, self.tof_response, mode="same")
        aug_tof0 = np.roll(aug_tof0,25) # convolution shift to the right
        aug_tof0 =  self.add_tof_noise_hf(aug_tof0)



        aug_tof1 = self.add_tof_noise(aug_tof1,self.num_tof_noise1)
        aug_tof1 = np.convolve(aug_tof1, self.tof_response, mode="same")
        aug_tof1 = np.roll(aug_tof1,25) # convolution shift to the right
        aug_tof1 =  self.add_tof_noise_hf(aug_tof1)

        return aug_tof0, aug_tof1

    def add_tof_noise(self,spectrum,num_noise_peaks):
        positions=np.random.randint(len(spectrum),size=num_noise_peaks)
        withspikes=spectrum+self.added_spikes(positions, len(spectrum))

        return withspikes

    def add_tof_noise_hf(self,spectrum, lower=0.00007, upper = 0.00014):
        """Add white noise to tof spectra, similar to real measurements"""
        # 0.00007 to 0.00014 from actual measured TOF spectra
        with_noise = np.abs(spectrum + np.random.uniform(lower,upper,1).item()*np.random.randn(len(spectrum)))

        return with_noise

    def discretized_spectrum(self, spectrum, num_points):
        from numpy import interp, zeros
        positions = self.discrete_positions(spectrum, num_points)

        disc_spec = self.added_spikes(positions, len(spectrum))

        return disc_spec

    @staticmethod
    @njit(fastmath=True)
    def added_spikes(positions, arr_length):
        '''simple linear interpolation and summation'''
        disc_spec = np.zeros(arr_length)
        for i in positions:
            valll = 8*np.random.rand()+0.5 # which heights are suitable? propably between 0.5 and 8.5
            (divval, modval) = np.divmod(i, 1)
            divval = int(divval)
            disc_spec[divval] += valll*(1-modval)
            disc_spec[divval+1] += valll*(modval)

        return disc_spec

    @staticmethod
    @njit(fastmath=True)
    def discrete_positions(spectrum, num_points):
        cumulative_spectrum = (np.cumsum(spectrum))/np.sum(spectrum)
        indices = np.arange(len(spectrum))
        discrete_positions = np.interp(np.random.rand(num_points), cumulative_spectrum, indices)

        return discrete_positions

# %%

class Streaked_Data(object):

    def __init__(self, 
                 pulse_props: Pulse_Properties, 
                 pulse_data: Pulse_Data, 
                 exp_env: Exp_Env,
                 streakspeed: float,
                 temp_offset: float = 0):

        from streaking_cal.statistics import weighted_avg_and_std

        self.pulse_props = pulse_props
        self.exp_env = exp_env     

        self.streakspeed = streakspeed
        self.pulse_props.temp_offset = temp_offset
        pulse_data.tOutput = cp.roll(pulse_data.tOutput, temp_offset)
        self.get_streaked_spectra(pulse_data = pulse_data, streak_speed = streakspeed) # calculate streaked spectra


#     @property for read access only members

    def get_streaked_spectra(self, pulse_data, streak_speed: float) -> None:

        from cupy.fft import fft

        def fs_in_au(t): return 41.3414*t  # from fs to a.u.

        # in V/m; shape of Vectorpotential determines: 232000 V/m = 1 meV/fs max streakspeed
        E0 = 3.2*232000*streak_speed # times 3.2 : frequency is 3.2 THz instead of 1
        ff1 = cp.flip(fft(pulse_data.tOutput*cp.exp(-1j*fs_in_au(pulse_data.tAxis)*
                                                (1/(2)*(self.pulse_props.p0*E0*self.exp_env.THz_p_times_A_vals_up +
                                                        1*E0**2*self.exp_env.THz_A_square_vals_up)))))
        ff2 = cp.flip(fft(pulse_data.tOutput*cp.exp(-1j*fs_in_au(pulse_data.tAxis)*
                                                (1/(2)*(self.pulse_props.p0*E0*self.exp_env.THz_p_times_A_vals_down +
                                                        1*E0**2*self.exp_env.THz_A_square_vals_down)))))

        streaked0 = cp.square(cp.abs(ff1))
        streaked1 = cp.square(cp.abs(ff2))


        streaked0 = interp(self.exp_env.sim_energies_gpu, pulse_data.enAxis, streaked0)
        streaked1 = interp(self.exp_env.sim_energies_gpu, pulse_data.enAxis, streaked1)
        xuvonly = interp(self.exp_env.sim_energies_gpu, pulse_data.enAxis,
                            cp.square(cp.abs(pulse_data.enOutput)))
        
        
        self._streakedspectra = np.asarray([cp.asnumpy(xuvonly),                                                                     
                                            cp.asnumpy(streaked0), 
                                            cp.asnumpy(streaked1)])
        

    def get_temp(self):
        return self.pulse_props.tOutput


    def get_spectra(self):
        '''returns streaked spectra as a numpy array'''
            
        return self._streakedspectra.copy()

    def get_augmented_spectra(self):
        '''Data augemtation to imitate: jitter between Streaking-Pulse and XUV-Pulse.
        Returns augemented spectra as a matrix.'''
        aug_spectra = np.asarray(self.get_spectra())


        shiftall = 0 # random wert ist ausgelagert

        shiftstr = 0 # np.random.randint(-120, 120)

        aug_spectra[0] = np.roll(aug_spectra[0], shiftall)
        aug_spectra[1] = np.roll(aug_spectra[1], shiftall-shiftstr)
        aug_spectra[2] = np.roll(aug_spectra[2], shiftall+shiftstr)


        return aug_spectra



# %%


class Data_Generator(Sequence):
    def __init__(self, x_set, y_set, batch_size, X, for_train=True):
        self.X=X
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size
        self.for_train = for_train

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        from source.process_stages import (measurement, to_eVs)

        batch = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]

        x = []
        for _,i in enumerate(batch):

            b1 = to_eVs(measurement(self.X[i]))

            x.append(b1.spectra)

        x = np.asarray(x)

        y = [self.X[i].get_temp() for i in batch]
        y = np.asarray(y)

        return x.reshape(-1, 3, x[0].shape[1], 1), np.array(y)

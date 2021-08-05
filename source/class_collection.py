# %%
from dataclasses import dataclass
from source.process_stages import measurement
# from typing import Tuple

import cupy as cp
import numpy as np
# import pandas as pd
# import scipy
# from streaking_cal.misc import interp
from cupy import interp
from numba import boolean, float64, njit, vectorize
from tensorflow.keras.utils import Sequence

h = 4.135667662  # in eV*fs

dt = np.dtype([('up', np.float32), ('down', np.float32)])
# import precomputed components of phi_el
# p*A_THz und A_THz^2 have been sampled at 2 zerocrossings ('up' and 'down') of A with p0 of 1 and E0 of 1
# to calculate these contributions for arbitrary values, the base values are multiplied by E0 / E0^2 and p0 / 1
p_times_A_vals = np.fromfile('./resources/m_paval.dat', dtype=dt)
p_times_A_vals_up = 1/h*cp.asarray(p_times_A_vals['up'])
p_times_A_vals_down = 1/h*cp.asarray(p_times_A_vals['down'])
del(p_times_A_vals)
A_square_vals = np.fromfile('./resources/m_aquadval.dat', dtype=dt)
A_square_vals_up = 1/h*cp.asarray(A_square_vals['up'])
A_square_vals_down = 1/h*cp.asarray(A_square_vals['down'])
del(A_square_vals)


streaking_dict = {}

measurement_env_dict = {}

# %%
a = "dd"
measurement_env_dict = {}
measurement_env_dict["ddd"]= a

# %%

# %%

def fs_in_au(t): return 41.3414*t  # from fs to a.u.
def eV_in_au(e): return 0.271106*np.sqrt(e)  # from eV to a.u.


# load noise peak for discretization of spectra
orig_tof_ens = np.genfromtxt("./resources/energies.csv", delimiter=',')


# df0=pd.read_csv("./FLASH-Spectra/0.0119464/"+"spec10.csv",header=None)
noisepeak = np.fromfile('./resources/noisepeak.dat', dtype="float64")
# noisepeak=(df0[1].values/sum(df0[1]))[orig_tof_ens<61]
noisepeak_gpu = cp.asarray(noisepeak)

# peak_max_y = orig_tof_ens[len(noisepeak)]-orig_tof_ens[0]
tof_ens = np.linspace(40, 110, 1401)
tof_ens_gpu = cp.asarray(tof_ens)


dt2 = np.dtype([('xuv', np.float64), ('up', np.float64), ('down', np.float64)])



# Pulse class returns temporal profile on basis of this time axis
standard_full_time = np.loadtxt('./resources/standard_time.txt')
standard_full_time = np.linspace(-250, 250, 512)

# %%
class InputMissingError(Exception):
    """Exception is raised if non sufficient input is provided to a method.
    
    
    Attributes:
        message -- explanation of the error
    """

    def __init__(self, message):
        self.message = message


# %%
@dataclass
class PulseData:
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



class PulseProperties:
    
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
        self.p0 = PulseProperties.eV_in_au(centralE)


    @staticmethod
    def fs_in_au(t): return 41.3414*t  # from fs to a.u.

    @staticmethod
    def eV_in_au(e): return 0.271106*np.sqrt(e)  # from eV to a.u.

# %%
class Energy_eV_Data():
    tof_params = [-755.6928301474567, 187.2222222222222, -39.8]
    vls_params = [1239.84, 0.0032, 11.41]
    energy_axis = np.arange(45,110.1,0.2)
    zeroindx = 674

    ionization_potential = 21.55 #Neon 2p 
    vls_pixels = np.arange(1024) + 1 # from Mathematica + indexcorrection
    vls_energies = 1239.84/(vls_pixels*0.0032 + 11.41)  # VLS pix 2 nm calibration 
    vls_energies -= ionization_potential

    def __init__(self, vls_data, tof_data, tof_times, pulse_props):
        self.vls_in_data = vls_data # measured data
        self.vls_in_len = len(self.vls_in_data)

        self.pulse_props = pulse_props

        self.tof_in_data = tof_data
        self.tof_in_times = tof_times

        self.tof_eVs = self.tof_params[0]**2/(self.tof_in_times - self.tof_params[1])**2 + self.tof_params[2]

        self.tof_energies = np.array(list(map(self.tof_to_eV,self.tof_in_times[self.zeroindx + 1:]*1e9)))  # OK

        self.spectra = np.asarray([self.vls_pix_to_eVenergies(self.vls_in_data, self.energy_axis),
                                   self.tof_to_eVenergies(self.tof_in_data[0], self.energy_axis),
                                   self.tof_to_eVenergies(self.tof_in_data[1], self.energy_axis) ])



    def vls_pix_to_eVenergies(self,vls,energies = None):
        '''interpolation to calculate a spectrum from a VLS signal'''
        if energies is None:
            energies = self.energy_axis
        spec =  np.interp(energies,self.vls_energies[::-1],vls[::-1],0,0)
        return spec

    def tof_to_eVenergies(self,tof,energies = None):
        '''interpolation and intensity correction to calculate a spectrum from a TOF signal'''
        if energies is None: 
            energies = self.energy_axis

        spec = -self.correctints(tof)
        spec = np.interp(energies,self.tof_energies[::-1],spec[::-1],0,0)
        return spec

    def correctints(self,spec):
        '''from TOF times to eV'''
        return 0.5 * self.tof_params[0] * spec[self.zeroindx + 1:]/(self.tof_energies + self.tof_params[2])**1.5

    def tof_to_eV(self,t):

        return self.tof_params[0]**2/(t - self.tof_params[1])**2 - self.tof_params[2]


# %%
class Raw_Data():
    
    # parameters from TOF calibration
    tof_params = np.asarray([-755.6928301474567, 187.2222222222, -39.8])

    tof_times = np.asarray([(i-1)/3600e6 for i in np.arange(2500)+1]) # OK
    zeroindx = 674 # OK

    real_tof_response = np.fromfile("./resources/TOF_response.dat", dtype="float64")
    real_tof_response = real_tof_response/np.sum(real_tof_response)
    ionization_potential = 21.55 #Neon 2p 

    vls_pixels = np.arange(1024) + 1 # from Mathematica + indexcorrection
    vls_enenergies = 1239.84/(vls_pixels*0.0032 + 11.41)  # VLS pix 2 nm calibration 
    vls_enenergies -= ionization_potential


    vls_pixels = np.arange(1024)  # pixels of spectrometer

    def __init__(self, spectra: np.ndarray, 
                 energies: np.ndarray,
                 pulse_props: PulseProperties):
        self.pulse_props = pulse_props
        self.energy_axis = energies

        self.spectra = spectra
        self.tof_response = self.get_random_response_curve()

        self.tof_energies = np.array(list(map(self.tof_to_eV,self.tof_times[self.zeroindx + 1:]*1e9)))  # OK


        self.calc_vls_spectrum()

        self.num_tof_noise0=int(0+np.random.rand()*3) # num of stray electrons in spectra
        self.num_tof_noise1=int(0+np.random.rand()*3)

    def tof_to_eV(self,t):

        return self.tof_params[0]**2/(t - self.tof_params[1])**2 - self.tof_params[2]


    def get_random_response_curve(self):
        response=np.abs(self.real_tof_response-0.015+0.03*np.random.rand(58))

        response= response/np.sum(response)
        return response

    def correctints(self,spec):
        '''from TOF times to eV'''
        return 0.5 * self.tof_params[0] * spec[self.zeroindx + 1:]/(self.tof_energies + self.tof_params[2])**1.5

    def uncorrectints(self,cspec):
        '''from eV to TOF times'''
        spec = cspec *(self.tof_energies + self.tof_params[[2]])**1.5/(0.5*self.tof_params[[0]])
        spec = np.pad(spec, (self.zeroindx + 1,0),'constant',constant_values=(0, 0))
        return spec

    def eVenergies_to_tof(self,spec, energies = None):
        '''interpolation and intensity correction to calculate a TOF signal from a spectrum'''
        if energies is None: 
            energies = self.energy_axis

        tof = np.interp(self.tof_energies,energies,spec,0,0)
        tof = -self.uncorrectints(tof)
        return tof

    def tof_to_eVenergies(self,tof,energies = None):
        '''interpolation and intensity correction to calculate a spectrum from a TOF signal'''
        if energies is None: 
            energies = self.energy_axis

        spec = -self.correctints(tof)
        spec = np.interp(energies,self.tof_energies[::-1],spec[::-1],0,0)
        return spec


    def calc_tof_traces(self):
        from numpy import argsort, asarray, take_along_axis

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
            energies = self.energy_axis
        vls = np.interp(self.vls_enenergies,energies,spec,0,0)
        return vls
    # VLS pixel to eV

    def vls_pix_to_eVenergies(self,vls,energies = None):
        '''interpolation to calculate a spectrum from a VLS signal'''
        if energies is None:
            energies = self.energy_axis
        spec =  np.interp(energies,self.vls_enenergies[::-1],vls[::-1],0,0)
        return spec



    def calc_vls_spectrum(self):
        from numpy import argsort, take_along_axis


        self.vls_signal = self.eVenergies_to_vls_pix(self.spectra[0])

        self.vls_signal = self.vls_signal/np.sum(self.vls_signal)


    def vls_finite_resolution(self,spectrum):
        from scipy import signal
        spectrum = np.convolve(spectrum,signal.gaussian(21, std=2),'same') # TODO is this (21+-2) correct?
        return spectrum

    def augment_vls(self):
        from numpy import roll

        aug_vls = self.vls_finite_resolution(self.vls_signal)
        aug_vls = self.add_tof_noise_hf(aug_vls,0.00009,0.00013) # real measured noise = 0.00011
        aug_vls = aug_vls/np.sum(aug_vls)

        return aug_vls

    def augment_tof(self):

        aug_tof0, aug_tof1 = self.calc_tof_traces()


        aug_tof0 = self.add_tof_noise(aug_tof0,self.num_tof_noise0)       
        aug_tof0 = np.convolve(aug_tof0, self.tof_response, mode="same")
        aug_tof0 = np.roll(aug_tof0,25) # convolution shift to the right
        aug_tof0 = aug_tof0/np.sum(aug_tof0)
        aug_tof0 =  self.add_tof_noise_hf(aug_tof0)
        aug_tof0 = aug_tof0/np.sum(aug_tof0)



        aug_tof1 = self.add_tof_noise(aug_tof1,self.num_tof_noise1)
        aug_tof1 = np.convolve(aug_tof1, self.tof_response, mode="same")
        aug_tof1 = np.roll(aug_tof1,25) # convolution shift to the right
        aug_tof1 = aug_tof1/np.sum(aug_tof1)
        aug_tof1 =  self.add_tof_noise_hf(aug_tof1)
        aug_tof1 = aug_tof1/np.sum(aug_tof1)        

        return aug_tof0, aug_tof1

    def get_raw_matrix(self):
        from numpy import pad, roll
        from numpy import sum as npsum

        vls_new = self.augment_vls()
        aug_tof0, aug_tof1 = self.augment_tof()

        tof_new0 = aug_tof0[self.zeroindx + 1:]
        tof_new1 = aug_tof1[self.zeroindx + 1:]

        vls_new = pad(vls_new, pad_width=(0, len(tof_new0)-len(self.vls_signal)))


        r = 0
        tof_new0 = roll(tof_new0, r) # roll, so that TOF and VLS are closer together
        tof_new1 = roll(tof_new1, r)


        return np.asarray([vls_new, tof_new0, tof_new1])

    def add_tof_noise(self,spectrum,num_noise_peaks):
        positions=np.random.randint(len(spectrum),size=num_noise_peaks)
        withspikes=spectrum+self.added_spikes(positions, len(spectrum))

        return withspikes

    def add_tof_noise_hf(self,spectrum, lower=0.00007, upper = 0.00014):
        """Add white noise to spectra, similar to real measurements"""
        # 0.00007 to 0.00014 from actual measured TOF spectra
        with_noise = np.abs(spectrum + np.random.uniform(lower,upper,1).item()*np.random.randn(len(spectrum)))

        return with_noise

    def energies_to_tof_times(self, energies_eV):  # in ns
        from numpy import sqrt
        TOF_times = self.tof_params[1]+ self.tof_params[0]**2/sqrt((self.tof_params[0]**2)*(
            energies_eV - self.tof_params[2]))  # funktioniert
        return TOF_times  # -min(TOF_times)

    def vls_pixel_to_energies(self, vls_pixel):
        return -21.5 + 1239.84/(11.41 + 0.0032*vls_pixel)

    def energies_to_vls_pixel(self, energies_eV):
        # calibration and 21.5 eV ionization
        return -3565.63 + 387450/(21.5 + energies_eV)


    def tof_signal_correction(self, signal):
        adjusted_signal = -4 * \
            (signal*(self.energy_axis +
                     self.tof_params[2])**1.5)/self.tof_params[0]
        return adjusted_signal


    def discretized_spectrum(self, spectrum, num_points):
        from numpy import interp, zeros
#         disc_spec=np.zeros(len(spectrum))
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
    
    def to_Measurement_Data(self):
        measurement_obj = Energy_eV_Data(self.augment_vls(), 
                                         self.augment_tof(),
                                         self.tof_times, 
                                         self.pulse_props) 


        return measurement_obj


# %%

class StreakedData(object):
    @staticmethod
    def fs_in_au(t): return 41.3414*t  # from fs to a.u.
    @staticmethod
    def eV_in_au(e): return 0.271106*np.sqrt(e)  # from eV to a.u.
    h = 4.135667662  # in eV*fs

    class Data_Nonexisting_Error(Exception):
        def __init__(self, message):
            self.message = message
#

    def __init__(self, pulse_props, pulse_data, streakspeed: float):

        from streaking_cal.statistics import weighted_avg_and_std

        # centralE, dE = weighted_avg_and_std(
        #     EnAxis.get(), cp.square(cp.abs(EnOutput)).get())

        self.pulse_props = pulse_props


        
        self.__is_low_res = False
        self.__streakspeed = streakspeed
        
        self._get_streaked_spectra(pulse_data = pulse_data, streak_speed = streakspeed) # calculate streaked spectra


#     @property for read access only members
#    
    def get_temp(self):
        return self.pulse_props.tOutput

    def get_tAxis(self):
        return self.pulse_props.tAxis

    def get_spec(self):
        if self.__is_low_res:
            return None
        else:
            return self.pulse_props.enOutput

    def get_eAxis(self):
        if self.__is_low_res:
            return None
        else:
            return self.pulse_props.enAxis

    def is_low_res(self):
        return self.__is_low_res

    def _get_streaked_spectra(self, pulse_data, streak_speed: float) -> None:
        if self.is_low_res == True:
            return None
        else:
            from cupy.fft import fft

            def fs_in_au(t): return 41.3414*t  # from fs to a.u.
            # def eV_in_au(e): return 0.271106*np.sqrt(e)  # from eV to a.u.

            # in V/m; shape of Vectorpotential determines: 232000 V/m = 1 meV/fs max streakspeed
            E0 = 232000*streak_speed
            ff1 = cp.flip(fft(pulse_data.tOutput*cp.exp(-1j*fs_in_au(pulse_data.tAxis)*
                                                 (1/(2)*(self.pulse_props.p0*E0*p_times_A_vals_up +
                                                         1*E0**2*A_square_vals_up)))))
            ff2 = cp.flip(fft(pulse_data.tOutput*cp.exp(-1j*fs_in_au(pulse_data.tAxis)*
                                                 (1/(2)*(self.pulse_props.p0*E0*p_times_A_vals_down +
                                                         1*E0**2*A_square_vals_down)))))

            streaked0 = cp.square(cp.abs(ff1))
            streaked1 = cp.square(cp.abs(ff2))


            streaked0 = interp(tof_ens_gpu, pulse_data.enAxis, streaked0)
            streaked1 = interp(tof_ens_gpu, pulse_data.enAxis, streaked1)
            xuvonly = interp(tof_ens_gpu, pulse_data.enAxis,
                             cp.square(cp.abs(pulse_data.enOutput)))
            
            
            self._streakedspectra = np.asarray([cp.asnumpy(xuvonly),                                                                     
                                                cp.asnumpy(streaked0), 
                                                cp.asnumpy(streaked1)])
            



    def get_spectra(self):
        '''returns streaked spectra as a numpy array'''
            
        return self._streakedspectra.copy()

    def get_augmented_spectra(self):
        '''Data augemtation to imitate: jitter between Streaking-Pulse and XUV-Pulse.
        Returns augemented spectra as a matrix.'''
        from numpy import asarray, roll
        from numpy.random import randint
        from sklearn.preprocessing import Normalizer

        aug_spectra = asarray(self.get_spectra())

#         to reduce dependancy from XUV-photon energy
        # shiftall = randint(-20, 20)
        shiftall = 0 # random wert ist ausgelagert
#         to account for jitter
        shiftstr = randint(-120, 120)

        aug_spectra[0] = roll(aug_spectra[0], shiftall)
        aug_spectra[1] = roll(aug_spectra[1], shiftall-shiftstr)
        aug_spectra[2] = roll(aug_spectra[2], shiftall+shiftstr)


        hnormalizer = Normalizer(norm="l1") # normalize area of all spectra to 1
        norm1 = hnormalizer.transform(aug_spectra) 

        return norm1


    def to_Raw_Data(self) -> Raw_Data:
        raw_obj = Raw_Data(self.get_augmented_spectra(), tof_ens, 
                           self.pulse_props)
        return raw_obj

    def to_file(self, writepath):
        if self.is_low_res():
            raise self.Data_Nonexisting_Error(
                'High resolution data is already deleted. Use to_file() before calculating streaked spectra or with get_spectra(keep_originals = True) ')
        else:
            self.get_temp().get().tofile(writepath)

    def get_streakspeed(self):
        return self.__streakspeed


# %%


class Datagenerator(Sequence):
    def __init__(self, x_set, y_set, batch_size, X, for_train=True):
        self.X=X
        self.x, self.y = x_set, y_set
#         self.pulses = pulses
        self.batch_size = batch_size
#         self.xdims = xdims
#         self.ydims = ydims
#         self.time=time
        self.for_train = for_train

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        import numpy as np
        from numpy import zeros
        from numpy.random import randint, uniform
        from scipy.ndimage import gaussian_filter
        from tensorflow.keras.utils import Sequence
        from source.process_stages import (measurement,
                                           to_eVs)

        batch = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
#         batch = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

#         x = [X[i].get_raw_matrix() for i in batch]

        x = []
        for _,i in enumerate(batch):
            # (xuv, str1, str2) = self.X[i].get_augmented_spectra(
            #     0, discretized=False)
            b1 = to_eVs(measurement(self.X[i]))

            # Raw_Data2(np.asarray((xuv, str1, str2)), tof_ens, self.X[i].get_temp(
            # ), num_electrons1=self.X[i].num_electrons1, num_electrons2=self.X[i].num_electrons2)


            x.append(b1.spectra)
#             x[ind]=b1.get_all_tof()
#             y.append(b1.get_temp())


#         x = [X[i].get_augmented_spectra(0,discretized=False) for i in batch]
        x = np.asarray(x)

        y = [self.X[i].get_temp() for i in batch]
        y = np.asarray(y)

        # .reshape(self.batch_size, -1)
        return x.reshape(-1, 3, x[0].shape[1], 1), np.array(y)

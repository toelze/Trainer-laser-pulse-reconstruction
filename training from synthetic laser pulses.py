# %%
# TODO: irgendetwas stimmt mit der Energieumrechnung nicht, das muss koriigiert werden. 
# TODO: Oder die Energieachse von Pulse muss erweitert werden, so auf bis 130 eV

# TODO: wie können die wichtigsten Daten elegant mit geliefert werden beim Erzeugen eines neuen Objekt?
# TODO: welche Daten sind bei jeder Instanz gleich / unterschiedlich?
# TODO Raw_Data2 zu Raw_Data wandeln (Raw_Data dabei löschen?)
# TODO neuronales Netz anpassen


import datetime
import os

import cupy as cp
import matplotlib.pyplot as plt
from numba.cuda import test
# from streaking_cal.misc import interp
import numpy as np
import pandas as pd
import scipy as sp
import tensorflow as tf
# from streaking_cal.misc import interp
from cupy import interp
from numba import boolean, float64, njit, vectorize
from numpy.random import rand, randint
from progressbar import ProgressBar
from scipy.interpolate import interp1d
from scipy.signal import gaussian
from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras.layers import (AveragePooling2D, BatchNormalization,
                                     Conv2D, Dense, Flatten,
                                     GlobalMaxPooling2D, Lambda, MaxPooling2D,
                                     Subtract)
from tensorflow.keras.optimizers import Adam
# from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras.regularizers import l1_l2

from streaking_cal.statistics import weighted_avg_and_std

# try:
#     # Disable all GPUS
#     tf.config.set_visible_devices([], 'GPU')
#     visible_devices = tf.config.get_visible_devices()
#     for device in visible_devices:
#         assert device.device_type != 'GPU'
# except:
#     # Invalid device or cannot modify virtual devices once initialized.
#     pass


# tf.config.experimental.set_lms_enabled(True)

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


mempool = cp.get_default_memory_pool()
# with cp.cuda.Device(0):
#     mempool.set_limit(size=800*1024**2)
# print(cp.get_default_memory_pool().get_limit())  # 1073741824


# from scipy.signal import find_peaks,peak_widths
# from scipy.ndimage import gaussian_filter
from source.class_collection import Datagenerator, Pulse, Raw_Data2

# def ws_reg(kernel):
#     if kernel.shape[0] > 0:
#         kernel_mean = tf.math.reduce_mean(
#             kernel, axis=[0, 1, 2], keepdims=True, name='kernel_mean')
#         kernel = kernel - kernel_mean
#         kernel_std = tf.math.reduce_std(
#             kernel, axis=[0, 1, 2], keepdims=True, name='kernel_std')
#         #     kernel_std = tf.keras.backend.std(kernel, axis=[0, 1, 2], keepdims=True)
#         kernel = kernel / (kernel_std + 1e-5)
#     else:
#         kernel = l1_l2(l1=1e-5, l2=1e-4)(kernel)
#     return kernel


# #
# def weighted_avg_and_std(values, weights):
#     import math
#     """
#     Return the weighted average and standard deviation.

#     values, weights -- Numpy ndarrays with the same shape.
#     """
#     average = np.average(values, weights=weights)
#     # Fast and numerically precise:
#     variance = np.average((values-average)**2, weights=weights)
#     return (average, math.sqrt(variance))


# load noise peak for discretization of spectra
dfe = pd.read_csv("./resources/energies.csv", header=None)
orig_tof_ens = dfe[0].values


# df0=pd.read_csv("./FLASH-Spectra/0.0119464/"+"spec10.csv",header=None)
noisepeak = np.fromfile('./resources/noisepeak.dat', dtype="float64")
# noisepeak=(df0[1].values/sum(df0[1]))[orig_tof_ens<61]
noisepeak_gpu = cp.asarray(noisepeak)

peak_max_y = orig_tof_ens[len(noisepeak)]-orig_tof_ens[0]
tof_ens = np.linspace(40, 110, 1401)
tof_ens_gpu = cp.asarray(tof_ens)


# progressbar for for loops


CentralEnergy = 73
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

def fs_in_au(t): return 41.3414*t  # from fs to a.u.
def eV_in_au(e): return 0.271106*np.sqrt(e)  # from eV to a.u.

p0 = eV_in_au(CentralEnergy)

# Pulse class returns temporal profile on basis of this time axis
standard_full_time = np.loadtxt('./resources/standard_time.txt')
standard_full_time = np.linspace(-250, 250, 512)

# background noise for data augmentation is read from actual measured spectra
measurednoise_train = np.loadtxt("./resources/measurednoise_train.txt")
measurednoise_val = np.loadtxt("./resources/measurednoise_val.txt")

print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))
print(tf.version.VERSION)

# %%
from source.class_collection import Datagenerator, Pulse, Raw_Data, Measurement_Data


#%%
# Array[0 + (x - 1)/(ADCFreq*10^6) &, Dimensions[hdf5tof1][[2]]]


class Raw_Data2():
    
    # parameters from TOF calibration
    tof_params = np.asarray([-755.6928301474567, 187.2222222222, -39.8])

    tof_times = np.asarray([(i-1)/3600e6 for i in np.arange(2500)+1]) # OK
    zeroindx = 674 # OK
    # TOF_to_eV = lambda t:  TOF_params[0]**2/(t - TOF_params[1])**2 - TOF_params[2]  # OK

    real_tof_response = np.fromfile("./resources/TOF_response.dat", dtype="float64")
    real_tof_response = real_tof_response/np.sum(real_tof_response)
    ionization_potential = 21.55 #Neon 2p 

    vls_pixels = np.arange(1024) + 1 # from Mathematica + indexcorrection
    vls_enenergies = 1239.84/(vls_pixels*0.0032 + 11.41)  # VLS pix 2 nm calibration 
    vls_enenergies -= ionization_potential

    # TOF_times = (cumsum(full(2500,1.))-1)/3.6e9 
    # TOF_times = TOF_times[675:]
    # linspace(0, 506+2/3, 1825)  # TOF times raw data
    

    vls_pixels = np.arange(1024)  # pixels of spectrometer

    def __init__(self, spectra, energies, temp_profile, num_electrons1=25, num_electrons2=25):
        self.num_electrons1 = num_electrons1
        self.num_electrons2 = num_electrons2
        self.energy_axis = energies
        # self.TOF_times = self.energies_to_TOF_times(energies)
        self.temp_profile = temp_profile
        self.spectra = spectra
        self.tof_response = self.get_random_response_curve()

        self.tof_energies = np.array(list(map(self.tof_to_eV,self.tof_times[self.zeroindx + 1:]*1e9)))  # OK


        # ll = self.TOF_times[self.zeroindx +1:]*1e9

        # print(ll[0])


        # self.TOF_energies = np.asarray([self.TOF_to_eV(i) for i in ll])


        self.calc_vls_spectrum()

        self.num_tof_noise0=int(0+np.random.rand()*3) # num of stray electrons in spectra
        self.num_tof_noise1=int(0+np.random.rand()*3)

    def tof_to_eV(self,t):

        return self.tof_params[0]**2/(t - self.tof_params[1])**2 - self.tof_params[2]


    def get_random_response_curve(self):
        response=np.abs(self.real_tof_response-0.015+0.03*np.random.rand(58))

        # resp_length=30;
        # tstd=2+9*np.random.rand();
        # noiselevel= 0.4*np.random.rand();
        # response=scipy.signal.gaussian(resp_length, std=tstd)
        # response=np.roll(response,np.random.randint(-(resp_length // 2)+tstd,(resp_length // 2)-tstd))
        # response+=np.abs(noiselevel*np.random.randn(resp_length))

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
        tof = -uncorrectints(tof)
        return tof

    def tof_to_eVenergies(self,tof,energies = None):
        '''interpolation and intensity correction to calculate a spectrum from a TOF signal'''
        if energies is None: 
            energies = self.energy_axis

        spec = -self.correctints(tof)
        spec = np.interp(energies,self.tof_energies[::-1],spec[::-1],0,0)
        return spec




    def calc_tof_traces(self):
        from numpy import argsort, take_along_axis, asarray

        tof_traces = self.spectra[1:]
        tof_traces = np.asarray([self.eVenergies_to_tof(tof_traces[0]),
                                 self.eVenergies_to_tof(tof_traces[1])])




        # self.TOF_times_sort_order = argsort(self.TOF_times, axis=0)
        # self.sorted_TOF_times = take_along_axis(
        #     self.TOF_times, self.TOF_times_sort_order, axis=0)

        # TOF_traces = asarray(
        #     list(map(self.TOF_signal_correction, self.spectra[1:])))
        # TOF_traces = asarray(list(map(self.resampled_TOF_signal, TOF_traces)))



        tof_traces[0] = self.discretized_spectrum(
            tof_traces[0], self.num_electrons1)
        # TOF_traces[0] = TOF_traces[0]/np.sum(TOF_traces[0])

        tof_traces[1] = self.discretized_spectrum(
            tof_traces[1], self.num_electrons2)
   
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

        # VLS_signal = self.spectra[0]

        # self.VLS_pixels = self.energies_to_VLS_pixel(self.energy_axis)
        # self.VLS_pixels_sort_order = argsort(self.VLS_pixels, axis=0)
        # self.sorted_VLS_pixels = take_along_axis(
        #     self.VLS_pixels, self.VLS_pixels_sort_order, axis=0)

        # self.VLS_signal = self.resampled_VLS_signal(self.VLS_signal)

        self.vls_signal = self.eVenergies_to_vls_pix(self.spectra[0])

        self.vls_signal = self.vls_signal/np.sum(self.vls_signal)

        # self.VLS_pixels = self.VLS_pixels

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
        from numpy import roll, pad
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



    def get_all_tof(self):
        """if every signal was measured over time-of-flight
        currently not used"""
        tof_matrix = self.get_raw_matrix()
        vls = self.vls_signal_to_energies()
        vls = self.tof_signal_correction(vls)
        vls = self.resampled_tof_signal(vls)

        tof_matrix[0] = vls

        return tof_matrix

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

    def vls_signal_to_energies(self):
        VLS_energies = self.vls_pixel_to_energies(self.vls_pixels)
        sort_order = np.argsort(VLS_energies, axis=0)
        VLS_energies = np.take_along_axis(VLS_energies, sort_order, axis=0)
        VLS_ordered = np.take_along_axis(self.vls_signal, sort_order, axis=0)
        VLS_resampled = np.roll(
            np.interp(self.energy_axis, VLS_energies, VLS_ordered, left=0, right=0), -50)
        # TODO wieso roll -50??

        return VLS_resampled

    # when calulating TOF_traces from energy spectra
    def tof_signal_correction(self, signal):
        adjusted_signal = -4 * \
            (signal*(self.energy_axis +
                     self.tof_params[2])**1.5)/self.tof_params[0]
        return adjusted_signal

    def resampled_vls_signal(self, VLS_signal):
        from numpy import take_along_axis, interp

        sorted_VLS_signal = take_along_axis(
            VLS_signal, self.vls_pixels_sort_order, axis=0)
        resampled_VLS_signal = interp(
            Raw_Data.vls_pixels, self.sorted_vls_pixels, sorted_vls_signal)

        return resampled_VLS_signal

    def resampled_tof_signal(self, TOF_signal):
        from numpy import take_along_axis, interp

        sorted_TOF_signal = take_along_axis(
            TOF_signal, self.tof_times_sort_order, axis=0)
        resampled_tof_signal = interp(
            1e9*Raw_Data.tof_times, self.sorted_tof_times, sorted_tof_signal)

        return resampled_TOF_signal

    def discretized_spectrum(self, spectrum, num_points):
        from numpy import interp, zeros
#         disc_spec=np.zeros(len(spectrum))
        positions = self.discrete_positions(spectrum, num_points)


#         for i in positions:
#             valll=np.random.rand()+1
#             (divval,modval)=divmod(i, 1)
#             divval=divval.astype("int")
#             disc_spec[divval]+=valll*(1-modval)
#             disc_spec[divval+1]+=valll*(modval)

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

    def get_temp(self):
        return self.temp_profile
    
    def to_Measurement_Data(self):
        measurement_obj = Measurement_Data(self.augment_vls(), self.augment_tof(),self.tof_times) # OK


        return measurement_obj

# %%
# vlspix = np.arange(1024) + 1 # from Mathematica + indexcorrection
# VLSens = 1239.84/((vlspix)*0.0032 + 11.41) - 21.55 


# eV to VLS pixel
def eVenergies_to_VLSpix(spec,energies = [0]):
    if len(energies) == 1:
        energies = tof_ens
    vls = np.interp(VLSens,energies,spec,0,0)
    return vls
# VLS pixel to eV

def VLSpix_to_eVenergies(vls,energies = [0]):
    if len(energies) == 1:
        energies = tof_ens
    spec =  np.interp(energies,VLSens[::-1],vls[::-1],0,0)
    return spec
VLSens
# %%

test = np.exp(-1/2 * ((np.arange(1024)+1)-300)**2/100**2)
plt.plot((eVenergies_to_VLSpix(VLSpix_to_eVenergies(test))-test)[2:])
# %%
# cc = eVenergies_to_VLSpix(raw_obj.spectra[0])
# cc = cc / sum(cc)
# plt.plot(cc)
plt.plot(raw_obj.vls_signal)
plt.plot(raw_obj.augment_vls())
plt.plot(raw_obj.augment_tof()[1][raw_obj.zeroindx + 1:])
print(sum(raw_obj.vls_signal))
# %%
# %%
nens= np.linspace(50,100,501)
plt.plot(nens,raw_obj.vls_pix_to_eVenergies(raw_obj.augment_vls(),nens))
# plt.plot(nens,raw_obj.tof_to_eVenergies(raw_obj.augment_tof()[0],nens))
# plt.plot(nens,raw_obj.tof_to_eVenergies(raw_obj.augment_tof()[1],nens))
# plt.plot(tof_ens,spp[1])
# plt.plot(tof_ens,spp[2])

plt.xlim([50,90])

#%%
nens= np.linspace(50,100,501)
# plt.plot(nens,raw_obj.vls_pix_to_eVenergies(raw_obj.augment_vls(),nens))
# plt.plot(nens,raw_obj.tof_to_eVenergies(raw_obj.augment_tof()[0],nens))
# plt.plot(nens,raw_obj.tof_to_eVenergies(raw_obj.augment_tof()[1],nens))

raw_obj.to_Measurement_Data().tof_in_data[1][675:].shape
# raw_obj.to_Measurement_Data().tof_in_times.shape
# raw_obj.to_Measurement_Data().tof_energies.shape

# %%
nens= np.linspace(50,100,501)
# plt.plot(nens,raw_obj.vls_pix_to_eVenergies(raw_obj.augment_vls(),nens))
# plt.plot(nens,raw_obj.tof_to_eVenergies(raw_obj.augment_tof()[1],nens))
ppp = raw_obj.to_Measurement_Data()
plt.plot(ppp.energy_axis ,ppp.spectra[0])
plt.plot(ppp.energy_axis ,ppp.spectra[1])
plt.plot(ppp.energy_axis ,ppp.spectra[2])

plt.xlim([50,90])

#%%
ppp.tof_in_times
Raw_Data2.tof_times

#%%
def correctints(spec):
    '''from TOF times to eV'''
    return 0.5 * TOF_params[0] * spec[zeroindx + 1:]/(TOF_energies + TOF_params[2])**1.5

def uncorrectints(cspec):
    '''from eV to TOF times'''
    spec = cspec *(TOF_energies + TOF_params[[2]])**1.5/(0.5*TOF_params[[0]])
    spec = np.pad(spec, (zeroindx+1,0),'constant',constant_values=(0, 0))
    return spec

def eVenergies_to_TOF(spec,energies):
    tof = np.interp(TOF_energies,energies,spec,0,0)
    tof = -uncorrectints(tof)
    return tof

def TOF_to_eVenergies(tof,energies):
    spec = -correctints(tof)
    spec = np.interp(energies,TOF_energies[::-1],spec[::-1],0,0)
    return spec


TOF_params = np.asarray([-755.6928301474567, 187.2222222222, -39.8])

TOF_times = np.asarray([(i-1)/3600e6 for i in np.arange(2500)+1]) # OK
zeroindx = 674 # OK
TOF_to_eV = lambda t:  TOF_params[0]**2/(t - TOF_params[1])**2 - TOF_params[2]  # OK
TOF_energies = np.array(list(map(TOF_to_eV,TOF_times[zeroindx + 1:]*1e9)))  # OK
# correctints = lambda spec: 0.5 * TOF_params[0] * spec[zeroindx + 1:]/(TOF_energies + TOF_params[2])**1.5 # OK
# uncorrectints = lambda cspec: cspec *(energies1 + TOF_params[[2]])**1.5/(0.5*TOF_params[[0]])
newens = np.arange(60,90.1,0.1)

test = np.asarray([-500*np.exp(-1/2 *(i - 1150)**2/20**2) for i in np.arange(2500)+1]) # OK
test2 = np.interp(newens,TOF_energies[::-1],correctints(test)[::-1]) # OK
test2 = test2/np.sum(test2) #OK



# way: eV -> toftimes: eV->interp-> eVnew->uncorrectint -> tof
# way tof -> correctints -> eVnew -> interp -> eV




# plt.plot(newens,test2)
# test2[140]
yy =np.argmax(test2)
# len(np.ones(2500)[zeroindx + 1 : -1])

spp =X[2].get_augmented_spectra(95)



raw_obj = Raw_Data2(spp, tof_ens, X[0].get_temp(),
                 num_electrons1=X[0].num_electrons1, num_electrons2= X[0].num_electrons2)

# eVenergies_to_TOF()
(xuv,str1,str2) = spp

# plt.plot(d.get_raw_matrix()[1])

str11=eVenergies_to_TOF(str1,tof_ens)


# plt.plot(-str11[::-1])
# plt.plot(0.0001*raw_obj.calc_tof_traces()[1])
# plt.xlim([0,500])
#  (0.5 params3[[1]] spec[[zeroindx + 1 ;; -1]])/(energyax11 + params3[[3]])^1.5

len(str11[zeroindx + 1:])

# %%

ff= TOF_to_eVenergies(eVenergies_to_TOF(raw_obj.spectra[2],np.linspace(40, 110, 1401)), tof_ens)
ff = ff/ sum(ff)
plt.plot(tof_ens,ff)
plt.plot(np.linspace(40, 110, 1401),raw_obj.spectra[2])

pp = raw_obj.eVenergies_to_tof(raw_obj.spectra[2])
pp = raw_obj.tof_to_eVenergies(pp)
pp = pp / sum(pp)
plt.plot(raw_obj.energy_axis,pp)

dd = raw_obj.augment_tof()[1]
dd = raw_obj.tof_to_eVenergies(dd,tof_ens)
dd = dd / sum(dd)
plt.plot(tof_ens,dd)


# %%
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
  try:
    tf.config.experimental.set_virtual_device_configuration(gpus[0], 
    [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2048)])
  except RuntimeError as e:
    print(e)
# %%
# timeit
# pbar = ProgressBar()
from tqdm import tqdm as pbar

num_pulses = 100000
streakspeed = 95  # meV/fs
X = [""]*num_pulses
y = [""]*num_pulses

for i in pbar(range(num_pulses),colour = 'red', ncols= 100):
    x1 = Pulse.from_GS(dT=np.random.uniform(70/2.355, 150/2.355), 
            dE=np.random.uniform(0.2/2.355, 1.8/2.355), 
            num_electrons1=np.random.randint(15, 40), 
            num_electrons2=np.random.randint(15, 40),
            centralE=np.random.uniform(65,75)
                       )
    x1.get_spectra(streakspeed, discretized=False)
    X[i] = x1
# %%
t= X[0].to_Raw_Data(95).to_Measurement_Data()
print(t.tof_eVs)
plt.plot(t.energy_axis,t.get_eV_tof()[0])
# %%
(xuv,str1,str2) = X[0].get_augmented_spectra(95)
d = X[0].to_Raw_Data(95)
t= d.to_Measurement_Data()
print(d.TOF_times)

(xuv,str1,str2)=X[0].get_augmented_spectra(95)
plt.plot(tof_ens,str1)
plt.plot(t.energy_axis,10*t.get_eV_tof()[0])
# print(X[3].to_Raw_Data(95).TOF_times)
print(t.tof_in_times)
print(Raw_Data.TOF_times*10**9)
# %%
class CenterAround(tf.keras.constraints.Constraint):
    #   """Constrains weight tensors to be centered around `ref_value`."""

    def __init__(self, ref_value):
        self.ref_value = ref_value

    def __call__(self, w):
        mean = tf.reduce_mean(w)
        return w - mean + self.ref_value

    def get_config(self):
        return {'ref_value': self.ref_value}
# %%
# %%
def maxLayer():
    return MaxPooling2D(pool_size=(8,8),strides=(1,8),padding="same")

def convLayer(filters):
    return Conv2D(filters=filters, kernel_size=(1, 7), activation="relu", strides=1, padding="same")

# %% 

convdim = 128

enc_inputs = Input(shape=(3, 301, 1), name="traces")


# HIER kernel_size=(3, 500) für bessere Ergebnisse bzw. kernel_size=(1, 500) für zeilenunabh. Mustererkennung
conv_out = Conv2D(convdim, kernel_size=(3, 250), activation="relu", strides=1, padding="same"
                  )(enc_inputs)



enc_output = GlobalMaxPooling2D()(conv_out)

encoder = tf.keras.Model(enc_inputs, enc_output, name="encoder")
encoder.summary()
# end of encoder
#%%

dec_inputs = Input(shape=(128), name="encoder_output")

# start of decoder
x = BatchNormalization()(dec_inputs)

x = Dense(256, activation="relu"
          #          , kernel_constraint=CenterAround(0)
          )(x)


dec_outputs = Dense(standard_full_time.shape[0], activation="softmax")(x)
decoder = tf.keras.Model(dec_inputs, dec_outputs, name="decoder")
decoder.summary()
# end of decoder
# %%

encoded = encoder(enc_inputs)
decoded = decoder(encoded)
merged_model = tf.keras.Model(enc_inputs, decoded, name="merged_model")

merged_model.summary()
# %%

wholeset = np.arange(len(X))

pulses_train, pulses_test, y_train, y_test = train_test_split(
    wholeset, wholeset, test_size=0.05, random_state=1)
params = {'batch_size': 250}
train_ds = Datagenerator(pulses_train, y_train, X=X, **params)
test_ds = Datagenerator(pulses_test, y_test, X=X, for_train=False, **params)

# time2=time[abs(time)<250]
# %%

opt = Adam(lr=5e-3, amsgrad=True) 
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.2, decay_steps=15000, decay_rate=0.99, staircase=True)
nadam = tf.keras.optimizers.Nadam(learning_rate=0.01)
adagrad = tf.keras.optimizers.Adagrad(
    learning_rate=lr_schedule, initial_accumulator_value=0.1)
merged_model.compile(optimizer="nadam", loss="KLDivergence",
              metrics=["accuracy", "mae"])

history = merged_model.fit(x=train_ds, validation_data=test_ds,
                    #                     use_multiprocessing=True,
                    #                     workers=4,
                    epochs=100
                    )
# %%
from tensorflow.keras.layers import (AveragePooling2D, BatchNormalization,
                                     Flatten, MaxPooling2D)

# %%

# ----------------------SAVE MODELS-------------------------

# merged_model.save('./models/RAW_mat_mergedm-95')
# encoder.save('./models/RAW_mat_encoder-95')
# decoder.save('./models/RAW_mat_decoder-95')


# ----------------------LOAD MODELS-------------------------

# merged_model = tf.keras.models.load_model('./models/RAW_mat_mergedm')
# encoder = tf.keras.models.load_model('./models/RAW_mat_encoder')
# decoder = tf.keras.models.load_model('./models/RAW_mat_decoder')


# %%
#"nadam"
#701 points, 5 epochs, 130000 pulses,val_loss=0.3551
#1401 points, 5 epochs, 130000 pulses,val_loss=0.3532
#model-split:1401 points 25 epochs, 130000 pulses val_loss= 0.3572 (0.3555-0.3572)
#model-normal:1401 points 25 epochs, 130000 pulses val_loss= 0.3562 (0.3554-0.3562)
#model-normal, all TOF:1401 points, 25 els, 25 epochs, 130000 pulses val_loss= 0.3565 (0.3565-0.3576)
#model-normal, all TOF:1401 points, 500 els, 25 epochs, 130000 pulses val_loss= 0.2935 (0.2930-0.2950)?
#model-normal:1401 points, 500 els, 25 epochs, 130000 pulses val_loss= 0.3019 (0.2921-0.3019)
#adam 1e-3
#model-normal:1401 points 25 epochs, 130000 pulses val_loss= 0.3550 (0.3550-0.3568)
# %%
testitems= test_ds.__getitem__(0)
preds=merged_model.predict(testitems[0])
y_test=testitems[1]
# %matplotlib inline
vv=26


plt.plot(standard_full_time,y_test[vv])
plt.plot(standard_full_time,preds[vv],'--')
# plt.plot(time,gaussian_filter(y_test[vv],10))

plt.figure()
plt.plot(standard_full_time,preds[vv],'orange')

plt.figure()
# plt.plot(tof_ens,testitems[0][vv][0])
plt.plot(np.arange(301),testitems[0][vv][0])
plt.plot(np.arange(301),testitems[0][vv][1])
plt.plot(np.arange(301),testitems[0][vv][2])
# plt.xlim([500,700])

# %%
import os
import re
from itertools import repeat

folder="./resources/raw_mathematica/down1/"
files=os.listdir(folder)

numbers=list(map(re.findall,repeat("[0-9]{5}"),files))
numbers=np.asarray(numbers)[:,0].astype("int32")
# %%
folder="./resources/raw_mathematica/up/"
files=os.listdir(folder)

numbers=list(map(re.findall,repeat("[0-9]{4}"),files))
numbers=np.asarray(numbers)[:,0].astype("int32")
#%%


tof1=[]
tof2=[]
vls=[]
testitems2=[]
for i in numbers:
    tof11=np.fromfile(folder+"TOF1-"+str(i)+".dat","float32")
    tof21=np.fromfile(folder+"TOF2-"+str(i)+".dat","float32")
    vls11=np.fromfile(folder+"VLS-"+str(i)+".dat","float32")
    tof11=np.roll(tof11,150)
    tof21=np.roll(tof21,150)
    vls11=np.pad(vls11,pad_width=(0, len(tof11)-len(vls11)))
    vls11=np.roll(vls11,0)
    testitems2.append([vls11,tof11,tof21])
testitems2=np.reshape(np.asarray(testitems2),[len(numbers),3,-1,1])

#%%
preds=merged_model.predict(testitems2)
# %matplotlib inline
vv=93

# plt.plot(time,gaussian_filter(y_test[vv],10))

plt.figure()
plt.plot(standard_full_time,
        preds[vv],
        'orange')
vsigma=weighted_avg_and_std(standard_full_time,preds[vv])
print([vsigma,2.35*vsigma[1]])
plt.figure()
plt.plot(np.arange(1825),testitems2[vv][1])
plt.plot(np.arange(1825),testitems2[vv][2])
plt.plot(np.arange(1825),testitems2[vv][0])
plt.xlim([500,700])


# %%
# encode synth pulses is batches of 300 and concatenate
from source.class_collection import Datagenerator, Pulse, Raw_Data

num_pulses = 10000
streakspeed = 95  # meV/fs
X = [""]*num_pulses
y = [""]*num_pulses

for i in pbar(range(num_pulses),colour = 'red', ncols= 100):
    x1 = Pulse.from_GS(dT=np.random.uniform(20/2.355, 120/2.355), 
            dE=np.random.uniform(0.2/2.355, 1.8/2.355), 
            num_electrons1=np.random.randint(15, 40), 
            num_electrons2=np.random.randint(15, 40),
            centralE=np.random.uniform(65,95)
                       )
    x1.get_spectra(streakspeed, discretized=False)
    X[i] = x1


wholeset = np.arange(len(X))

pulses_train, pulses_test, y_train, y_test = train_test_split(
    wholeset, wholeset, test_size=0.95, random_state=1)
params = {'batch_size': 300}
train_ds = Datagenerator(pulses_train, y_train, X=X, **params)
test_ds = Datagenerator(pulses_test, y_test, X=X, for_train=False, **params)


encoded_t0 = encoded_t
lower_dim_input0 = lower_dim_input
encoded_t =[encoder(test_ds.__getitem__(i)[0]) for i in range(30)]
encoded_t = np.concatenate(encoded_t, axis = 0)

# encoded_t2=encoder(testitems2[:300])
#%%
from sklearn.decomposition import PCA
# pca = PCA(n_components=2)
# pca.fit(encoded_t)
lower_dim_input=pca.transform(encoded_t)
synth_low= lower_dim_input
lower_dim_input2=pca.transform(encoded_t2)
measures_low = lower_dim_input2


COM = lambda elements2d: np.mean(elements2d,axis=0)

@njit(fastmath=True)
def dist(coord1,coord2):
    return np.sqrt(np.sum(np.square(coord1-coord2)))


COM_synth=COM(synth_low)
COM_measures=COM(measures_low)

COM_rise=COM_synth-COM_measures

dist(COM_synth,COM_measures)

intermed_pos = [COM_measures + (i+1)*COM_rise/400 for i in range(1000)]

synth_argmins=[]
for i_pos in intermed_pos:
    i_distances= [dist(i_pos,synth_pos) for synth_pos in synth_low]
    # print(list(map(np.unique, np.argmin(i_distances))))
    synth_argmins.append(np.argmin(i_distances).astype(int))

synth_argmins
indexes = np.unique(synth_argmins, return_index=True)[1]
ordered_list_of_nearest_members=[synth_argmins[index] for index in sorted(indexes)]


plt.scatter(lower_dim_input2[:,0],lower_dim_input2[:,1], c='r')
plt.scatter(lower_dim_input0[:,0],lower_dim_input0[:,1], c= 'y')
plt.scatter(lower_dim_input[:,0],lower_dim_input[:,1])


# TODO welche Pulse wie nah sind, verändert sich mit jedem Abruf der Spektren, es muss also dort zufällig variiert werden

# %%
np.random.randint(0,100)
# %%

output_t =[merged_model(test_ds.__getitem__(i)[0]) for i in range(30)]
output_t = np.concatenate(output_t, axis = 0)



# %%
def find_2dist(x1,y1,x2,y2):
    return np.sqrt(np.square(x1-x2)+np.square(y1-y2))


# %%
# x = -0.2, y = -0.1
nearest_sample_index=np.argmin(find_2dist(lower_dim_input[:,0],lower_dim_input[:,1],-0.2,-0.1))

# %%

decoder.summary()

(encoded_t)
# %%
#  DISPLAY THE SYNTH SAMPLE THAT IS NEAREST TO THE MEASURED ONES

temp_nearest=decoder(tf.convert_to_tensor([pca.inverse_transform(lower_dim_input[nearest_sample_index])]))[0]

plt.figure()
plt.plot(standard_full_time,
        temp_nearest,
        'orange')
vsigma=weighted_avg_and_std(standard_full_time,temp_nearest)
print([vsigma,2.35*vsigma[1]])
# plt.figure()
# plt.plot(np.arange(1825),testitems2[vv][1])
# plt.plot(np.arange(1825),testitems2[vv][2])
# plt.plot(np.arange(1825),testitems2[vv][0])
# plt.xlim([500,700])


# %%
plt.plot(np.arange(1825),testitems[0][vv][2])
plt.plot(np.arange(1825),testitems[0][vv][1])
plt.plot(np.arange(1825),testitems[0][vv][0])

plt.xlim([1,1500])

#%%
spec_stat_measured= np.asarray([[weighted_avg_and_std(np.arange(len(testitems2[0,0])),np.abs(testitems2[i,j].reshape(-1,))) 
            for j in np.arange(0,3)] 
            for i in np.arange(len(testitems2))])
# %%
plt.figure()
plt.hist(spec_stat_measured[:,0,0],np.arange(400,1000,10))
plt.hist(spec_stat_measured[:,1,0],np.arange(400,1000,10))
plt.hist(spec_stat_measured[:,2,0],np.arange(400,1000,10))
# %%
spec_stat= np.asarray([[weighted_avg_and_std(np.arange(len(testitems[0][0,0])),np.abs(testitems[0][i,j].reshape(-1,))) 
            for j in np.arange(0,3)] 
            for i in np.arange(len(testitems[0]))])
# %%
# plt.figure()
plt.hist(spec_stat[:,0,0],np.arange(400,1000,10))
plt.hist(spec_stat[:,1,0],np.arange(400,1000,10))
plt.hist(spec_stat[:,2,0],np.arange(400,1000,10))

# %%
len(testitems)
spec_stat[:,1,0]-spec_stat[:,2,0]
# %%
plt.hist(testitems2[:,:,1].reshape(-1,),100)
# %%
numbers
# %%
from sklearn.manifold import TSNE
# encoded_t = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
tsne = TSNE(n_components=3)
# tsne.fit(encoded_t)

t_lower_dim_input=tsne.fit_transform(np.concatenate((encoded_t, encoded_t2.numpy())))
# t_lower_dim_input2=tsne.fit_transform(encoded_t2)

synth_low = t_lower_dim_input[:9000]
measures_low = t_lower_dim_input[9000:]

# plt.scatter(synth_low[:,0],synth_low[:,1])
# plt.scatter(measures_low[:,0],measures_low[:,1], c='r')

COM = lambda elements2d: np.mean(elements2d,axis=0)

@njit(fastmath=True)
def dist(coord1,coord2):
    return np.sqrt(np.sum(np.square(coord1-coord2)))



COM_synth=COM(synth_low)
COM_measures=COM(measures_low)

COM_rise=COM_synth-COM_measures
intermed_pos = [COM_measures + (i+1)*COM_rise/400 for i in range(1000)]


# for all the positions find the nearest synth_pulse
synth_argmins=[]
for i_pos in intermed_pos:
    # all distances to intermediate positions
    i_distances= [dist(i_pos,synth_pos) for synth_pos in synth_low]
    # print(list(map(np.unique, np.argmin(i_distances))))
    synth_argmins.append(np.argmin(i_distances).astype(int))


indexes = np.unique(synth_argmins, return_index=True)[1] # filter out doublettes
ordered_list_of_nearest_members=[synth_argmins[index] for index in sorted(indexes)]

# plt.scatter(synth_low[ordered_list_of_nearest_members,0],synth_low[ordered_list_of_nearest_members,1], c = 'y')


# plt.scatter(t_lower_dim_input[:,0],t_lower_dim_input[:,1], c=np.concatenate((np.full(300, 0),np.full(327, 1))))

#  YELLOW = measured
#  PURPLE = synth



# %%
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
fig = plt.figure()
ax = plt.axes(projection='3d')

ax.scatter3D(synth_low[:,0],synth_low[:,1],synth_low[:,2])
ax.scatter3D(measures_low[:,0],measures_low[:,1], measures_low[:,2], c='r')
ax.scatter3D(synth_low[ordered_list_of_nearest_members,0],synth_low[ordered_list_of_nearest_members,1],synth_low[ordered_list_of_nearest_members,2], c= 'y')
plt.figure()

# %%
import numpy as np
from numba import njit, vectorize, float64, boolean
from tensorflow.keras.utils import Sequence
# from streaking_cal.misc import interp
from cupy import interp
import cupy as cp

import pandas as pd

import scipy

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


dt2 = np.dtype([('xuv', np.float64), ('up', np.float64), ('down', np.float64)])
# measured_spectra = []
# for i in range(1, 110):
#     lll = np.fromfile('./resources/files_mathematica/' +
#                       str(i)+'.dat', dtype=dt2)
#     xuv0 = np.interp(tof_ens, orig_tof_ens, lll['xuv'], left=0, right=0)
#     xuv0 = xuv0/sum(xuv0)
#     up0 = np.interp(tof_ens, orig_tof_ens, lll['up'], left=0, right=0)
#     up0 = up0/sum(up0)
#     down0 = np.interp(tof_ens, orig_tof_ens, lll['down'], left=0, right=0)
#     down0 = down0/sum(down0)
#     measured_spectra.append(np.array([xuv0, up0, down0]))
# measured_spectra = np.asarray(measured_spectra)


# background noise for data augmentation is read from actual measured spectra
measurednoise_train = np.loadtxt("./resources/measurednoise_train.txt")
measurednoise_val = np.loadtxt("./resources/measurednoise_val.txt")


# Pulse class returns temporal profile on basis of this time axis
standard_full_time = np.loadtxt('./resources/standard_time.txt')
standard_full_time = np.linspace(-250, 250, 512)


class Raw_data():
    from numpy import asarray, linspace, arange, full, cumsum
    
    # parameters from TOF calibration
    TOF_params = asarray([-755.6928301474567, 187.2222222222, -39.8])
    TOF_response = np.fromfile("./resources/TOF_response.dat", dtype="float64")
    TOF_response = TOF_response/np.sum(TOF_response)

    TOF_times = (cumsum(full(2500,1.))-1)/3.6e9 
    TOF_times = TOF_times[675:]
    # linspace(0, 506+2/3, 1825)  # TOF times raw data
    

    VLS_pixels = arange(1024)  # pixels of spectrometer

    def __init__(self, spectra, energies, temp_profile, num_electrons1=25, num_electrons2=25):
        self.num_electrons1 = num_electrons1
        self.num_electrons2 = num_electrons2
        self.energy_axis = energies
        self.TOF_times = self.energies_to_TOF_times(energies)
        self.temp_profile = temp_profile
        self.spectra = spectra
        self.TOF_response = self.get_random_response_curve()

        self.calc_vls_spectrum()

        self.num_TOF_noise0=np.int(0+np.random.rand()*3) # num of stray electrons in spectra
        self.num_TOF_noise1=np.int(0+np.random.rand()*3)

    def get_random_response_curve(self):
        response=np.abs(Raw_data.TOF_response-0.015+0.03*np.random.rand(58))

        # resp_length=30;
        # tstd=2+9*np.random.rand();
        # noiselevel= 0.4*np.random.rand();
        # response=scipy.signal.gaussian(resp_length, std=tstd)
        # response=np.roll(response,np.random.randint(-(resp_length // 2)+tstd,(resp_length // 2)-tstd))
        # response+=np.abs(noiselevel*np.random.randn(resp_length))

        response= response/np.sum(response)
        return response

    def calc_tof_traces(self):
        from numpy import argsort, take_along_axis, asarray

        self.TOF_times_sort_order = argsort(self.TOF_times, axis=0)
        self.sorted_TOF_times = take_along_axis(
            self.TOF_times, self.TOF_times_sort_order, axis=0)

        TOF_traces = asarray(
            list(map(self.TOF_signal_correction, self.spectra[1:])))
        TOF_traces = asarray(list(map(self.resampled_TOF_signal, TOF_traces)))
        TOF_traces[0] = self.discretized_spectrum(
            TOF_traces[0], self.num_electrons1)
        # TOF_traces[0] = TOF_traces[0]/np.sum(TOF_traces[0])

        TOF_traces[1] = self.discretized_spectrum(
            TOF_traces[1], self.num_electrons2)
        # TOF_traces[1] = TOF_traces[1]/np.sum(TOF_traces[1])


#         self.TOF_times = Raw_data.TOF_times

        return TOF_traces

    def calc_vls_spectrum(self):
        from numpy import argsort, take_along_axis

        self.VLS_signal = self.spectra[0]

        self.VLS_pixels = self.energies_to_VLS_pixel(self.energy_axis)
        self.VLS_pixels_sort_order = argsort(self.VLS_pixels, axis=0)
        self.sorted_VLS_pixels = take_along_axis(
            self.VLS_pixels, self.VLS_pixels_sort_order, axis=0)

        self.VLS_signal = self.resampled_VLS_signal(self.VLS_signal)
        self.VLS_signal = self.VLS_signal/np.sum(self.VLS_signal)

        self.VLS_pixels = Raw_data.VLS_pixels

    def vls_finite_resolution(self,spectrum):
        from scipy import signal
        spectrum = np.convolve(spectrum,signal.gaussian(21, std=2),'same')
        return spectrum


    def get_raw_matrix(self):
        from numpy import roll, pad
        from numpy import sum as npsum

        TOF_traces = self.calc_tof_traces()

        vls_new = pad(self.VLS_signal,
                      pad_width=(0, len(TOF_traces[0])-len(self.VLS_signal)))
        vls_new = self.vls_finite_resolution(vls_new)
        vls_new = self.add_tof_noise_hf(vls_new,0.00009,0.00013) # real measured noise = 0.00011
        vls_new = vls_new/np.sum(vls_new)
        vls_new = roll(vls_new, 0)


        tof_new0 = roll(TOF_traces[0], 150) # roll, so that TOF and VLS are closer together
        tof_new1 = roll(TOF_traces[1], 150)

        tof_new0 =  self.add_tof_noise(tof_new0,self.num_TOF_noise0)
        tof_new1 =  self.add_tof_noise(tof_new1,self.num_TOF_noise1)


        tof_new0 = np.convolve(tof_new0, self.TOF_response, mode="same")
        tof_new0 = np.roll(tof_new0,-20) # convolution shift to the right
        tof_new0 = tof_new0/np.sum(tof_new0)
        tof_new0 =  self.add_tof_noise_hf(tof_new0)
        tof_new0 = tof_new0/np.sum(tof_new0)



        tof_new1 = np.convolve(tof_new1, self.TOF_response, mode="same")
        tof_new1 = np.roll(tof_new1,-20) # convolution shift to the right
        tof_new1 = tof_new1/np.sum(tof_new1)
        tof_new1 =  self.add_tof_noise_hf(tof_new1)
        tof_new1 = tof_new1/np.sum(tof_new1)



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
        VLS = self.VLS_signal_to_energies()
        VLS = self.TOF_signal_correction(VLS)
        VLS = self.resampled_TOF_signal(VLS)

        tof_matrix[0] = VLS

        return tof_matrix

    def energies_to_TOF_times(self, energies_eV):  # in ns
        from numpy import sqrt
        TOF_times = self.TOF_params[1]+ self.TOF_params[0]**2/sqrt((self.TOF_params[0]**2)*(
            energies_eV + self.TOF_params[2]))  # funktioniert
        return TOF_times  # -min(TOF_times)

    def VLS_pixel_to_energies(self, vls_pixel):
        return -21.5 + 1239.84/(11.41 + 0.0032*vls_pixel)

    def energies_to_VLS_pixel(self, energies_eV):
        # calibration and 21.5 eV ionization
        return -3565.63 + 387450/(21.5 + energies_eV)

    def VLS_signal_to_energies(self):
        VLS_energies = self.VLS_pixel_to_energies(self.VLS_pixels)
        sort_order = np.argsort(VLS_energies, axis=0)
        VLS_energies = np.take_along_axis(VLS_energies, sort_order, axis=0)
        VLS_ordered = np.take_along_axis(self.VLS_signal, sort_order, axis=0)
        VLS_resampled = np.roll(
            np.interp(self.energy_axis, VLS_energies, VLS_ordered, left=0, right=0), -50)

        return VLS_resampled

    # when calulating TOF_traces from energy spectra
    def TOF_signal_correction(self, signal):
        adjusted_signal = -4 * \
            (signal*(self.energy_axis +
                     self.TOF_params[2])**1.5)/self.TOF_params[0]
        return adjusted_signal

    def resampled_VLS_signal(self, VLS_signal):
        from numpy import take_along_axis, interp

        sorted_VLS_signal = take_along_axis(
            VLS_signal, self.VLS_pixels_sort_order, axis=0)
        resampled_VLS_signal = interp(
            Raw_data.VLS_pixels, self.sorted_VLS_pixels, sorted_VLS_signal)

        return resampled_VLS_signal

    def resampled_TOF_signal(self, TOF_signal):
        from numpy import take_along_axis, interp

        sorted_TOF_signal = take_along_axis(
            TOF_signal, self.TOF_times_sort_order, axis=0)
        resampled_TOF_signal = interp(
            1e9*Raw_data.TOF_times, self.sorted_TOF_times, sorted_TOF_signal)

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
            valll = 8*np.random.rand()+0.5 # which heights are suitable? propably between 0.5 and 4.5
            (divval, modval) = np.divmod(i, 1)
            divval = np.int(divval)
            disc_spec[divval] += valll*(1-modval)
            disc_spec[divval+1] += valll*(modval)

        return disc_spec

    @staticmethod
    @njit(fastmath=True)
    def discrete_positions(spectrum, num_points):
        cumulative_spectrum = (np.cumsum(spectrum))/np.sum(spectrum)
        indices = np.arange(len(spectrum))
        discrete_positions = np.interp(np.random.rand(
            num_points), cumulative_spectrum, indices)

        return discrete_positions

    def get_temp(self):
        return self.temp_profile

# %%


class Pulse(object):
    @staticmethod
    def fs_in_au(t): return 41.3414*t  # from fs to a.u.
    @staticmethod
    def eV_in_au(e): return 0.271106*np.sqrt(e)  # from eV to a.u.
    h = 4.135667662  # in eV*fs

    class Data_Nonexisting_Error(Exception):
        def __init__(self, message):
            self.message = message
#

    def __init__(self, EnAxis, EnOutput, TAxis, TOutput, num_electrons1, num_electrons2, dE=0, dT=0):
        from streaking_cal.statistics import weighted_avg_and_std

        centralE, dE = weighted_avg_and_std(
            EnAxis.get(), cp.square(cp.abs(EnOutput)).get())
        self.p0 = eV_in_au(centralE)

        if dT == 0:
            dT = weighted_avg_and_std(
                TAxis.get(), cp.square(cp.abs(TOutput)).get())[1]

        self.dE = dE
        self.dT = dT
        self.__spec = EnOutput
        self.__eAxis = EnAxis
        self.__temp = TOutput
        self.__tAxis = TAxis
        self.num_electrons1 = num_electrons1
        self.num_electrons2 = num_electrons2
        self.__is_low_res = False
        self.__streakspeed = 0
        del(EnAxis)
        del(EnOutput)
        del(TAxis)
        del(TOutput)

#     @property for read access only members
    @classmethod
    def from_GS(cls, dE=0.35, dT=35, centralE=73, num_electrons1=25, num_electrons2=25):
        from streaking_cal.GetSASE import GetSASE_gpu as GS
#         from streaking_cal.statistics import weighted_avg_and_std

        (EnAxis, EnOutput, TAxis, TOutput) = GS(CentralEnergy=centralE,
                                                dE_FWHM=2.355*dE*2**0.5,
                                                dt_FWHM=2.355*dT*2**0.5,
                                                onlyT=False)
        EnAxis = EnAxis.astype("float32")
        TAxis = TAxis.astype("float32")
        EnOutput = EnOutput.astype("complex64")
        TOutput = TOutput.astype("complex64")

        return cls(EnAxis, EnOutput, TAxis, TOutput, num_electrons1, num_electrons2, dE, dT)

    @classmethod
    def from_file(cls, path_temp, num_electrons1=25, num_electrons2=25):
        TOutput = cp.fromfile(path_temp, dtype="complex64")
        TAxis = cp.fromfile("./GS_Pulses/axis/time_axis.dat", dtype="float32")

        EnOutput = cp.fft.ifft(TOutput)
        EnAxis = cp.linspace(0, 140, 32*1024)

        return cls(EnAxis, EnOutput, TAxis, TOutput, num_electrons1, num_electrons2)

#     certain entries may be deleted after calculation of measured spectra
    def get_temp(self):
        return self.__temp

    def get_tAxis(self):
        return self.__tAxis

    def get_spec(self):
        if self.__is_low_res:
            return None
        else:
            return self.__spec

    def get_eAxis(self):
        if self.__is_low_res:
            return None
        else:
            return self.__eAxis

    def is_low_res(self):
        return self.__is_low_res

    def __get_streaked_spectra(self, streakspeed):
        if self.is_low_res == True:
            return None
        else:
            from cupy.fft import fft

            def fs_in_au(t): return 41.3414*t  # from fs to a.u.
            # def eV_in_au(e): return 0.271106*np.sqrt(e)  # from eV to a.u.

            # in V/m; shape of Vectorpotential determines: 232000 V/m = 1 meV/fs max streakspeed
            E0 = 232000*streakspeed
            ff1 = cp.flip(fft(self.__temp*cp.exp(-1j*fs_in_au(self.__tAxis)*(1/(2)*(self.p0*E0*p_times_A_vals_up +
                                                                                    1*E0**2*A_square_vals_up)))))
            ff2 = cp.flip(fft(self.__temp*cp.exp(-1j*fs_in_au(self.__tAxis)*(1/(2)*(self.p0*E0*p_times_A_vals_down +
                                                                                    1*E0**2*A_square_vals_down)))))

            spectrum1 = cp.square(cp.abs(ff1))
            spectrum2 = cp.square(cp.abs(ff2))

    #         ff1=ff1/(cp.sum(cp.square(cp.abs(ff1))))
    #         ff1=ff2/(cp.sum(cp.square(cp.abs(ff2))))

            return spectrum1, spectrum2

    # @staticmethod
    # @vectorize([float64(float64, boolean)])
    # def get_ind_vals(new_y, start):
    #     if not(start):
    #         new_y += peak_max_y
    #     idx = np.abs(tof_ens - new_y).argmin()
    #     if start:
    #         if tof_ens[idx] < new_y:
    #             idx += 1
    #     else:
    #         if tof_ens[idx] > new_y+peak_max_y:
    #             idx -= 1
    #     return idx

    # @staticmethod
    # def discretized_spectrum(spectrum, num_points):
    #     from numpy import interp
    #     spectrum = cp.asnumpy(spectrum)
    #     vals = Pulse.discrete_positions(spectrum, num_points)
    #     st = Pulse.get_ind_vals(vals, True).astype("int")
    #     en = Pulse.get_ind_vals(vals, False).astype("int")
    #     new_spec = np.zeros(len(tof_ens))

    #     orig_tof_no_off = orig_tof_ens[:10]-orig_tof_ens[0]
    #     for start, end, valq in zip(st, en, vals):
    #         intvals = tof_ens[start:end]
    #         new_int = np.interp(intvals, orig_tof_no_off +
    #                             valq, Pulse.single_discrete_spike())
    #         new_spec[start:end] += new_int
    #     new_spec = new_spec/sum(new_spec)
    #     return new_spec

    @staticmethod
    def discretized_spectrum(spectrum, num_points):
        # from numpy import interp
        # spectrum = cp.asnumpy(spectrum)
        # vals = Pulse.discrete_positions(spectrum, num_points)
        # st = Pulse.get_ind_vals(vals, True).astype("int")
        # en = Pulse.get_ind_vals(vals, False).astype("int")
        # new_spec = np.zeros(len(tof_ens))

        # orig_tof_no_off = orig_tof_ens[:10]-orig_tof_ens[0]
        # for start, end, valq in zip(st, en, vals):
        #     intvals = tof_ens[start:end]
        #     new_int = np.interp(intvals, orig_tof_no_off +
        #                         valq, Pulse.single_discrete_spike())
        #     new_spec[start:end] += new_int
        # new_spec = new_spec/sum(new_spec)
        return spectrum


#     def __discretized_spectrum(self,spectrum,num_points):
#         ll=len(self.__single_discrete_spike())
#         sval=self.__discrete_positions(spectrum,num_points)
#         sval_ens=interp(cp.array(sval),cp.arange(len(tof_ens)),tof_ens_gpu)
#         new_spec=cp.zeros(len(tof_ens))
#         one_spike_int = lambda r : interp(tof_ens_gpu
#                                           ,cp.asarray(orig_tof_ens[:ll]-orig_tof_ens[7])+r
#                                           ,self.__single_discrete_spike())
#         all_spikes=cp.array([one_spike_int(i) for i in sval_ens])
#         new_spec= cp.sum(all_spikes,axis=0)

#         return new_spec


#     @staticmethod
#     def single_discrete_spike():
#         from numpy.random import rand
#         from numpy import pad, floor, ceil
# #         padval=(len(orig_tof_ens)-len(noisepeak))/2
#         single_spike = 0.4*rand(10)+gauss_spike
#     #     single_spike= pad(single_spike, (floor(padval).astype(int),
#     #                                      ceil(padval).astype(int)), 'constant', constant_values=(0))
#         return single_spike

#     @staticmethod
#     def discrete_positions(spectrum, num_points):
#         from scipy.interpolate import interp1d
#         from numpy import interp as npinterp
#         from numpy.random import rand
#         from numpy import cumsum
#         from numpy import round as np_round
#         spectrum = cp.asnumpy(spectrum)
#         cumulative_spectrum = (np.cumsum(spectrum))/sum(spectrum)
#         indices = np.arange(len(spectrum))
#         discrete_positions = npinterp(
#             rand(num_points), cumulative_spectrum, indices)
# #         discrete_positions=mint(rand(num_points))
#         discrete_positions = npinterp(
#             discrete_positions, np.arange(len(tof_ens)), tof_ens)

#     #     discrete_positions=np_round(discrete_positions)
#         return discrete_positions

#     two methods (training and validation) from distinct original measurements to introduce
#     background noise into the spectra


    def __makenoise_train(self, maxv):
        '''background noise used for training'''
        from numpy.random import randint
        from numpy import roll
        hnoise = measurednoise_train*cp.asnumpy(maxv)
        return roll(hnoise, np.random.randint(1, len(tof_ens)))[:len(tof_ens)]

    def __makenoise_val(self, maxv):
        '''background noise used for validation'''
        from numpy.random import randint
        from numpy import roll
        hnoise = measurednoise_val*cp.asnumpy(maxv)
        return roll(hnoise, np.random.randint(1, len(tof_ens)))[:len(tof_ens)]

    def get_spectra(self, streakspeed_in_meV_per_fs, keep_originals=False, discretized=True):
        '''returns streaked spectra, measured with "number_electronsx" simulated electrons or nondiscretized as a tuple'''
        from streaking_cal.statistics import weighted_avg_and_std

        if not(self.is_low_res()):

            (streaked1, streaked2) = self.__get_streaked_spectra(
                streakspeed_in_meV_per_fs)

            streaked1 = interp(tof_ens_gpu, self.__eAxis, streaked1)
            streaked2 = interp(tof_ens_gpu, self.__eAxis, streaked2)
            xuvonly = interp(tof_ens_gpu, self.__eAxis,
                             cp.square(cp.abs(self.__spec)))

            if not(keep_originals):
                self.__eAxis = None
                self.__spec = None
                t_square = cp.square(cp.abs(self.__temp))
                t_mean, _ = weighted_avg_and_std(
                    self.__tAxis.get(), t_square.get())
                self.__temp = interp(cp.asarray(
                    standard_full_time), self.__tAxis-t_mean, t_square).get()
                self.__temp = self.__temp/cp.sum(self.__temp)
                self.__tAxis = standard_full_time
                self.__is_low_res = True
                self.__streakedspectra = np.asarray(
                    (xuvonly.get(), streaked1.get(), streaked2.get()))
                self.__streakspeed = streakspeed_in_meV_per_fs
                self.__tAxis = standard_full_time

            if discretized:
                streaked1 = self.discretized_spectrum(
                    streaked1, self.num_electrons1)
                streaked2 = self.discretized_spectrum(
                    streaked2, self.num_electrons2)

            self.__streakspeed = streakspeed_in_meV_per_fs

            return cp.asnumpy(xuvonly), cp.asnumpy(streaked1), cp.asnumpy(streaked2)

        elif discretized:
            (xuvonly, streaked1, streaked2) = self.__streakedspectra
            streaked1 = self.discretized_spectrum(
                streaked1, self.num_electrons1)
            streaked2 = self.discretized_spectrum(
                streaked2, self.num_electrons2)

            return cp.asnumpy(xuvonly), cp.asnumpy(streaked1), cp.asnumpy(streaked2)

        else:
            return self.__streakedspectra.copy()

    def get_augmented_spectra(self, streakspeed_in_meV_per_fs, for_training=False, discretized=True):
        '''data augemtation to imitate: shift of XUV-energy, jitter between Streaking and XUV, background signal
        values here are handcrafted for the (reduced) chosen resolution of self.__eAxis and might 
        need be adjusted in the future
        ;Returns augemented spectra as a matrix'''
        from numpy.random import randint
        from numpy import asarray, roll
        from sklearn.preprocessing import Normalizer

        aug_spectra = asarray(self.get_spectra(
            streakspeed_in_meV_per_fs, discretized=discretized))

#         to reduce dependancy from XUV-photon energy
        # shiftall = randint(-20, 20)
        shiftall = 0 # random wert ist ausgelagert
#         to account for jitter
        shiftstr = randint(-120, 120)

        aug_spectra[0] = roll(aug_spectra[0], shiftall)
        aug_spectra[1] = roll(aug_spectra[1], shiftall-shiftstr)
        aug_spectra[2] = roll(aug_spectra[2], shiftall+shiftstr)

#         add background signal from actual measurements
#         if for_training:
#             aug_spectra[1]=aug_spectra[1]+self.__makenoise_train(max(aug_spectra[1]))
#             aug_spectra[2]=aug_spectra[2]+self.__makenoise_train(max(aug_spectra[2]))

#         else:
#             aug_spectra[1]=aug_spectra[1]+self.__makenoise_val(max(aug_spectra[1]))
#             aug_spectra[2]=aug_spectra[2]+self.__makenoise_val(max(aug_spectra[2]))


#         aug_spectra[1]=np.convolve(noisepeak,aug_spectra[1],mode="same")
#         aug_spectra[2]=np.convolve(noisepeak,aug_spectra[2],mode="same")

#             normalize to area 1
        hnormalizer = Normalizer(norm="l1")
        norm1 = hnormalizer.transform(aug_spectra)

        return norm1

    def to_file(self, writepath):
        if self.is_low_res():
            raise self.Data_Nonexisting_Error(
                'High resolution data is already deleted. Use to_file() before calculating streaked spectra or with get_spectra(keep_originals = True) ')
        else:
            self.get_temp().get().tofile(writepath)

    def get_streakspeed(self):
        return self.__streakspeed

    def to_raw_data(self):
        pass

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
        from tensorflow.keras.utils import Sequence
        from scipy.ndimage import gaussian_filter
        from numpy.random import uniform, randint
        import numpy as np
        from numpy import zeros

        batch = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
#         batch = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

#         x = [X[i].get_raw_matrix() for i in batch]

        x = []
        for _,i in enumerate(batch):
            (xuv, str1, str2) = self.X[i].get_augmented_spectra(
                0, discretized=False)
            b1 = Raw_data(np.asarray((xuv, str1, str2)), tof_ens, self.X[i].get_temp(
            ), num_electrons1=self.X[i].num_electrons1, num_electrons2=self.X[i].num_electrons2)
            x.append(b1.get_raw_matrix())
#             x[ind]=b1.get_all_tof()
#             y.append(b1.get_temp())


#         x = [X[i].get_augmented_spectra(0,discretized=False) for i in batch]
        x = np.asarray(x)

        y = [self.X[i].get_temp() for i in batch]
        y = np.asarray(y)

        # .reshape(self.batch_size, -1)
        return x.reshape(-1, 3, x[0].shape[1], 1), np.array(y)
# %%

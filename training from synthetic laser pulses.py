# %%
# TODO: Warum ist VLS nicht über TOF Daten in Simulationen? Warum ist das gemessene Rauschen nicht 
# wie in den Simulation? Sind die Messdaten korrekt? Bringt es evtl. etwas einen anderen Run anzugucken?


import datetime

from streaking_cal.statistics import weighted_avg_and_std  
# from streaking_cal.misc import interp
from cupy import interp
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, BatchNormalization, Flatten
from tensorflow.keras.layers import Subtract, GlobalMaxPooling2D, Conv2D, Dense, Lambda
from tensorflow.keras import Input
from scipy.signal import gaussian
from progressbar import ProgressBar
# from streaking_cal.misc import interp
import numpy as np
from numpy.random import randint, rand

import pandas as pd
import os
import scipy as sp
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from numba import vectorize, float64, boolean, njit


import tensorflow as tf
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

# from tensorflow_addons.layers import WeightNormalization
from tensorflow.keras.regularizers import l1_l2
# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)

# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))


import cupy as cp
mempool = cp.get_default_memory_pool()
# with cp.cuda.Device(0):
#     mempool.set_limit(size=800*1024**2)
# print(cp.get_default_memory_pool().get_limit())  # 1073741824


# from scipy.signal import find_peaks,peak_widths
# from scipy.ndimage import gaussian_filter
from source.class_collection import Pulse, Raw_data, Datagenerator



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
import tensorflow as tf
# import cupy as cp

print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))
print(tf.version.VERSION)
# cupy.fusion

# %%
dt2 = np.dtype([('xuv', np.float64), ('up', np.float64), ('down', np.float64)])
measured_spectra = []
for i in range(1, 110):
    lll = np.fromfile('./resources/files_mathematica/' +
                      str(i)+'.dat', dtype=dt2)
    xuv0 = np.interp(tof_ens, orig_tof_ens, lll['xuv'], left=0, right=0)
    xuv0 = xuv0/sum(xuv0)
    up0 = np.interp(tof_ens, orig_tof_ens, lll['up'], left=0, right=0)
    up0 = up0/sum(up0)
    down0 = np.interp(tof_ens, orig_tof_ens, lll['down'], left=0, right=0)
    down0 = down0/sum(down0)
    measured_spectra.append(np.array([xuv0, up0, down0]))
measured_spectra = np.asarray(measured_spectra)
# %%
TOF_instrument_function = np.fromfile(
    './resources/files_mathematica/instrument_function.dat', dtype=np.float64)
TOF_instrument_function -= TOF_instrument_function[0]
for num, i in enumerate(TOF_instrument_function):
    if i < 0:
        TOF_instrument_function[num] = 0
TOF_instrument_function = TOF_instrument_function[TOF_instrument_function > 0]

TOF_instrument_function = np.fromfile(
    './resources/TOF_response2.dat', dtype=np.float32)

# %%
x1 = Pulse.from_GS(dT=np.random.uniform(10/2.355, 120/2.355), 
            centralE=71.5,
            dE=np.random.uniform(0.2/2.355, 1.8/2.355), 
            num_electrons1=np.random.randint(15, 31), 
            num_electrons2=np.random.randint(15, 31)
                       )

# %%
# enss=[]
# for i in np.arange(1024)+1:
#     enss.append(b1.VLS_pixel_to_energies(i))
# %%
sss=x1.get_augmented_spectra(0,discretized=False)
plt.plot(enss,2.3*b1.VLS_signal)
plt.plot(tof_ens,sss[0])
plt.plot(tof_ens,10*sss[1])

plt.plot(tof_ens,10*sss[2])

plt.xlim([67,85])
# %%

# %%
(xuv, str1, str2) = x1.get_augmented_spectra(95, discretized=False)
b1 = Raw_data(np.asarray((xuv, str1, str2)), tof_ens, x1.get_temp(), 
            num_electrons1=x1.num_electrons1, num_electrons2=x1.num_electrons2)
rm=b1.get_raw_matrix()
plt.plot(rm[0])
plt.plot(rm[1])
plt.plot(rm[2])
plt.xlim([400,800])

# %%
# timeit
pbar = ProgressBar()

num_pulses = 100000
streakspeed = 95  # meV/fs
X = [""]*num_pulses
y = [""]*num_pulses

for i in pbar(range(num_pulses)):
    x1 = Pulse.from_GS(dT=np.random.uniform(10/2.355, 120/2.355), 
            dE=np.random.uniform(0.2/2.355, 1.8/2.355), 
            num_electrons1=np.random.randint(15, 85), 
            num_electrons2=np.random.randint(15, 85),
            centralE=np.random.uniform(70,76)
                       )
    x1.get_spectra(streakspeed, discretized=False)
    X[i] = x1



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
def maxLayer():
    return MaxPooling2D(pool_size=(8,8),strides=(1,8),padding="same")

def convLayer(filters):
    return Conv2D(filters=filters, kernel_size=(1, 7), activation="relu", strides=1, padding="same")

# %%

convdim = 128

inputs = Input(shape=(3, 1825, 1), name="traces")



conv_out = Conv2D(convdim, kernel_size=(3, 500), activation="relu", strides=1, padding="same"
                  )(inputs)


# # x = MaxPooling2D(pool_size=(3, 3),strides=(1,2), padding="valid")(conv_out)
# # x2 = AveragePooling2D(pool_size=(3, 3),strides=(1,2), padding="valid")(conv_out)

# # x = Subtract()([x, x2])

x = GlobalMaxPooling2D()(conv_out)

# x = tf.keras.layers.LeakyReLU()(x)

# x = Flatten()(x)

x = BatchNormalization()(x)

x = Dense(256, activation="relu"
          #          , kernel_constraint=CenterAround(0)
          )(x)

# x= Dense(10,activation="relu"
# #          , kernel_constraint=CenterAround(0)
#         )(x)

# x= Dense(100,activation="relu"
# #          , kernel_constraint=CenterAround(0)
#         )(x)


outputs = Dense(standard_full_time.shape[0], activation="softmax")(x)

model = tf.keras.Model(inputs, outputs, name="mynet")
model.summary()
# %%

wholeset = np.arange(len(X))

pulses_train, pulses_test, y_train, y_test = train_test_split(
    wholeset, wholeset, test_size=0.05, random_state=1)
params = {'batch_size': 200}
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
model.compile(optimizer="nadam", loss="KLDivergence",
              metrics=["accuracy", "mae"])

history = model.fit(x=train_ds, validation_data=test_ds,
                    #                     use_multiprocessing=True,
                    #                     workers=4,
                    epochs=80
                    )
# %%
from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D, BatchNormalization, Flatten

# %%
# model.save('./models/RAW_mat_15-30els_70-76eV')

# model = tf.keras.models.load_model('./models/RAW_mat_15-30els_70-76eV')

# from numba import cuda 
# device = cuda.get_current_device()
# device.reset()
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
testitems= test_ds.__getitem__(1)
preds=model.predict(testitems[0])
y_test=testitems[1]
# %matplotlib inline
vv=11


plt.plot(standard_full_time,y_test[vv])
plt.plot(standard_full_time,preds[vv],'--')
# plt.plot(time,gaussian_filter(y_test[vv],10))

plt.figure()
plt.plot(standard_full_time,preds[vv],'orange')

plt.figure()
# plt.plot(tof_ens,testitems[0][vv][0])
plt.plot(np.arange(1825),testitems[0][vv][2])
plt.plot(np.arange(1825),testitems[0][vv][1])
plt.plot(np.arange(1825),testitems[0][vv][0])
plt.xlim([200,700])
# %%
import os
import re
from itertools import repeat

files=os.listdir("./resources/raw_mathematica/up/")
numbers=list(map(re.findall,repeat("[0-9]{4}"),files))
numbers=np.asarray(numbers)[:,0].astype("int32")

# %%



tof1=[]
tof2=[]
vls=[]
for i in numbers:
    tof11=np.fromfile("./resources/raw_mathematica/up/TOF1-"+str(i)+".dat","float32")
    tof21=np.fromfile("./resources/raw_mathematica/up/TOF2-"+str(i)+".dat","float32")
    vls11=np.fromfile("./resources/raw_mathematica/up/VLS-"+str(i)+".dat","float32")
    tof11=np.roll(tof11,150)
    tof21=np.roll(tof21,150)
    vls11=np.pad(vls11,pad_width=(0, len(tof11)-len(vls11)))
    tof1.append(tof11)
    tof2.append(tof21)
    vls.append(vls11)

testitems=np.reshape(np.asarray([vls,tof1,tof2]),[len(numbers),3,-1,1])


# %%

preds=model.predict(testitems)
# %matplotlib inline
vv=10


# plt.plot(time,gaussian_filter(y_test[vv],10))

plt.figure()
plt.plot(standard_full_time,preds[vv],'orange')
weighted_avg_and_std(standard_full_time,preds[vv])
# %%
plt.plot(np.arange(1825),testitems[vv][1])
plt.plot(np.arange(1825),testitems[vv][2])
plt.plot(np.arange(1825),testitems[vv][0])
plt.xlim([1,1500])
# %%

# %%

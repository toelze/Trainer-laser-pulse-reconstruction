# %%

# TODO: warum 326 statt 301 energien
# TODO: wie können die wichtigsten Daten elegant mit geliefert werden beim Erzeugen eines neuen Objekt?
# TODO: welche Daten sind bei jeder Instanz gleich / unterschiedlich?
# TODO: composition: Klassen halten Daten für Pulsparameter, für statische/ variable Messumgebung,
# TODO Raw_Data2 zu Raw_Data wandeln (Raw_Data dabei löschen?)
# TODO neuronales Netz anpassen

# TODO: grafiken erzeugen : Genauigkeit über time-bandwidth-product
# TODO:                     Genauigkeit über num_electrons


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
from source.class_collection import Datagenerator, StreakedData, Raw_Data2, PulseProperties

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

# def fs_in_au(t): return 41.3414*t  # from fs to a.u.
# def eV_in_au(e): return 0.271106*np.sqrt(e)  # from eV to a.u.

# p0 = eV_in_au(CentralEnergy)

# Pulse class returns temporal profile on basis of this time axis
standard_full_time = np.loadtxt('./resources/standard_time.txt')
standard_full_time = np.linspace(-250, 250, 512)

# # background noise for data augmentation is read from actual measured spectra
# measurednoise_train = np.loadtxt("./resources/measurednoise_train.txt")
# measurednoise_val = np.loadtxt("./resources/measurednoise_val.txt")

print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))
print(tf.version.VERSION)

# %%
from source.class_collection import Datagenerator, StreakedData, Raw_Data, Measurement_Data



# %%
# vlspix = np.arange(1024) + 1 # from Mathematica + indexcorrection
# VLSens = 1239.84/((vlspix)*0.0032 + 11.41) - 21.55 


# %%
# timeit
# pbar = ProgressBar()
from tqdm import tqdm as pbar

num_pulses = 100000
streakspeed = 95  # meV/fs
X = [""]*num_pulses
y = [""]*num_pulses

for i in pbar(range(num_pulses),colour = 'red', ncols= 100):
    pulse_props= PulseProperties(fwhm_t = np.random.uniform(70, 150),  # in fs
                                 fwhm_E = np.random.uniform(0.2, 1.8), # in eV
                                 num_electrons0 = np.random.randint(15, 40), 
                                 num_electrons1 = np.random.randint(15, 40), 
                                 centralE = np.random.uniform(65,75))

    x1 = pulse_props.to_StreakedData_from_GetSASE(streakspeed = streakspeed)
    # x1.get_spectra(streakspeed, discretized=False)
    X[i] = x1
    
# %%
pp = X[25]
dd = pp.to_Raw_Data().to_Measurement_Data()
ff = dd.spectra


plt.plot(pp.pulse_props.tAxis,pp.pulse_props.tOutput)
plt.figure()
plt.plot(dd.energy_axis,ff[1])
plt.plot(dd.energy_axis,ff[2])
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
testitems= train_ds.__getitem__(0)
preds=merged_model.predict(testitems[0])
y_test=testitems[1]
# %matplotlib inline
vv=75


plt.plot(standard_full_time,y_test[vv])
plt.plot(standard_full_time,preds[vv],'--')
# plt.plot(time,gaussian_filter(y_test[vv],10))

plt.figure()
plt.plot(standard_full_time,preds[vv],'orange')

plt.figure()
# plt.plot(tof_ens,testitems[0][vv][0])
plt.plot(np.arange(326),testitems[0][vv][0])
plt.plot(np.arange(326),testitems[0][vv][1])
plt.plot(np.arange(326),testitems[0][vv][2])
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
from source.class_collection import Datagenerator, StreakedData, Raw_Data

num_pulses = 10000
streakspeed = 95  # meV/fs
X = [""]*num_pulses
y = [""]*num_pulses

for i in pbar(range(num_pulses),colour = 'red', ncols= 100):
    x1 = StreakedData.from_GS(dT=np.random.uniform(20/2.355, 120/2.355), 
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

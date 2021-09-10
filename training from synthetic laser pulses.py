# %%

# TODO: gemessene Daten sind wieder nur schlecht durch simulierte angehähert, woran liegt das?
# TODO: import 'to_eVs_from_file' funktioniert nicht TOFs sind falsch skaliert
# TODO: strshift auf letzte Klasse verlagern
# TODO neuronales Netz anpassen; functionen auslagern
# TODO: self.x und self.y in Datagenerator loswerden

# TODO: Grafik: welche Periodendauern werden wie rekonstruiert? (testdaten)
# TODO: grafiken erzeugen : Genauigkeit über time-bandwidth-product
# TODO:                     Genauigkeit über num_electrons

# TODO: delete files not used


import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from tensorflow.keras import Input
from tensorflow.keras.layers import (Reshape, BatchNormalization,
                                     Conv2D, Dense, Flatten, Softmax,
                                     GlobalMaxPooling2D, Conv1DTranspose,
                                     ReLU)
from tensorflow.keras.optimizers import Adam

from source.class_collection import (Data_Generator, Pulse_Properties)
from source.process_stages import streaking

from streaking_cal.statistics import weighted_avg_and_std

# set max used memory by cupy
# mempool = cp.get_default_memory_pool()
# with cp.cuda.Device(0):
#     mempool.set_limit(size=800*1024**2)
# print(cp.get_default_memory_pool().get_limit())  # 1073741824


CentralEnergy = 73


print("Num GPUs Available: ", len(
    tf.config.experimental.list_physical_devices('GPU')))
print(tf.version.VERSION)

# %%

from tqdm import tqdm as pbar
from source.process_stages import get_exp_env

num_pulses = 100000
streakspeed = 95  # meV/fs
X = [""]*num_pulses
y = [""]*num_pulses
exp_env = get_exp_env()

for i in pbar(range(num_pulses),colour = 'red', ncols= 100):
    pulse_props= Pulse_Properties(fwhm_t = np.random.uniform(60, 180),  # in fs
                                 fwhm_E = np.random.uniform(0.2, 1.8), # in eV
                                 num_electrons0 = np.random.randint(15, 40), 
                                 num_electrons1 = np.random.randint(15, 40), 
                                 centralE = np.random.uniform(CentralEnergy-3,CentralEnergy+3))

    x1 = streaking(streakspeed = np.random.uniform(streakspeed-5,streakspeed+5), 
                   exp_env = exp_env, pulse_props = pulse_props)

    X[i] = x1
# %%
# create figures for talk
# temp. profile from GetSASE
import cupy as cp
from streaking_cal.GetSASE import GetSASE_gpu as GS
(enAxis, enOutput, tAxis, tOutput) = GS(CentralEnergy=pulse_props.centralE,
                                                dE_FWHM=2.355*pulse_props.dE*2**0.5,
                                                dt_FWHM=2.355*pulse_props.dT*2**0.5,
                                                onlyT=False)


plt.plot(cp.abs(tOutput).get())
plt.savefig("getSASE.svg")

# %%
# streaked spectra figure
from source.process_stages import (measurement, to_eVs)

nn=14
n0 = X[nn].get_augmented_spectra()[0]
n0 = n0 / sum(n0)
n1 = X[nn].get_augmented_spectra()[1]
n1 = n1 / sum(n1)
n2 = X[nn].get_augmented_spectra()[2]
n2 = n2 / sum(n2)
plt.plot(n0)
plt.plot(n1)
plt.plot(n2)
plt.xlim([545,750])

plt.savefig("streaked_spectra2.svg")
plt.figure()


# discretized spectra
meas= measurement(X[nn])
n0 = np.roll(meas.calc_tof_traces()[0],-610)
n0 = n0 / sum(n0)
n1 = np.roll(meas.calc_tof_traces()[1],-610)
n1 = n1 / sum(n1)
n2 = meas.eVenergies_to_vls_pix(meas.spectra[0]) # VLS
n2 = n2 / sum(n2)

plt.plot(n0)
plt.plot(n1)
plt.plot(n2)
plt.xlim([490,600])

plt.savefig("discretized_spectra2.svg")
plt.figure()

# final spectra
teV=to_eVs(meas).spectra

plt.plot(teV[0])
plt.plot(teV[1])
plt.plot(teV[2])
plt.xlim([80,190])

plt.savefig("final_spectra2.svg")

# %%
plt.imshow(np.log(teV+0.01),aspect=25, cmap = 'Greens', interpolation = 'none')
plt.savefig("input_data.svg")

# %%
# test, what is the effect of shot noise?
tof_traces = meas.calc_tof_traces()
tof_traces_n = meas.add_tof_noise(tof_traces[0],50)

plt.plot(tof_traces[0])
plt.plot(tof_traces_n)
plt.xlim([1000,1200])
# %%

wholeset = np.arange(len(X))

pulses_train, pulses_test, y_train, y_test = train_test_split(
    wholeset, wholeset, test_size=0.05, random_state=1)
params = {'batch_size': 300}
train_ds = Data_Generator(pulses_train, y_train, X=X, **params)
test_ds = Data_Generator(pulses_test, y_test, X=X, for_train=False, **params)

# %%
def conv_Encoder(inputs: tf.keras.Input, convdim) -> tf.keras.Model:
    conv_out = Conv2D(convdim, 
                      kernel_size=(3, 250), 
                      activation="linear", 
                      strides=1, 
                      padding="same")(inputs)
    
    x = GlobalMaxPooling2D()(conv_out)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    outputs = Dense(convdim, activation="relu")(x)
    

    return tf.keras.Model(inputs, outputs, name="encoder")

def dense_Encoder(inputs: tf.keras.Input) -> tf.keras.Model:
    x = Flatten()(inputs)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    
    outputs = Dense(128, activation="linear")(x)

    


    return tf.keras.Model(inputs, outputs, name="encoder")


def convT_Decoder(inputs: tf.keras.Input, outputsize: int) -> tf.keras.Model:
    if outputsize % 16 != 0:
        raise(Exception('outputsize must be a multiple of 16!'))
    
    num_filters = 16
    
    # x = BatchNormalization()(inputs)
    x = Reshape((1,-1))(inputs)
    x = Conv1DTranspose(filters = num_filters, kernel_size=outputsize // num_filters)(x)
    x = Flatten()(x)
    outputs = Softmax()(x)
    
    return tf.keras.Model(inputs, outputs, name="decoder")

def dense_Decoder(inputs: tf.keras.Input, outputsize: int) -> tf.keras.Model:
    x = Dense(256, activation="relu")(inputs)
    outputs = Dense(outputsize, activation="softmax")(x)
    
    return tf.keras.Model(inputs, outputs, name="decoder")



# %%
convdim = 256

specdim = test_ds.__getitem__(0)[0].shape[2]

enc_inputs = Input(shape=(3, specdim, 1), name="traces")

encoder = conv_Encoder(enc_inputs,convdim)
encoder.summary()

dec_inputs = Input(shape=(convdim), name="encoder_output")

decoder = dense_Decoder(dec_inputs, outputsize=512)
decoder.summary()


# %%
encoded = encoder(enc_inputs)
decoded = decoder(encoded)
merged_model = tf.keras.Model(enc_inputs, decoded, name="merged_model")

merged_model.summary()

# time2=time[abs(time)<250]
# %%

opt = Adam(lr=5e-3, amsgrad=True) 
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.2, decay_steps=15000, decay_rate=0.99, staircase=True)
nadam = tf.keras.optimizers.Nadam(learning_rate=0.006)
adagrad = tf.keras.optimizers.Adagrad(
    learning_rate=lr_schedule, initial_accumulator_value=0.1)
merged_model.compile(optimizer="nadam", 
                    #  loss="KLDivergence",
                    loss = 'categorical_crossentropy',
                    metrics=["accuracy", "mae", "KLDivergence"])

history = merged_model.fit(x=train_ds, validation_data=test_ds,
                                        # use_multiprocessing=True,
                    #                     workers=4,
                    epochs=3
                    )

# %%

train_ds.__getitem__(0)[0].shape[1:]
# %%

# ----------------------SAVE MODELS-------------------------

# merged_model.save('./models/3.2THz-eV-95-merged')
# encoder.save('./models/3.2THz-eV-95-encoder')
# decoder.save('./models/3.2THz-eV-95-decoder')


# ----------------------LOAD MODELS-------------------------

# merged_model = tf.keras.models.load_model('./models/3.2THz-eV-95-merged')
# encoder = tf.keras.models.load_model('./models/3.2THz-eV-95-encoder')
# decoder = tf.keras.models.load_model('./models/3.2THz-eV-95-decoder')
# %%



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
enax=np.arange(45,110.1,0.2)
testitems= train_ds.__getitem__(0)
preds=merged_model.predict(testitems[0])
y_test=testitems[1]
# %matplotlib inline
vv=27


plt.plot(exp_env.reconstruction_time,y_test[vv])
plt.plot(exp_env.reconstruction_time,preds[vv],'--')
# plt.plot(time,gaussian_filter(y_test[vv],10))

plt.figure()
plt.plot(exp_env.reconstruction_time,preds[vv],'orange')

plt.figure()
# plt.plot(tof_ens,testitems[0][vv][0])
plt.plot(enax,testitems[0][vv][0])
plt.plot(enax,testitems[0][vv][1])
plt.plot(enax,testitems[0][vv][2])
# plt.xlim([500,700])

# %%
plt.figure()
plt.plot(exp_env.reconstruction_time,preds[vv],'orange')
plt.savefig("output.svg")

plt.figure()
plt.plot(exp_env.reconstruction_time,y_test[vv])
plt.savefig("ref_output.svg")
# %%
import os
import re
from itertools import repeat
from source.process_stages import to_eVs_from_file
# %%
folder="./resources/raw_mathematica/up/"
files=os.listdir(folder)

numbers=list(map(re.findall,repeat("[0-9]{4}"),files))
numbers=np.asarray(numbers)[:,0].astype("int32")

X_exp = [to_eVs_from_file(i,exp_env) for i in numbers]
# %%
X_exp[0].spectra.shape
# %%
plt.plot(X_exp[0].spectra[2])
#%%


# tof1=[]
# tof2=[]
# vls=[]
# testitems2=[]
# for i in numbers:
#     tof11=np.fromfile(folder+"TOF1-"+str(i)+".dat","float32")
#     tof21=np.fromfile(folder+"TOF2-"+str(i)+".dat","float32")
#     vls11=np.fromfile(folder+"VLS-"+str(i)+".dat","float32")
#     tof11=np.roll(tof11,150)
#     tof21=np.roll(tof21,150)
#     vls11=np.pad(vls11,pad_width=(0, len(tof11)-len(vls11)))
#     vls11=np.roll(vls11,0)
#     testitems2.append([vls11,tof11,tof21])
# testitems2=np.reshape(np.asarray(testitems2),[len(numbers),3,-1,1])
# %%
testitems2 = [i.spectra for i in X_exp]
testitems2=np.reshape(np.asarray(testitems2),[len(numbers),3,-1,1])
#%%
preds=merged_model.predict(testitems2)

# %%
# %matplotlib inline
vv=26

# plt.plot(time,gaussian_filter(y_test[vv],10))

plt.figure()
plt.plot(exp_env.reconstruction_time,
        preds[vv],
        'orange')
vsigma=weighted_avg_and_std(exp_env.reconstruction_time,preds[vv])
print([vsigma,2.35*vsigma[1]])
plt.figure()
plt.plot(exp_env.reconstruction_energies,testitems2[vv][1])
plt.plot(exp_env.reconstruction_energies,testitems2[vv][2])
plt.plot(exp_env.reconstruction_energies,testitems2[vv][0])
# plt.xlim([500,700])


# %%
# encode synth pulses is batches of 300 and concatenate
from source.class_collection import Data_Generator, Streaked_Data, Raw_Data

num_pulses = 10000
streakspeed = 95  # meV/fs
X = [""]*num_pulses
y = [""]*num_pulses

for i in pbar(range(num_pulses),colour = 'red', ncols= 100):
    x1 = Streaked_Data.from_GS(dT=np.random.uniform(20/2.355, 120/2.355), 
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
train_ds = Data_Generator(pulses_train, y_train, X=X, **params)
test_ds = Data_Generator(pulses_test, y_test, X=X, for_train=False, **params)


# encoded_t0 = encoded_t
# lower_dim_input0 = lower_dim_input
# encoded_t =[encoder(test_ds.__getitem__(i)[0]) for i in range(30)]
# encoded_t = np.concatenate(encoded_t, axis = 0)

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
plt.plot(exp_env.reconstruction_time,
        temp_nearest,
        'orange')
vsigma=weighted_avg_and_std(exp_env.reconstruction_time,temp_nearest)
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

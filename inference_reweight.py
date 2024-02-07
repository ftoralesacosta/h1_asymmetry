import numpy as np
import tensorflow as tf
from tqdm import tqdm
import yaml
import os
import sys

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model

from unfold_hvd import weighted_binary_crossentropy
from unfold_hvd import reweight


# CONFIG FILE
if len(sys.argv) > 1:
    CONFIG_FILE = sys.argv[1]
else:
    CONFIG_FILE = "./configs/config.yaml"

config = yaml.safe_load(open(CONFIG_FILE))
print(f"\nLoaded {CONFIG_FILE}\n")

mc = config['mc']  # Rapgap, Django, Pythia
run_type = config['run_type']  # nominal, bootstrap, systematic
main_dir = config['main_dir']
model_dir = config['model_dir']

LABEL = config['identifier']
ID = f"{mc}_{run_type}_{LABEL}"
if config['is_test']:
    ID = ID + "_TEST"  # avoid overwrites of nominal

N_iter = config['n_iterations']
N_passes = config['n_passes']

# Need to save each pass, and it would be nice to save pass_avgs :]


# ========================================
# Tensorflow Init

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ======= Load the thetaG and unknownS ======
theta0_G = np.load(f"npy_inputs/{ID}_Theta0_G.npy")
print("\n\nLoaded theta0_G")
theta_unknown_S = np.load(f"npy_inputs/{ID}_theta_unknown_S.npy")
print("Loaded theta_unknown_S")
weights_MC_sim = np.load(f"npy_inputs/{ID}_weights_mc_sim.npy")
print("Loaded weights MC\n")

#Get StandardScalar from raw data, apply to Rapgap in each iteration later   

# ======= Load the Model ======
ID_File = ID # FIXME will add be np.arrange(n_bootstraps)
num_observables = 8
if config['asymm_vars']:
    num_observables = 12

inputs = Input((num_observables, ))
hidden_layer_1 = Dense(50, activation='relu')(inputs)
dropoutlayer = Dropout(0.1)(hidden_layer_1)
hidden_layer_2 = Dense(100, activation='relu')(dropoutlayer)
hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
outputs = Dense(1, activation='sigmoid')(hidden_layer_3)
model = Model(inputs=inputs, outputs=outputs)


model.compile(loss=weighted_binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=2e-6),
              metrics=['accuracy'])

pass_avgs = np.zeros( (N_iter, len(theta0_G[:,0])) )

for p in tqdm(range(N_passes)):

    for i in range(N_iter):

        model_name = f'{model_dir}/{ID}/{ID_File}_Pass{p}/iter_{i}_step2_checkpoint'
        print(f"\n\nLoading Model {model_name}")
        model.load_weights(model_name)

        # model = tf.keras.models.load_model(Model_Name)
        print(f"Success!")

        weights_push = weights_MC_sim * reweight(theta0_G, model, verbose = 1)
        print(f"Mean Weight = {np.mean(weights_push)}")
        print(f"STDV Weight = {np.std(weights_push)} ")
        # print("Shape of NN Weights = ",np.shape(weights_push))
        # print(f"Weights in Pass {p}:")
        # print(weights_push)

        # file = f"./weights/{LABEL}_Pass{p}_Step2_Weights.npy"
        # np.save(file, weights_push)
        pass_avgs[i] += weights_push

    pass_avgs[i] = pass_avgs[i]/N_passes

np.save(f"./weights/{LABEL}_{ID}_pass_avgs.npy", pass_avgs)
print("Weights saved to ./weights/")

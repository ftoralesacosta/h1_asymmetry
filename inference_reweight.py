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

n_iter = config['n_iterations']
n_passes = config['n_passes']

# Need to save each pass, and it would be nice to save pass_avgs :]


# ========================================
# Tensorflow Init
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


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
n_passes = 1
num_observables = 8

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

for p in tqdm(range(n_passes)):

    Model_Name = f'{model_dir}/{ID}/{ID_File}_Pass{p}/iter_4_step2_checkpoint'
    print(f"\n\nLoading Model {Model_Name}")
    model.load_weights(Model_Name)

    # model = tf.keras.models.load_model(Model_Name)
    print(f"Success!")

    weights_push = weights_MC_sim * reweight(theta0_G, model, verbose = 1)
    print(f"Mean Weight = {np.mean(weights_push)}")
    print(f"STDV Weight = {np.std(weights_push)} ")
    print(f"Weights in iteration {i}:")
    print(weights_push)

# FIXME: Bootstrapping needs the npy random seed in ID_FILE. Its in the filename
# model_name = f'{model_dir}/{ID}/{ID_File}/iter_{i}_step2_checkpoint'


# ============================================
# ================== OLD =====================
# ============================================

#scaler_data = StandardScaler()
#scaler_data.fit(theta_unknown_S)
#print("\n Length of theta0G =",len(theta0_G),"\n")
## run_iter = 4
#bootstrap_weights = []
#for seed in tqdm(np.concatenate([range(1,30),range(34,45),[46,47,48,49],range(54,66),range(80,86),range(100,106),range(120,126)])):

#    #Make sure to reset weights
#    NNweights_step2 = np.ones(len(theta0_G))
#    NNweights_step2_hold = np.ones(len(theta0_G))

#    for run_iter in range(5):
#        print(
#            "Loading /clusterfs/ml4hep/yxu2/inputfiles/fullscan_stat/models/Rapgap_nominal_iteration_"
#                +str(run_iter)+"_"+str(seed)+"_step2")

#        mymodel = tf.keras.models.load_model(
#            "/clusterfs/ml4hep/yxu2/inputfiles/fullscan_stat/models/Rapgap_nominal_iteration"+str(run_iter)+"_"+str(seed)+"_step2", compile=False)

#        NNweights_step2_hold = mymodel.predict(scaler_data.transform(theta0_G),batch_size=10000)
#        NNweights_step2_hold = NNweights_step2_hold/(1.-NNweights_step2_hold)
#        NNweights_step2_hold = NNweights_step2_hold[:,0]
#        NNweights_step2_hold = np.squeeze(np.nan_to_num(NNweights_step2_hold,posinf=1))
#        NNweights_step2_hold[pass_truth==0] = 1.
#        NNweights_step2 = NNweights_step2_hold*NNweights_step2

#        tf.keras.backend.clear_session()

#    bootstrap_weights.append(NNweights_step2)

#bootstrap_weights = np.asarray(bootstrap_weights)
#np.save("bootstrap_rapgap_weights.npy",bootstrap_weights)



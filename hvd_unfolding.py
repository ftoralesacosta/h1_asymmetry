import sys
import numpy as np
import pandas as pd
import gc
import time
import os
import yaml
import tensorflow as tf
import tensorflow.keras
import horovod.tensorflow.keras as hvd

# from get_np_arrays import get_kinematics
from unfold_hvd import MultiFold
from unfold_hvd import MASK_VAL


hvd.init()

gpus = tf.config.list_physical_devices('GPU')
print(f"HVD: {hvd.rank()+1} / {hvd.size()}")
print(f"local HVD (no +1): {hvd.local_rank()} / {hvd.size()}")
print(f"gpus: ", gpus)
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
    # if gpus:
    #     tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')


print("="*50)
print("MASK VAL = ", MASK_VAL)
print("-"*50)
print("GPUs = ",gpus)
print("-"*50)
print(f"HVD: {hvd.rank()+1} / {hvd.size()}")
print("="*50)


print(f"\nTotal hvd size {hvd.size()}, rank: {hvd.rank()}, local size: {hvd.local_size()}, local rank{hvd.local_rank()}\n")


if len(sys.argv) > 1:
    CONFIG_FILE = sys.argv[1]
else:
    CONFIG_FILE = "./configs/config.yaml"

config = yaml.safe_load(open(CONFIG_FILE))
print(f"\nLoaded {CONFIG_FILE}\n")

processed_dir = config['main_dir']
model_folder = config['model_dir']
inputs_dir = config['data_dir']

mc_type = config['mc']  # Rapgap, Django, Pythia
run_type = config['run_type']  # nominal, bootstrap, systematic
LABEL = config['identifier']
ID = f"{mc_type}_{run_type}_{LABEL}"

if config['is_test']:
    ID = ID + "_TEST"  # avoid overwrites of nominal

ID_File = ID

print(f"Running on MC sample {mc_type} with setting, {run_type}")
print(f"\n\n ID = {ID} \n\n")

save_dir = model_folder
try:
    os.mkdir(f"{save_dir}/{ID}/")
except OSError as error:
    print(error)

# Copy the config used to the output dir
os.system(f"cp {CONFIG_FILE} {save_dir}/{ID}")

# BOOTSTRPPING SEED
np_seed = 0
if len(sys.argv) > 2:
    np_seed = int(float(sys.argv[2]))
    np.random.seed(np_seed)


add_asymmetry = config['asymm_vars']
leading_jets_only = True
num_observables = 8

if config['asymm_vars']:
    num_observables = 12

NEVENTS = -1
n_epochs = config['n_epochs']
NIter = config['n_iterations']
NPasses = config['n_passes']

if config['is_test']:
    NEVENTS = 100_000  # usually not enough for results
    n_epochs = 4
    NIter = 5
    NPasses = 2

# theta0_G = np.load(f"./npy_inputs/Rapgap_nominal_Perlmutter_Bootstrap_Theta0_G.npy")[hvd.rank():NEVENTS:hvd.size()]
# theta0_S = np.load(f"./npy_inputs/Rapgap_nominal_Perlmutter_Bootstrap_Theta0_S.npy")[hvd.rank():NEVENTS:hvd.size()]
# theta_unknown_S = np.load(f"./npy_inputs/Rapgap_nominal_Perlmutter_Bootstrap_theta_unknown_S.npy")[hvd.rank():NEVENTS:hvd.size()]

theta0_G = np.load(f"npy_inputs/{ID}_Theta0_G.npy")[hvd.rank():NEVENTS:hvd.size()]
theta0_S = np.load(f"npy_inputs/{ID}_Theta0_S.npy")[hvd.rank():NEVENTS:hvd.size()]
theta_unknown_S = np.load(f"npy_inputs/{ID}_theta_unknown_S.npy")[hvd.rank():NEVENTS:hvd.size()]

# Set Initial Data Weights for BOOTSTRAPPING
dataw = None
if (np_seed != 0):
    dataw = np.random.poisson(1, len(theta_unknown_S))
    print("Doing Bootstrapping")
    print(f"10 Weights = {dataw[:10]}")
    ID_File += str(np_seed)
else:
    print("Not doing bootstrapping")


# try:
#     os.mkdir(f"{save_dir}/{ID}/{ID_File}/")
# except OSError as error:
#     print(error)


print("="*50)
print("L: 187")
print("NaN in Theta0_S = ", np.isnan(theta0_S).any() )
print("NaN in Theta0_G = ", np.isnan(theta0_G).any() )
print("NaN in Data = ", np.isnan(theta_unknown_S).any() )
print("-"*50)
print("INF in Theta0_S = ", np.inf in theta0_S)
print("INF in Theta0_G = ", np.inf in theta0_G)
print("INF in Data = ", np.inf in theta_unknown_S)
print("="*50)


for p in range(NPasses):

    start = 0.1
    if hvd.rank() == 0:
        start = time.time()
        print(f"Unfolding Pass {p}...")


    M_F = MultiFold(num_observables, NIter,
                    theta0_G, theta0_S,
                    theta_unknown_S, 
                    ID, ID_File+f"_Pass{p}", 
                    save_dir, n_epochs,
                    None, dataw)

    weights, models, history = M_F.unfold()
    tf.keras.backend.clear_session()

    # save step 2 first, more important :]
    np.save(f"{save_dir}/{ID}/{ID_File}_Pass{p}_Step2_Weights.npy", weights[:, 1:2, :])
    np.save(f"{save_dir}/{ID}/{ID_File}_Pass{p}_Step2_MODEL_W.npy", models)
    np.save(f"{save_dir}/{ID}/{ID_File}_Pass{p}_Step2_History.npy", weights[:, 1:2, :])

    np.save(f"{save_dir}/{ID}/{ID_File}_Pass{p}_Step1_Weights.npy", weights[:, 0:1, :])
    np.save(f"{save_dir}/{ID}/{ID_File}_Pass{p}_Step1_History.npy", weights[:, 0:1, :])

    if hvd.rank() == 0:
        print(f"Pass {p} took {time.time() - start} seconds \n")

    # weights = None
    # models = None
    # history = None

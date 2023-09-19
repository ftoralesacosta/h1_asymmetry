'''
run like python refactor_unfolding.py Rapgap nominal  0
[command] [Rapgap or Django] [run_type: nominal, sys_0...] [BOOTSTRAPPING Seed]
'''

import sys
import numpy as np
import pandas as pd
import gc
import time
import os
import tensorflow as tf
import tensorflow.keras
from get_np_arrays import get_kinematics

from unfold_hvd import *
print("="*50)
print("MASK VAL = ", MASK_VAL)
print("="*50)


import horovod.tensorflow.keras as hvd
hvd.init()

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

print("-"*15)
print("GPUs = ",gpus)
print(f"HVD: {hvd.rank()+1} / {hvd.size()}")
print("-"*15)

print("Running on MC sample", sys.argv[1], "with setting", sys.argv[2])
if (sys.argv[1] == "Django" and sys.argv[2] == 'closure'):
    sys.exit("closure test must be run with Rapgap as MC")

# mc = "Rapgap"
mc = sys.argv[1]
# LABEL = f"{mc}_HVDRapgap_Aug30"
LABEL = f"{mc}_HVDRapgap_Sep11"
ID = f"{sys.argv[1]}_{sys.argv[2]}_{LABEL}"
print(f"\n\n ID = {ID} \n\n")

save_dir = "/pscratch/sd/f/fernando/h1_models"
try:
    os.mkdir(f"{save_dir}/{ID}/")
except OSError as error:
    print(error)


# tf.random.set_seed(int(sys.argv[3]))
np.random.seed(int(sys.argv[3]))

test = False
add_asymmetry = False
leading_jets_only = True
num_observables = 8

NEVENTS = -1
n_epochs = 1000
NIter = 10
NPasses = 5

if test:
    NEVENTS = 1_000_000
    # NEVENTS = -1
    n_epochs = 10
    NIter = 10
    NPasses = 2

inputs_dir = "/pscratch/sd/f/fernando/h1_data"

if sys.argv[2] == 'closure':
    data = pd.read_pickle(f"{inputs_dir}/Django_nominal.pkl")[hvd.rank():NEVENTS:hvd.size()]
    mc = pd.read_pickle(f"{inputs_dir}/{sys.argv[1]}_nominal.pkl")[hvd.rank():NEVENTS:hvd.size()]
    #see if remainder is larger than batch_size
    # Unfoldes Django to Rapgap
else:
    mc = pd.read_pickle(f"{inputs_dir}/{sys.argv[1]}_{sys.argv[2]}.pkl")[hvd.rank():NEVENTS:hvd.size()]
    data = pd.read_pickle(f"{inputs_dir}/Data_nominal.pkl")[hvd.rank():NEVENTS:hvd.size()]

print(f"MC = {inputs_dir}/{sys.argv[1]}_{sys.argv[2]}.pkl")
print(f"Data = {inputs_dir}/Data_nominal.pkl")

print(np.shape(mc))
print(np.shape(data))

if (leading_jets_only):
    njets_tot = len(data["e_px"])
    data = data.loc[(slice(None), 0), :]
    mc = mc.loc[(slice(None), 0), :]
    print("Number of subjets cut = ",
          njets_tot-len(data["e_px"]), " / ",
          len(data["jet_pt"]))

reco_vars = ['e_px', 'e_py', 'e_pz',
             'jet_pt', 'jet_eta', 'jet_phi',
             'jet_dphi', 'jet_qtnorm']

gen_vars = ['gene_px', 'gene_py', 'gene_pz',
            'genjet_pt', 'genjet_eta', 'genjet_phi',
            'genjet_dphi', 'genjet_qtnorm']

# Load the Data
theta_unknown_S = data[reco_vars].to_numpy()
theta0_S = mc[reco_vars].to_numpy()
theta0_G = mc[gen_vars].to_numpy()

weights_MC_sim = mc['wgt']
pass_reco = np.array(mc['pass_reco'])
pass_truth = np.array(mc['pass_truth'])
pass_fiducial = np.array(mc['pass_fiducial'])


#Can fix the number of training steps in model.fit

print("THE LENGTH OF THE ARRAYS IS = ", len(pass_fiducial))

print(f"\n\n SHAPE OF theta_unknown_S {np.shape(theta_unknown_S)} \n\n")
print(f"\n\n SHAPE OF theta0_S {np.shape(theta0_S)} \n\n")

for ivar in range(8):
    print(f"theta0_S Variable {reco_vars[ivar]} = ",
          theta0_S[pass_reco == 1][:5, ivar])
print()
for ivar in range(8):
    print(f"theta0_S Variable (Truth) {gen_vars[ivar]} = ",
          theta0_S[pass_truth == 1][:5, ivar])
print()
for ivar in range(8):
    print(f"theta0_G Variable (Truth) {gen_vars[ivar]} = ",
          theta0_G[pass_truth == 1][:5, ivar])

# Add directly the asymmetry angle to the unfolding.
if add_asymmetry:

    num_observables = 11

    asymm_kinematics = np.asarray(get_kinematics(theta_unknown_S)).T[:, :-1]
    theta_unknown_S = np.append(theta_unknown_S, asymm_kinematics, 1)

    Sasymm_kinematics = np.asarray(get_kinematics(theta0_S)).T[:, :-1]
    theta0_S = np.append(theta0_S, Sasymm_kinematics, 1)

    Gasymm_kinematics = np.asarray(get_kinematics(theta0_G)).T[:, :-1]
    theta0_G = np.append(theta0_G, Gasymm_kinematics, 1)

    print(f"\n\n SHAPE OF theta0_S {np.shape(theta0_S)} \n\n")

    for ivar in range(11):
        print(f"theta0_S Variable {ivar} = ",
              theta0_S[pass_reco == 1][:5, ivar])
    print()

    for ivar in range(11):
        print(f"theta0_S Variable (Truth) {ivar} = ",
              theta0_S[pass_truth == 1][:5, ivar])
    print()

    for ivar in range(11):
        print(f"theta0_G Variable (Truth) {ivar} = ",
              theta0_G[pass_truth == 1][:5, ivar])

del mc
gc.collect()


print("-"*50)
print("L: 184")
print("NaN in Theta0_S =        ", np.nan in theta0_S )
print("NaN in Theta0_G =        ", np.nan in theta0_G )
print("NaN in theta_unknown_S = ", np.nan in theta_unknown_S )

print("INF in Theta0_S =        ", np.inf in theta0_S)
print("INF in Theta0_G =        ", np.inf in theta0_G)
print("INF in theta_unknown_S = ", np.inf in theta_unknown_S)

# DATA CLEANING
# Mask value = MASK_VAL
#FIXME: MAKE SURE KINEMATICS WON'T NATURALLY BE MASK VALUE
theta0_S[:, 0][pass_reco == 0] = MASK_VAL
theta0_G[:, 0][pass_truth == 0] = MASK_VAL

theta0_S[theta0_S == np.inf] = MASK_VAL
theta0_G[theta0_G == np.inf] = MASK_VAL
theta_unknown_S[theta_unknown_S == np.inf] = MASK_VAL

np.nan_to_num(theta0_S, copy=False, nan =MASK_VAL)
np.nan_to_num(theta0_G, copy=False, nan =MASK_VAL)
np.nan_to_num(theta_unknown_S, copy=False, nan =MASK_VAL)

print("-"*50)
print("L: 184")
print("NaN in Theta0_S =        ", np.nan in theta0_S )
print("NaN in Theta0_G =        ", np.nan in theta0_G )
print("NaN in theta_unknown_S = ", np.nan in theta_unknown_S )

print("INF in Theta0_S =        ", np.inf in theta0_S)
print("INF in Theta0_G =        ", np.inf in theta0_G)
print("INF in theta_unknown_S = ", np.inf in theta_unknown_S)
print("="*50)

for p in range(NPasses):

    start = time.time()
    print(f"Unfolding Pass {p}...")

    mfold = MultiFold(num_observables, NIter,
                      theta0_G, theta0_S,
                      theta_unknown_S, n_epochs)

    weights, models, history = mfold.unfold()

    tf.keras.backend.clear_session()

    np.save(f"{save_dir}/{ID}/{ID}_Pass{p}_Step2_Weights.npy", weights[:, 1:2, :])
    np.save(f"{save_dir}/{ID}/{ID}_Pass{p}_Step2_History.npy", weights[:, 1:2, :])

    np.save(f"{save_dir}/{ID}/{ID}_Pass{p}_Step1_Weights.npy", weights[:, 0:1, :])
    np.save(f"{save_dir}/{ID}/{ID}_Pass{p}_Step1_History.npy", weights[:, 0:1, :])

    print(f"Pass {p} took {time.time() - start} seconds \n")

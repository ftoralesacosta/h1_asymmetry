import sys
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import gc
import time
import os

import tensorflow as tf
import tensorflow.keras

from get_np_arrays import get_kinematics

from unfold import multifold

from unfold import MASK_VAL
print("MASK_VAL = ", MASK_VAL)

os.environ['CUDA_VISIBLE_DEVICES'] = "2"

'''
run like python refactor_unfolding.py Rapgap nominal  0
[command] [Rapgap or Django] [run_type: nominal, sys_0...] [BOOTSTRAPPING Seed]
'''
print("Running on MC sample", sys.argv[1], "with setting", sys.argv[2])

if (sys.argv[1] == "Django" and sys.argv[2] == 'closure'):
    sys.exit("closure test must be run with Rapgap as MC")


physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
print("GPUs = ", physical_devices)

# mc = "Rapgap"
# mc = "Django"
mc = sys.argv[1]
LABEL = f"{mc}_SingleWorker_Q2_Cut"
# LABEL = f"{mc}_AddKinematics"
# LABEL = f"{mc}_BootStrap_Nov20"
ID = f"{sys.argv[1]}_{sys.argv[2]}_{LABEL}"
print(f"\n\n ID = {ID} \n\n")

save_dir = "../h1_models"
try:
    os.mkdir(f"{save_dir}/{ID}/")
except OSError as error:
    print(error)


# tf.random.set_seed(int(sys.argv[3]))
np_seed = int(float(sys.argv[3]))
np.random.seed(np_seed)

add_asymmetry = False
leading_jets_only = True
num_observables = 8

NEVENTS = -1
n_epochs = 1000
NIter = 5
NPasses = 5

test = False
if test:
    LABEL = LABEL+"_TEST"
    NEVENTS = -1
    n_epochs = 10
    NIter = 5
    NPasses = 1


# inputs_dir = "/pscratch/sd/f/fernando/h1_data"
inputs_dir = "/clusterfs/ml4hep/yxu2/unfolding_mc_inputs"

if sys.argv[2] == 'closure':
    data = pd.read_pickle(f"{inputs_dir}/Django_nominal.pkl")
    mc = pd.read_pickle(f"{inputs_dir}/{sys.argv[1]}_nominal.pkl")
    #see if remainder is larger than batch_size

else:
    mc = pd.read_pickle(f"{inputs_dir}/{sys.argv[1]}_{sys.argv[2]}.pkl")[:NEVENTS]
    data = pd.read_pickle(f"{inputs_dir}/Data_nominal.pkl")[:NEVENTS]


# Check MC and Data
print(f"MC = {inputs_dir}/{sys.argv[1]}_{sys.argv[2]}.pkl")
print(f"Data = {inputs_dir}/Data_nominal.pkl")
print(np.shape(mc))
print(np.shape(data))


# Cut subleading Jets
if (leading_jets_only):
    njets_tot = len(data["e_px"])
    data = data.loc[(slice(None), 0), :]
    mc = mc.loc[(slice(None), 0), :]
    print("Number of subjets cut = ",
          njets_tot-len(data["e_px"]), " / ",
          len(data["jet_pt"]))

# print("\n\nFirst 5 Events = ", mc.head(5))
# print("Last 5 Events = ", mc.tail(5))

gen_Q2 = mc['gen_Q2'].to_numpy()
gen_underQ2 = gen_Q2 < 100
print("length of Q2 array = ", np.shape(gen_Q2))

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

print("THE LENGTH OF THE ARRAYS IS =", len(pass_fiducial))

print(f"\n\n SHAPE OF theta_unknown_S {np.shape(theta_unknown_S)} \n\n")
print(f"\n\n SHAPE OF theta0_S {np.shape(theta0_S)} \n\n")

# Set Initial Data Weights for BOOTSTRAPPING
dataw = None
if (np_seed != 0):
    dataw = np.random.poisson(1, len(theta_unknown_S))
    print("Doing Bootstrapping")
else:
    print("Not doing bootstrapping")

# Option for unfolding on more vars
for ivar in range(num_observables):
    print(f"theta0_S Variable {reco_vars[ivar]} = ",
          theta0_S[pass_reco == 1][:5, ivar])
print()
for ivar in range(num_observables):
    print(f"Data Variable {reco_vars[ivar]} = ",
          theta_unknown_S[:5, ivar])

for ivar in range(num_observables):
    print(f"theta0_S Variable (Truth) {gen_vars[ivar]} = ",
          theta0_S[pass_truth == 1][:5, ivar])

print()
for ivar in range(num_observables):
    print(f"theta0_G Variable (Truth) {gen_vars[ivar]} = ",
          theta0_G[pass_truth == 1][:5, ivar])

print("NaN in Data = ", np.isnan(theta_unknown_S).any())
print("NaN in Theta0_S (RECO) = ", np.isnan(theta0_S[pass_reco==1]).any())
#print("NaN in Theta0_S(Truth)(NOT USED)= ",\
#np.isnan(theta0_S[pass_truth==1]).any())
print("NaN in Theta0_G (Truth)= ", np.isnan(theta0_G[pass_truth==1]).any())

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

#  sys.exit("\n\nEXITING\n\n")

# DATA CLEANING
# Mask value = -10



#FIXME: MAKE SURE KINEMATICS WON'T NATURALLY BE MASK VALUE
# It's almost impossible for a value to be exactly the MASK_val

theta0_S[:, 0][pass_reco == 0] = MASK_VAL
theta0_G[:, 0][pass_truth == 0] = MASK_VAL

theta0_S[:, 0][gen_underQ2] = MASK_VAL
theta0_G[:, 0][gen_underQ2] = MASK_VAL
# theta_unknown_S[data_underQ2] = MASK_VAL

theta0_S[theta0_S == np.inf] = MASK_VAL
theta0_G[theta0_G == np.inf] = MASK_VAL
theta_unknown_S[theta_unknown_S == np.inf] = MASK_VAL

np.nan_to_num(theta0_S, copy=False, nan =MASK_VAL)
np.nan_to_num(theta0_G, copy=False, nan =MASK_VAL)
np.nan_to_num(theta_unknown_S, copy=False, nan =MASK_VAL)


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

del mc
gc.collect()


for p in range(NPasses):

    start = time.time()
    print(f"Unfolding Pass {p}...")

    weights, models, history = multifold(num_observables, NIter,
                                         theta0_G, theta0_S,
                                         theta_unknown_S, n_epochs,
                                         None, dataw)

    tf.keras.backend.clear_session()

    np.save(f"{save_dir}/{ID}/{ID}_Pass{p}_Step2_Weights.npy", weights[:, 1:2, :])
    np.save(f"{save_dir}/{ID}/{ID}_Pass{p}_Step2_History.npy", weights[:, 1:2, :])

    np.save(f"{save_dir}/{ID}/{ID}_Pass{p}_Step1_Weights.npy", weights[:, 0:1, :])
    np.save(f"{save_dir}/{ID}/{ID}_Pass{p}_Step1_History.npy", weights[:, 0:1, :])

    print(f"Pass {p} took {time.time() - start} seconds \n")

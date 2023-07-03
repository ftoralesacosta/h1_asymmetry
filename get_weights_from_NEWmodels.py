import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
# from tqdm import tqdm
import sys
import os

from get_np_arrays import get_kinematics

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

label = sys.argv[1]
p = int(sys.argv[2])
mc_type = sys.argv[3]

mc = pd.read_pickle(f"/clusterfs/ml4hep/yxu2/unfolding_mc_inputs/{mc_type}_nominal.pkl")
data = pd.read_pickle("/clusterfs/ml4hep/yxu2/unfolding_mc_inputs/Data_nominal.pkl")

# Apply Cuts
cut_subleading_jet = True
if cut_subleading_jet:
    mc = mc.loc[(slice(None), 0), :]
    data = data.loc[(slice(None), 0), :]

pass_fiducial = np.array(mc['pass_fiducial'])
pass_truth = np.array(mc['pass_truth'])

theta0_G = mc[['gene_px','gene_py','gene_pz','genjet_pt','genjet_eta','genjet_phi','genjet_dphi','genjet_qtnorm']].to_numpy()
# Q2 = theta0_G[:,8]
# Q2 = Q2[pass_fiducial==1]
# theta0_G = theta0_G[:,0:8]

theta_unknown_S = data[['e_px','e_py','e_pz','jet_pt','jet_eta','jet_phi','jet_dphi','jet_qtnorm']].to_numpy()

# Directly train on and run inference on Aymmetry Variables
add_asymmetry = True
if add_asymmetry:

    asymm_kinematics = np.asarray(get_kinematics(theta_unknown_S)).T
    theta_unknown_S = np.append(theta_unknown_S, asymm_kinematics, 1)

    Gasymm_kinematics = np.asarray(get_kinematics(theta0_G)).T
    theta0_G = np.append(theta0_G, Gasymm_kinematics, 1)

# Get StandardScalar from raw data, apply to Rapgap in each iteration later
scaler_data = StandardScaler()
scaler_data.fit(theta_unknown_S)
print("\n Length of theta0G =", len(theta0_G), "\n")

# Rapgap_nominal_pass_testingnominalIteration_28Pass4model
n_passes = 5
n_iter = 30
dir = "/global/ml4hep/spss/ftoralesacosta/new_models/"
# base = "Rapgap_nominal_pass_testingnominalIteration_"
# base = "Rapgap_nominal_June10_Iter30_Pass5nominalIteration_"

# label = "June10_Iter30"
# base = f"Rapgap_nominal_{label}_Pass{p}nominalIteration_"  # see +specific, L 15
base = f"{mc_type}_nominal_{label}_Pass{p}nominalIteration_"  # see +specific, L 15
# p = 4
all_weights = []

# for p in range(n_passes):

pass_weights = []
NNweights_step2 = np.ones(len(theta0_G))

for i in range(n_iter):

        # Make sure to reset weights
        NNweights_step2 = np.ones(len(theta0_G))
        NNweights_step2_hold = np.ones(len(theta0_G))

        specific = f"{i}Pass{p}model"
        model_name = dir+base+specific
        print(f"Loading {model_name}")

        mymodel = tf.keras.models.load_model(model_name, compile=False)

        NNweights_step2_hold = mymodel.predict(scaler_data.transform(theta0_G),
                                               batch_size=10000)

        NNweights_step2_hold = NNweights_step2_hold/(1.-NNweights_step2_hold)
        NNweights_step2_hold = NNweights_step2_hold[:, 0]
        NNweights_step2_hold = np.squeeze(np.nan_to_num(NNweights_step2_hold,
                                                        posinf=1))
        NNweights_step2_hold[pass_truth == 0] = 1.
        NNweights_step2 = NNweights_step2_hold*NNweights_step2

        tf.keras.backend.clear_session()

        # if (i > 0):
        # NNweights_step2 *= np.asarray(pass_weights[i-1])

        pass_weights.append(NNweights_step2)
        # End Iteration Loop

pass_weights = np.asarray(pass_weights)
np.save(dir+label+f"_{mc_type}_pass{p}_weights.npy", pass_weights)

# all_weights.append(pass_weights)
# End Pass Loop

# all_weights = np.array(all_weights)
# np.save(dir+label+"all_rapgap_weights.npy", all_weights)

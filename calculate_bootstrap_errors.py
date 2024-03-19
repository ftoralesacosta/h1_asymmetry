import numpy as np
import matplotlib.pyplot as plt
import sys
import yaml

import pickle

sys.path.append('../')
from process_functions import *

from matplotlib import style
#style.use('/global/home/users/ftoralesacosta/dotfiles/scientific.mplstyle')
colors = ['#348ABD','#C70039','#FF5733','#FFC300','#65E88F','#40E0D0']



# config_name = "pscratch/sd/f/fernando/h1_models/Rapgap_nominal_Perlmutter_Bootstrap/perlmutter_config.yaml"
if len(sys.argv) <= 1:
    CONFIG_FILE = "./configs/config.yaml"
else:
    CONFIG_FILE = sys.argv[1]

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

q_max = 10.0
q_perp_bins = np.asarray(config['q_bins'])
N_Bins = len(q_perp_bins)-1
keys=["q_perp","phi","cos1","cos2","cos3"]
print(q_perp_bins)

N_Events = -1

  # Load npy Files
cuts_h1rpgp       = np.load(f'{main_dir}/npy_files/{ID}_cuts.npy')[:N_Events]
jet_pT_h1rpgp     = np.load(f'{main_dir}/npy_files/{ID}_jet_pT.npy')[:N_Events][cuts_h1rpgp]
q_perp_h1rpgp     = np.load(f'{main_dir}/npy_files/{ID}_q_perp.npy')[:N_Events][cuts_h1rpgp]
asymm_phi_h1rpgp  = np.load(f'{main_dir}/npy_files/{ID}_asymm_angle.npy')[:N_Events][cuts_h1rpgp]
weights_h1rpgp    = np.load(f'{main_dir}/npy_files/{ID}_weights.npy')[:N_Events][cuts_h1rpgp] #this is already nn step2 * mc
mc_weights_h1rpgp = np.load(f"{main_dir}/npy_files/{ID}_mc_weights.npy")[:N_Events][cuts_h1rpgp]
#nn_weights_h1rpgp = np.load(f"{processed_dir}/npy_files/{ID}_nn_weights.npy")

# boot_ensemble = np.load("./weights/Perlmutter_Bootstrap_bootstrap_weights.npy")
print(f"Loading ./weights/{LABEL}_bootstrap_weights.npy")
boot_ensemble = np.load(f"./weights/{LABEL}_bootstrap_weights.npy")

skips = []
bootstrap_errors = get_bootstrap_errors(boot_ensemble,q_perp_h1rpgp,
                                        q_perp_bins,asymm_phi_h1rpgp,
                                        cuts_h1rpgp,skips,title="")


print(bootstrap_errors["cos1"])
print(np.shape(bootstrap_errors))

file = open('bootstrap_errors.pkl', 'wb')
pickle.dump(bootstrap_errors, file,protocol=pickle.HIGHEST_PROTOCOL)
file.close()

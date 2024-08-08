import numpy as np
import matplotlib.pyplot as plt
import sys
import yaml

import pickle
from icecream import ic
from tqdm import tqdm

sys.path.append('../')
# from process_functions import *

from matplotlib import style
#style.use('/global/home/users/ftoralesacosta/dotfiles/scientific.mplstyle')
colors = ['#348ABD','#C70039','#FF5733','#FFC300','#65E88F','#40E0D0']


def get_bootstrap_errors(boot_ensemble, phi_bins, asymm_phi,
                         q_perp, cuts, q_range='low',skips=[], title=""):

    N_Bootstraps = np.shape(boot_ensemble)[0]
    print(f"N_Bootstraps = {N_Bootstraps}")
    N_Bins = len(phi_bins)-1

    q_avg = np.zeros((N_Bootstraps, N_Bins))
    phi_avg = np.zeros((N_Bootstraps, N_Bins))
    # jetpT_avg = np.zeros((N_Bootstraps,N_Bins))

    # Just Used for Plotting:
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    phi_centers = (phi_bins[:-1]+phi_bins[1:])/2.0
    phi_width = (phi_bins[1]+phi_bins[0])/2
    axes = np.ravel(axes)


    init_weights = boot_ensemble[0]
    n_samples = min(len(init_weights), len(cuts))
    asymm_phi = asymm_phi[:n_samples]
    cuts = cuts[:n_samples]
    q_perp = q_perp[:n_samples]

    if q_range == 'high':
        cuts = np.logical_and(cuts, q_perp > 2.0)

    else:
        cuts = np.logical_and(cuts, q_perp <= 2.0)

    asymm_phi = asymm_phi[cuts]

    for istrap in tqdm(range(N_Bootstraps)):

        weights = boot_ensemble[istrap]
        weights = weights[:n_samples][cuts]
        # This is what fundamentally changes per iteration

        if skips:
            if istrap in skips:
                print("Skipping", istrap)
                continue

        phi_w = np.cos(asymm_phi)*weights
        # jetpT_w = jetpT*weights

        phi_avg[istrap], _ = np.histogram(asymm_phi,bins=phi_bins,
                                           weights=weights,density=True)

        axes[0].errorbar(phi_centers,phi_avg[istrap],xerr=phi_width,alpha=0.2)

        axes[0].set_ylabel("$\phi$")

        axes[0].set_xlabel("$q_\perp$")

    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"./plots/PHI_q{q_range}_Bootstrap_Ensemble.pdf")

    phi_errors = np.zeros(N_Bins)

    for ibin in range(N_Bins):
        #std and avg. over ITERATIONS, per bin
        phi_errors[ibin] = np.nanstd(phi_avg[:,ibin])/np.nanmean(phi_avg[:,ibin])
        # print(f"Cos1 = {np.nanstd(cos1_avg[:,ibin])} / {np.nanmean(cos1_avg[:,ibin])} = {cos1_errors[ibin]}")

        phi_errors[ibin] = np.nanstd(phi_avg[:, ibin])
        # This means bootstrap errors are saved as ABSOLUTE errors

    bootstrap_errors = {}
    bootstrap_errors["phi"] = phi_errors

    return bootstrap_errors



if len(sys.argv) <= 1:
    CONFIG_FILE = "./configs/perlmutter_nominal.yaml"
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
print(q_perp_bins)

N_Events = -1

n_phi_bins = config['n_phi_bins']
phi_bins = np.linspace(0, 3.14159, n_phi_bins+1)
print(phi_bins)

  # Load npy Files
cuts_h1rpgp       = np.load(f'{main_dir}/npy_files/{ID}_cuts.npy')[:N_Events]
jet_pT_h1rpgp     = np.load(f'{main_dir}/npy_files/{ID}_jet_pT.npy')[:N_Events]
q_perp_h1rpgp     = np.load(f'{main_dir}/npy_files/{ID}_q_perp.npy')[:N_Events]
asymm_phi_h1rpgp  = np.load(f'{main_dir}/npy_files/{ID}_asymm_angle.npy')[:N_Events]
weights_h1rpgp    = np.load(f'{main_dir}/npy_files/{ID}_weights.npy')[:N_Events] 
mc_weights_h1rpgp = np.load(f"{main_dir}/npy_files/{ID}_mc_weights.npy")[:N_Events]
#nn_weights_h1rpgp = np.load(f"{processed_dir}/npy_files/{ID}_nn_weights.npy")

# boot_ensemble = np.load("./weights/Perlmutter_Bootstrap_bootstrap_weights.npy")
# print(f"Loading ./weights/{LABEL}_bootstrap_weights.npy")
# boot_ensemble = np.load(f"./weights/{LABEL}_bootstrap_weights.npy")

print(f"Loading ./weights/{LABEL}_bootstrap_weights.npy")
boot_ensemble = np.load(f"./weights/Perlmutter_March11_bootstrap_weights.npy")

skips = []
q_range = 'high'
bootstrap_errors = get_bootstrap_errors(boot_ensemble, phi_bins, asymm_phi_h1rpgp,
                                        q_perp_h1rpgp, cuts_h1rpgp, q_range, skips, title="")



print(bootstrap_errors["phi"])
print(np.shape(bootstrap_errors))

file = open(f'PHI_q{q_range}_bootstrap_errors.pkl', 'wb')
pickle.dump(bootstrap_errors, file,protocol=pickle.HIGHEST_PROTOCOL)
file.close()

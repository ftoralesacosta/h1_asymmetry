import sys
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import yaml
sys.path.append("..")
from get_np_arrays import *

CONFIG_FILE = "/clusterfs/ml4hep_nvme2/ftoralesacosta/h1_asymmetry/config.yaml"
config = yaml.safe_load(open(CONFIG_FILE))
print(f"\nLoaded {CONFIG_FILE}\n")

mc = config['mc']  # Rapgap, Django, Pythia
run_type = config['run_type']  # nominal, bootstrap, systematic
processed_dir = config['main_dir']


# processed_dir = "/clusterfs/ml4hep_nvme2/ftoralesacosta/h1_asymmetry"
# mc = "Rapgap"
# type = "nominal"
LABEL = f"{mc}_{run_type}_{mc}_SingleWorker_Q2_Cut"


cuts_h1rpgp       = np.load(f'{processed_dir}/npy_files/{LABEL}_cuts.npy')
jet_pT_h1rpgp     = np.load(f'{processed_dir}/npy_files/{LABEL}_jet_pT.npy')
cuts_h1rpgp = np.ones(len(jet_pT_h1rpgp), dtype=bool)
jet_pT_h1rpgp     = np.load(f'{processed_dir}/npy_files/{LABEL}_jet_pT.npy')[cuts_h1rpgp]
q_perp_h1rpgp     = np.load(f'{processed_dir}/npy_files/{LABEL}_q_perp.npy')[cuts_h1rpgp]
asymm_phi_h1rpgp  = np.load(f'{processed_dir}/npy_files/{LABEL}_asymm_angle.npy')[cuts_h1rpgp]
weights_h1rpgp    = np.load(f'{processed_dir}/npy_files/{LABEL}_weights.npy')
mc_weights_h1rpgp = np.load(f"{processed_dir}/npy_files/{LABEL}_mc_weights.npy")
nn_weights_h1rpgp = np.load(f"{processed_dir}/npy_files/{LABEL}_nn_weights.npy")

cuts_h1rpgp = np.load(f'{processed_dir}/npy_files/{LABEL}_cuts.npy')

plt.rcParams.update({'font.size': 12})
fig = plt.figure(figsize=(10, 10)) 
gs = gridspec.GridSpec(2, 1, height_ratios=[2,1]) 
ax0 = plt.subplot(gs[0])
ax0.yaxis.set_ticks_position('both')
ax0.xaxis.set_ticks_position('both')
ax0.tick_params(direction="in",which="both")
ax0.minorticks_on()
plt.xticks(fontsize=0)
plt.yticks(fontsize=20)
#bin_for_angle = np.linspace(0,3,18)
bin_for_angle = np.linspace(-1,1,51)

cosphi = np.cos(asymm_phi_h1rpgp)


a_sim,b_sim,c=plt.hist(cosphi[cuts_h1rpgp], bins = bin_for_angle,
                       weights=mc_weights_h1rpgp[cuts_h1rpgp],
                       density=True,histtype="step",color="red",ls=":",label="Rapgap Gen.")

a_sim,b_sim,c=plt.hist(cosphi[cuts_h1rpgp], bins = bin_for_angle,
                       weights=weights_h1rpgp,
                       density=True,histtype="step",color="blue",ls=":",label="Rapgap Unfolded")

plt.ylabel("Normalized to unity",fontsize=20)
plt.title("Asymmetry Distribution ",loc="left",fontsize=20)
plt.text(0.65, 0.95,'H1 internal', horizontalalignment='center', verticalalignment='center',
         transform = ax0.transAxes, fontsize=25, fontweight='bold')
plt.legend(frameon=False,fontsize=15,loc='upper left')
plt.locator_params(axis='x', nbins=5)
#plt.yscale("log")
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)

#plt.xlabel(r"$cos(\phi) = q_T\cdot e_{p_T}/(|q_T|\cdot|e_{p_T}|)$",fontsize=15)
plt.xlabel(r'cos($\phi) = (\vec{q}_\perp \cdot \vec{P}_\perp)\ /\ |\vec{q}_\perp| |\vec{P}_\perp|$',fontsize=20)

plt.savefig("Rapgap_Angle_Distribution.pdf")
plt.tight_layout()
plt.show()

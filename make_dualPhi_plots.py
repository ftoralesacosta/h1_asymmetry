'''This code is based off of 'make_gifs.py' '''
import sys
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import yaml
from get_np_arrays import *
from process_functions import *

colors = ['#348ABD', '#C70039', '#FF5733', '#FFC300', '#65E88F', '#40E0D0']


# CONFIG FILE
if len(sys.argv) > 1:
    CONFIG_FILE = sys.argv[1]
else:
    CONFIG_FILE = "./configs/perlmutter_nominal.yaml"

config = yaml.safe_load(open(CONFIG_FILE))
print(f"\nLoaded {CONFIG_FILE}\n")

mc = config['mc']  # Rapgap, Django, Pythia
run_type = config['run_type']  # nominal, bootstrap, systematic
processed_dir = config['main_dir']
NIter = config['n_iterations']
q_perp_bins = [0.0, 2.0, 8.0]

LABEL = config['identifier']
ID = f"{mc}_{run_type}_{LABEL}"


n_phi_bins = config['n_phi_bins']
phi_bins = np.linspace(0, 3.14159, n_phi_bins+1)
edges=np.stack((phi_bins[:-1],phi_bins[1:])).T

# Load npy Files
cuts_h1rpgp       = np.load(f'{processed_dir}/npy_files/{ID}_cuts.npy')
jet_pT_h1rpgp     = np.load(f'{processed_dir}/npy_files/{ID}_jet_pT.npy')[cuts_h1rpgp]
q_perp_h1rpgp     = np.load(f'{processed_dir}/npy_files/{ID}_q_perp.npy')[cuts_h1rpgp]
asymm_phi_h1rpgp  = np.load(f'{processed_dir}/npy_files/{ID}_asymm_angle.npy')[cuts_h1rpgp]
weights_h1rpgp    = np.load(f'{processed_dir}/npy_files/{ID}_weights.npy')[cuts_h1rpgp]
mc_weights_h1rpgp = np.load(f"{processed_dir}/npy_files/{ID}_mc_weights.npy")[cuts_h1rpgp]
nn_weights_h1rpgp = np.load(f"{processed_dir}/npy_files/{ID}_nn_weights.npy")

print("Printing LENGTHS")
print("Cut Len", len(cuts_h1rpgp))
print("NN len", len(nn_weights_h1rpgp))
print("Applied Cuts", len(asymm_phi_h1rpgp))

# Load Django
if len(sys.argv) > 2:
    Django_CONFIG_FILE = sys.argv[2]
else:
    Django_CONFIG_FILE = "./configs/perlmutter_Django.yaml"

Django_config = yaml.safe_load(open(Django_CONFIG_FILE))
print(f"\nLoaded {Django_CONFIG_FILE}\n")

mc = Django_config['mc']  # Rapgap, Django, Pythia
run_type = Django_config['run_type']  # nominal, bootstrap, systematic
processed_dir = Django_config['main_dir']
NIter = Django_config['n_iterations']
q_perp_bins = Django_config['q_bins']
q_perp_bins = [0.0, 2.0, 8.0]

LABEL = Django_config['identifier']
DjID = f"{mc}_{run_type}_{LABEL}"
# DjID = "Django_nominal_Perlmutter_django"
cuts_h1djgo       = np.load(f'{processed_dir}/npy_files/{DjID}_cuts.npy')
jet_pT_h1djgo     = np.load(f'{processed_dir}/npy_files/{DjID}_jet_pT.npy')[cuts_h1djgo]
q_perp_h1djgo     = np.load(f'{processed_dir}/npy_files/{DjID}_q_perp.npy')[cuts_h1djgo]
asymm_phi_h1djgo  = np.load(f'{processed_dir}/npy_files/{DjID}_asymm_angle.npy')[cuts_h1djgo]
weights_h1djgo    = np.load(f'{processed_dir}/npy_files/{DjID}_weights.npy')[cuts_h1djgo]
mc_weights_h1djgo = np.load(f"{processed_dir}/npy_files/{DjID}_mc_weights.npy")[cuts_h1djgo]
nn_weights_h1djgo = np.load(f"{processed_dir}/npy_files/{DjID}_nn_weights.npy")


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


plt.hist(cosphi, bins = bin_for_angle,
                       weights=mc_weights_h1rpgp,
                       density=True,histtype="step",
                       color="red",ls=":",label="Rapgap Gen.")

plt.hist(cosphi, bins = bin_for_angle,
                       weights=weights_h1rpgp,
                       density=True,histtype="step",
                       color="blue",ls=":",label="Rapgap Unfolded")

plt.ylabel("Normalized to unity",fontsize=20)
plt.title("Asymmetry Distribution ",loc="left",fontsize=20)
plt.text(0.65, 0.95,'H1 internal',
         horizontalalignment='center', verticalalignment='center',
         transform = ax0.transAxes, fontsize=25, fontweight='bold')

plt.legend(frameon=False,fontsize=15,loc='upper left')
plt.locator_params(axis='x', nbins=5)
#plt.yscale("log")
plt.yticks(fontsize=20)
plt.xticks(fontsize=20)

#plt.xlabel(r"$cos(\phi) = q_T\cdot e_{p_T}/(|q_T|\cdot|e_{p_T}|)$",fontsize=15)
plt.xlabel(r'cos($\phi) = (\vec{q}_\perp \cdot \vec{P}_\perp)\ /\ |\vec{q}_\perp| |\vec{P}_\perp|$',fontsize=20)

plt.savefig(f"plots/{ID}_Angle_Distribution.pdf")
plt.tight_layout()
plt.show()


Rapgap_Iterations = {}
Django_Iterations = {}

pass_avg_weights = np.load(f"./weights/{ID}_pass_avgs.npy")
django_pass_avg_weights = np.load(f"./weights/{DjID}_pass_avgs.npy")

size_init = min(len(pass_avg_weights[0]), len(cuts_h1rpgp))
cuts_h1rpgp = cuts_h1rpgp[:size_init]

size_init = min(len(django_pass_avg_weights[0]), len(cuts_h1djgo))
cuts_h1djgo = cuts_h1djgo[:size_init]
# q_perp_h1rpgp = q_perp_h1rpgp[:size_init]
# asymm_phi_h1rpgp = asymm_phi_h1rpgp[:size_init]
# mc_weights_h1rpgp = mc_weights_h1rpgp[:size_init]


print("NN weight  SHAPE = ", np.shape(pass_avg_weights))
print("q_perp     SHAPE = ", np.shape(q_perp_h1rpgp))
print("asymm_phi  SHAPE = ", np.shape(asymm_phi_h1rpgp))
print("mc weights SHAPE = ", np.shape(mc_weights_h1rpgp))
print("cuts       SHAPE = ", np.shape(cuts_h1rpgp))
print()

bootstrap_errors = []
sys_errors = []
for q_range in ['low','high']:

    stat_file = open(f'PHI_q{q_range}_bootstrap_errors.pkl', 'rb')
    sys_file = open(f'./pkls/PHI_{q_range}_uncertainties.pkl', 'rb')

    bootstrap_errors.append(pickle.load(stat_file)['phi'])
    sys_errors.append(pickle.load(sys_file)['total']['phi'])

    sys_file.close()
    stat_file.close()

print(sys_errors)

for i in range(NIter):
    phi_bins = np.linspace(0,3.1416,9)
    phi_bins = np.linspace(0,3.1416,13)
    h1_rpgp_phi = {}
    h1_djgo_phi = {}
    rpgp_phi = {}
    djgo_phi = {}

    uf_weights = pass_avg_weights[i]
    print("Weight check = ", np.mean(uf_weights))

    phi_inside_qperp(h1_rpgp_phi, q_perp_bins, phi_bins,
                     q_perp_h1rpgp , asymm_phi_h1rpgp,
                     pass_avg_weights[i][cuts_h1rpgp])

    phi_inside_qperp(rpgp_phi, q_perp_bins, phi_bins, 
                     q_perp_h1rpgp ,asymm_phi_h1rpgp, 
                     mc_weights_h1rpgp)

    phi_inside_qperp(h1_djgo_phi, q_perp_bins, phi_bins,
                     q_perp_h1djgo ,asymm_phi_h1djgo,
                     django_pass_avg_weights[i][cuts_h1djgo])

    phi_inside_qperp(djgo_phi, q_perp_bins, phi_bins,
                     q_perp_h1djgo ,asymm_phi_h1djgo,
                     mc_weights_h1djgo)


    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(9, 4), constrained_layout=True)
    axes = axes.ravel()


    #you have iterate through the q_perp ranges with ix.range
    # you have a low and high pkl for systematics and statistics. Make a np array where the
    #first index is ix, and then you can simply plot the errors!


    for ix, ax in enumerate(axes):
        ax.errorbar(h1_rpgp_phi["bin_centers"], h1_rpgp_phi[str(ix)], yerr=bootstrap_errors[ix],
                color='black', label="H1 Data", ls='none', ms=4, marker='o')
                # color=colors[4], label="H1 OmniFold [Rapgap]")

        #FIXME work on this next
        for x in range(len(edges)):
            ax.fill_between(edges[x],h1_rpgp_phi[str(ix)][x]-sys_errors[ix][x],
                            h1_rpgp_phi[str(ix)][x]+sys_errors[ix][x], alpha=0.2,color='k',)

        # ax.plot(rpgp_phi["bin_centers"], rpgp_phi[str(ix)],
        #         color=colors[2], linestyle="--", label="Rapgap")

        offset = 0.1
        ax.plot(rpgp_phi['bin_centers'], rpgp_phi[str(ix)], label="RAPGAP",ls='none',marker="P",
                         fillstyle='none',markeredgewidth=2,ms=10,alpha=0.9, color=colors[2])
        offset = -0.1
        ax.plot(djgo_phi["bin_centers"]+offset, djgo_phi[str(ix)],label="DJANGOH",ls='none',marker="D",
                         fillstyle='none',markeredgewidth=2,ms=7,alpha=0.8, color="darkorange")

        # ax.plot(h1_djgo_phi["bin_centers"], h1_djgo_phi[str(ix)],
        #         color=colors[5], label="H1 OmniFold [Django]")

        # ax.plot(djgo_phi["bin_centers"], djgo_phi[str(ix)],
        #         color="darkorange", linestyle="--", label="Django")

        if (ix == 0):
            ax.legend(fontsize=12)
        if (ix % 4 == 0):
            ax.set_ylabel("Normalized Counts")

        ax.set_xlabel(r"$\phi$ [rad]")
        ax.set_ylim(0, 0.6)

        ax.set_title(r"$%i < q_\perp < %i$ [GeV]" % (q_perp_bins[ix],
                                                     q_perp_bins[ix+1]),
                     fontsize=20)

    # plt.suptitle(fr"$\phi$ Distributions [Iteration {i}]", fontsize=25)
    plt.savefig(f"./plots/{ID}_Phi_Distributions_Iteration_{i}.png")

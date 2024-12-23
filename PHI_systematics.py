import sys
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt

from get_np_arrays import npy_from_pkl
from process_functions import averages_in_qperp_bins
from matplotlib.ticker import AutoMinorLocator
from matplotlib import style

from scipy import interpolate
from scipy import ndimage

from icecream import ic
ic.configureOutput(includeContext=True)

# style.use('/global/home/users/ftoralesacosta/dotfiles/scientific.mplstyle')
style.use('~/dotfiles/scientific.mplstyle')


def process_sys_npy(keys, suffix, mc_name='Rapgap'):
    # for key in range(len(keys)):
    for key in keys:

        if 'sys' not in key:
            continue

        mc_label = f"{mc_name}_{key}_{suffix}"

        print("on sample:", mc_name, key, "\n")
        npy_from_pkl(mc_label, run_type=key)


def process_QED_npy(qed_keys):
    # QED_path = "/clusterfs/ml4hep_nvme2/ftoralesacosta/h1_check/h1_lepton_jet_asymmetry/"
    QED_path = "/pscratch/sd/f/fernando/h1_data/"
    for QED_label in qed_keys:
        QED_file = QED_path + QED_label + ".pkl"
        npy_from_pkl(QED_label, load_NN=False, pkl_path=QED_file)
        # Unfolding NOT applied in QED systematic. Note 'False'


def get_sys_dictionaries(keys, suffix, do_sys_calc=False, phi_bins=None,
                         q_range='low',mc='Rapgap', npy_folder='./npy_files',
                         pkl_dir='./pkls'):

    sys_file = 'sys_variations'

    if not do_sys_calc:
        with open(f'{pkl_dir}/PHI_{sys_file}.pkl', 'rb') as pkl_file:
            return pickle.load(pkl_file)

    #check phi_binning
    if phi_bins is None:
        phi_bins = np.linspace(0, 3.14159, 13)

    q_min=0.0
    q_max=2.0
    if q_range=='high':
        q_min=2.0
        q_max=8.0

    # Creates nested dicitonary for systematics

    sys_dictionaries = {}

    for key in keys:
        if key == "QED": continue

        LABEL = f"{mc}_{key}_{suffix}"
        if key == 'model':
            # LABEL = f"Django_nominal_{suffix}"
            LABEL = "Django_nominal_Perlmutter_django_March13"
            LABEL = "Django_nominal_Perlmutter_Interactive"

        if LABEL == "Rapgap_nominal_perlmutter_March13":
            LABEL = "Rapgap_nominal_Perlmutter_March13"
        print(f"Grabbing {LABEL}")

        cuts       = np.load(f"{npy_folder}/{LABEL}_cuts.npy")
        q_perp     = np.load(f"{npy_folder}/{LABEL}_q_perp.npy")
        cut_on_q = np.logical_and(q_perp > q_min, q_perp <= q_max)
        cuts = np.logical_and(cut_on_q, cuts)
        weights    = np.load(f"./weights/{LABEL}_pass_avgs.npy")[-1]
        d_len = min( len(cuts), len(weights) )

        cuts       = cuts[:d_len]
        weights    = weights[:d_len][cuts]
        q_perp     = np.load(f"{npy_folder}/{LABEL}_q_perp.npy")[:d_len][cuts]
        asymm_phi  = np.load(f"{npy_folder}/{LABEL}_asymm_angle.npy")[:d_len][cuts]

        inner_dict = {}
        inner_dict['phi'],_ = np.histogram(asymm_phi,bins=phi_bins,
                                           weights=weights,density=True)
        inner_dict['bin_centers'] = (phi_bins[:-1] + phi_bins[1:])/2
        sys_dictionaries[key] = inner_dict

    print()
    print(f"Outer {sys_dictionaries.keys()}")
    print(f"Inner {sys_dictionaries[keys[0]].keys()}")

    pickle_dict(sys_dictionaries, f'PHI_{q_range}_{sys_file}.pkl', folder=pkl_dir)
    return sys_dictionaries


def get_QED_dict(qed_keys, do_qed_calc=True, phi_bins=None,
                 q_range='low', npy_folder='./npy_files', pkl_dir='./pkls'):
    # yields difference of Rapgap and Django w and w/o Rad

    qed_file = 'qed_variations'

    if not do_qed_calc:
        with open(f'{pkl_dir}/PHI_{qed_file}.pkl', 'rb') as pkl_file:
            return pickle.load(pkl_file)

    #check phi_binning
    if phi_bins is None:
        phi_bins = np.linspace(0, 3.14159, 13)

    q_min=0.0
    q_max=2.0
    if q_range=='high':
        q_min=2.0
        q_max=8.0

    qed_dict = {}

    for key in qed_keys:
        LABEL = key
        inner_dict = {}
        cuts       = np.load(f"{npy_folder}/{LABEL}_cuts.npy")
        q_perp     = np.load(f"{npy_folder}/{LABEL}_q_perp.npy")
        cut_on_q = np.logical_and(q_perp > q_min, q_perp <= q_max)
        cuts = np.logical_and(cut_on_q, cuts)

        q_perp     = q_perp[cuts]
        asymm_phi  = np.load(f"{npy_folder}/{LABEL}_asymm_angle.npy")[cuts]
        weights    = np.load(f"{npy_folder}/{LABEL}_weights.npy")[cuts]
        inner_dict = {}
        inner_dict['phi'],_ = np.histogram(asymm_phi,bins=phi_bins,
                                           weights=weights, density=True)
        inner_dict['bin_centers'] = (phi_bins[:-1] + phi_bins[1:])/2
        qed_dict[key] = inner_dict

    pickle_dict(qed_dict, f'PHI_{q_range}_{qed_file}.pkl', folder=pkl_dir)
    return qed_dict


def get_QED_sys(qed_dict):
    qed_sys = {}

    for inner_key in qed_dict['Rapgap_nominal_Rady'].keys():

        rapgap_difference = qed_dict['Rapgap_nominal_Rady'][inner_key] - \
                            qed_dict['Rapgap_nominal_noRady'][inner_key]
        django_difference = qed_dict['Django_nominal_Rady'][inner_key] - \
                            qed_dict['Django_nominal_noRady'][inner_key]

        smooth_qed = True

        if smooth_qed:
            diff = np.abs(rapgap_difference-django_difference)
            sigma = 1.0
            q_g1d = ndimage.gaussian_filter1d(q_perp_bins, sigma)
            diff_g1d = ndimage.gaussian_filter1d(diff, sigma)
            print("\n\n LINE 130 \n\n")
            print(len(q_g1d))
            print(len(diff_g1d))
            qed_sys[inner_key] = diff_g1d

        else:
            qed_sys[inner_key] = np.abs(rapgap_difference-django_difference)

    return qed_sys


def plot_QED(qed_dict, qed_sys, savedir = './plots/systematics'):
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.ravel()

    for i in range(3):
        key = "cos%i" % (i+1)
        key = "phi"
        string = r"$\langle\cos(%i\phi)\rangle$" % (i+1)
        string = string.replace("1", "")

        for mc in ["Rapgap", "Django"]:
            bin_centers = qed_dict[f"{mc}_nominal_Rady"]["bin_centers"]

            axes[i].plot(bin_centers, qed_dict[f"{mc}_nominal_Rady"][key],
                         label=f"{mc} QED Rad.")

            axes[i].plot(bin_centers, qed_dict[f"{mc}_nominal_noRady"][key],
                         linestyle='--', label=f"{mc} No QED Rad.")

            axes[i+3].plot(bin_centers, qed_dict[f"{mc}_nominal_Rady"][key] - \
                           qed_dict[f"{mc}_nominal_noRady"][key],
                           label=f"{mc} QED difference")

        axes[i+3].plot(bin_centers, qed_sys[key],color="blue",label="Rapgap-Django QED difference")

        # Plotsmanship
        # axes[i].set_ylim(-0.4, 0.5)
        axes[i].set_ylabel(string, fontsize=28)

        # axes[i+3].set_ylim(-0.1,0.05)
        axes[i+3].set_ylabel("counts",fontsize=28)
        axes[i+3].set_xlabel("$\phi_\perp$ [rad.]",fontsize=28)

    axes[2].legend(fontsize=18)
    axes[5].legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{savedir}/PHI_QED_plot.pdf")

    return


# def QED_systematic(qed_dict):

#     qed_sys = get_QED_sys(qed_dict)
#     plot_QED(qed_dict, qed_sys)

#     return qed_sys


def pickle_dict(dic, save_name, folder='.'):
    file = open(f'{folder}/{save_name}', 'wb')
    pickle.dump(dic, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()


def get_uncertainties(sys_keys, nested_sys_dict, qed_dict):

    sys_uncertainties = {}
    print(nested_sys_dict.keys())
    nominal = nested_sys_dict['nominal']

    for syst in sys_keys:
        # if syst == 'nominal': continue
        if syst == "QED":
            sys_uncertainties[syst] = get_QED_sys(qed_dict)
            plot_QED(qed_dict, sys_uncertainties[syst])
            continue
            # also plots QED

        inner_dict = {}
        for inner_key in nominal.keys():

            if inner_key == 'bin_centers': 
                inner_dict[inner_key] = nominal[inner_key]
                continue

            print("Dictionaries in get_uncertanties function")
            ic(inner_key)
            # ic(nominal[inner_key])
            # ic(nested_sys_dict[syst][inner_key])
            inner_dict[inner_key] = np.abs(nominal[inner_key] -
                                            nested_sys_dict[syst][inner_key])
            ic(inner_dict)
        sys_uncertainties[syst] = inner_dict

    return sys_uncertainties


def sum_in_quadruture(sys_dict):
    # sum uncertainties in quadrature

    keys = sys_dict.keys()
    kinematics = sys_dict['nominal'].keys()
    total_sys = {k: 0 for k in kinematics}  # init to 0

    for kinematic in kinematics:
        for key in keys:
            total_sys[kinematic] += sys_dict[key][kinematic]**2

        total_sys[kinematic] = np.sqrt(total_sys[kinematic])

    sys_dict['total'] = total_sys

    return


def plot_variations(sys_dict, color_dict, plot_labels):
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    for i in range(3):
        kin = "phi"
        for syst in sys_dict.keys():
            if syst == "QED": continue

            axes[i].errorbar(sys_dict[syst]["bin_centers"], sys_dict[syst][kin],
                             label=plot_labels[syst], color=color_dict[syst], linewidth=3)

            string = r"$\cos(%i\phi)$" % (i+1)
            string = string.replace("1", "")
            axes[i].set_ylabel("counts", fontsize=28)
            axes[i].set_xlabel("$\phi_\perp$ [rad.]", fontsize=28)
            # axes[i].set_ylim(-0.25,0.75)
            axes[0].legend(fontsize=15)


    plt.tight_layout()
    plt.savefig("./plots/systematics/PHI_systematics_unfolded_separately.pdf")

    return


def plot_uncertanties(uncertainties, color_dict, plot_labels):

    # keys = ["q_perp", "cos1", "cos2", "cos3"]
    fig,axes = plt.subplots(1,3,figsize=(22,7))

    plot_model = True
    plot_bootstrap = False
    plot_original_diff = False

    plot_sys0 = True
    plot_sys1 = True
    plot_sys5 = True
    plot_sys7 = True
    plot_sys11 = True
    plot_sysQED = True

    # axes[0].set_ylim(-0.01, 0.15)
    # axes[1].set_ylim(-0.01, 0.15)
    # axes[2].set_ylim(-0.01, 0.15)

    for i in range(3):
        kin="phi"

        for syst in uncertainties.keys():

            if syst == 'nominal': continue
            axes[i].errorbar(uncertainties['nominal']["bin_centers"],
                             uncertainties[syst][kin],
                             label=plot_labels[syst],
                             color=color_dict[syst], linewidth=3)

        # if (plot_bootstrap):
        #     axes[i].errorbar(nominal["q_perp"], np.abs(abs_bootstrap_errors[key]),
        #             label="Stat. Uncertainty", color="grey", linestyle="-", linewidth=3)

            # axes[0].set_ylim(-0.01,1.5)
            # axes[1].set_ylim(-0.01,1.5)
            # axes[2].set_ylim(-0.01,1.5)

        string = r"$\cos(%i\phi)$ Error [abs.]" % (i+1)
        string = string.replace("1", "")
        axes[i].set_ylabel('counts')
        axes[i].set_xlabel("$\phi_\perp [rad.]$")

    axes[0].text(0.04, 0.92, "H1", transform=axes[0].transAxes, fontsize=22) 

    plt.tight_layout()
    plt.legend(fontsize=18)
    plt.savefig("./plots/systematics/PHI_absError_systematics_unfolded_separately.pdf")

    return


def plot_systematics(sys_dict, uncertainties, color_dict, plot_labels):
    ''' Plots distributions and differences in a subplot'''

    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(22, 12), constrained_layout=True,
                         gridspec_kw={'height_ratios': [4, 3]})

    axes = np.ravel(axes)

    for i in range(3):
        kin = "phi"

        for syst in sys_dict.keys():

            if syst == "QED": continue

            ic(sys_dict[syst])
            if syst=='nominal':
                axes[i].errorbar(sys_dict[syst]["bin_centers"], sys_dict[syst][kin],
                                label=plot_labels[syst], color=color_dict[syst], linewidth=2, linestyle='--')
            else:
                axes[i].errorbar(sys_dict[syst]["bin_centers"], sys_dict[syst][kin],
                                 label=plot_labels[syst], color=color_dict[syst], linewidth=2, linestyle='-')

        string = r"$\langle\ \cos(%i\phi)\ \rangle$"%(i+1)
        string = string.replace("1", "")
        axes[i].set_title(string)
        axes[i].set_ylabel("Counts")
        # axes[i].set_ylim(-0.2, 0.45)
        axes[i].text(0.04, 0.92, "H1", transform=axes[i].transAxes, fontsize=24)
        axes[i].yaxis.set_minor_locator(AutoMinorLocator(5))
        axes[i].xaxis.set_minor_locator(AutoMinorLocator(5))
        axes[i].axes.set_xticklabels("")

        # Replot nominal on top, don't want to reorg dict
        axes[i].errorbar(sys_dict['nominal']["bin_centers"],
                         sys_dict['nominal'][kin],
                         color=color_dict['nominal'],
                         linewidth=2, linestyle='--', alpha=0.7)

        axes[i].legend(fontsize=18, frameon=False, loc='upper right')
        # axes[i].legend(fontsize=18, frameon=False, loc="upper right", 
        #                bbox_to_anchor=(-0.022, 0.92))


        # Lower Panel Difference Plots
        # Separate loops: Above skips QED (not unfolded)
        # Here, we skip nominal (difference plots)
        for syst in uncertainties.keys():
            if syst == 'nominal':
                axes[i+3].errorbar(uncertainties['nominal']["bin_centers"],
                                   uncertainties[syst][kin],
                                   color=color_dict[syst], linewidth=2,
                                   linestyle='--',alpha=0.2)

            elif syst == 'QED' or syst == 'total':
                axes[i+3].errorbar(uncertainties['nominal']["bin_centers"],
                                   uncertainties[syst][kin],
                                   label=plot_labels[syst],
                                   color=color_dict[syst], linewidth=2)
            else:
                axes[i+3].errorbar(uncertainties['nominal']["bin_centers"],
                                   uncertainties[syst][kin],
                                   color=color_dict[syst], linewidth=2)


        axes[i+3].legend(fontsize=14, frameon=False, loc='upper right')
        axes[i+3].set_ylabel('Uncertainty (Abs.)')
        axes[i+3].set_xlabel("$\phi_\perp$ [rad.]")
        # axes[i+3].set_ylim(-0.015, 0.105)
        axes[i+3].yaxis.set_minor_locator(AutoMinorLocator(5))
        axes[i+3].xaxis.set_minor_locator(AutoMinorLocator(5))

    plt.tight_layout()
    plt.savefig("./plots/systematics/PHI_systematics_dual_plot.pdf")

    return

# Define Dictionary for labels in plotting. Maps code to Systematic
plot_labels = {}
plot_labels['nominal'] = 'Nominal'
plot_labels['model'] = 'Model (Prior)'
plot_labels['sys_0'] = 'HFS scale (in jet)'
plot_labels['sys_1'] = 'HFS scale (remainder)'
plot_labels['sys_5'] = 'HFS $\phi$ angle'
plot_labels['sys_7'] = 'Lepton energy scale'
plot_labels['sys_11'] = 'Lepton $\phi$ angle'
plot_labels['QED']  = 'QED rad corr.'
plot_labels['total']  = 'Total Systematic'

keys = ['nominal', 'model', 'QED', 'sys_0', 'sys_1', 'sys_5', 'sys_7', 'sys_11']

colors = ['black', '#348ABD', 'blue', '#C70039',
          '#FF5733', '#FFC300', '#65E88F', '#40E0D0']


colors = ['black', '#348ABD', 'blue', '#5a03fc', '#C70039',
          '#FF5733', '#FFC300', '#65E88F']

#8403fc -- purple
# 5a03fc
#4AFC03

color_dict = {keys[i]: colors[i] for i in range(len(colors))}
color_dict['total'] = 'grey'

qed_keys = ["Rapgap_nominal_Rady", "Rapgap_nominal_noRady",
            "Django_nominal_Rady", "Django_nominal_noRady"]


# Load the Config File
if len(sys.argv) <= 1:
    CONFIG_FILE = "./configs/config.yaml"
else:
    CONFIG_FILE = sys.argv[1]

config = yaml.safe_load(open(CONFIG_FILE))
print(f"\nLoaded {CONFIG_FILE}\n")


# Load Settings
suffix = config['identifier']
q_perp_bins = config['q_bins']
n_phi_bins = config['n_phi_bins']
phi_bins = np.linspace(0, 3.14159, n_phi_bins+1)

main_dir = config['main_dir']

# Save NPY arrays
reprocess_systematics = False
if reprocess_systematics:
    process_sys_npy(keys, suffix, 'Rapgap')

reprocess_QED_npy = False
if reprocess_QED_npy:
    process_QED_npy(qed_keys)


q_range = 'low'
# q_range = 'high'

# Save dictionary ingredients
do_sys_calc = True
nested_sys_dict = get_sys_dictionaries(keys, suffix, do_sys_calc,
                                       phi_bins,q_range=q_range)

do_qed_calc = True
qed_dict = get_QED_dict(qed_keys, do_qed_calc, phi_bins, q_range=q_range)


# Consolidate all the uncertainties
uncertainties = get_uncertainties(keys, nested_sys_dict, qed_dict)
ic(uncertainties)
sum_in_quadruture(uncertainties)  # appends 'total' to dict
pickle_dict(uncertainties, f'PHI_{q_range}_uncertainties.pkl', folder='./pkls')


plot_variations(nested_sys_dict, color_dict, plot_labels)
plot_uncertanties(uncertainties, color_dict, plot_labels)
plot_systematics(nested_sys_dict, uncertainties, color_dict, plot_labels)

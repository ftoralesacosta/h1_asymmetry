import sys
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt

from get_np_arrays import npy_from_pkl
from process_functions import averages_in_qperp_bins

from matplotlib import style
# style.use('/global/home/users/ftoralesacosta/dotfiles/scientific.mplstyle')
style.use('~/dotfiles/scientific.mplstyle')


def process_sys_npy(keys, suffix, mc_name='Rapgap'):
    for key in range(len(keys)):
        if key == 'QED': continue
        # QED and nominal are done separetly

        mc_label = f"{mc_name}_{key}_{mc_name}_{suffix}"

        print("on sample:", mc_name, key, "\n")
        npy_from_pkl(mc_label, run_type=key)


def process_QED_npy(qed_keys):
    QED_path = "/clusterfs/ml4hep_nvme2/ftoralesacosta/h1_check/h1_lepton_jet_asymmetry/"
    for QED_label in qed_keys:
        npy_from_pkl(QED_label, False, QED_path)
        # Unfolding NOT applied in QED systematic. Note 'False'


def get_sys_dictionaries(keys, suffix, do_sys_calc=False, mc='Rapgap',
                          npy_folder='./npy_files', pkl_dir='./pkls'):

    sys_file = 'sys_variations'

    if not do_sys_calc:
        with open(f'{pkl_dir}/{sys_file}.pkl', 'rb') as pkl_file:
            return pickle.load(pkl_file)

    # Creates nested dicitonary for systematics

    sys_dictionaries = {}

    for key in keys:
        if key == "QED": continue

        LABEL = f"{mc}_{key}_{mc}_{suffix}"
        if key == 'model':
            LABEL = f"Django_nominal_Django_{suffix}"

        print(f"Grabbing {LABEL}")

        inner_dict = {}
        cuts       = np.load(f"{npy_folder}/{LABEL}_cuts.npy")
        q_perp     = np.load(f"{npy_folder}/{LABEL}_q_perp.npy")[cuts]
        asymm_phi  = np.load(f"{npy_folder}/{LABEL}_asymm_angle.npy")[cuts]
        weights    = np.load(f"./weights/{LABEL}_pass_avgs.npy")[-1][cuts]
       # weights   = np.load(f"{npy_folder}/{LABEL}_weights.npy")

        averages_in_qperp_bins(inner_dict, q_perp_bins, q_perp, asymm_phi, weights)
        sys_dictionaries[key] = inner_dict

    print()
    print(f"Outer {sys_dictionaries.keys()}")
    print(f"Inner {sys_dictionaries[keys[0]].keys()}")

    pickle_dict(sys_dictionaries, f'{sys_file}.pkl', folder=pkl_dir)
    return sys_dictionaries


def get_QED_dict(qed_keys, do_qed_calc=True,
                 npy_folder='./npy_files', pkl_dir='./pkls'):
    # yields difference of Rapgap and Django w and w/o Rad

    qed_file = 'qed_variations'

    if not do_qed_calc:
        with open(f'{pkl_dir}/{qed_file}.pkl', 'rb') as pkl_file:
            return pickle.load(pkl_file)

    qed_dict = {}

    for key in qed_keys:
        LABEL = key
        inner_dict = {}
        cuts       = np.load(f"{npy_folder}/{LABEL}_cuts.npy")
        q_perp     = np.load(f"{npy_folder}/{LABEL}_q_perp.npy")[cuts]
        asymm_phi  = np.load(f"{npy_folder}/{LABEL}_asymm_angle.npy")[cuts]
        weights    = np.load(f"{npy_folder}/{LABEL}_weights.npy")[cuts]

        averages_in_qperp_bins(inner_dict, q_perp_bins, q_perp, asymm_phi, weights)
        qed_dict[key] = inner_dict

    pickle_dict(qed_dict, f'{qed_file}.pkl', folder=pkl_dir)
    return qed_dict


def get_QED_sys(qed_dict):
    qed_sys = {}

    for inner_key in qed_dict['Rapgap_nominal_Rady'].keys():

        rapgap_difference = qed_dict['Rapgap_nominal_Rady'][inner_key] - \
                            qed_dict['Rapgap_nominal_noRady'][inner_key]
        django_difference = qed_dict['Django_nominal_Rady'][inner_key] - \
                            qed_dict['Django_nominal_noRady'][inner_key]

        qed_sys[inner_key] = np.abs(rapgap_difference-django_difference)

    return qed_sys


def plot_QED(qed_dict, qed_sys, savedir = './plots/systematics'):
    fig, axes = plt.subplots(2, 3, figsize=(22, 12))
    axes = axes.ravel()

    for i in range(3):
        key = "cos%i" % (i+1)
        string = r"$\langle\cos(%i\phi)\rangle$" % (i+1)
        string = string.replace("1", "")

        for mc in ["Rapgap", "Django"]:
            q_perp = qed_dict[f"{mc}_nominal_Rady"]["q_perp"]

            axes[i].plot(q_perp, qed_dict[f"{mc}_nominal_Rady"][key],
                         label=f"{mc} QED Rad.")

            axes[i].plot(q_perp, qed_dict[f"{mc}_nominal_noRady"][key],
                         linestyle='--', label=f"{mc} No QED Rad.")

            axes[i+3].plot(q_perp, qed_dict[f"{mc}_nominal_Rady"][key] - \
                           qed_dict[f"{mc}_nominal_noRady"][key],
                           label=f"{mc} QED difference")

        axes[i+3].plot(q_perp, qed_sys[key],color="blue",label="Rapgap-Django QED difference")

        # Plotsmanship
        axes[i].set_ylim(-0.4, 0.5)
        axes[i].set_ylabel(string, fontsize=28)

        axes[i+3].set_ylim(-0.1,0.05)
        axes[i+3].set_ylabel(string,fontsize=28)
        axes[i+3].set_xlabel("$q_\perp$ [GeV]",fontsize=28)

    axes[2].legend(fontsize=18)
    axes[5].legend(fontsize=18)
    plt.tight_layout()
    plt.savefig(f"{savedir}/QED_plot.pdf")

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
    print("LINE 171")
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

            if inner_key == 'q_perp': 
                inner_dict[inner_key] = nominal[inner_key]
                continue

            inner_dict[inner_key] = np.abs(nominal[inner_key] -
                                            nested_sys_dict[syst][inner_key])
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


def plot_variations(sys_dict, color_dict, plot_label):
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    for i, kin in enumerate(["cos1", "cos2", "cos3"]):
        for syst in sys_dict.keys():
            if syst == "QED": continue

            # print(sys_dict[syst]['q_perp'])
            axes[i].errorbar(sys_dict[syst]["q_perp"], sys_dict[syst][kin],
                             label=plot_label[syst], color=color_dict[syst], linewidth=3)

            string = r"$\cos(%i\phi)$" % (i+1)
            string = string.replace("1", "")
            axes[i].set_ylabel(string, fontsize=28)
            axes[i].set_xlabel("$q_\perp$ [GeV]", fontsize=28)
            axes[0].legend(fontsize=15)


    plt.tight_layout()
    plt.savefig("./plots/systematics/systematics_unfolded_separately.pdf")

    return


def plot_uncertanties(uncertainties, color_dict):

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

    axes[0].set_ylim(-0.01, 0.15)
    axes[1].set_ylim(-0.01, 0.15)
    axes[2].set_ylim(-0.01, 0.15)

    for i, kin in enumerate(["cos1", "cos2", "cos3"]):

        for syst in uncertainties.keys():

            if syst == 'nominal': continue
            print("L263: ")
            print(syst, uncertainties[syst]['q_perp'])
            axes[i].errorbar(uncertainties['nominal']["q_perp"],
                             uncertainties[syst][kin],
                             label=syst,
                             color=color_dict[syst], linewidth=3)

        # if (plot_bootstrap):
        #     axes[i].errorbar(nominal["q_perp"], np.abs(abs_bootstrap_errors[key]),
        #             label="Stat. Uncertainty", color="grey", linestyle="-", linewidth=3)

            # axes[0].set_ylim(-0.01,1.5)
            # axes[1].set_ylim(-0.01,1.5)
            # axes[2].set_ylim(-0.01,1.5)

        string = r"$\cos(%i\phi)$ Error [abs.]" % (i+1)
        string = string.replace("1", "")
        axes[i].set_ylabel(string)
        axes[i].set_xlabel("$q_\perp [GeV]$")

    axes[0].text(0.04, 0.92, "H1 Preliminary", transform=axes[0].transAxes, fontsize=22) 

    plt.tight_layout()
    plt.legend(fontsize=20)
    plt.savefig("./plots/systematics/absError_systematics_unfolded_separately.pdf")

    return


# Define Dictionary for labels in plotting. Maps code to Systematic
plot_label = {}
plot_label['sys_0'] = 'HFS scale (in jet)'
plot_label['sys_1'] = 'HFS scale (remainder)'
plot_label['sys_5'] = 'HFS $\phi$ angle'
plot_label['sys_7'] = 'Lepton energy scale'
plot_label['sys_11'] = 'Lepton $\phi$ angle'
plot_label['QED']  = 'QED rad corr.'

keys = ['nominal', 'model', 'QED', 'sys_0', 'sys_1', 'sys_5', 'sys_7', 'sys_11']

colors = ['black', '#348ABD', 'blue', '#C70039',
          '#FF5733', '#FFC300', '#65E88F', '#40E0D0']
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
N_bins = len(q_perp_bins)


# Save NPY arrays
reprocess_systematics = False
if reprocess_systematics:
    process_sys_npy(keys, suffix, 'Rapgap')

reprocess_QED_npy = False
if reprocess_QED_npy:
    process_QED_npy(qed_keys)


# Save dictionary ingredients
do_sys_calc = True
nested_sys_dict = get_sys_dictionaries(keys, suffix, do_sys_calc)

do_qed_calc = False
qed_dict = get_QED_dict(qed_keys, do_qed_calc)


# Consolidate all the uncertainties
uncertainties = get_uncertainties(keys, nested_sys_dict, qed_dict)
sum_in_quadruture(uncertainties)  # appends 'total' to dict
pickle_dict(uncertainties, 'uncertainties.pkl', folder='./pkls')


plot_variations(nested_sys_dict, color_dict, plot_label)
plot_uncertanties(uncertainties, color_dict, plot_label)

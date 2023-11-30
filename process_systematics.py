import sys
import yaml
import pickle
import numpy as np
import matplotlib.pyplot as plt

from get_np_arrays import npy_from_pkl
from process_functions import averages_in_qperp_bins

from matplotlib import style
style.use('/global/home/users/ftoralesacosta/dotfiles/scientific.mplstyle')


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
        weights    = np.load(f"{npy_folder}/{LABEL}_weights.npy")

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


def QED_systematic(qed_dict):

    qed_sys = get_QED_sys(qed_dict)
    plot_QED(qed_dict, qed_sys)

    return qed_sys


def pickle_dict(dic, save_name, folder='.'):
    file = open(f'{folder}/{save_name}', 'wb')
    pickle.dump(dic, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()


# Define Dictionary for labels in plotting. Maps code to Systematic
label = {}
label['sys_0'] = 'HFS scale (in jet)'
label['sys_1'] = 'HFS scale (remainder)'
label['sys_5'] = 'HFS $\phi$ angle'
label['sys_7'] = 'Lepton energy scale'
label['sys_11'] = 'Lepton $\phi$ angle'
label['QED']  = 'QED rad corr.'

keys = ['nominal', 'model', 'QED', 'sys_0', 'sys_1', 'sys_5', 'sys_7', 'sys_11']

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
do_sys_calc = False
nested_sys_dict = get_sys_dictionaries(keys, suffix, do_sys_calc)

do_qed_calc = False
qed_dict = get_QED_dict(qed_keys, do_qed_calc)

def get_systematics(sys_keys, nested_sys_dict):

    systematics = {}
    print(nested_sys_dict.keys())
    nominal = nested_sys_dict['nominal']

    for syst in sys_keys:
        if sys == "QED": continue

        inner_dict = {}
        for inner_key in nominal.keys():
            inner_dict[inner_dict] = np.abs(nominal[inner_key] -
                                            systematics[syst][inner_key])
        systematics[syst] = inner_dict

    return systematics



systematics = get_systematics(keys, nested_sys_dict)
systematics["QED"] = QED_systematic(qed_dict)  # also plots QED
# qed_sys = QED_systematic(qed_dict)  # also plots QED

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import sys
from matplotlib import style
sys.path.insert(0, '../')
from process_functions import *
from tqdm import tqdm
import yaml

# style.use('/global/home/users/ftoralesacosta/dotfiles/scientific.mplstyle')
colors = ['#348ABD', '#C70039', '#FF5733', '#FFC300', '#65E88F', '#40E0D0']


# Load the Config File
if len(sys.argv) <= 1:
    CONFIG_FILE = "./configs/config.yaml"
else:
    CONFIG_FILE = sys.argv[1]

config = yaml.safe_load(open(CONFIG_FILE))
print(f"\nLoaded {CONFIG_FILE}\n")

mc = config['mc']  # Rapgap, Django, Pythia
run_type = config['run_type']  # nominal, bootstrap, systematic
run_identifier = config['identifier']
processed_dir = config['main_dir']

LABEL = f"{mc}_{run_type}_{run_identifier}"
if config['is_test']:
    LABEL = LABEL + "_TEST"  # avoid overwrites of nominal

cuts_h1rpgp       = np.load(f'{processed_dir}/npy_files/{LABEL}_cuts.npy')
jet_pT_h1rpgp     = np.load(f'{processed_dir}/npy_files/{LABEL}_jet_pT.npy')
cuts_h1rpgp = np.ones(len(jet_pT_h1rpgp), dtype=bool)
jet_pT_h1rpgp     = np.load(f'{processed_dir}/npy_files/{LABEL}_jet_pT.npy')[cuts_h1rpgp]
q_perp_h1rpgp     = np.load(f'{processed_dir}/npy_files/{LABEL}_q_perp.npy')[cuts_h1rpgp]
asymm_phi_h1rpgp  = np.load(f'{processed_dir}/npy_files/{LABEL}_asymm_angle.npy')[cuts_h1rpgp]
# weights_h1rpgp    = np.load(f'{processed_dir}/npy_files/{LABEL}_weights.npy')
weights_h1rpgp    = np.load(f"./weights/{LABEL}_pass_avgs.npy")
mc_weights_h1rpgp = np.load(f"{processed_dir}/npy_files/{LABEL}_mc_weights.npy")[cuts_h1rpgp]
nn_weights_h1rpgp = np.load(f"{processed_dir}/npy_files/{LABEL}_nn_weights.npy")[cuts_h1rpgp]


folder = f"{processed_dir}/../h1_models/{LABEL}"
django_weights = np.ones(0)
rapgap_weights = np.ones(0)

plot_avg = False

N_Events = 1_000_000
N_Events = -1
NIter = config['n_iterations']
# NIter = 20
N_passes = config['n_passes']
# sum_over_passes = np.zeros((NIter, N_Events))

n_bins = 100
colors = cm.magma(np.linspace(0.2, 0.9, N_passes))
colors = colors[::-1]

d_colors = cm.viridis(np.linspace(0.2, 0.9, N_passes))
d_colors = d_colors[::-1]

django_stdevs = np.zeros((NIter))

file_init = f"{folder}/{LABEL}_Pass0_Step2_Weights.npy"
weights_init = np.load(file_init)
print("Length of weights = ", len(weights_init[0,0]))
size_init = len(weights_init[0, 0])
print("Size init = ", size_init)

stdevs = np.zeros((N_passes, NIter))
means = np.zeros((N_passes, NIter))

pass_avgs = np.zeros((NIter, size_init))
print("PASS AVGE SHAPE = ", np.shape(pass_avgs))
cuts = cuts_h1rpgp[:size_init]
print("Length of cuts = ", len(cuts))


fig, axes = plt.subplots(nrows=3, ncols=4,
# fig, axes = plt.subplots(nrows=10, ncols=10,
                         figsize=(20, 14), sharex=True, sharey=True)
# list = [axes,]
axes = np.asarray([axes,])
axes = axes[0, :, :]
axes = np.ravel(axes)

for p in tqdm(range(0, N_passes)):

    file = f"{folder}/{LABEL}_Pass{p}_Step2_Weights.npy"
    pass_p = np.load(file, allow_pickle=True)[:, 0]
    # print("WEIGHT in LOOP = ", np.shape(pass_p))

    for i in range(0, NIter):

        weights = pass_p[i]
        print(np.mean(weights), np.std(weights))
        stdevs[p][i] = np.std(weights)
        means[p][i] = np.mean(weights)

        # Plot
        axes[i].hist(weights[cuts], bins=np.linspace(0, 2, n_bins),
                     label=f"Pass {p}",
                     alpha=0.5, color=d_colors[p],
                     linewidth=1.5, histtype='step')

        axes[i].legend(fontsize=5, ncol=2)
        axes[i].set_title(f"{mc} Iteration {i}", fontsize=10)

        # Stats
        pass_avgs[i] += weights[cuts]

    pass_avgs[i] = pass_avgs[i]/N_passes

    # if plot_avg:
    #     axes[i].hist(pass_avgs[i][cuts], bins=np.linspace(0, 2, n_bins),
    #                  label=f"Mean Iteration {i}", alpha=0.9, color='k',
    #                  linestyle='--', linewidth=0.5, histtype='step')
    #     axes[i].text(1.0, 190, "-- Pass Average", fontsize=7)


plt.suptitle("Pre-Averaged Distributions", fontsize=25)
plt.savefig(f"./plots/{LABEL}_Passes.pdf")

np.save(f"./weights/{LABEL}_pass_avgs.npy", pass_avgs)
np.save(f"./weights/{LABEL}_stds.npy", stdevs)
np.save(f"./weights/{LABEL}_means.npy", means)

# Plot Standard Deviations
fig = plt.figure(figsize=(10, 10))
iterations = np.linspace(0, NIter, NIter, endpoint=False)
print(np.shape(stdevs))
print(iterations)

for p in range(N_passes):
    plt.scatter(iterations, stdevs[p],
                color=d_colors[p], label=f"Pass {p}")
    plt.ylim(0, 0.6)
    plt.xlim(-0.5,NIter+0.5)
    plt.legend(fontsize=15)
    plt.ylabel("$\sigma_{weights}$")
    plt.xlabel("Iteration")
plt.savefig(f"./plots/{LABEL}_stdevs.pdf")

# Plot Means
fig = plt.figure(figsize=(10, 10))
for p in range(N_passes):
    plt.scatter(iterations, means[p],
                color=colors[p], label=f"Pass {p}")
    plt.ylim(0, 2.1)
    plt.xlim(-0.5,NIter+0.5)
    plt.legend(fontsize=15)
    plt.ylabel("$\mu_{weights}$")
    plt.xlabel("Iteration")
plt.savefig(f"./plots/{LABEL}_means.pdf")

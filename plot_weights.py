import numpy as np
import matplotlib.pyplot as plt

# from copy import copy
# from matplotlib.colors import LogNorm
# from matplotlib import gridspec
# from matplotlib.ticker import AutoMinorLocator
# import pickle
from matplotlib.pyplot import cm
import sys
from matplotlib import style
sys.path.insert(0, '../')
from process_functions import *
from tqdm import tqdm

style.use('/global/home/users/ftoralesacosta/dotfiles/scientific.mplstyle')
colors = ['#348ABD', '#C70039', '#FF5733', '#FFC300', '#65E88F', '#40E0D0']


processed_dir = "/clusterfs/ml4hep_nvme2/ftoralesacosta/h1_check/h1_lepton_jet_asymmetry"
# Data Unfolded From Django
cuts_h1djgo       = np.load(f'{processed_dir}/npy_files/cuts.npy')
jet_pT_h1djgo     = np.load(f'{processed_dir}/npy_files/jet_pT.npy')[cuts_h1djgo]
q_perp_h1djgo     = np.load(f'{processed_dir}/npy_files/q_perp.npy')[cuts_h1djgo]
asymm_phi_h1djgo  = np.load(f'{processed_dir}/npy_files/asymm_angle.npy')[cuts_h1djgo]
weights_h1djgo    = np.load(f'{processed_dir}/npy_files/weights.npy')[cuts_h1djgo]
mc_weights_h1djgo = np.load(f"{processed_dir}/npy_files/mc_weights.npy")[cuts_h1djgo]
nn_weights_h1djgo = np.load(f"{processed_dir}/npy_files/nn_weights.npy")[cuts_h1djgo]

# Data Unfolded From Rapgap
cuts_h1rpgp       = np.load(f'{processed_dir}/npy_files/from_rapgap_cuts.npy')
jet_pT_h1rpgp     = np.load(f'{processed_dir}/npy_files/from_rapgap_jet_pT.npy')[cuts_h1rpgp]
q_perp_h1rpgp     = np.load(f'{processed_dir}/npy_files/from_rapgap_q_perp.npy')[cuts_h1rpgp]
asymm_phi_h1rpgp  = np.load(f'{processed_dir}/npy_files/from_rapgap_asymm_angle.npy')[cuts_h1rpgp]
weights_h1rpgp    = np.load(f'{processed_dir}/npy_files/from_rapgap_weights.npy')[cuts_h1rpgp]
mc_weights_h1rpgp = np.load(f"{processed_dir}/npy_files/from_rapgap_mc_weights.npy")[cuts_h1rpgp]
nn_weights_h1rpgp = np.load(f"{processed_dir}/npy_files/from_rapgap_nn_weights.npy")[cuts_h1rpgp]


mc = "Rapgap"
# label = f"{mc}_nominal_Aug14Test_unfoldpy"
label = "Rapgap_nominal_Aug18Test_unfoldpy"
# label = "Rapgap_nominal_Aug19Test_unfoldpy"
folder = f"/global/ml4hep/spss/ftoralesacosta/new_models/{label}"

django_weights = np.ones(0)
rapgap_weights = np.ones(0)

plot_avg = False

N_Events = 1_000_000
NIter = 100
# NIter = 20
N_passes = 4
sum_over_passes = np.zeros((NIter, N_Events))

n_bins = 100
colors = cm.magma(np.linspace(0.2, 0.9, N_passes))
colors = colors[::-1]

d_colors = cm.viridis(np.linspace(0.2, 0.9, N_passes))
d_colors = d_colors[::-1]

django_stdevs = np.zeros((NIter))

file_init = f"{folder}/{label}_Pass0_Step2_Weights.npy"
weights_init = np.load(file_init)[:N_Events]
size_init = len(weights_init[0, 0])

stdevs = np.zeros((N_passes, NIter))
means = np.zeros((N_passes, NIter))

pass_avgs = np.zeros((NIter, size_init))
cuts = cuts_h1rpgp[:size_init]
print("Length of cuts = ", len(cuts))

# fig, axes = plt.subplots(nrows=4, ncols=5,
fig, axes = plt.subplots(nrows=10, ncols=10,
                         figsize=(20, 14), sharex=True, sharey=True)
# list = [axes,]
axes = np.asarray([axes,])
axes = axes[0, :, :]
axes = np.ravel(axes)

for p in tqdm(range(0, N_passes)):

    file = f"{folder}/{label}_Pass{p}_Step2_Weights.npy"
    pass_p = np.load(file, allow_pickle=True)[:, 0, :N_Events]
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
        pass_avgs[i] += weights

    pass_avgs[i] = pass_avgs[i]/N_passes

    if plot_avg:
        axes[i].hist(pass_avgs[i][cuts], bins=np.linspace(0, 2, n_bins),
                     label=f"Mean Iteration {i}", alpha=0.9, color='k',
                     linestyle='--', linewidth=0.5, histtype='step')
        axes[i].text(1.0, 190, "-- Pass Average", fontsize=7)


plt.suptitle("Pre-Averaged Distributions", fontsize=25)
plt.savefig(f"./plots/{label}_Passes.pdf")
np.save(f"./weights/{label}_pass_avgs.npy", pass_avgs)
np.save(f"./weights/{label}_stds.npy", stdevs)
np.save(f"./weights/{label}_means.npy", means)

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
plt.savefig(f"./plots/{label}_stdevs.pdf")

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
plt.savefig(f"./plots/{label}_means.pdf")

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import yaml
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm

from sklearn.preprocessing import StandardScaler
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model

from unfold_hvd import weighted_binary_crossentropy
from unfold_hvd import reweight


# CONFIG FILE
if len(sys.argv) > 1:
    CONFIG_FILE = sys.argv[1]
else:
    CONFIG_FILE = "./configs/config.yaml"

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

N_iter = config['n_iterations']
N_passes = config['n_passes']

N_bootstraps = 100
# N_bootstraps = 10
# N_bootstraps = 2
# N_passes = 2

if 'bootstrap' not in ID and 'Bootstrap' not in ID:
    print("CONFIGURATION FAIL: Need to specify bootstrap in config")
    sys.exit("This runs only on bootstrap ensembles")
# Need to save each pass, and it would be nice to save pass_avgs :]

# Django_nominal_Perlmutter_django_Bootstrap

# ========================================
# Tensorflow Init

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ======= Load the thetaG and unknownS ======
theta0_G = np.load(f"npy_inputs/{ID}_Theta0_G.npy")
print("\n\nLoaded theta0_G")
theta_unknown_S = np.load(f"npy_inputs/{ID}_theta_unknown_S.npy")
print("Loaded theta_unknown_S")
weights_MC_sim = np.load(f"npy_inputs/{ID}_weights_mc_sim.npy")
print("Loaded weights MC\n")

#Get StandardScalar from raw data, apply to Rapgap in each iteration later   

# ======= Load the Model ======
ID_File = ID # FIXME will add be np.arrange(n_bootstraps)
num_observables = 8
if config['asymm_vars']:
    num_observables = 12

inputs = Input((num_observables, ))
hidden_layer_1 = Dense(50, activation='relu')(inputs)
dropoutlayer = Dropout(0.1)(hidden_layer_1)
hidden_layer_2 = Dense(100, activation='relu')(dropoutlayer)
hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
outputs = Dense(1, activation='sigmoid')(hidden_layer_3)
model = Model(inputs=inputs, outputs=outputs)


model.compile(loss=weighted_binary_crossentropy,
              optimizer=tf.keras.optimizers.Adam(learning_rate=2e-6),
              metrics=['accuracy'])




# ======= For Plotting ======
stdevs = np.zeros((N_bootstraps, N_passes, N_iter))
means  = np.zeros((N_bootstraps, N_passes, N_iter))

n_bins = 100
colors = cm.magma(np.linspace(0.2, 0.9, N_passes))
colors = colors[::-1]
d_colors = cm.viridis(np.linspace(0.2, 0.9, N_passes))
d_colors = d_colors[::-1]

# Plot for weights
fig, axes = plt.subplots(nrows=3, ncols=4,
                         figsize=(20, 14),
                         sharex=True, sharey=True)
axes = np.asarray([axes, ])
axes = np.ravel(axes[0, :, :])

# array to save
bootstrap_weights = []
skips = []
skips = [4]


for b in tqdm(range(N_bootstraps)):
    bootstrap_name = ID_File+f"{b+1}"
    if b+1 in skips:
        print(f"Skipping Bootstrap {b+1} \n")
        continue
    print(f"Bootstrap {b+1}\n")
    print(bootstrap_name)

    pass_avgs = np.zeros( (N_iter, len(theta0_G[:,0])) )
    # pass_avgs = np.zeros(N_iter, len(theta0_G[:,0]))

    for p in range(N_passes):

        for i in range(N_iter):

            # if i!= N_iter-1: continue  # just need last iter
            if i!= N_iter-2: continue  # not confident in checpointing  last iter

            model_name = f'{model_dir}/{ID}/{bootstrap_name}_Pass{p}/iter_{i}_step2_checkpoint'
            print(f"Loading Model {model_name}")
            model.load_weights(model_name)

            weights_push = weights_MC_sim * reweight(theta0_G, model, verbose = 0)
            print(f"Mean Weight = {np.mean(weights_push)}")
            print(f"STDV Weight = {np.std(weights_push)} ")

            print("indecies = ")
            print(b, p, i)
            print(np.shape(stdevs))
            stdevs[b][p][i] = np.std(weights_push)
            means[b][p][i] = np.mean(weights_push)

            # ====== PLOTTING ======
            axes[i].hist(weights_push, bins=np.linspace(0, 2, n_bins),
                         label=f"Pass {p}",
                         alpha=0.5, color=d_colors[p],
                         linewidth=1.5, histtype='step')

            axes[i].legend(fontsize=5, ncol=2)
            axes[i].set_title(f"{mc} Iteration {i}", fontsize=10)

            print("Push W               = ", weights_push[:5])
            pass_avgs[i] += weights_push
            print(f"Avg over {p} passes = ", pass_avgs[i][:5]/(p+1))

        # pass_avgs[i] = pass_avgs[i]/N_passes
        # print("\npass_avg_weight = ",pass_avgs[i], "\n")

    pass_avgs[-2] = pass_avgs[-2]/N_passes
    print("\n Avg outside of ITER loop  = ",pass_avgs[-2][:5], "\n")

    # if b % 10 == 0:
    np.save(f"./weights/{bootstrap_name}_pass_avgs.npy", pass_avgs)

    bootstrap_weights.append(pass_avgs[-2]) #save the last iteration

np.save(f"./weights/{LABEL}_bootstrap_weights.npy",bootstrap_weights)
print("Weights saved to ./weights/")

# === Save Weight Figures ==== 
plt.suptitle("Pre-Averaged Distributions", fontsize=25)
plt.savefig(f"./plots/{LABEL}_Passes.pdf")

np.save(f"./weights/{LABEL}_pass_avgs.npy", pass_avgs)
np.save(f"./weights/{LABEL}_stds.npy", stdevs)
np.save(f"./weights/{LABEL}_means.npy", means)



# ==================== Plotting STDEV and MEANS ======================#
# Plot Standard Deviations
fig = plt.figure(figsize=(10, 10))
iterations = np.linspace(0, N_iter, N_iter, endpoint=False)
print(np.shape(stdevs))
print(iterations)

for b in tqdm(range(1,N_bootstraps+1,10)):
    bootstrap_name = ID_File+f"{b}"
    for p in range(N_passes):
        plt.scatter(iterations, stdevs[b][p],
                    color=d_colors[p], label=f"Pass {p}")
        plt.ylim(0, 0.6)
        plt.xlim(-0.5,N_iter+0.5)
        plt.legend(fontsize=15)
        plt.ylabel("$\sigma_{weights}$")
        plt.xlabel("Iteration")
        plt.title(f"Stdv Weight vs Iteration [B{b}]")
    plt.savefig(f"./plots/{bootstrap_name}_stdevs.pdf")


# Plot Means
    fig = plt.figure(figsize=(10, 10))
    for p in range(N_passes):
        plt.scatter(iterations, means[b][p],
                    color=colors[p], label=f"Pass {p}")
        plt.ylim(0, 2.1)
        plt.xlim(-0.5,N_iter+0.5)
        plt.legend(fontsize=15)
        plt.ylabel("$\mu_{weights}$")
        plt.xlabel("Iteration")
        plt.title(f"Mean Weight vs Iteration [B{b}]")
    plt.savefig(f"./plots/{bootstrap_name}_means.pdf")

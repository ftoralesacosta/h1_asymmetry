import sys

'''
run like python unfold_fullstats.py Rapgap nominal  0
[command] [Rapgap or Django] [run_type: nominal, sys_0...] [BOOTSTRAPPING Seed]
'''

print("Running on MC sample",sys.argv[1],"with setting",sys.argv[2])

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
from matplotlib import gridspec
import time

from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler
from get_np_arrays import get_kinematics

import os
# os.environ['CUDA_VISIBLE_DEVICES']=sys.argv[3] #"1"
os.environ['CUDA_VISIBLE_DEVICES']= "0"

physical_devices = tf.config.list_physical_devices('GPU') 
tf.config.experimental.set_memory_growth(physical_devices[0], True)

# LABEL = "Feb20_NewNominal"
# LABEL = "Feb20_sys"+sys.argv[2]
# LABEL = "rerunning_systematics_"+sys.argv[2]
# LABEL = f"July1_Iter30_Pass{pass_i}"+sys.argv[2]
# LABEL = f"AdditionalKinematics_Iter30_Pass{pass_i}"+sys.argv[2]
LABEL = "TestRefactor"
ID = f"{sys.argv[1]}_{sys.argv[2]}_{LABEL}"  # Basic string. Add Pass and Iter
print(f"\n\n {ID} \n\n")
pass_i = int(sys.argv[4])

tf.random.set_seed(int(sys.argv[3]))
np.random.seed(int(sys.argv[3]))

def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1) # event weights
    y_true = tf.gather(y_true, [0], axis=1) # actual y_true for loss

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    t_loss = -weights * ((y_true) * K.log(y_pred) +
                         (1 - y_true) * K.log(1 - y_pred))
    return K.mean(t_loss)


def reweight(events):
    f = model.predict(events, batch_size=10000)
    weights = f / (1. - f)
    return np.squeeze(np.nan_to_num(weights, posinf=1))


# Read in the data
NEVENTS = -1
# NEVENTS = 100_000
inputs_dir = "/clusterfs/ml4hep/yxu2/unfolding_mc_inputs"
mc = pd.read_pickle(f"{inputs_dir}/{sys.argv[1]}_{sys.argv[2]}.pkl")[:NEVENTS]
data = pd.read_pickle(f"{inputs_dir}/Data_nominal.pkl")[:NEVENTS]

print(f"MC = {inputs_dir}/{sys.argv[1]}_{sys.argv[2]}.pkl")
print(f"Data = {inputs_dir}/Data_nominal.pkl")

print(np.shape(mc))
print(np.shape(data))

output_dir = "/global/ml4hep/spss/ftoralesacosta/new_models"
output_dir = output_dir+f"/{ID}"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

leading_jets_only = True
if (leading_jets_only):
    njets_tot = len(data["e_px"])
    data = data.loc[(slice(None), 0), :]
    mc = mc.loc[(slice(None), 0), :]
    print("Number of subjets cut = ",njets_tot-len(data["e_px"])," / ",len(data["jet_pt"]))

reco_vars = [
    'e_px', 'e_py', 'e_pz',
    'jet_pt', 'jet_eta', 'jet_phi',
    'jet_dphi', 'jet_qtnorm']

gen_vars = ['gene_px', 'gene_py', 'gene_pz',
            'genjet_pt', 'genjet_eta', 'genjet_phi',
            'genjet_dphi', 'genjet_qtnorm']

# Load the Data
theta_unknown_S = data[reco_vars].to_numpy()
theta0_S = mc[reco_vars].to_numpy()
theta0_G = mc[gen_vars].to_numpy()

# Add directly the asymmetry angle to the unfolding.
add_asymmetry = False
if add_asymmetry:

    asymm_kinematics = np.asarray(get_kinematics(theta_unknown_S)).T
    theta_unknown_S = np.append(theta_unknown_S, asymm_kinematics,1)

    Sasymm_kinematics = np.asarray(get_kinematics(theta0_S)).T
    theta0_S = np.append(theta0_S, Sasymm_kinematics,1)

    Gasymm_kinematics = np.asarray(get_kinematics(theta0_G)).T
    theta0_G = np.append(theta0_G, Gasymm_kinematics,1)


weights_MC_sim = mc['wgt']
pass_reco = np.array(mc['pass_reco'])
pass_truth = np.array(mc['pass_truth'])
pass_fiducial = np.array(mc['pass_fiducial'])
print("THE LENGTH OF THE ARRAYS IS =",len(pass_fiducial))

del mc
gc.collect()

#Early stopping
earlystopping = EarlyStopping(patience=10,
                              verbose=True,
                              restore_best_weights=True)

#Now, for the unfolding!
nepochs = 2000
# nepochs = 2

bins = np.logspace(np.log10(0.03),np.log10(3.03),9) - 0.03
bins = bins[1:]
bins[0] = 0.0
starttime = time.time()


# Set up the model
input_shape = np.shape(theta0_G)[1]
inputs = Input((input_shape, ))
# inputs = Input((8, ))
hidden_layer_1 = Dense(50, activation='relu')(inputs)
hidden_layer_2 = Dense(100, activation='relu')(hidden_layer_1)
hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
outputs = Dense(1, activation='sigmoid')(hidden_layer_3)
mymodel = Model(inputs=inputs, outputs=outputs)

# logfile
logfile = open(f"{output_dir}/{ID}_Pass{pass_i}.txt", "w")
logfile.write("Running on MC sample "+sys.argv[1]+" with setting "+sys.argv[2]+" seed "+sys.argv[3]+"Pass%i"%(pass_i)+"\n")

# Set Initial Data Weights
dataw = np.ones(len(theta_unknown_S))
if (int(sys.argv[3]) != 0):
    dataw = np.random.poisson(1, len(theta_unknown_S))
    print("Doing Bootstrapping")
else:
    print("Not doing bootstrapping")

# Set 'Truth' Weights
NNweights_step2 = np.ones(len(theta0_S))

NIter = 30
for iteration in range(NIter):

        mymodel = Model(inputs=inputs, outputs=outputs)
        # Process the data
        print("on iteration=",iteration," processing data for step 1, time elapsed=",time.time()-starttime)
        logfile.write("on iteration="+str(iteration)+" processing data for step 1, time elapsed="+str(time.time()-starttime)+"\n")

        xvals_1 = np.concatenate([theta0_S[pass_reco==1], theta_unknown_S])
        yvals_1 = np.concatenate([np.zeros(len(theta0_S[pass_reco==1])),np.ones(len(theta_unknown_S))])
        weights_1 = np.concatenate([NNweights_step2[pass_reco==1]*weights_MC_sim[pass_reco==1],dataw])

        scaler_data = StandardScaler()
        scaler_data.fit(theta_unknown_S)
        xvals_1 = scaler_data.transform(xvals_1)

        X_train_1, X_test_1, Y_train_1, Y_test_1, w_train_1, w_test_1 = train_test_split(xvals_1, yvals_1, weights_1,test_size=0.5)
        del xvals_1, yvals_1, weights_1
        gc.collect()

        Y_train_1 = np.stack((Y_train_1, w_train_1), axis=1)
        Y_test_1 = np.stack((Y_test_1, w_test_1), axis=1)
        del w_train_1, w_test_1
        gc.collect()

        print("on iteration=",iteration," done processing data for step 1, time elapsed=",time.time()-starttime)
        print("data events = ",len(X_train_1[Y_train_1[:,0]==1]))
        print("MC events = ",len(X_train_1[Y_train_1[:,0]==0]))

        logfile.write("Iter ="+str(iteration)+" done processing data for step 1, time="+str(time.time()-starttime)+"\n")
        logfile.write("data events="+str(len(X_train_1[Y_train_1[:, 0]==1]))+"\n")
        logfile.write("MC events = "+str(len(X_train_1[Y_train_1[:, 0]==0]))+"\n")

        # Step 1
        print("on step 1, time elapsed=", time.time()-starttime)
        logfile.write("step 1, time elapsed="+str(time.time()-starttime)+"\n")

        # opt = tf.keras.optimizers.Adam(learning_rate=2e-6)
        opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        mymodel.compile(loss=weighted_binary_crossentropy,
                          optimizer=opt,
                          metrics=['accuracy'])


        print("done with step 1, time elapsed=",time.time()-starttime)
        logfile.write("done with step 1, time elapsed="+str(time.time()-starttime)+"\n")

        # Now, let's do some checking.

        mypred = mymodel.predict(scaler_data.transform(np.nan_to_num(theta0_S,posinf=0,neginf=0)),batch_size=10000)
        mypred = mypred/(1.-mypred)
        mypred = mypred[:,0]
        mypred = np.squeeze(np.nan_to_num(mypred,posinf=1))


##############
### STEP 2 ###
##############

        print("Step 2, time elapsed=", time.time()-starttime)
        logfile.write("time for step 2, time elapsed="+str(time.time()-starttime)+"\n")

        xvals_2 = np.concatenate([theta0_G[pass_truth==1],theta0_G[pass_truth==1]])
        yvals_2 = np.concatenate([np.zeros(len(theta0_G[pass_truth==1])),np.ones(len(theta0_G[pass_truth==1]))])

        xvals_2 = scaler_data.transform(xvals_2)

        NNweights = mymodel.predict(scaler_data.transform(np.nan_to_num(theta0_S[pass_truth==1],posinf=0,neginf=0)), batch_size=10000)
        NNweights = NNweights/(1.-NNweights)
        NNweights = NNweights[:, 0]
        NNweights = np.squeeze(np.nan_to_num(NNweights, posinf=1))
        NNweights[pass_reco[pass_truth == 1] == 0] = 1.
        weights_2 = np.concatenate([NNweights_step2[pass_truth==1]*weights_MC_sim[pass_truth==1],NNweights*NNweights_step2[pass_truth==1]*weights_MC_sim[pass_truth==1]])

        X_train_2, X_test_2, Y_train_2, Y_test_2, w_train_2, w_test_2 = train_test_split(xvals_2, yvals_2, weights_2,test_size=0.5)
        del xvals_2, yvals_2, weights_2
        gc.collect()

        Y_train_2 = np.stack((Y_train_2, w_train_2), axis=1)
        Y_test_2 = np.stack((Y_test_2, w_test_2), axis=1)
        del w_train_2, w_test_2
        gc.collect()

        print("on iteration=",iteration," done processing data for step 2, time elapsed=",time.time()-starttime)
        print("MC events = ",len(X_train_1[Y_train_1[:,0]==1]))
        print("MC events = ",len(X_train_1[Y_train_1[:,0]==0]))

        logfile.write("on iteration="+str(iteration)+" done processing data for step 2, time elapsed="+str(time.time()-starttime)+"\n")
        logfile.write("MC events = "+str(len(X_train_1[Y_train_1[:,0]==1]))+"\n")
        logfile.write("MC events = "+str(len(X_train_1[Y_train_1[:,0]==0]))+"\n")

        # step 2
        # opt = tf.keras.optimizers.Adam(learning_rate=5e-6)
        opt = tf.keras.optimizers.Adam(learning_rate=1e-4)
        mymodel.compile(loss=weighted_binary_crossentropy,
                          optimizer=opt,
                          metrics=['accuracy'])

        hist_s2 =  mymodel.fit(X_train_2, Y_train_2,
                               epochs=nepochs,
                               batch_size=100000,
                               validation_data=(X_test_2, Y_test_2),
                               callbacks=[earlystopping],
                               verbose=1)

        print("on iteration=",iteration," finished step 2; time elapsed=",time.time()-starttime)
        logfile.write("on iteration="+str(iteration)+" finished step 2; time elapsed="+str(time.time()-starttime)+"\n")


        NNweights_step2_hold = mymodel.predict(scaler_data.transform(theta0_G),
                                               batch_size=10000)
        NNweights_step2_hold = NNweights_step2_hold/(1.-NNweights_step2_hold)
        NNweights_step2_hold = NNweights_step2_hold[:, 0]
        NNweights_step2_hold = np.squeeze(np.nan_to_num(NNweights_step2_hold, posinf=1))
        NNweights_step2_hold[pass_truth == 0] = 1.
        NNweights_step2 = NNweights_step2_hold*NNweights_step2

        # Save Model
        tf.keras.models.save_model(mymodel,f"{output_dir}/{ID}_Iteration{iteration}_Pass{pass_i}"+"_model")

        # Now, let's do some checking.

        ###
        # Loss
        ###

        fig = plt.figure(figsize=(7, 5))
        gs = gridspec.GridSpec(1, 1, height_ratios=[1])
        ax0 = plt.subplot(gs[0])
        ax0.yaxis.set_ticks_position('both')
        ax0.xaxis.set_ticks_position('both')
        ax0.tick_params(direction="in", which="both")
        ax0.minorticks_on()
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)

        plt.plot(np.array(hist_s2.history['loss']), label="loss")
        plt.plot(np.array(hist_s2.history['val_loss']), label="val. loss", ls=":")
        plt.xlabel("Epoch", fontsize=20)
        plt.ylabel("Loss", fontsize=20)
        plt.title("OmniFold iteration "+str(iteration)+", step 2",loc="left",fontsize=20)
        plt.text(0.05, 1.15, 'H1', horizontalalignment='center', verticalalignment='center', transform = ax0.transAxes, fontsize=25, fontweight='bold')
        plt.legend(frameon=False, fontsize=20)
        plt.locator_params(axis='x', nbins=5)
        plt.xscale("log")
        fig.savefig(f"{output_dir}/{ID}_Iteration{iteration}_Pass{pass_i}_Step2_loss.pdf", bbox_inches='tight')


        print("done with the "+str(iteration)+"iteration, time elapsed=",time.time()-starttime)
        logfile.write("done with the "+str(iteration)+"iteration, time elapsed="+str(time.time()-starttime)+"\n")

        tensorflow.keras.backend.clear_session()
        del mymodel

        # end OF Loop

        print("\n\nStep 2 Weights = ", NNweights_step2,"\n\n")
        print(f"Step 2 Weights Mean: {np.mean(NNweights_step2)}")
        print(f"Step 2 Weights STD: {np.std(NNweights_step2)}")

        np.save(f"{output_dir}/{ID}_Iteration{iteration}_Pass{pass_i}_NNweights_step2_Pass.npy",
                NNweights_step2)

    # tf.keras.models.save_model(mymodel,"./models/Jan20_"+sys.argv[1]+"_"+sys.argv[2]+"_"+"Pass%i"%(pass_i)+"model")
logfile.close()
print("FINISHED")

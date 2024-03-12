from __future__ import absolute_import, division, print_function
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt

import horovod.tensorflow.keras as hvd 
import horovod.tensorflow
# hvd.init() #done in hvd_unfolding.py

MASK_VAL = -10

def weighted_binary_crossentropy(y_true, y_pred):
    weights = tf.gather(y_true, [1], axis=1)  # event weights
    y_true = tf.gather(y_true, [0], axis=1)  # actual y_true for loss

    # Clip the prediction value to prevent NaN's and Inf's
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1. - epsilon)
    t_loss = -weights * ((y_true) * K.log(y_pred) +
        (1 - y_true) * K.log(1 - y_pred))
    return K.mean(t_loss)


# def reweight(events, model, verbose=0):
#     f = model.predict(events, batch_size=2000, verbose=verbose)
#     if verbose >= 1:
#         print("\n\nWeights in reweight_function = ",f)
#     weights = f / (1. - f)
#     return np.squeeze(np.nan_to_num(weights))

def reweight(events, model, verbose=0):
    f = np.nan_to_num(model.predict(events, batch_size=2000, verbose=verbose),posinf=1,neginf=0)
    if verbose >= 1:
        print("\n\nWeights in reweight_function = ", f)
    weights = f / (1. - f)
    # weights = weights[:,0]
    return np.squeeze(np.nan_to_num(weights, posinf=1))

class MultiFold():
    def __init__(self,
                 num_observables, iterations,
                 theta0_G, theta0_S, 
                 theta_unknown_S,
                 ID, ID_File, save_dir,
                 n_epochs=1000,
                 weights_MC_sim=None, 
                 weights_MC_data=None,
                 verbose=1):

        self.num_observables = num_observables 
        self.iterations = iterations 
        self.theta0_G = theta0_G
        self.theta0_S = theta0_S 
        self.theta_unknown_S = theta_unknown_S
        self.ID = ID
        self.ID_File = ID_File
        self.save_dir = save_dir
        self.n_epochs = n_epochs
        self.weights_MC_sim = weights_MC_sim
        self.weights_MC_data = weights_MC_data
        self.verbose = verbose

        print("\nIN MULTIFOLD CLASS INIT")
        print( "NaN in Theta0_S =        ", np.nan in theta0_S )
        print( "NaN in Theta0_G =        ", np.nan in theta0_G )
        print( "NaN in theta_unknown_S = ", np.nan in theta_unknown_S )
        print(np.nanmean(theta0_S))
        print("="*50)

        # def multifold_hvd(num_observables, iterations,
        #       theta0_G, theta0_S, theta_unknown_S,
        #       n_epochs=1000,
        #       weights_MC_sim=None, weights_MC_data=None,
        #       verbose=1):


    def unfold(self):

        print(f"HVD in unfold: {hvd.rank()+1} / {hvd.size()}")

        if self.weights_MC_sim is None:
            self.weights_MC_sim = np.ones(len(self.theta0_S))

        if self.weights_MC_data is None:
            self.weights_MC_data = np.ones(len(self.theta_unknown_S))


        self.theta0 = np.stack([self.theta0_G, self.theta0_S], axis=1)
        labels0 = np.zeros(len(self.theta0))
        labels1 = np.ones(len(self.theta0_G))
        labels_unknown = np.ones(len(self.theta_unknown_S))

        xvals_1 = np.concatenate((self.theta0_S, self.theta_unknown_S))
        yvals_1 = np.concatenate((labels0, labels_unknown))

        xvals_2 = np.concatenate((self.theta0_G, self.theta0_G))
        yvals_2 = np.concatenate((labels0, labels1))

        weights = np.empty(shape=(self.iterations, 2, len(self.theta0_G)))
        models = {}

        inputs = Input((self.num_observables, ))
        hidden_layer_1 = Dense(50, activation='relu')(inputs)
        dropoutlayer = Dropout(0.1)(hidden_layer_1)
        hidden_layer_2 = Dense(100, activation='relu')(dropoutlayer)
        hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
        outputs = Dense(1, activation='sigmoid')(hidden_layer_3)
        model = Model(inputs=inputs, outputs=outputs)
        # model2 = Model(inputs=inputs, outputs=outputs)

        # Two separate models ensures less bias

        mean_mc_sim = np.mean(self.weights_MC_sim)
        mean_mc_sim = 1.0
        weights_pull = self.weights_MC_sim/mean_mc_sim
        weights_push = self.weights_MC_sim/mean_mc_sim

        history = {}
        history['step1'] = []
        history['step2'] = []

        # Horovod Optimizer. Scale LR by number of workers, hvd.size()
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-6 * hvd.size()) #sqrt(size())
        optimizer = hvd.DistributedOptimizer(optimizer)


        model.compile(loss=weighted_binary_crossentropy,
                       optimizer=tf.keras.optimizers.Adam(learning_rate=2e-6),
                       metrics=['accuracy'])

        # model.compile(loss=weighted_binary_crossentropy,
        #                optimizer=tf.keras.optimizers.Adam(learning_rate=2e-6),
        #                metrics=['accuracy'])
        # # Not using 2-model thing for now


        for i in range(self.iterations):
            if hvd.rank() == 0:
                print("ITERATION: {}".format(i + 1))
                print("STEP 1...")

            weights_1 = np.concatenate((weights_push, self.weights_MC_data))

            X_train_1, X_test_1, Y_train_1, Y_test_1, w_train_1, w_test_1 = \
            train_test_split(xvals_1, yvals_1, weights_1)

            if hvd.rank() == 0:
                print("NaN in X_train_1 = ", np.nan in X_train_1)
                print("NaN in Y_train_1 = ", np.nan in Y_train_1)
                print("NaN in w_train_1 = ", np.nan in w_train_1)
                print()
                print("NaN in X_test_1 = ", np.nan in X_test_1)
                print("NaN in Y_test_1 = ", np.nan in Y_test_1)
                print("NaN in w_test_1 = ", np.nan in w_test_1)

            Y_train_1 = np.stack((Y_train_1, w_train_1), axis=1)
            Y_test_1 = np.stack((Y_test_1, w_test_1), axis=1)

            # X_train_1, X_test_1, Y_train_1, Y_test_1, w_train_1, w_test_1 = \
            # train_test_split(xvals_1, yvals_1, weights_1, 
            #                  test_size=0.1, random_state=42)

            if hvd.rank() == 0:
                print("weights_pull Step 1 mean = ", np.mean(weights_pull))
                print("weights_1 mean = ", np.mean(weights_1))
                print("w_train_1 mean = ", np.mean(w_train_1))
                print("Y_train_1 mean = ", np.mean(Y_train_1))
                print("="*20, "Running model step 1 fit", "="*20)


            batch_size = 2000

            self.verbose = 2 if hvd.rank()==0 else 0

            callbacks = [
                hvd.callbacks.BroadcastGlobalVariablesCallback(0), 
                #check if the early stopping is avg of all or just 1 gpu
                hvd.callbacks.MetricAverageCallback(),
                ReduceLROnPlateau(patience=5, min_lr=1e-7,verbose=self.verbose), #cos schedule might cool
                # EarlyStopping(patience=20, restore_best_weights=True)]
                EarlyStopping(
                    monitor='val_loss', 
                    min_delta=0.0005,
                    patience=10,
                    mode='auto',
                    verbose=2,
                    baseline=None)]


            hist_s1 = model.fit(X_train_1[X_train_1[:, 0] != MASK_VAL],
                                Y_train_1[X_train_1[:, 0] != MASK_VAL],
                                epochs=self.n_epochs,
                                batch_size=batch_size,
                                validation_data=(X_test_1[X_test_1[:, 0] != MASK_VAL],
                                                 Y_test_1[X_test_1[:, 0] != MASK_VAL]),
                                callbacks=callbacks,
                                verbose=self.verbose)



            if hvd.rank() == 0:
                print("Weights before setting MASK = ",weights_pull[self.theta0_S[:,0] == MASK_VAL])
                fig = plt.figure(figsize=(10,7))
                plt.hist(weights_pull[self.theta0_S[:,0] == MASK_VAL],bins=np.linspace(0,2,20))
                plt.savefig(f"plots/{self.ID}_Iter{i}_step1_Weights.png")

            # weights_pull = weights_push * \
            #     reweight(self.theta0_S, model, self.verbose)
            # OLD
            # weights_pull[self.theta0_S[:, 0] == MASK_VAL] = 1.
            # NEW
            # weights_pull[self.theta0_S[:, 0] == MASK_VAL] = weights_push[self.theta0_S[:, 0] == MASK_VAL]

            if hvd.rank() == 0:
                print("\n\nSTEP 1 REWEIGHT")

            new_weights = reweight(self.theta0_S, model, self.verbose)
            new_weights[self.theta0_S[:, 0] == MASK_VAL] = 1.
            weights_pull = weights_push * new_weights

            if hvd.rank() == 0:
                history['step1'].append(hist_s1)
                weights[i, :1, :] = weights_pull
                models[i, 1] = model.get_weights()
                print("STEP 2...")

            weights_2 = np.concatenate((self.weights_MC_sim, weights_pull))
            # having the original weights_MC_sim here means it learns
            # the weights FROM SCRATCH

            X_train_2, X_test_2, Y_train_2, Y_test_2, w_train_2, w_test_2 = \
            train_test_split(xvals_2, yvals_2, weights_2)

            if hvd.rank() == 0:
                print("NaN in X_train_2 = ", np.nan in X_train_2)
                print("NaN in Y_train_2 = ", np.nan in Y_train_2)
                print("NaN in w_train_2 = ", np.nan in w_train_2)
                print()
                print("NaN in X_test_2 = ", np.nan in X_test_2)
                print("NaN in Y_test_2 = ", np.nan in Y_test_2)
                print("NaN in w_test_2 = ", np.nan in w_test_2)

            # zip ("hide") the weights with the labels
            Y_train_2 = np.stack((Y_train_2, w_train_2), axis=1)
            Y_test_2 = np.stack((Y_test_2, w_test_2), axis=1)

            if hvd.rank() == 0:
                print("weights_pull mean = ", np.mean(weights_pull))
                print("weights_2 mean = ", np.mean(weights_2))
                print("w_train_2 mean = ", np.mean(w_train_2))
                print("Y_train_2 mean = ", np.mean(Y_train_2))
                print("="*20, "Running model step2 fit", "="*20)

            if hvd.rank() == 0:
                model_name = f'{self.save_dir}/{self.ID}/{self.ID_File}/iter_{i}_step2_checkpoint'
                callbacks.append(ModelCheckpoint(model_name, monitor='val_loss', save_best_only=True,
                                                 mode='auto',period=1,save_weights_only=True))

            hist_s2 = model.fit(X_train_2[X_train_2[:, 0] != MASK_VAL],
                                Y_train_2[X_train_2[:, 0] != MASK_VAL],
                                epochs=self.n_epochs,
                                batch_size=batch_size,
                                validation_data=(X_test_2[X_test_2[:, 0] != MASK_VAL],
                                                 Y_test_2[X_test_2[:, 0] != MASK_VAL]),
                                callbacks=callbacks,
                                verbose=self.verbose)

            #steps per epoch
            # steps_per = nevents / (ngpus * batch_size)

            if hvd.rank() == 0:
                print("\n\nSTEP 2 REWEIGHT")
            weights_push = self.weights_MC_sim * \
                reweight(self.theta0_G, model, self.verbose)
            #all gpus see the same model. The Data is split.
            #important: Here we can see that we are applying MC Weights!
            # no need to do this again later!

            fig = plt.figure(figsize=(10,7))
            plt.hist(weights_push[self.theta0_S[:,0] == MASK_VAL],bins=np.linspace(0,2,20))
            plt.savefig(f"plots/{self.ID}_Iter{i}_step1_Weights.png")

            if hvd.rank() == 0:
                print(f"Weights for Iteration {i} =", weights[i, 1:2, :])

            models[i, 2] = model.get_weights() #same model on ALL GPUS. HOROVOD FTW! 

            weights[i, 1:2, :] = weights_push*mean_mc_sim #plot and use. Take a look at weights_push normalization
            # weights[i, 1:2, :] = weights_push
            # V keeps normalization the same, ensuring the cross section doesn't change. 
            # want it to be the same as the sum of mc_weights.
            history['step2'].append(hist_s2)

        K.clear_session()

        return weights, models, history

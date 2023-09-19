from __future__ import absolute_import, division, print_function
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow.keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense, Input, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping

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


def multifold(num_observables, iterations,
              theta0_G, theta0_S, theta_unknown_S,
              n_epochs=1000,
              weights_MC_sim=None, weights_MC_data=None,
              verbose=1):

    print("\nIN MULTIFOLD FUNCTION")
    print( "NaN in Theta0_S = ", np.isnan(theta0_S).any() )
    print( "NaN in Theta0_G = ", np.isnan(theta0_G).any() )
    print( "NaN in theta_unknown_S = ", np.isnan(theta_unknown_S).any() )
    print(np.nanmean(theta0_S))
    print("="*60)

    if weights_MC_sim is None:
        weights_MC_sim = np.ones(len(theta0_S))

    if weights_MC_data is None:
        weights_MC_data = np.ones(len(theta_unknown_S))

    theta0 = np.stack([theta0_G, theta0_S], axis=1)
    labels0 = np.zeros(len(theta0))
    # theta_unknown = np.stack([theta_unknown_S, theta_unknown_S], axis=1)
    labels1 = np.ones(len(theta0_G))
    labels_unknown = np.ones(len(theta_unknown_S))


    xvals_1 = np.concatenate((theta0_S, theta_unknown_S))
    yvals_1 = np.concatenate((labels0, labels_unknown))

    xvals_2 = np.concatenate((theta0_G, theta0_G))
    yvals_2 = np.concatenate((labels0, labels1))

    weights = np.empty(shape=(iterations, 2, len(theta0_G)))
    models = {}

    inputs = Input((num_observables, ))
    hidden_layer_1 = Dense(50, activation='relu')(inputs)
    dropoutlayer = Dropout(0.1)(hidden_layer_1)
    # hidden_layer_2 = Dense(50, activation='relu')(dropoutlayer)
    hidden_layer_2 = Dense(100, activation='relu')(dropoutlayer)
    hidden_layer_3 = Dense(50, activation='relu')(hidden_layer_2)
    outputs = Dense(1, activation='sigmoid')(hidden_layer_3)
    model = Model(inputs=inputs, outputs=outputs)

    earlystopping = EarlyStopping(patience=5,
                                  verbose=verbose,
                                  restore_best_weights=True)

    # from NN (DCTR)
    def reweight(events):
        f = model.predict(events, batch_size=2000)
        print("\nf IN REWEIGHT = ",f)
        weights = f / (1. - f)
        print("\nWEIGHTS IN REWEIGHT = ",weights)
        return np.squeeze(np.nan_to_num(weights))

    weights_pull = weights_MC_sim
    weights_push = weights_MC_sim

    # weights_pull = np.ones(len(theta0_S))
    # weights_push = np.ones(len(theta0_S))

    history = {}
    history['step1'] = []
    history['step2'] = []

    for i in range(iterations):
        print("ITERATION: {}".format(i + 1))
        print("STEP 1...")

        weights_1 = np.concatenate((weights_push, weights_MC_data))

        X_train_1, X_test_1, Y_train_1, Y_test_1, w_train_1, w_test_1 = \
            train_test_split(xvals_1, yvals_1, weights_1)

        print("Nan in X, Y, weights1")
        print(np.isnan(X_train_1).any())
        print(np.isnan(Y_train_1).any())
        print(np.isnan(weights_1).any())
        print()

        Y_train_1 = np.stack((Y_train_1, w_train_1), axis=1)
        Y_test_1 = np.stack((Y_test_1, w_test_1), axis=1)

        print("X_train_1 = ", X_train_1)
        print("Y_train_1 = ", Y_train_1)
        print("w_train_1 = ", w_train_1)
        print("xvals_1 = ", xvals_1)
        print("yvals_1 = ", yvals_1, "\n")

        print("weights_pull Step 1 mean = ", np.mean(weights_pull))
        print("weights_1 mean = ", np.mean(weights_1))
        print("w_train_1 mean = ", np.mean(w_train_1))
        print("Y_train_1 mean = ", np.mean(Y_train_1))

        batch_size = 2000
        model.compile(loss=weighted_binary_crossentropy,
                      optimizer=tf.keras.optimizers.Adam(learning_rate=2e-6),
                      # optimizer='Adam',
                      metrics=['accuracy'])

        # print("X Train var0 = ",X_train_1[:,0])
        # print("X Train var1 = ",X_train_1[:,1])

        hist_s1 = model.fit(X_train_1[X_train_1[:, 0] != MASK_VAL],
                            Y_train_1[X_train_1[:, 0] != MASK_VAL],
                            epochs=n_epochs,
                            batch_size=batch_size,
                            validation_data=(X_test_1[X_test_1[:, 0] != MASK_VAL],
                                             Y_test_1[X_test_1[:, 0] != MASK_VAL]),
                            callbacks=[earlystopping],
                            verbose=verbose)

        # history['step1'].append(hist_s1)
        print("...STEP1")
        weights_pull = weights_push * reweight(theta0_S)
        weights_pull[theta0_S[:, 0] == MASK_VAL] = 1
        weights[i, :1, :] = weights_pull
        print("\nWEIGHTS PULL = ", weights_pull)
        models[i, 1] = model.get_weights()

        print("STEP 2...")
        weights_2 = np.concatenate((weights_MC_sim, weights_pull))
        # having the original weights_MC_sim here means it learns
        # the weights FROM SCRATCH

        # weights_2 = np.concatenate((np.ones(len(theta0_G)), weights_pull))


        X_train_2, X_test_2, Y_train_2, Y_test_2, w_train_2, w_test_2 = \
            train_test_split(xvals_2, yvals_2, weights_2)

        # zip ("hide") the weights with the labels
        Y_train_2 = np.stack((Y_train_2, w_train_2), axis=1)
        Y_test_2 = np.stack((Y_test_2, w_test_2), axis=1)

        print("weights_pull mean = ", np.mean(weights_pull))
        print("weights_2 mean = ", np.mean(weights_2))
        print("w_train_2 mean = ", np.mean(w_train_2))
        print("Y_train_2 mean = ", np.mean(Y_train_2))

        model.compile(loss=weighted_binary_crossentropy,
                      optimizer='Adam',
                      metrics=['accuracy'])

        hist_s2 = model.fit(X_train_2[X_train_2[:, 0] != MASK_VAL],
                            Y_train_2[X_train_2[:, 0] != MASK_VAL],
                            epochs=n_epochs,
                            batch_size=batch_size,
                            validation_data=(X_test_2[X_test_2[:, 0] != MASK_VAL],
                                             Y_test_2[X_test_2[:, 0] != MASK_VAL]),
                            callbacks=[earlystopping],
                            verbose=verbose)

        # history['step2'].append(hist_s2)

        # weights_push = reweight(theta0_G)
        print("...STEP2")
        weights_push = weights_MC_sim * reweight(theta0_G)

        weights[i, 1:2, :] = weights_push
        print(f"Weights for Iteration {i} =", weights[i, 1:2, :])
        models[i, 2] = model.get_weights()

    K.clear_session()
    return weights, models, history

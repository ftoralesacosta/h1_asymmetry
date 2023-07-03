import sys

import numpy as np
import pandas as pd
import gc
import time
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import h5py
from unfold import weighted_binary_crossentropy

import uproot as ur

def npy_from_pkl(label,load_NN=True,pkl_path="",model_path="",keys=[]):

    if (pkl_path == ""):
        pkl_path = "/clusterfs/ml4hep/yxu2/unfolding_mc_inputs/"

    if (model_path == ""):
        model_path = "/clusterfs/ml4hep_nvme2/ftoralesacosta/disjets/inputfiles/fullscan3/models/"

    #DATA LOADING
    if ".pkl" in pkl_path:
        print("Loading PKL directly from argument: ",pkl_path)
        mc = pd.read_pickle(pkl_path)

    else:
        print("Loading PKL "+pkl_path+label+".pkl")
        mc = pd.read_pickle(pkl_path+label+".pkl") 




    leading_jets_only = True
    if (leading_jets_only):
        njets_tot = len(mc["e_px"])
        mc = mc.loc[(slice(None),0), :]
        print("Number of subjets cut = ",njets_tot-len(mc["jet_pt"])," / ",len(mc["jet_pt"]))

    if not keys:
        keys = ['gene_px','gene_py','gene_pz','genjet_pt','genjet_eta',
                   'genjet_phi','genjet_dphi','genjet_qtnorm']

    theta0_G = mc[keys].to_numpy()
    weights_MC_sim = mc['wgt']
    pass_reco = np.array(mc['pass_reco'])
    pass_truth = np.array(mc['pass_truth'])
    pass_fiducial = np.array(mc['pass_fiducial'])
    del mc
    _ = gc.collect()


    #WEIGHTS
    NNweights_step2 = np.ones(len(theta0_G))

    if (load_NN):
        print("Loading Data for standard scalar: "+pkl_path+"Data_nominal.pkl")
        data = pd.read_pickle(pkl_path+"Data_nominal.pkl")

        if (leading_jets_only):
            data = data.loc[(slice(None),0), :]

        theta_unknown_S = data[['e_px','e_py','e_pz','jet_pt','jet_eta',
                            'jet_phi','jet_dphi','jet_qtnorm']].to_numpy()
        scaler_data = StandardScaler()
        scaler_data.fit(theta_unknown_S)
        del data
        _ = gc.collect()

        print("Loading Model "+model_path+label)

        for i in range(5):

            print("Iteration"+str(i)+" Step 2")
                
            mymodel = tf.keras.models.load_model(model_path+label+"_iteration"+str(i)+"_step2",compile=False)
            NNweights_step2_hold = mymodel.predict(scaler_data.transform(theta0_G),batch_size=10000)
            NNweights_step2_hold = NNweights_step2_hold/(1.-NNweights_step2_hold)
            NNweights_step2_hold = NNweights_step2_hold[:,0]
            NNweights_step2_hold = np.squeeze(np.nan_to_num(NNweights_step2_hold,posinf=1))
            NNweights_step2_hold[pass_truth==0] = 1.
            NNweights_step2 = NNweights_step2_hold*NNweights_step2

    weights = weights_MC_sim*NNweights_step2

    # KINEMATICS
    q_perp_mag, jet_pT_mag, asymm_phi, jet_qT_norm = get_kinematics(theta0_G)
    
    # CUTS
    cuts = get_cuts(pass_fiducial, q_perp_mag, jet_pT_mag, asymm_phi, jet_qT_norm)


    np.save('npy_files/'+label+'_cuts.npy',cuts)
    np.save('npy_files/'+label+'_jet_pT.npy',jet_pT_mag)
    np.save('npy_files/'+label+'_q_perp.npy',q_perp_mag)
    np.save('npy_files/'+label+'_asymm_angle.npy',asymm_phi)
    np.save('npy_files/'+label+'_weights.npy',weights)
    np.save('npy_files/'+label+'_nn_weights.npy',NNweights_step2)
    np.save('npy_files/'+label+'_mc_weights.npy',weights_MC_sim)


#Primarily for Loading ROOT files, e.g. PYTHIA
def get_npy_from_ROOT(label,file_name="",tree_name="Tree",keys=[]):
    #DATA LOADING
    print("Loading ROOT Tree  "+file_name+":"+tree_name)
    events = ur.open("%s:%s"%(file_name,tree_name))

    if not keys:
        keys = ['gen_lep_px','gen_lep_py','gen_lep_pz','gen_jet_pt','gen_jet_eta', 'gen_jet_phi']
        #get_kinematics expects variables in specific order... should have passed a dictionary...
        #we do not use 'dphi' and 'qt_norm' in pythia (or in general for this analysis)

    print("Looking for Keys:  ",keys)
    print("Keys from ROOT file:  ", events.keys())

    mc = events.arrays(library="pd")

    theta0_G = mc[keys].to_numpy()
    q_perp_mag, jet_pT_mag, asymm_phi, jet_qT_norm = get_kinematics(theta0_G)
    pass_fiducial = np.ones(len(theta0_G[:,0]))
    jet_qT_norm = q_perp_mag/np.sqrt(mc["Q2"].to_numpy())
    cuts = get_cuts(pass_fiducial, q_perp_mag, jet_pT_mag, asymm_phi, jet_qT_norm)

    weights_MC_sim = mc['weight']
    weights = weights_MC_sim

    np.save('npy_files/'+label+'_cuts.npy',cuts)
    np.save('npy_files/'+label+'_jet_pT.npy',jet_pT_mag)
    np.save('npy_files/'+label+'_q_perp.npy',q_perp_mag)
    np.save('npy_files/'+label+'_asymm_angle.npy',asymm_phi)
    np.save('npy_files/'+label+'_weights.npy',weights)
    np.save('npy_files/'+label+'_mc_weights.npy',weights_MC_sim)

def get_kinematics(theta0_G):
    print("Calculating q_perp, asymm_phi, and jet_pT")

    e_px = theta0_G[:,0]
    e_py = theta0_G[:,1]
    e_pT = np.array([e_px,e_py])

    jet_pT_mag = theta0_G[:,3]
    jet_phi = theta0_G[:,5]
    
    jet_px = np.multiply(jet_pT_mag, np.cos(jet_phi))
    jet_py = np.multiply(jet_pT_mag, np.sin(jet_phi))
    jet_pT = np.array([jet_px,jet_py])

    
    q_perp_vec = jet_pT + e_pT
    q_perp_mag = np.linalg.norm(q_perp_vec,axis=0)
    P_perp_vec = (e_pT-jet_pT)/2
    P_perp_mag = np.linalg.norm(P_perp_vec,axis=0)

    q_dot_P = q_perp_vec[0,:]*P_perp_vec[0,:] + q_perp_vec[1,:]*P_perp_vec[1,:]
    
    cosphi = (q_dot_P)/(q_perp_mag*P_perp_mag)
    asymm_phi = np.arccos(cosphi)


    #For consistency with previous analysis
    if np.shape(theta0_G)[1]>7:
        jet_qT_norm = theta0_G[:,7] #[not to be confused with q_Perp!]

    else: 
        jet_qT_norm = np.ones(len(theta0_G[:,0]))
        print("WARNING: jet_qT_norm set to {1.0}. Be careful cutting on this!\n")
        #jet_qT norm grandfathered in from the disjets repo
        #see https://github.com/miguelignacio/disjets/blob/1ed6f8f4d572e2bc1d7916a6cc1491fb05e2f176/FinalReading/dataloader.py#L109
        #temp.eval('jet_qtnorm = jet_qt/sqrt(Q2)', inplace=True

    print("Mean q_perp_mag = ",np.mean(q_perp_mag))
    return q_perp_mag, jet_pT_mag, asymm_phi, jet_qT_norm


def get_cuts(pass_fiducial, q_perp_mag, jet_pT_mag, asymm_phi, jet_qT):
    print("Getting Cut Mask")

    # pT_cut = jet_pT_mag > 10.
    pT_cut = jet_pT_mag > 20. #Test only for Feb 17
    q_over_pT_cut = q_perp_mag/jet_pT_mag < 0.3 #Kyle guessed ~0.3, may need variation
    qT_cut = np.where((jet_qT<0.25), True, False)
    phi_nan_cut = ~np.isnan(asymm_phi)
    #q_perp_cut = q_perp_mag < 10.0 #q_perp_max
    
    cut_arrays = [pass_fiducial,
        pT_cut,
        q_over_pT_cut,
        qT_cut,
        phi_nan_cut]
    
    cuts = np.ones(len(pT_cut))
    
    for cut in cut_arrays:
        cuts = np.logical_and(cuts,cut)

    print("Cut Length OK = ",len(q_perp_mag)==len(cuts))

    return cuts

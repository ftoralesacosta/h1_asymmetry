import sys

import numpy as np
import pandas as pd
import gc
import time
# import tensorflow as tf
# from sklearn.preprocessing import StandardScaler
import h5py
# from unfold import weighted_binary_crossentropy

import uproot as ur

# 1 make sure q2 is obtained
# 2 load nn weights from new npy files
# 3 write averaging function
# 4 make sure Q2 cuts applied in get cuts
# 5 save np files with label from mc_name


def npy_from_pkl(label, load_NN=True, pkl_path="", mc="Rapgap", type="nominal", keys=[]):

    if (pkl_path == ""):
        pkl_path = "/pscratch/sd/f/fernando/h1_data"

    #DATA LOADING
    if ".pkl" in pkl_path:
        print("Loading PKL directly from argument: ",pkl_path)
        mc = pd.read_pickle(pkl_path)

    else:
        print("Loading PKL ", f"{pkl_path}/{mc}_{type}.pkl")
        mc = pd.read_pickle(f"{pkl_path}/{mc}_{type}.pkl") 


    print("MC SHAPE = ", np.shape(mc))
    leading_jets_only = True
    if (leading_jets_only):
        njets_tot = len(mc["e_px"])
        mc = mc.loc[(slice(None),0), :]
        print("Number of subjets cut = ",njets_tot-len(mc["jet_pt"])," / ",len(mc["jet_pt"]))

    print("MC SHAPE = ", np.shape(mc))
    if not keys:
        keys = ['gene_px','gene_py','gene_pz','genjet_pt','genjet_eta',
                   'genjet_phi','genjet_dphi','genjet_qtnorm', 'gen_Q2']

    theta0_G = mc[keys].to_numpy()
    print("shape of theta0_G = ", np.shape(theta0_G) )
    weights_MC_sim = mc['wgt']
    pass_reco = np.array(mc['pass_reco'])
    pass_truth = np.array(mc['pass_truth'])
    pass_fiducial = np.array(mc['pass_fiducial'])
    del mc
    _ = gc.collect()


    home = '/pscratch/sd/f/fernando/h1_models'
    # NN_step2_weights = np.load(f"../weights/{label}_pass_avgs.npy")[-1]
    # NN_step2_weights = np.load(f"../h1_models/{label}/{label}_Pass0_Step2_Weights.npy")[-1]
    NN_step2_weights = np.load(f"{home}/{label}/{label}_Pass0_Step2_Weights.npy")[-1,0]
    # Q^2 Cut already applied to NN weights in unfolding procedure!!!

    # NN_step2_weights = np.load(f"/pscratch/sd/f/fernando/h1_models/Rapgap_nominal_Rapgap_HVDRapgap_Sep25/Rapgap_nominal_Rapgap_HVDRapgap_Sep25_Pass0_Step2_Weights.npy")[-1,0]
    # NN_step2_weights = np.load(f"/pscratch/sd/f/fernando/h1_models/Rapgap_nominal_Rapgap_HVDRapgap_NCCL_Oct17/Rapgap_nominal_Rapgap_HVDRapgap_NCCL_Oct17_Pass0_Step2_Weights.npy")[-1,0]

    # take the last iteration, MultiFold learns from SCRATCH here
    print("WEIGHTS SHAPE = ", np.shape(NN_step2_weights))

    # KINEMATICS
    q_perp_mag, jet_pT_mag, asymm_phi, jet_qT_norm, Q2 = get_kinematics(theta0_G)
    

    # CUTS
    cuts = get_cuts(pass_fiducial, pass_truth, q_perp_mag, jet_pT_mag, 
                    asymm_phi, jet_qT_norm, Q2)

    print("Cuts SHAPE = ", np.shape(cuts))
    weights = weights_MC_sim[cuts] 
    print("SHAPE after Cuts = ", np.shape(weights))
    weights *= NN_step2_weights

    np.save('npy_files/'+label+'_cuts.npy',cuts)
    np.save('npy_files/'+label+'_jet_pT.npy',jet_pT_mag)
    np.save('npy_files/'+label+'_q_perp.npy',q_perp_mag)
    np.save('npy_files/'+label+'_Q2.npy',Q2)
    np.save('npy_files/'+label+'_asymm_angle.npy',asymm_phi)
    np.save('npy_files/'+label+'_weights.npy',weights)
    np.save('npy_files/'+label+'_nn_weights.npy',NN_step2_weights)
    np.save('npy_files/'+label+'_mc_weights.npy',weights_MC_sim)


#Primarily for Loading ROOT files, e.g. PYTHIA
def get_npy_from_ROOT(label,file_name="",tree_name="Tree",keys=[]):

    #DATA LOADING
    print("Loading ROOT Tree  "+file_name+":"+tree_name)
    events = ur.open("%s:%s"%(file_name,tree_name))

    if not keys:
        keys = ['gen_lep_px','gen_lep_py','gen_lep_pz',
                'gen_jet_pt','gen_jet_eta', 'gen_jet_phi', 'Q2']

        # get_kinematics expects variables in specific order... 
        # should have passed a dictionary...
        # we do not use 'dphi' and 'qt_norm' in pythia 
        # (or in general for this analysis)

    print("Looking for Keys:  ",keys)
    print("Keys from ROOT file:  ", events.keys())

    mc = events.arrays(library="pd")

    theta0_G = mc[keys].to_numpy()
    q_perp_mag, jet_pT_mag, asymm_phi, jet_qT_norm, Q2 = get_kinematics(theta0_G)
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

    Q2 = theta0_G[:,-1]
    
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
    if np.shape(theta0_G)[1]>7+1:  #+1 added Q^2 Oct 2023
        jet_qT_norm = theta0_G[:,7] #[not to be confused with q_Perp!]

    else: 
        jet_qT_norm = np.ones(len(theta0_G[:,0]))
        print("WARNING: jet_qT_norm set to {1.0}. Be careful cutting on this!\n")
        #jet_qT norm grandfathered in from the disjets repo
        #see https://github.com/miguelignacio/disjets/blob/1ed6f8f4d572e2bc1d7916a6cc1491fb05e2f176/FinalReading/dataloader.py#L109
        #temp.eval('jet_qtnorm = jet_qt/sqrt(Q2)', inplace=True

    print("Mean q_perp_mag = ",np.mean(q_perp_mag))
    print("Mean Q^2 = ",np.mean(Q2))
    print("Mean jet_qT_norm = ",np.mean(jet_qT_norm))
    return q_perp_mag, jet_pT_mag, asymm_phi, jet_qT_norm, Q2


def get_cuts(pass_fiducial, pass_truth, q_perp_mag, jet_pT_mag, asymm_phi, jet_qT, genQ2=None):
    print("Getting Cut Mask")

    # pT_cut = jet_pT_mag > 10.
    pT_cut = jet_pT_mag > 20. #Test only for Feb 17
    q_over_pT_cut = q_perp_mag/jet_pT_mag < 0.3 #Kyle guessed ~0.3, may need variation
    # qT_cut = np.where((jet_qT<0.25), True, False)
    qT_cut = jet_qT < 0.25
    phi_nan_cut = ~np.isnan(asymm_phi)
    # print("PHI Not NaN",np.any(phi_nan_cut))

    # q_perp_cut = q_perp_mag < 10.0 #q_perp_max
    
    jet_pT_mag_nan = ~np.isnan(jet_pT_mag)
    Q2_cut = genQ2 > 100

    cut_arrays = [
        Q2_cut,
        jet_pT_mag_nan,
        pass_truth,
        pass_fiducial,
        pT_cut,
        q_over_pT_cut,
        qT_cut,
        phi_nan_cut]
    
    cuts = np.ones(len(pT_cut))
    
    # print("Length Test = ", len(jet_pT_mag[cuts]))
    for cut in cut_arrays:
        cuts = np.logical_and(cuts,cut)
        print(np.any(cuts))
        print("Length = ", np.shape(jet_pT_mag[cuts]))

    print("Cut Length OK = ",len(q_perp_mag)==len(cuts))

    print('Mean Q2 = ', np.mean(genQ2))
    print('Min Q2 after other cuts = ', np.min(genQ2[cuts]) )
    if genQ2 is not None:
        Q2_cut = genQ2 > 100
        print("Cutting on Q^2")
        cuts = np.logical_and(cuts, Q2_cut)

    return cuts

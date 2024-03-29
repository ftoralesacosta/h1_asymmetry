import sys
import numpy as np
import pandas as pd
import gc
import uproot as ur
import yaml
from unfold import MASK_VAL
print("MASK VAL = ", MASK_VAL)

# 1 make sure q2 is obtained
# 2 load nn weights from new npy files
# 3 write averaging function
# 4 make sure Q2 cuts applied in get cuts
# 5 save np files with label from mc_name


def npy_from_pkl(label, main_dir=".", pass_avg=True, suffix="", 
                 load_NN=True, pkl_path="", mc="Rapgap", 
                 run_type="nominal", use_theta0_S = False, keys=[]):

    if (pkl_path == ""):
        pkl_path = "/pscratch/sd/f/fernando/h1_data"
        # pkl_path = "/clusterfs/ml4hep/yxu2/unfolding_mc_inputs/"

    if 'bootstrap' in run_type:
        run_type = 'nominal'

    # DATA LOADING
    if ".pkl" in pkl_path:
        print("Loading PKL directly from argument: ", pkl_path)
        mc = pd.read_pickle(pkl_path)

    else:
        print("Loading PKL ", f"{pkl_path}/{mc}_{run_type}.pkl")
        mc = pd.read_pickle(f"{pkl_path}/{mc}_{run_type}.pkl")

    print("MC SHAPE = ", np.shape(mc))
    leading_jets_only = True
    if (leading_jets_only):
        njets_tot = len(mc["e_px"])
        mc = mc.loc[(slice(None), 0), :]
        print("Number of subjets cut = ",
              njets_tot-len(mc["jet_pt"]), " / ", len(mc["jet_pt"]))

    if not keys:
        keys = ['gene_px', 'gene_py', 'gene_pz', 'genjet_pt', 'genjet_eta',
                'genjet_phi', 'genjet_dphi', 'genjet_qtnorm', 'gen_Q2']

    # print("First 5 Events = ", mc.head(5))
    # print("Last 5 Events = ", mc.tail(5))

    print("*"*30)

    NN_step2_weights = mc['wgt']
    if load_NN:
        NN_step2_weights = np.load(f"./weights/{label}_pass_avgs.npy")[-1]
        # FIXME: There's a bug in checkpointing the last iteration...
        # See the inference script, but theres something wrong with the LAST checkpoint
        # Regardless of how many iterations. This is still 5 passes, just iter N-1
        # NN_step2_weights = np.load(f"./weights/{label}_pass_avgs.npy")[0]  #Test March 12 2024

    # if pass_avg:  # depricated March 2024 (will always want pass avg...
    #     NN_step2_weights = np.load(f"./weights/{label}_pass_avgs.npy")[-1]
        # Taking the last element grabs the last omnifold ITERATION
        # take the last iteration, MultiFold learns from SCRATCH

    print("WEIGHTS SHAPE = ", np.shape(NN_step2_weights))
    NEVENTS = min(len(NN_step2_weights), len(mc))
    NN_step2_weights = NN_step2_weights[:NEVENTS]
    mc = mc[:NEVENTS]
    # ^ ensures the lengths are the same. ^
    # Was previously checked, if mismatch, the TAIL is
    # cut. So taking the first N events matches eventns to weights


    #FIXME: rm this. Test March 11 2024
    # NN_step2_weights = np.load(f"./weights/{ID}_pass_avgs.npy")[3]

    theta0_G = mc[keys].to_numpy()

    print("shape of theta0_G = ", np.shape(theta0_G) )
    weights_MC_sim = mc['wgt']
    pass_reco = np.array(mc['pass_reco'])
    pass_truth = np.array(mc['pass_truth'])
    pass_fiducial = np.array(mc['pass_fiducial'])


    del mc
    _ = gc.collect()

    # NN_step2_weights = np.load(f"../weights/{label}_pass_avgs.npy")[-1]
    # NN_step2_weights = np.load(f"../h1_models/{label}/{label}_Pass0_Step2_Weights.npy")[-1]

    # Q^2 Cut already applied to NN weights in unfolding procedure!!!

    # NN_step2_weights = np.load(f"/pscratch/sd/f/fernando/h1_models/Rapgap_nominal_Rapgap_HVDRapgap_Sep25/Rapgap_nominal_Rapgap_HVDRapgap_Sep25_Pass0_Step2_Weights.npy")[-1,0]
    # NN_step2_weights = np.load(f"/pscratch/sd/f/fernando/h1_models/Rapgap_nominal_Rapgap_HVDRapgap_NCCL_Oct17/Rapgap_nominal_Rapgap_HVDRapgap_NCCL_Oct17_Pass0_Step2_Weights.npy")[-1,0]

    # take the last iteration, MultiFold learns from SCRATCH here

    # KINEMATICS
    q_perp_mag, jet_pT_mag, asymm_phi, jet_qT_norm, Q2 = get_kinematics(theta0_G)
    print("q_perp = ", q_perp_mag[:10], q_perp_mag[-10:])
    print("asymm_phi = ", asymm_phi[:10], asymm_phi[-10:])
    print("Q2 = ", Q2[:10], Q2[-10:])


    # CUTS
    cuts = get_cuts(pass_fiducial, pass_truth, q_perp_mag, jet_pT_mag,
                    asymm_phi, jet_qT_norm, Q2)


    # weights = weights_MC_sim * NN_step2_weights
    weights = NN_step2_weights  # mc_weights already applied. Look at inference script!

    print("Cuts SHAPE = ", np.shape(cuts))
    print("SHAPE after Cuts = ", np.shape(weights[cuts]))

    if use_theta0_S:
        label = label+"_RECO"


    # npy_dir = '/global/ml4hep/spss/ftoralesacosta/h1_asymmetry/npy_files/'
    npy_dir = main_dir + '/npy_files'
    np.save(f'{npy_dir}/{label}_cuts.npy', cuts)
    np.save(f'{npy_dir}/{label}_jet_pT.npy', jet_pT_mag)
    np.save(f'{npy_dir}/{label}_q_perp.npy', q_perp_mag)
    np.save(f'{npy_dir}/{label}_Q2.npy', Q2)
    np.save(f'{npy_dir}/{label}_asymm_angle.npy', asymm_phi)
    np.save(f'{npy_dir}/{label}_weights.npy', weights)
    np.save(f'{npy_dir}/{label}_nn_weights.npy', NN_step2_weights)
    np.save(f'{npy_dir}/{label}_mc_weights.npy', weights_MC_sim)


# Primarily for Loading ROOT files, e.g. PYTHIA
def get_npy_from_ROOT(label, file_name="", tree_name="Tree", keys=[]):

    #DATA LOADING
    print("Loading ROOT Tree  "+file_name+":"+tree_name)
    events = ur.open("%s:%s" % (file_name, tree_name))

    if not keys:
        keys = ['gen_lep_px', 'gen_lep_py', 'gen_lep_pz',
                'gen_jet_pt', 'gen_jet_eta', 'gen_jet_phi', 'Q2']

        # get_kinematics expects variables in specific order...
        # should have passed a dictionary...
        # we do not use 'dphi' and 'qt_norm' in pythia
        # (or in general for this analysis)

    print("Looking for Keys: ", keys)
    print("Keys from ROOT file: ", events.keys())

    mc = events.arrays(library="pd")

    theta0_G = mc[keys].to_numpy()
    q_perp_mag, jet_pT_mag, asymm_phi, jet_qT_norm, Q2 = get_kinematics(theta0_G)
    pass_fiducial = np.ones(len(theta0_G[:, 0]))
    jet_qT_norm = q_perp_mag/np.sqrt(mc["Q2"].to_numpy())
    cuts = get_cuts(pass_fiducial, q_perp_mag, jet_pT_mag, asymm_phi, jet_qT_norm)

    weights_MC_sim = mc['weight']
    weights = weights_MC_sim

    np.save('npy_files/'+label+'_cuts.npy', cuts)
    np.save('npy_files/'+label+'_jet_pT.npy', jet_pT_mag)
    np.save('npy_files/'+label+'_q_perp.npy', q_perp_mag)
    np.save('npy_files/'+label+'_asymm_angle.npy', asymm_phi)
    np.save('npy_files/'+label+'_weights.npy', weights)
    np.save('npy_files/'+label+'_mc_weights.npy', weights_MC_sim)


def get_kinematics(theta0_G):
    print("Calculating q_perp, asymm_phi, and jet_pT")

    e_px = theta0_G[:, 0]
    e_py = theta0_G[:, 1]
    e_pT = np.array([e_px, e_py])

    jet_pT_mag = theta0_G[:, 3]
    jet_phi = theta0_G[:, 5]

    Q2 = theta0_G[:, -1]

    jet_px = np.multiply(jet_pT_mag, np.cos(jet_phi))
    jet_py = np.multiply(jet_pT_mag, np.sin(jet_phi))
    jet_pT = np.array([jet_px, jet_py])

    q_perp_x = jet_px + e_px
    q_perp_y = jet_py + e_py

    P_perp_x = jet_px - e_px
    P_perp_y = jet_py - e_py

    #NEW
    # q_perp_mag = np.sqrt( np.square(q_perp_x) + np.square(q_perp_y) )
    # P_perp_mag = np.sqrt( np.square(P_perp_x) + np.square(P_perp_y) )

    # q_dot_P = q_perp_x*P_perp_x + q_perp_y*P_perp_y
    # cosphi = (q_dot_P)/(q_perp_mag*P_perp_mag)
    # asymm_phi = np.arccos(cosphi)

    #OLD
    q_perp_vec = jet_pT + e_pT
    q_perp_mag = np.linalg.norm(q_perp_vec, axis=0)
    P_perp_vec = (e_pT-jet_pT)/2
    P_perp_mag = np.linalg.norm(P_perp_vec, axis=0)

    q_dot_P = q_perp_vec[0, :]*P_perp_vec[0, :] + q_perp_vec[1, :]*P_perp_vec[1, :]
    cosphi = (q_dot_P)/(q_perp_mag*P_perp_mag)
    asymm_phi = np.arccos(cosphi)

    # For consistency with previous analysis
    # if np.shape(theta0_G)[1]>7+1:  # +1 added Q^2 Oct 2023 #removed Dec 2023
    jet_qT_norm = theta0_G[:, 7] # [not to be confused with q_Perp!]

    # else: 
    #     jet_qT_norm = np.ones(len(theta0_G[:, 0]))
    #     print("WARNING: jet_qT_norm set to {1.0}. Be careful cutting on this!\n")
    # jet_qT_norm = theta0_G[:, 7] # [not to be confused with q_Perp!]
        # jet_qT norm grandfathered in from the disjets repo
        # see https://github.com/miguelignacio/disjets/blob/1ed6f8f4d572e2bc1d7916a6cc1491fb05e2f176/FinalReading/dataloader.py#L109
        # temp.eval('jet_qtnorm = jet_qt/sqrt(Q2)', inplace=True

    print("Mean q_perp_mag = ", np.mean(q_perp_mag))
    print("Mean Q^2 = ", np.mean(Q2))
    print("Mean jet_qT_norm = ", np.mean(jet_qT_norm))
    return q_perp_mag, jet_pT_mag, asymm_phi, jet_qT_norm, Q2


def get_cuts(pass_fiducial, pass_truth, q_perp_mag,
             jet_pT_mag, asymm_phi, jet_qT, gen_Q2):
    print("Getting Cut Mask")

    # pT_cut = jet_pT_mag > 10.
    # q_over_pT_cut = q_perp_mag/jet_pT_mag < 0.3  # Kyle guessed ~0.3
    # qT_cut = np.where((jet_qT < 0.25), True, False)
    # Q2_cut = gen_Q2 > 150
    # phi_nan_cut = ~np.isnan(asymm_phi)
    # jet_pT_mag_nan = ~np.isnan(jet_pT_mag)

    pT_cut = jet_pT_mag > 10.
    q_over_pT_cut = q_perp_mag/jet_pT_mag < 0.3
    qT_cut = np.where((jet_qT < 0.25), True, False)
    phi_nan_cut = ~np.isnan(asymm_phi)
    jet_pT_mag_nan = ~np.isnan(jet_pT_mag)
    Q2_cut = gen_Q2 > 150

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
    print("Checking Cut Sub Masks")
    for cut in cut_arrays:
        cuts = np.logical_and(cuts, cut)
        # print(np.any(cuts))
        print("Length = ", len(jet_pT_mag[cuts]), "/", len(jet_pT_mag))

    print("Cut Length OK = ", len(q_perp_mag) == len(cuts))
    print("MIN Q^2 AFTER CUT = ", np.min(gen_Q2[cuts]))

    return cuts

# ==============================================================================================
# ==============================================================================================
# ==============================================================================================
# ==============================================================================================
# ==============================================================================================


def npy_from_npy(label, save_dir, pass_avg=True, suffix="", load_NN=True,
                 pkl_path="", mc="Rapgap", use_theta0_S = False,
                 run_type="nominal", keys=[]):

    theta0_G = np.load(f"npy_inputs/{ID}_Theta0_G.npy")
    if use_theta0_S:
        theta0_G = np.load(f"npy_inputs/{ID}_Theta0_S.npy")


    pass_reco = np.load(f"npy_inputs/{ID}_pass_reco.npy")
    pass_truth = np.load(f"npy_inputs/{ID}_pass_truth.npy")
    print("PASS TRUTH  = ", np.mean(pass_truth))
    pass_fiducial = np.load(f"npy_inputs/{ID}_pass_fiducial.npy")
    weights_MC_sim = np.load(f"npy_inputs/{ID}_weights_mc_sim.npy")
    # theta0_S = np.load(f"npy_inputs/{ID}_Theta0_S.npy")
    # theta_unknown_S = np.load(f"npy_inputs/{ID}_theta_unknown_S.npy")

    print("Checking BOOLS")
    print(pass_reco)
    print(pass_truth)
    print(pass_fiducial)
    print(weights_MC_sim)

    print("shape of theta0_G = ", np.shape(theta0_G))
    print("shape of pass_reco  = ", np.shape(pass_reco))
    print("shape of pass_fiducial  = ", np.shape(pass_fiducial))

    if pass_avg:
        NN_step2_weights = np.load(f"./weights/{ID}_pass_avgs.npy")[-1]
        NN_step2_weights = np.load(f"./weights/{ID}_pass_avgs.npy")[0]
        # Taking the last element grabs the last omnifold ITERATION
        # take the last iteration, MultiFold learns from SCRATCH


    # KINEMATICS
    q_perp_mag, jet_pT_mag, asymm_phi, jet_qT_norm, Q2 = get_kinematics(theta0_G)
    # This is really important. We don't unfold Q2 in this study, so I do not have it in theta0_G
    # That is output from process_data.py. We load it here separatley before the cuts
    # in get_npy_from pkl loads the genQ from the pkl file. We don't have that for npy
    # Didn't want to fix the return Q2 for older notebook. Can remove this maybe after validation
    Q2 = np.load(f"npy_inputs/{ID}_genQ2.npy")


    print("JET QT NORM = ", jet_qT_norm)
    # CUTS
    cuts = get_cuts(pass_fiducial, pass_truth, q_perp_mag,
                    jet_pT_mag, asymm_phi, jet_qT_norm, Q2)

    mask_cut = theta0_G[:, 0] != MASK_VAL
    cuts = np.logical_and(cuts, mask_cut)

    print("Cuts SHAPE = ", np.shape(cuts))
    print("RAPGAP SHAPE = ", np.shape(weights_MC_sim))
    print("NN_step2_Weights LEN = ", np.shape(NN_step2_weights))
    print()
    print("RAPGAP AFTER CUTS = ", np.shape(weights_MC_sim[cuts]))
    print("NN_step2_Weights AFTER CUTS = ", np.shape(NN_step2_weights[cuts]))

    # weights = weights_MC_sim[cut_Q2]
    # weights = weights_MC_sim
    # print("SHAPE after Cuts = ", np.shape(weights))
    weights = NN_step2_weights

    # TO DO: make sure to cut on MASK_VAL

    npy_dir = save_dir + '/npy_files'

    if use_theta0_S:
        label = label+"_RECO"

    np.save(f'{npy_dir}/{label}_cuts.npy', cuts)
    np.save(f'{npy_dir}/{label}_jet_pT.npy', jet_pT_mag)
    np.save(f'{npy_dir}/{label}_q_perp.npy', q_perp_mag)
    np.save(f'{npy_dir}/{label}_Q2.npy', Q2)
    np.save(f'{npy_dir}/{label}_asymm_angle.npy', asymm_phi)
    np.save(f'{npy_dir}/{label}_weights.npy', weights)
    np.save(f'{npy_dir}/{label}_nn_weights.npy', NN_step2_weights)
    np.save(f'{npy_dir}/{label}_mc_weights.npy', weights_MC_sim)



if __name__ == '__main__':

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

    LABEL = config['identifier']
    ID = f"{mc}_{run_type}_{LABEL}"

    pass_avg = True
    if pass_avg:
        print("USING PASS AVGERAGED WEIGHTS")

    load_NN = True
    if 'bootstrap' in run_type:
        load_NN = False
    print("Calculating Kinematics")
    # npy_from_npy(ID, main_dir, pass_avg, mc=mc)
    # npy_from_npy(ID, main_dir, pass_avg, mc=mc, use_theta0_S=True)  # reco
    print("Doing Theta0_G)")
    npy_from_pkl(ID, main_dir, pass_avg, mc=mc,
                 run_type=run_type, load_NN=load_NN)

    print("Doing Theta0_S)")
    npy_from_pkl(ID, main_dir, pass_avg, mc=mc, run_type=run_type,
                 load_NN=load_NN, use_theta0_S = True)

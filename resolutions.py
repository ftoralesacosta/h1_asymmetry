import sys
import yaml
import numpy as np
import pandas as pd
import uproot as ur
from copy import copy
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from get_np_arrays import get_kinematics
from get_np_arrays import get_cuts
from unfold import MASK_VAL

def ModPi(input_angle):
    return np.arccos(np.cos(input_angle - np.pi))


    
def calculate_resolution(binning, reco, truth):

    #Goal: slices defined by bin of truth, filled with prediction distributions
    N = len(binning)+1
    indecies = np.digitize(truth,binning)
    max_count = ((np.bincount(indecies).max()))
    slices = np.empty((N,max_count))
    slices.fill(np.nan)

    counter = np.zeros(N,int)
    avg_truth = np.zeros(N,float)

    pred_over_truth = np.zeros(N,float)

    for i in range(len(reco)):
        
        #if (truth[i] > E_Max): continue
        bin = indecies[i]
        slices[bin][counter[bin]] = reco[i] #slices[bin, element inside bin]
        counter[bin]+=1
        avg_truth[bin]+=truth[i]
        pred_over_truth[bin] += reco[i]/truth[i]
        

    #Resoluton: stdev(pred)/avg_truth    
    avg_truth = avg_truth/counter
    stdev_pred = np.nanstd(slices,axis=1)
    resolution = stdev_pred

    pred_over_truth = pred_over_truth/counter

    return avg_truth, resolution, slices, pred_over_truth

def plot_reco_gen(reco, gen, binning, title=""):
    plt.figure(figsize=(10,7))
    plt.hist(reco, label="Reco", bins=binning, histtype='step')
    plt.hist(gen, label="Gen", bins=binning, histtype='step')
    plt.title(title)
    plt.legend()
    plt.savefig(f"./plots/{title}.pdf")

def plot_resolution(avg_truths, resolution, title="", var="", first_bin = 0, last_bin= -1, ax=None):


    if ax == None:
        ax = plt.subplot(1,1,1)

    # fig=plt.figure(figsize=(14,10))
    ax.set_title(title)
    ax.set_ylabel(var + " Resolution")
    ax.set_xlabel(var + " Truth")
    # ax.set_xticks(fontsize=20)
    # ax.set_yticks(fontsize=20)
    ax.set_ylim(0,2*min(np.nanmax(resolution), 20)+0.01*np.nanmin(resolution))
    #plt.xlim(0,8)
    #plt.ylim(0,3)
    ax.tick_params(direction='in',right=True,top=True,length=10)
    ax.errorbar(avg_truths[first_bin:last_bin],
                 resolution[first_bin:last_bin],
                 linestyle="-",linewidth=2.0,capsize=4,
                 capthick=1.2,elinewidth=1.2,ecolor='black',
                 marker="o",color='dodgerblue',alpha=0.7)

    # plt.savefig(f"./plots/{title}_Resolution.pdf")


# For use in response matrix
def draw_identity_line(axes, *line_args, **line_kwargs):
    identity, = axes.plot([], [], *line_args, **line_kwargs)
    def callback(axes):
        low_x, high_x = axes.get_xlim()
        low_y, high_y = axes.get_ylim()
        low = max(low_x, low_y)
        high = min(high_x, high_y)
        identity.set_data([low, high], [low, high])
    callback(axes)
    axes.callbacks.connect('xlim_changed', callback)
    axes.callbacks.connect('ylim_changed', callback)
    return axes

def plot_response(reco, truth, label, truth_bins, 
                  take_log = False, density=False, 
                  ylabel="phi Reco", plot_offset = 5.0):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 10), constrained_layout=True)
    cmap = copy(plt.cm.plasma)
    cmap.set_bad(cmap(0))


    truth_counts = 1.0
    if density:
        truth_counts, bins = np.histogram(truth, bins=truth_bins)
        
    h, xedges, yedges = np.histogram2d(truth, reco, bins=[truth_bins, truth_bins])
    h = h/truth_counts

    pcm = ax.pcolormesh(xedges, yedges, h.T, cmap=cmap, norm=LogNorm(vmin=1.0e3,vmax=1.1e5), rasterized=True)

    if density:
        pcm = ax.pcolormesh(xedges, yedges, h.T, cmap=cmap, rasterized=True)

    cb = fig.colorbar(pcm, ax=ax, pad=0)
    cb.ax.tick_params(labelsize=20)
    ax.set_xlabel("Truth $\phi$",fontsize=22)
    ax.set_ylabel("Reco $\phi$",fontsize=25)
    title_string = r"$\phi_\mathrm{Truth} vs. \phi_\mathrm{Reco}$"
    if density: title_string += " (Density)"
    ax.set_title(title_string,fontsize=30)
        

    draw_identity_line(ax, color='cyan', linewidth=2, alpha=0.5, label="Ideal")
    ax.legend(loc="upper left")
    fig.text(0.95,-0.05,label,transform=ax.transAxes,fontsize=10)


    plt.savefig(f"./plots/{label}_Respons_Matrix.pdf")

# ============= MAIN ==============
if __name__=="__main__":
# =============  Load Config  =============

    if len(sys.argv) > 1:
        CONFIG_FILE = sys.argv[1]
    else:
        CONFIG_FILE = "./configs/config.yaml"

    config = yaml.safe_load(open(CONFIG_FILE))
    print(f"\nLoaded {CONFIG_FILE}\n")

    mc_type = config['mc']  # Rapgap, Django, Pythia
    run_type = config['run_type']  # nominal, bootstrap, systematic
    processed_dir = config['main_dir']
    NIter = config['n_iterations']
    q_perp_bins = config['q_bins']

    LABEL = config['identifier']
    ID = f"{mc_type}_{run_type}_{LABEL}"

    NEVENTS = 100_000


    reco_vars = ['e_px', 'e_py', 'e_pz',
                 'jet_pt', 'jet_eta', 'jet_phi',
                 'jet_dphi', 'jet_qtnorm']

    gen_vars = ['gene_px', 'gene_py', 'gene_pz',
                'genjet_pt', 'genjet_eta', 'genjet_phi',
                'genjet_dphi', 'genjet_qtnorm']

    theta0_G = np.load(f"./npy_inputs/{ID}_Theta0_G.npy")
    theta0_S = np.load(f"./npy_inputs/{ID}_Theta0_S.npy")

    MASK_CUT = np.logical_and(theta0_G[:,0] != MASK_VAL, theta0_S[:,0] != MASK_VAL)
    theta0_G = theta0_G[MASK_CUT]
    theta0_S = theta0_S[MASK_CUT]

    # ====== Electron =======
    reco_e_px = theta0_S[:,0]
    reco_e_py = theta0_S[:,1]
    reco_e_pz = theta0_S[:,2]
    reco_e_pt = np.sqrt(reco_e_px**2 + reco_e_py**2)
    reco_e_phi = np.arccos(reco_e_px / reco_e_pt)
    reco_e_eta  = np.arcsinh(reco_e_pz / reco_e_pt)

    gen_e_px = theta0_G[:,0]
    gen_e_py = theta0_G[:,1]
    gen_e_pz = theta0_G[:,2]
    gen_e_pt = np.sqrt(gen_e_px**2 + gen_e_py**2)
    gen_e_phi = np.arccos(gen_e_px/gen_e_pt)
    gen_e_eta  = np.arcsinh(gen_e_pz / gen_e_pt)

    # ====== Jet =======
    reco_jet_pt = theta0_S[:,3]
    reco_jet_eta = theta0_S[:,4]
    reco_jet_phi  = theta0_S[:,5]

    gen_jet_pt = theta0_G[:,3]
    gen_jet_eta = theta0_G[:,4]
    gen_jet_phi  = theta0_G[:,5]


    assert(np.shape(gen_jet_pt) == np.shape(reco_jet_pt))

    # ====== Get Dictionary of variables ======
    res_vars = {}
    res_vars["e_phi"] = [reco_e_phi, gen_e_phi]
    res_vars["e_eta"] = [reco_e_eta, gen_e_eta]
    res_vars["e_pt"] = [reco_e_pt, gen_e_pt]
    res_vars["jet_phi"] = [reco_jet_phi, gen_jet_phi]
    res_vars["jet_eta"] = [reco_jet_eta, gen_jet_eta]
    res_vars["jet_pt"] = [reco_jet_pt, gen_jet_pt]


    # ====== Define Binning =======
    phi_binning = np.linspace(0, np.pi, 13)
    pt_binning  = np.linspace(0, 150 , 151)
    qperp_binning=np.linspace(0, 3.1,   32)

    eta_binning = np.linspace(0, 10   , 11)
    x_binning = np.linspace(0, 150, 151 )
    z_binning = np.linspace(0, 150, 151 )

    # avg_gen_e_px, stdevs_gen_e_px, slices_gen_e_px, scale_gen_e_px = \
    # calculate_resolution(x_binning, reco_e_px, gen_e_px)
    # print(stdevs_gen_e_px)

    # avg_gen_e_pz, stdevs_gen_e_pz, slices_gen_e_pz, scale_gen_e_pz = \
    # calculate_resolution(z_binning, reco_e_pz, gen_e_pz)
    # print(stdevs_gen_e_pz)


    # avg_gen_e_phi, stdevs_gen_e_phi, slices_gen_e_phi, scale_gen_e_phi = \
    # calculate_resolution(phi_binning, ModPi(reco_e_phi), ModPi(gen_e_phi))
    # plot_resolution(avg_gen_e_phi, stdevs_gen_e_phi, title="electron_Phi", var="$\phi^e$")
    # # plot_response(reco_e_phi, gen_e_phi,"./", phi_binning, density=True)
    # print(stdevs_gen_e_phi)
    
    # avg_gen_jet_phi, stdevs_gen_jet_phi, slices_gen_jet_phi, scale_gen_jet_phi = \
    # calculate_resolution(phi_binning, ModPi(reco_jet_phi), ModPi(gen_jet_phi))
    # plot_resolution(avg_gen_jet_phi, stdevs_gen_jet_phi, title="jet_Phi", var="$\phi^\mathrm{jet}$",last_bin=-2)
    # # plot_response(reco_jet_phi, gen_jet_phi,"./", phi_binning, density=True)
    # print(stdevs_gen_jet_phi)

    # avg_gen_jet_pt, stdevs_gen_jet_pt, slices_gen_jet_pt, scale_gen_jet_pt = \
    # calculate_resolution(pt_binning, reco_jet_pt, gen_jet_pt)
    # plot_resolution(avg_gen_jet_pt, stdevs_gen_jet_pt, title="jet_pT", var="$p_\mathrm{T}^e$")
    # # plot_response(reco_pt, gen_jet_pt,"./",pt_binning, density=True)
    # print(stdevs_gen_jet_pt)




    # Calculated Variables, with and without Cuts
    # When weights are not applied, these are GEN/TRUTH LEVEL
    cuts_h1rpgp       = np.load(f'{processed_dir}/npy_files/{ID}_cuts.npy')
    jet_pT_h1rpgp     = np.load(f'{processed_dir}/npy_files/{ID}_jet_pT.npy')
    # cuts_h1rpgp = np.ones(len(jet_pT_h1rpgp), dtype=bool)
    jet_pT_h1rpgp     = np.load(f'{processed_dir}/npy_files/{ID}_jet_pT.npy')[cuts_h1rpgp]
    q_perp_h1rpgp     = np.load(f'{processed_dir}/npy_files/{ID}_q_perp.npy')[cuts_h1rpgp]
    asymm_phi_h1rpgp  = np.load(f'{processed_dir}/npy_files/{ID}_asymm_angle.npy')[cuts_h1rpgp]
    weights_h1rpgp    = np.load(f'{processed_dir}/npy_files/{ID}_weights.npy')[cuts_h1rpgp]
    mc_weights_h1rpgp = np.load(f"{processed_dir}/npy_files/{ID}_mc_weights.npy")[cuts_h1rpgp]
    nn_weights_h1rpgp = np.load(f"{processed_dir}/npy_files/{ID}_nn_weights.npy")

    ID = ID+"_RECO"
    reco_cuts_h1rpgp       = np.load(f'{processed_dir}/npy_files/{ID}_cuts.npy')
    reco_jet_pT_h1rpgp     = np.load(f'{processed_dir}/npy_files/{ID}_jet_pT.npy')
    # reco_cuts_h1rpgp = np.ones(len(jet_pT_h1rpgp), dtype=bool)
    reco_jet_pT_h1rpgp     = np.load(f'{processed_dir}/npy_files/{ID}_jet_pT.npy')[cuts_h1rpgp]
    reco_q_perp_h1rpgp     = np.load(f'{processed_dir}/npy_files/{ID}_q_perp.npy')[cuts_h1rpgp]
    reco_asymm_phi_h1rpgp  = np.load(f'{processed_dir}/npy_files/{ID}_asymm_angle.npy')[cuts_h1rpgp]
    reco_weights_h1rpgp    = np.load(f'{processed_dir}/npy_files/{ID}_weights.npy')[cuts_h1rpgp]
    reco_mc_weights_h1rpgp = np.load(f"{processed_dir}/npy_files/{ID}_mc_weights.npy")[cuts_h1rpgp]
    reco_nn_weights_h1rpgp = np.load(f"{processed_dir}/npy_files/{ID}_nn_weights.npy")


    # avg_gen_asymm, stdevs_asymm, slices_asymm, scale_asymm = \
    # calculate_resolution(phi_binning, ModPi(reco_asymm_phi_h1rpgp), ModPi(asymm_phi_h1rpgp))
    # plot_resolution(avg_gen_asymm, stdevs_asymm, title='AsymmetryAnglePhi',
    #                 var="phi_asymmetry", last_bin=-1)
    # print(stdevs_asymm)

    # plot_resolution(avg_gen_asymm, stdevs_asymm, title='AsymmetryAnglePhi_NOCUTS',
    # plot_response(reco_jet_phi, gen_jet_phi,"./", phi_binning, density=True)

    # avg_gen_pt, stdevs_pt, slices_pt, scale_pt = \
    # calculate_resolution(pt_binning, reco_jet_pT_h1rpgp, jet_pT_h1rpgp)
    # # plot_resolution(avg_gen_pt, stdevs_pt, title='pT_resolution_wCuts',
    # plot_resolution(avg_gen_pt, stdevs_pt, title='pT_resolution_noCuts',
    #                 var="jet pT", last_bin=-1)
    # # plot_resolution(avg_gen_jet_pt, stdevs_gen_jet_pt, title="jet_pT", var="$p_\mathrm{T}^e$")
    # # plot_response(reco_jet_phi, gen_jet_phi,"./", phi_binning, density=True)
    # print(stdevs_pt)

    avg_gen_qperp, stdevs_qperp, slices_qperp, scale_qperp = \
    calculate_resolution(qperp_binning, reco_q_perp_h1rpgp, q_perp_h1rpgp)
    print("*"*20)
    print("Plotting Resolutions")
    plot_resolution(avg_gen_qperp, stdevs_qperp, title='LeptonJet_qPerp',
                    var="q_perp", last_bin=-1)
    print(stdevs_qperp)
    plot_reco_gen(reco_q_perp_h1rpgp, q_perp_h1rpgp, qperp_binning, title="q_perp_distributions" )
    # plot_resolution(avg_gen_qperp, stdevs_qperp, title='qperpetryAnglePhi_NOCUTS',)


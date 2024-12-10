import pickle
import numpy as np
from process_functions import averages_in_qperp_bins
from get_np_arrays import get_cuts

import numpy as np

import numpy as np

def calculate_jet_qT(E_px, E_py, E_pz, Jet_px, Jet_py, Jet_pz):
    E_electron = 27.6  # Incoming electron energy in GeV

    # Step 1: Compute scattered electron energy E'_E
    E_E = np.sqrt(E_px**2 + E_py**2 + E_pz**2)
    
    # Step 2: Compute virtual photon components q = k - k'
    q_x = -E_px
    q_y = -E_py
    q_z = -E_electron - E_pz
    q_E = E_electron - E_E

    # Step 3: Compute unit vector in q direction
    q_magnitude = np.sqrt(q_x**2 + q_y**2 + q_z**2)
    q_unit_x = q_x / q_magnitude
    q_unit_y = q_y / q_magnitude
    q_unit_z = q_z / q_magnitude

    # Step 4: Compute jet momentum components
    Jet_p = np.sqrt(Jet_px**2 + Jet_py**2 + Jet_pz**2)

    # Step 5: Compute components of jet momentum transverse to q
    p_dot_qunit = Jet_px * q_unit_x + Jet_py * q_unit_y + Jet_pz * q_unit_z
    p_perp_x = Jet_px - p_dot_qunit * q_unit_x
    p_perp_y = Jet_py - p_dot_qunit * q_unit_y
    p_perp_z = Jet_pz - p_dot_qunit * q_unit_z

    # Step 6: Compute magnitude of transverse momentum
    jet_qT = np.sqrt(p_perp_x**2 + p_perp_y**2 + p_perp_z**2)

    return jet_qT

#def calculate_jet_qT_old(l_incoming, l_scattered, p_jet):
#    """
#    Calculate the jet qT in Deep Inelastic Scattering (DIS).

#    Parameters:
#    l_incoming: Incoming electron 4-vector [E_e, p_e_x, p_e_y, p_e_z]
#    l_scattered: Scattered electron 4-vector [E_e', p_e'_x, p_e'_y, p_e'_z]
#    p_jet: Jet 3-momentum vector [p_jet_x, p_jet_y, p_jet_z]

#    Returns:
#    float: The jet qT (transverse momentum relative to the virtual photon direction)
#    """
#    # Convert inputs to NumPy arrays
#    l_incoming = np.array(l_incoming)
#    l_scattered = np.array(l_scattered)
#    p_jet = np.array(p_jet)
    
#    #Calculate the virtual photon 4-momentum q^mu = l^mu - l'^mu
#    q_mu = l_incoming - l_scattered  # [q0, qx, qy, qz]
#    q_vec = q_mu[1:]  # [qx, qy, qz]
#    q_vec_mag = np.linalg.norm(q_vec)

#    #Project the jet momentum onto q_hat
#    q_hat = q_vec / q_vec_mag
#    p_jet_dot_q_hat = np.dot(p_jet, q_hat)
#    p_jet_parallel = p_jet_dot_q_hat * q_hat
    
#    q_T_vec = p_jet - p_jet_parallel
#    q_T = np.linalg.norm(q_T_vec)
    
#    return q_T


def txt_to_dataDict(filename):
    with open(filename, 'r') as f:
        # Read and process the header line
        header_line = f.readline().strip()
        header_line = header_line.strip('[]')  # Remove the square brackets
        headers = [h.strip() for h in header_line.split(',')]

        # Load the data starting from the second line
        data = np.loadtxt(f, delimiter=',')

    # Create a dictionary mapping headers to data columns
    data_dict = {header: data[:, idx] for idx, header in enumerate(headers)}

    return data_dict


# Example usage
if __name__ == "__main__":
    # l_incoming = [10.0, 0.0, 0.0, 10.0]
    # l_scattered = [8.0, 1.0, 1.0, 7.8]
    # p_jet = [3.0, -1.0, 2.0] # Jet momentum 3-vector [p_jet_x, p_jet_y, p_jet_z]
    # # Calculate the jet qT
    # jet_qT = calculate_jet_qT(l_incoming, l_scattered, p_jet)
    # print(f"Jet qT: {jet_qT:.3f} GeV")

    # sherpa_data = txt_to_dataDict('sherpa_events.txt')
    # filename = './theory_files/sherpa_asymm.pkl'

    # sherpa_data = txt_to_dataDict('sherpa_events_justNLO.txt')
    # filename = './theory_files/sherpa_asymm_justNLO.pkl'

    sherpa_data = txt_to_dataDict('sherpa_eventsLO.txt')
    filename = './theory_files/sherpa_asymmLO.pkl'

    q_perp_bins = [ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  8., 10.]

    q_perp = sherpa_data['q_perp']
    asymm_phi = sherpa_data['asymm_phi']
    weights = sherpa_data['MC Event Weight']
    Q2 = sherpa_data['Q2']
    jet_pT_mag = np.sqrt(sherpa_data['Jet_px']**2 + sherpa_data['Jet_py']**2)
    print('mean jet pT', np.nanmean(jet_pT_mag))

    jet_qT = calculate_jet_qT(sherpa_data['E_px'],
                              sherpa_data['E_py'],
                              sherpa_data['E_pz'],
                              sherpa_data['Jet_px'],
                              sherpa_data['Jet_py'],
                              sherpa_data['Jet_pz'])

    jet_qT_norm = jet_qT / Q2

    print('qt = ', np.nanmean(jet_qT))
    print('qt_norm =', np.nanmean(jet_qT_norm))

    pass_fiducial = np.ones(len(weights))
    pass_truth = np.ones(len(weights))
    cuts = get_cuts(pass_fiducial, pass_truth, q_perp,
                    jet_pT_mag, asymm_phi, jet_qT_norm, Q2)
    print('from txt: ',np.nanmean(q_perp), np.nanmean(asymm_phi))

    sherpa_asymm = {}
    averages_in_qperp_bins(sherpa_asymm, q_perp_bins, q_perp[cuts], asymm_phi[cuts],
                           weights[cuts], print_keys=True)
    print('from dict: ',np.nanmean(sherpa_asymm['q_perp']), np.nanmean(sherpa_asymm['phi']))
    print('cos1 = ',sherpa_asymm['cos1'])
    print('cos2 = ',sherpa_asymm['cos2'])
    print('cos3 = ',sherpa_asymm['cos3'])

    with open(filename, 'wb') as file:
        pickle.dump(sherpa_asymm, file, protocol=pickle.HIGHEST_PROTOCOL)
    print("\nFILENAME = ", filename)




#[Q2, y, E_px, E_py, E_pz, E_eta, Jet_px, Jet_py, Jet_pz, Jet_eta, asymm_phi, q_perp, cos1, cos2, cos3, MC Event Weight]

''' We have q_perp and the cos1-3. We need the average in q_perp bins function'''

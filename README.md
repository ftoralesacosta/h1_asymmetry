# h1_asymmetry
Azimuthal Lepton Jet Asymmetry Physics Analysis using ML for Unfolding

## Running:
There are two version of the unfolding script, and unfolding engine. One with Horovod and one without.
To run the simple scripts, run:

    1. `process_data.py [configs/config_file.yaml]`
        - This saves Theta_G, Theta_S, and Theta_unknown_S as npy files
    2. `python no_pkl_unfolding.py [configs/config_file.yaml]`
        - This runs the unfolding, calling unfold.py
    3. `python inference_reweight.py [configs/config_file.yaml]`
        - This runs inference on the full dataset, reweights to produce step2 push weights,
        and averages over passes
    4. `python get_np_arrays.py [configs/config_file.yaml]`
        - This calculates the physics kinematic observables of interest
    4. `python plot_weights.py [configs/config_file.yaml]'
        - This plots averages and standard deviations of the weight distributions. Very important for 
        validation and debugging before looking at physics.
    5. `python make_gifs.py [configs/config_file.yaml]`
        - This generates several phi_asymm plots and cos[n*phi] plots, and generates a gif

The Horovod versions are run similarly, but bullet 1. may need you to run with `horovodrun -p 8 python hvd_unfolding.py [configs/config_file.yaml] `



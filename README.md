# h1_asymmetry
Azimuthal Lepton Jet Asymmetry Physics Analysis using ML for Unfolding

## Running:
There are two version of the unfolding script, and unfolding engine. One with Horovod and one without.
To run the simple scripts, run:

    1. `process_data.py [configs/config_file.yaml]`
        - This saves Theta_G, Theta_S, and Theta_unknown_S as npy files
    2. `python no_pkl_unfolding.py [configs/config_file.yaml]`
        - This runs the unfolding, calling unfold.py
    3. `python plot_weights.py [configs/config_file.yaml]`
        - This averages over passes, and saves useful plots of the Step2 weights. Very important for validation and debugging
    4. `python make_gifs.py [configs/config_file.yaml]`
        - This generates several phi_asymm plots and cos[n*phi] plots, and generates a gif

The Horovod versions are run similarly, but bullet 1. may need you to run with `horovodrun -p 8 python hvd_unfolding.py [configs/config_file.yaml] `



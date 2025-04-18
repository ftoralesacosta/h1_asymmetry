# h1_asymmetry
Azimuthal Lepton Jet Asymmetry Physics Analysis using ML for Unfolding

## Running:
There are two version of the unfolding script, and unfolding engine. One with Horovod and one without.
To run the simple scripts, run:

    1. `process_data.py [configs/config_file.yaml]`
        - This saves Theta_G, Theta_S, and Theta_unknown_S as npy files
    2. `python hvd_train.py [configs/config_file.yaml]`
        - This runs the unfolding, calling unfold.py
    <!-- 3. `python inference_reweight.py [configs/config_file.yaml]` -->
    3. 'python plot_weights.py [configs/config_file.yaml]'
        - This runs inference on the full dataset, reweights to produce step2 push weights,
        and averages over passes. Plots a lot.
    4. `python get_np_arrays.py [configs/config_file.yaml]`
        - This calculates the physics kinematic observables of interest
        - Most importantly, this applies CUTS
        validation and debugging before looking at physics.
    5. in the notebooks directory, the main notebook for the resultss is *Plot_Asymmetry.ipynb*
    6. `python make_gifs.py [configs/config_file.yaml]`
        - This generates several phi_asymm plots and cos[n*phi] plots, and generates a gif

When running an perlmutter, you will need to load two modules:

`module load cpe/23.03`
`module load tensorflow/2.6.0`

Single GPU tasks should work well, despite supporting Horovod. For bootstrapping, make sure to run a job array. Use srun in interactive nodes

import os
import tensorflow.keras.backend as K
n_passes = 5

# The zero below is the random seed for bootstraping. 0 skips bootstrapping
mc_type = "Rapgap"
# mc_type = "Django"
for i_pass in range(n_passes):
    command_string = f"python unfold_fullstats.py {mc_type} nominal 0 {i_pass}"

    print(command_string)
    os.system(command_string+f" &> ./logs/log_{i_pass}.txt")

    # os.system("python unfold_fullstats_boot.py Rapgap bootstrap_%i %i"%(i,myrand))
    K.clear_session()

# %%
######
# Copyright (c) 2025 Rong Chen (rong.chen.mail@gmail.com)
# All rights reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Particle segmentation: development 
######

import numpy as np
import pandas as pd
import os
from multiprocessing import Pool
from functools import partial
from itertools import product
import random

from acva_seg_particle_core import seg_particle_core
from acva_seg_particle_core import seg_particle_3d
from acva_toolbox import get_centered_subset, conf_to_csv, remove_image
from acva_toolbox import hp_selection


# %%
# for a given f_set and configuration file, we run the analysis pipeline
# this function will get the configuration file based on home_folder and channel_folder
def mode_conditional_run(home_folder, channel_folder, f_set):
    conf_file = os.path.join(home_folder, channel_folder + "_res", "conf_seg_particle.csv")
    if os.path.isfile(conf_file):
        param = pd.read_csv(conf_file)
        erosionIte = int(param["value"][0])
        flatFieldSiz = int(param["value"][1])
        hatSiz = int(param["value"][2])
        valThZ = param["value"][3]
        binStrSiz = int(param["value"][4])
        blockSiz = int(param["value"][5])
        print(erosionIte, flatFieldSiz, hatSiz, valThZ, binStrSiz, blockSiz)

        partial_func = partial(seg_particle_core, conf_file=conf_file)
        with Pool() as pool:
            pool.map(partial_func, f_set)
        rrr = os.path.join(home_folder, channel_folder + "_res")
        b_set = [os.path.join(rrr, f) for f in os.listdir(rrr) if f.endswith(".png")]
        seg_particle_3d(b_set, block_size=blockSiz)
    else:
        print("Error: the configuration file does not exist.")
    

# %%
def mode_hyper_param_tuning(home_folder, channel_folder, f_subset, conf_space, rsp_limit):
    # random sampling the conf space
    res = []
    n = min(rsp_limit, int(0.1 * len(conf_space)))
    sampled = random.sample(range(len(conf_space)), n)
    for item in sampled:
        # prepare the configuration file
        conf_current = conf_space[item]
        conf_file = os.path.join(home_folder, channel_folder + '_res', "conf_seg_particle.csv")
        conf_to_csv(conf_current, conf_file)

        remove_image(os.path.join(home_folder, channel_folder + '_res'))

        mode_conditional_run(home_folder, channel_folder, f_subset)
        ttt = pd.read_csv(os.path.join(home_folder, channel_folder + '_res', "all_objects.csv"))
        total_vol = ttt["volume"].to_numpy().sum()
        conf_current["U"] = total_vol
        res.append(conf_current)

    # select a configuration
    rrr = pd.DataFrame(res)
    fname = os.path.join(home_folder, channel_folder + '_res', "000_u.csv")
    rrr.to_csv(fname, index=False)
    conf_selected = hp_selection(rrr)
    print(conf_selected)
    conf_file = os.path.join(home_folder, channel_folder + '_res', "conf_seg_particle.csv")
    conf_to_csv(conf_selected, conf_file)

    remove_image(os.path.join(home_folder, channel_folder + '_res'))

import sys
if __name__ == "__main__":
    from multiprocessing import freeze_support
    # freeze_support() is required on Windows/macOS when using multiprocessing with the spawn start method
    # On Linux, the default start method is fork, and freeze_support() does nothing.
    freeze_support()

    if len(sys.argv) < 2:
        print("Usage: python my.py <exp_file>")
        sys.exit(1)
    
    exp_file = sys.argv[1]  
    random.seed(888)
    rsp_limit = 30
    subset_span = 5

    f_set = []
    if(os.path.isfile(exp_file)):
        prj = pd.read_csv(exp_file)
        home_folder = prj["folder"][0]
        channel_folder = prj["folder"][1]
        data_folder = os.path.join(home_folder, channel_folder)
        for r, _, f in os.walk(data_folder):
            for file in f:
                file_path = os.path.join(r, file)
                f_set.append(file_path)

        rrr = os.path.join(home_folder, channel_folder + '_res')
        if not os.path.exists(rrr):
            os.makedirs(rrr)
            opMode = "hyper_param_tuning"
        else:
            # if the result folder exists, then it MUST have a conf file.
            # create the clean result folder; remove all image files if they exist.
            remove_image(rrr)
            opMode = "full_run"
    else:
        print("Error: the experiment file does not exist. ")

    print(data_folder, opMode)
    if opMode == "full_run":
        mode_conditional_run(home_folder, channel_folder, f_set)
    elif opMode == "hyper_param_tuning":
        conf_df = pd.read_csv("conf_seg_particle_space.csv")
        print(conf_df)
        erosion_iteration = conf_df["start"][0]
        flat_field_size_range = np.arange(conf_df["start"][1], conf_df["end"][1] + 1, 2)
        white_hat_size = conf_df["start"][2]
        threshold_z_range = np.arange(conf_df["start"][3], conf_df["end"][3] + 0.1, 0.25)
        binary_structure_size_range = np.arange(conf_df["start"][4], conf_df["end"][4] + 1, 2)
        block_size = conf_df["start"][5]
        conf_space = [{
                "erosion_iteration": erosion_iteration,
                "flat_field_size": f,
                "white_hat_size": white_hat_size,
                "threshold_z": z,
                "binary_structure_size": b,
                "block_size": block_size,
            }
            for f, z, b in product(flat_field_size_range, threshold_z_range, binary_structure_size_range)]

        f_subset = get_centered_subset(f_set, span=subset_span) # different from the full run, we use a subset
        mode_hyper_param_tuning(home_folder, channel_folder, f_subset, conf_space, rsp_limit)
    else:
        print("unknown mode")



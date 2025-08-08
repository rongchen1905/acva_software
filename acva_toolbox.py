######
# Copyright (c) 2025 Rong Chen (rong.chen.mail@gmail.com)
# All rights reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Toolbox
######

import numpy as np
import pandas as pd
import os
import csv

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import NearestNeighbors

# write log. this is useful for distributed computing
# parameters: 
# msg: the message
def write_log(msg):
    with open("run.log", "a") as log_file:
        log_file.write("{0}\n".format(msg))


# get a subset of f_set that is a list, LTS
# parameters: 
# f_set: the list
# span: the center is the middle point of f_set, expand it left and right by span.
# returns: the subset of the list
def get_centered_subset(f_set, span=10):
    n = len(f_set)
    center = n // 2
    start = max(0, center - span)
    end = min(n, center + span + 1)  # +1 because slice is exclusive on the right
    return f_set[start:end]


# hyperparameter selection based on stability
# parameters: 
# rrr: a data frame. The last column is the outcome and other columns are hyperparameters.
# returns: the selected configuration, hp_selected
def hp_selection(rrr):
    # remove constant columns (i.e., with zero variance)
    rrr_clean = rrr.loc[:, rrr.nunique() > 1]
    # separate outcome (U) and hyperparameters (all others)
    U = rrr_clean.iloc[:, -1]  # assuming U is the last column
    H = rrr_clean.iloc[:, :-1]

    eff = []
    for col in H.columns:
        x = H[[col]].values  # ensure it's 2D
        model = LinearRegression().fit(x, U)
        U_pred = model.predict(x)
        r2 = r2_score(U, U_pred)
        eff.append((col, r2))

    eff.sort(key=lambda x: x[1], reverse=True)
    for name, r2 in eff:
        print(f"{name}: R2 = {r2:.4f}")
    top_h1, top_h2 = eff[0][0], eff[1][0]

    H_clean = H[[top_h1, top_h2]]
    scaler = MinMaxScaler()
    H_scaled = scaler.fit_transform(H_clean)

    k = 5
    graph_run = 1
    while graph_run > 0:
        nbrs = NearestNeighbors(n_neighbors=k).fit(H_scaled)
        # compute the graph of k-Neighbors, the graph is a sparse matrix in csr format
        knn_graph = nbrs.kneighbors_graph(mode='connectivity')
        mutual_knn = knn_graph.minimum(knn_graph.T).tocsr()
        neighbors = [mutual_knn[i].indices for i in range(mutual_knn.shape[0])]
        local_curv = np.array([
            U[i] - np.mean(U[neighbors[i]]) if len(neighbors[i]) > 0 else -1
            for i in range(len(U))
        ])
    
        if np.any(local_curv == -1):
            k = k+1
            graph_run = graph_run + 1
        else:
            graph_run = -1

    curv_idx = np.argmax(np.abs(local_curv))
    hp_selected = rrr.iloc[curv_idx, :-1]
    return hp_selected


# save a configuration to a csv file
# parameters: 
# conf: the configuration. It's a dictionary.
# fff: the csv file name
def conf_to_csv(conf, fff):
    with open(fff, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["name", "value"])  # header
        for key, value in conf.items():
            writer.writerow([key, value])


# clean the workspace and remove all images (png and tif)
# parameters:
# rrr: the workspace, it is a folder path
def remove_image(rrr):
    file_list = [f for f in os.listdir(rrr) if f.endswith(".png") or f.endswith(".tif")]
    for file_name in file_list:
        file_path = os.path.join(rrr, file_name)
        if os.path.isfile(file_path):
            os.remove(file_path)
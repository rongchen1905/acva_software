######
# Copyright (c) 2025 Rong Chen (rong.chen.mail@gmail.com)
# All rights reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Particle segmentation core
######

import numpy as np
import pandas as pd
import os

import imageio.v3 as imageio
from skimage.filters import gaussian, threshold_otsu
from scipy import ndimage
from skimage import measure
import SimpleITK as sitk

# The 2D operation of particle segmentation, LTS
# fpath is a single file
# conf_file is the operation configuration
# (fpath, conf_file) are parameters of the segmentation core function
# debugging and test of this function is based on I/O 
def seg_particle_core(fpath, conf_file):
    if(os.path.isfile(conf_file)):
        param = pd.read_csv(conf_file)
        erosionIte = int(param["value"][0])
        flatFieldSiz = int(param["value"][1])
        hatSiz = int(param["value"][2])
        valThZ = param["value"][3]
        binStrSiz = int(param["value"][4])
    else:
        print("No configuration file!")

    img = imageio.imread(fpath)
    # prepare binary image that is based on thresholding z-value
    imgB = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)

    # compute 1% and 99.5% quantiles
    qlow, qhigh = np.quantile(img, [0.01, 0.995])
    # clean outliers in the intensity
    imgC = np.clip(img, qlow, qhigh)
    # get brain masks
    # mask - foreground, extMask - extended foreground, effMask - effective mask
    if erosionIte > 0:
        val = threshold_otsu(imgC)
        mask = imgC>val
        extMask = gaussian(mask, sigma=10)
        extMask = extMask>0.2
        effMask =  ndimage.binary_erosion(extMask, iterations=int(erosionIte))
    else:
        effMask = imgC>0 # disable foreground detection

    # non-uniform background correction (flat-field correction)
    if flatFieldSiz > 0:
        field = gaussian(imgC, sigma=flatFieldSiz)
        imgCorrected = imgC/field
    else:
        imgCorrected = imgC
    
    imgCorrected[effMask==0] = 0

    # top hat
    imgTopHat = imgCorrected - ndimage.grey_opening(imgCorrected, size=(hatSiz, hatSiz))

    zIn = imgTopHat
    ttt = zIn[zIn>0]
    m = ttt.mean()
    s = ttt.std()
    z = (zIn-m)/s
    z[effMask==0] = 0
    z[z<0] = 0
    imgB[z>valThZ] = 1
    # smooth imgB
    sss = np.ones((binStrSiz, binStrSiz), dtype=bool)
    imgB_clean = ndimage.binary_closing(ndimage.binary_opening(imgB, structure=sss), structure=sss)

    resFolder = os.path.dirname(fpath) + "_res"
    subj = os.path.basename(fpath)
    name, _ = os.path.splitext(subj)
    ttt = f"b_{name}.png"
    imageio.imwrite(os.path.join(resFolder, ttt), (imgB_clean * 255).astype(np.uint8), compress_level=9)

    #print(type(z), z.dtype, z.min(), z.max())


# The 3D operation of particle segmentation, LTS
# b_set: a list of files. We use a list because this function is for 3D
# block_size: the block size for online calculation
def seg_particle_3d(b_set, block_size):
    total = len(b_set)
    for block_start in range(0, total, block_size):
        block_end = min(block_start + block_size, total)
        b_active = b_set[block_start:block_end]
        #print(b_active)

        # assemble 3D binary object map
        b_rep = b_active[0]
        imgR = imageio.imread(b_rep)
        imgB_clean = np.zeros((len(b_active), imgR.shape[0], imgR.shape[1]))
        for iii, bbb in enumerate(b_active):
            imgB_clean[iii, :, :] = imageio.imread(bbb)

        resFolder = os.path.dirname(b_rep)
        subj = os.path.basename(b_rep)
        name, _ = os.path.splitext(subj)
        ttt = f"report_{name}.csv"

        labels = measure.label(imgB_clean)
        imgObj = sitk.GetImageFromArray(labels.astype(int))
        shape_stats = sitk.LabelShapeStatisticsImageFilter()
        shape_stats.ComputeOrientedBoundingBoxOn()
        shape_stats.Execute(imgObj)

        # clean labels and remove particles with extreme volumes
        objVol = np.array([ (shape_stats.GetPhysicalSize(i)) for i in shape_stats.GetLabels()])
        q1 = np.percentile(objVol, 25)
        q3 = np.percentile(objVol, 75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 3 * iqr
        relabelMap =  { i : 0 for i in shape_stats.GetLabels() if shape_stats.GetPhysicalSize(i) < lower or 
                    shape_stats.GetPhysicalSize(i) > upper }
        imgObjClean = sitk.ChangeLabel(imgObj, changeMap=relabelMap)
        shape_stats.Execute(imgObjClean)

        stats_list = [ (shape_stats.GetPhysicalSize(i),
                        shape_stats.GetElongation(i),
                        shape_stats.GetFlatness(i),
                        *shape_stats.GetCentroid(i)) for i in shape_stats.GetLabels()]
        cols=["volume", "elongation", "flatness", "x", "y","z"]
        stats = pd.DataFrame(data=stats_list, columns=cols)
        stats["z"] += block_start
        stats.to_csv(os.path.join(resFolder, ttt), index=False) 

    # combine report files
    report_files = [f for f in os.listdir(resFolder) if f.startswith("report_") and f.endswith(".csv")]
    ttt = [pd.read_csv(os.path.join(resFolder, f)) for f in report_files]
    combined_df = pd.concat(ttt, ignore_index=True)
    #combined_df.describe()
    combined_df.to_csv(os.path.join(resFolder, "all_objects.csv"), index=False)
    for f in report_files:
        os.remove(os.path.join(resFolder, f))


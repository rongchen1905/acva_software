# %%
######
# Copyright (c) 2025 Rong Chen (rong.chen.mail@gmail.com)
# All rights reserved.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Particle segmentation: test 
######

import numpy as np
import pandas as pd
import os

import imageio.v3 as imageio
from skimage.filters import gaussian, threshold_otsu
from skimage import measure
from scipy import ndimage
import SimpleITK as sitk


# %%
f_set = []

if(os.path.isfile("exp_sample.csv")):
    prj = pd.read_csv("exp_sample.csv")
    home_folder = prj["folder"][0]
    channel_folder = prj["folder"][1]
    data_folder = os.path.join(home_folder, channel_folder)
    print(data_folder)
    for r, _, f in os.walk(data_folder):
        for file in f:
            file_path = os.path.join(r, file)
            f_set.append(file_path)
    #print(f_set)
    # create the clean result folder, remove all files in that folder if it exists
    rrr = os.path.join(home_folder, channel_folder + '_res')
    if not os.path.exists(rrr):
        os.makedirs(rrr)
    else:
        #file_list = os.listdir(rrr)
        file_list = [f for f in os.listdir(rrr) if f.endswith(".tif")]
        for file_name in file_list:
            file_path = os.path.join(rrr, file_name)
            if os.path.isfile(file_path):
                os.remove(file_path)    
else:
    print("Error: no experiments.")


# %%
# prepare data for the segmentation core
# fpath is a single file, conf_file is the operation configuration
# (fpath, conf_file) are parameters of the segmentation core function

fpath = f_set[0]

conf_file = os.path.join(home_folder, channel_folder + '_res', "conf_seg_particle.csv")
if(os.path.isfile(conf_file)):
        param = pd.read_csv(conf_file)
        erosionIte = param["value"][0]
        flatFieldSiz = param["value"][1]
        hatSiz = param["value"][2]
        valThZ = param["value"][3]
        binStrSiz = param["value"][4]
else:
        erosionIte = 60
        flatFieldSiz = 20
        hatSiz = 40
        valThZ = 3
        binStrSiz = 3

# for test purpose, we fix this to get the identical result
erosionIte = 60
flatFieldSiz = 20
hatSiz = 40
valThZ = 1
binStrSiz = 5

print(fpath)
print(erosionIte, flatFieldSiz, hatSiz, valThZ, binStrSiz)

# %%
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

# %%
# here is the 3D part. All these operation should be for a stack of images because this is 3D.
# However, for test purpose, we use a 2D image: imgB

labels = measure.label(imgB_clean)
imgObj = sitk.GetImageFromArray(labels.astype(int))
shape_stats = sitk.LabelShapeStatisticsImageFilter()
shape_stats.ComputeOrientedBoundingBoxOn()
shape_stats.Execute(imgObj)

# clean labels and remove particles with extreme large volumes
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
               shape_stats.GetFlatness(i)) for i in shape_stats.GetLabels()]
cols=["volume", "elongation", "flatness"]
stats = pd.DataFrame(data=stats_list, columns=cols)
stats.describe()
#stats.to_csv("000_partcle.csv", index=False) 

# %%
if abs(np.percentile(objVol, 50) - 102) < 1:
    print("Happy ending.")
else:
    print("Failed")

# %%
#ttt = labels
#ttt = sitk.GetArrayViewFromImage(imgObjClean)
#print(type(ttt), ttt.dtype, ttt.min(), ttt.max())
#imageio.imwrite("000_ttt.tif", ttt.astype(np.uint8))
#imageio.imwrite("000_ttt.tif", ttt.astype(np.uint16))
#imageio.imwrite("000_ttt.tif", ttt.astype(np.float32)) 


# %%
# to compare different implementations of the white top hat algorithm. 
# the processing speed could be quite different while the results are similar

#imgA = imgCorrected

#from skimage.morphology import ball
# create 3D ball structure
#radius = 40
#s = ball(radius)
# take only the upper half of the ball
#h = int((s.shape[1] + 1) / 2)
# flat the 3D ball to a weighted 2D disc
#s = s[:h, :, :].sum(axis=0)
# rescale image into 0-255
#s = (255 * (s - s.min())) / (s.max() - s.min())
#selem = s.astype(np.uint8)  # ensure integer type
#print(type(selem), selem.dtype, selem.min(), selem.max())
#imgTopHat = imgA - ndimage.grey_opening(imgA, footprint=selem)

# %%




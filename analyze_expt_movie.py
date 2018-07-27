"""Analyze a movie of electron microscopy files (dm4 format)."""



import matplotlib.pyplot as plt
from glob import glob
import numpy as np
import keras
from keras.utils import multi_gpu_model
import tensorflow as tf
from temnn.knet import net
from temnn.net.dataset import DataEntry,DataSet
from temnn.net.mods import local_normalize
from pyqstem.imaging import CTF
import matplotlib.pyplot as plt
# Peak detection
from stm.preprocess import normalize
from stm.feature.peaks import find_local_peaks, refine_peaks
from skimage.morphology import disk
from scipy.spatial import cKDTree as KDTree
import sys
import os
import shutil
#from collections import deque
import hyperspy.api as hs
import skimage.io

#input = 'Au-CeO2__200_CO_7_250'
#filetype = '.png'
#sampling = 0.21
#graph_dir = 'graphs-110-peismovie-final'
#graph_path = os.path.join(graph_dir, 'clusters-19.h5')


input = 'Au-CeO2_HVAC14'
filetype = '.dm4'
sampling = 0.105
graph_dir = 'graphs-110-thomas-negdefocus'
graph_path = os.path.join(graph_dir, 'clusters-19.h5')
threshold = 0.3

cnn = None
cnn_size = (-1, -1)

def analyze_tree(input, filetype='.dm4'):
    assert not input.endswith('/')

    output1 = input + '_prediction'
    output2 = input + '_atoms'
    for o in (output1, output2):
        if not os.path.exists(o):
            print("Creating folder", o)
            os.makedirs(o)
        # Keep a copy of this script for reference
        shutil.copy2(__file__, o)

    for root, dirs, files in os.walk(input):
        dirs.sort()
        files.sort()
        for f in files:
            if f.endswith(filetype):
                infile = os.path.join(root, f)
                print("Analyzing", infile, flush=True)
                prediction, atoms = analyze(infile)
                assert root.startswith(input)
                bname = os.path.splitext(f)[0]
                outdir = output1 + root[len(input):]
                predname = os.path.join(outdir, bname+'_prediction')
                print("  -> {}.npz".format(predname), flush=True)
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                np.savez_compressed(predname, prediction=prediction)
                outdir = output2 + root[len(input):]
                atomname = os.path.join(outdir, bname+'_atoms')
                print("  -> {}.npy".format(atomname), flush=True)
                if not os.path.exists(outdir):
                    os.makedirs(outdir)
                np.save(atomname, atoms)

def analyze(filename):
    global cnn, cnn_size, graph_path
    if filename.endswith('.dm4') or filename.endswith('*.dm3'):
        # Electron microscopy file - read with HyperSpy
        f = hs.load(filename)
        # Image size must be divisible by 8
        image_size = f.data.shape
        image_size = tuple(np.array(image_size) // 8 * 8)
        image = f.data[:image_size[0],:image_size[1]]
    else:
        #Assume an image file, read with scikit-image
        f = skimage.io.imread(filename)
        image = skimage.color.rgb2gray(f)
        image_size = image.shape
        image_size = tuple(np.array(image_size) // 8 * 8)
        image = image[:image_size[0],:image_size[1]]

    # Load CNN if not yet loaded or not fitting.
    if cnn_size != image_size:
        print("Loading CNN of size {} from {}".format(image_size, graph_path))
        cnn = load_CNN(graph_path, image_size)
        cnn_size = image_size
        
    # Normalize the image so it can be recognized
    img2 = local_normalize(image, 120, 120)
    img2.shape = (1, ) + image_size + (1,)

    # Make the prediction
    prediction = cnn.predict(img2)[0,:,:,0]

    # Find the peaks in the output from the CNN
    distance = int(2.5 / sampling)
    peaks = find_local_peaks(prediction, min_distance=distance, 
                             threshold=threshold, exclude_border=10,
                             exclude_adjacent=True)
    peaks = refine_peaks(normalize(prediction), peaks, 
                         disk(2), model='polynomial')

    return prediction, peaks

def load_CNN(graph_path, size, num_gpus=1):
    "Load the Keras neural net, and return a Model."
    kernel_num=32
    image_features=1
    num_classes=1

    if num_gpus == 1:
        x = keras.Input(shape=size+(image_features,))
        model = net.graph(x, output_features=num_classes)
        model.load_weights(graph_path)
    else:
        with tf.device('/cpu:0'):
            x = keras.Input(shape=size+(image_features,))
            model = net.graph(x, output_features=num_classes)
            model.load_weights(graph_path)
        model = multi_gpu_model(model, gpus=num_gpus)

    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    
    return model




analyze_tree(input, filetype=filetype)

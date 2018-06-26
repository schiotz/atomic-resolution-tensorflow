"""Calculate precision and recall on the testset."""

from glob import glob
import numpy as np
#import tensorflow as tf
import keras
from keras.utils import multi_gpu_model
import tensorflow as tf
from temnn.knet import net
#from temdata.dataset import DataEntry,DataSet
from temnn.net.dataset import DataEntry,DataSet
from pyqstem.imaging import CTF
import matplotlib.pyplot as plt
# Peak detection
from stm.preprocess import normalize
from stm.feature.peaks import find_local_peaks, refine_peaks
from skimage.morphology import disk
from scipy.spatial import cKDTree as KDTree
import sys
import os
from collections import deque
from multiprocessing import Pool

# Data folders
data_dir = "data/cluster-100-single-class/"
validation_dir = "data/cluster-100-single-class-test/"

if len(sys.argv) >= 2:
    graph_dir = sys.argv[1]
else:
    graph_dir = 'kgraphs'

graph_path = os.path.join(graph_dir, 'clusters-{:d}.h5')
result = os.path.join(graph_dir, 'noisecurve.dat')

# Microscope parameters
#sampling=0.11953 #244.8/2048
sampling=0.10
Cs=-15e4
defocus=90
focal_spread=30
blur=1.5
#dose=5*10**2
dose = 0
#mtf_param=[1,0,4.89683027e-01,2.34644273e+00]
mtf_param=[1,0,0.45,2.5]


num_gpus = 1
batch_size = 8 * num_gpus

def load(data_dir):
    "Load data folder."
    #models=sorted(glob(data_dir+"model/model_*.cfg"))
    waves=sorted(glob(data_dir+"wave/wave_*.npz"))
    #labels=sorted(glob(data_dir+"label/label_*.npy"))
    points=sorted(glob(data_dir+"points/points_*.npz"))
    #entries=[DataEntry(model,wave,label) for model,wave,label in zip(models,waves,labels)]
    entries = [DataEntry(wave=w, points=p) for w,p in zip(waves,points)]

    return DataSet(entries)

def load_CNN(graph_path, num_gpus=1):
    "Load the Keras neural net, and return a Model."
    size=(360,360)
    #size=(328,328)
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
    
    return (x, model)

def makeimage(entry, size):
    entry.load()    
    ctf=CTF(defocus=defocus,Cs=Cs,focal_spread=focal_spread)
    entry.create_image(ctf,sampling,blur,dose,mtf_param)
    entry.create_label(sampling, width = int(.4/sampling), num_classes=False)

    entry.local_normalize(12./sampling, 12./sampling)

    shape = entry._image.shape[1:3]
    assert not ((size[0] > shape[0]) != (size[1] > shape[1]))
    if shape[0] > size[1]:
        assert shape[1] >= size[1]
        entry.random_crop(size, sampling)
    elif shape[0] < size[1]:
        assert shape[1] <= size[1]
        entry.pad(size)
    else:
        assert shape[1] == size[1]
        
    image,label=entry.as_tensors()
    entry.reset()
    
    return image, label

# Use multiprocessing to generate many sample datasets
class MakeImages:
    def __init__(self, data, imagesize):
        self.data = data
        self.precomputed = []
        self.batchsize = 100
        self.imagesize = np.array(imagesize)

    def precompute(self):
        #print("Precomputing {} images.".format(self.batchsize), flush=True)
        entries = self.data.next_batch(self.batchsize, shuffle=False)
        imagesizes = self.imagesize[np.newaxis,:] * np.ones(self.batchsize, int)[:,np.newaxis]
        with Pool() as pool:
            self.precomputed = deque(pool.starmap(makeimage,  zip(entries, imagesizes)))

    def next_example(self):
        if not self.precomputed:
            self.precompute()
        return self.precomputed.popleft()

    def get_all_examples(self):
        "Get an example from each data point."
        n = self.data.num_examples
        entries = self.data.next_batch(n, shuffle=False)
        imagesizes = self.imagesize[np.newaxis,:] * np.ones(n, int)[:,np.newaxis]
        images = []
        labels = []
        with Pool() as pool:
            for img, lbl in pool.starmap(makeimage, zip(entries, imagesizes)):
                images.append(img)
                labels.append(lbl)
        return np.concatenate(images), np.concatenate(labels)
    

def precision_recall(predicted, target, distance=6.0):
    """Precision and recall for peak positions"""
    # Precision: Number of correctly predicted peaks 
    # divided by number of target peaks
    if len(predicted) == 0:
        return (0.0, 1.0)
    if len(target) == 0:
        return (1.0, 0.0)
    tree = KDTree(target)
    x = tree.query(predicted, distance_upper_bound=distance)[0]
    precision = (x <= distance).sum() / len(predicted)
    # Recall: Number of target peaks that were found
    # divided by total number of target peaks
    tree = KDTree(predicted)
    x = tree.query(target, distance_upper_bound=distance)[0]
    recall = (x <= distance).sum() / len(target)
    return (precision, recall)

def evaluate_result(inference, label):
    "Evaluate the prediction for an image."
    # Find the peaks
    infer_peaks = find_local_peaks(inference[:,:,0], min_distance=25, 
                                   threshold=0.6, exclude_border=10,
                                   exclude_adjacent=True)
    label_peaks = find_local_peaks(label[:,:,0], min_distance=25, 
                                   threshold=0.6, exclude_border=10,
                                   exclude_adjacent=True)

    # Refine the peaks
    region = disk(2)
    infer_refined = refine_peaks(normalize(inference[:,:,0]), infer_peaks, 
                                region, model='polynomial')
    label_refined = refine_peaks(normalize(label[:,:,0]), label_peaks, 
                                region, model='polynomial')
    return precision_recall(infer_refined, label_refined)

image_size = (360,360)
data_train = load(data_dir)
imagestream_train = MakeImages(data_train, image_size)
n_train = data_train.num_examples
print("Number of training images:", n_train)
data_valid = load(validation_dir)
imagestream_valid = MakeImages(data_valid, image_size)
n_valid = data_valid.num_examples
print("Number of validation images:", n_train)

# Find the latest CNN
print("Looking for CNNs in files matching", graph_path)
i = 1
while os.path.exists(graph_path.format(i)):
    gr = graph_path.format(i)
    i += 1
print("Using CNN parameters in", gr)
x, model = load_CNN(gr, num_gpus)

with open(result, "wt") as outfile:
    for step, dose in enumerate((1e2, 2e2, 5e2, 1e3, 2e3, 5e3, 1e4)):
        print("Evaluating dose", dose, flush=True)
        
        linedata = [dose]
        for (n, imagestream) in ((n_train, imagestream_train), (n_valid, imagestream_valid)):
            #n = 25
            result = []

            print("Getting all images", flush=True)
            images, labels = imagestream.get_all_examples()
            print("Making preditions with CNN.", flush=True)
            predictions = model.predict(np.array(images), batch_size=batch_size)

            # Now we have an array with predicted images (predictions) and
            # one with expected images (labels).  We now need to calculate
            # precision and recall in parallel

            print("Processing predictions.", flush=True)
            with Pool() as pool:
                result = list(pool.starmap(evaluate_result, zip(predictions, labels)))
            
            result = np.array(result)
            precision = result[:,0].mean()
            recall = result[:,1].mean()
            linedata.extend((precision, recall))
        line = "{:5.1f}  {:8.6f}  {:8.6f}  {:8.6f}  {:8.6f}".format(*tuple(linedata))
        print("*****", line, flush=True)
        print(line, file=outfile, flush=True)

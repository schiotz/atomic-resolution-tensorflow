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
from evaluatepeaks import precision_recall, evaluate_result
import sys
import os
from collections import deque
from multiprocessing import Pool
import shutil
from natsort import natsorted

# Data folders
data_dir = "data/cluster-110-single-class/"
validation_dir = "data/cluster-110-single-class-test/"
#validation_dir = data_dir  # A big no-no

if len(sys.argv) >= 2:
    graph_dir = sys.argv[1]
else:
    graph_dir = 'kgraphs'

graph_path = os.path.join(graph_dir, 'clusters-*.h5')
result = os.path.join(graph_dir, 'scalecurve.dat')

# Microscope parameters
#sampling=0.11953 #244.8/2048
#sampling=0.088
Cs=-5e4
defocus=90
focal_spread=30
blur=1.5
#dose=5*10**2
dose = 5e2
#mtf_param=[1,0,4.89683027e-01,2.34644273e+00]
mtf_param=[1,0,0.38,2.5]


num_gpus = 1
batch_size = 8 * num_gpus

if 'LSB_MAX_NUM_PROCESSORS' in os.environ:
    maxcpu = int(os.environ['LSB_MAX_NUM_PROCESSORS'])
    print("Setting max number of CPUs to", maxcpu, flush=True)
else:
    maxcpu = None

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
    size=(248,248)
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
        with Pool(maxcpu) as pool:
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
        with Pool(maxcpu) as pool:
            for img, lbl in pool.starmap(makeimage, zip(entries, imagesizes)):
                images.append(img)
                labels.append(lbl)
        return np.concatenate(images), np.concatenate(labels)
    

image_size = (248,248)
data_train = load(data_dir)
imagestream_train = MakeImages(data_train, image_size)
n_train = data_train.num_examples
print("Number of training images:", n_train)
data_valid = load(validation_dir)
imagestream_valid = MakeImages(data_valid, image_size)
n_valid = data_valid.num_examples
print("Number of validation images:", n_valid)

# Keep a copy of this script for reference
shutil.copy2(__file__, graph_dir)

# Find the latest CNN
print("Looking for CNNs in files matching", graph_path)
print("Looking for CNNs in files matching", graph_path)
gr = list(natsorted(glob(graph_path)))[-1]
print("Using CNN parameters in", gr)
x, model = load_CNN(gr, num_gpus)

with open(result, "wt") as outfile:
    for step, sampling in enumerate(np.arange(0.1, 0.35, 0.01)):
        print("Evaluating sampling", sampling, flush=True)
        
        linedata = [sampling]
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
            with Pool(maxcpu) as pool:
                result = pool.starmap(evaluate_result,
                                      zip(predictions, labels, [sampling]*len(labels)))
            
            result = np.array(result)
            precision = result[:,0].mean()
            recall = result[:,1].mean()
            linedata.extend((precision, recall))
        line = "{:8.4f}  {:8.6f}  {:8.6f}  {:8.6f}  {:8.6f}".format(*tuple(linedata))
        print("*****", line, flush=True)
        print(line, file=outfile, flush=True)

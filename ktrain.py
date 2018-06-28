import os
from datetime import datetime
from glob import glob
import numpy as np
import matplotlib
matplotlib.use('AGG')

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import tensorflow as tf
import keras
from keras.utils import multi_gpu_model
from pyqstem.imaging import CTF
#from temnn.net import net
from temnn.knet import net
from temnn.net.dataset import DataSet, DataEntry
from temnn.net.labels import create_label
import sys
import os
import time
import platform
from collections import deque
from multiprocessing import Pool
import shutil

debug = 50

def load(data_dir):
    
    waves = sorted(glob(data_dir+"wave/wave_*.npz"))
    points = sorted(glob(data_dir+"points/points_*.npz"))

    entries = [DataEntry(wave=w, points=p) for w,p in zip(waves,points)]
    
    return DataSet(entries)

def show_example(image, label, text, filename):
    fig = plt.figure(figsize=(14,7))
    ax = fig.add_subplot(1, 2, 1)

    im = ax.imshow(image[0,:,:,0], interpolation='nearest', cmap='gray')
    divider = make_axes_locatable(ax)
    cax1 = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax = cax1)

    ax = fig.add_subplot(1, 2, 2)
    im = ax.imshow(label[0,:,:,0], cmap='jet')
    divider = make_axes_locatable(ax)
    cax2 = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im, cax = cax2)
        
    plt.tight_layout()
    #plt.show()
    fig.savefig(filename+'.png', bbox_inches='tight')
    with open(filename+'.txt', "wt") as info:
        info.write(text)
    plt.close(fig)
    
class randomscale:
    def __init__(self, rnd):
        self.rnd = rnd
        self.n = 0
    def __call__(self, low=0.0, high=1.0):
        r = self.rnd[self.n]
        self.n += 1
        return (high - low) * r + low
    def randint(self, low, high):
        r = self()
        ri = int(np.floor((high - low) * r) + low)
        assert low <= ri < high
        return ri

def makeimage(entry, size, imgnum, rndnums):
    """Make a TEM image.

    entry: A data entry containing at least an exit wave function.

    size:  Size of desired image in pixels (2-tuple).

    imgnum: Sequential number used when plotting in debug mode.

    rndnums: XX random numbers (uniformly in [0;1[).  This prevents
             trouble with random numbers when multiprocessing.
    """
    rnd = randomscale(rndnums)
    
    sampling = rnd(.09,.11)
    #Cs = rnd(-20,20) * 1e4
    Cs = rnd(-20,-5) * 1e4
    defocus = rnd(0,200)
    focal_spread = rnd(20,40)
    
    aberrations={'a22' : rnd(0, 50), 
                'phi22' : rnd(0, 2 * np.pi),
                } #'a40' : 1.4 * 10**6}   # Looks like it is replacing Cs in the code!
    
    dose = 10**rnd(2,4)
    
    c1=rnd(.95,1)   # 1.0 in paper
    c2=rnd(0,.01)  # c1 in paper
    c3=rnd(.4,.6)  # c2 in paper
    c4=rnd(2.,3.)  # c3 in paper
    
    mtf_param=[c1,c2,c3,c4]
    
    blur = rnd(0, 2.5) #rnd(5,7)
    
    entry.load()
    
    ctf=CTF(defocus=defocus,Cs=Cs,focal_spread=focal_spread,aberrations=aberrations)
    
    entry.create_image(ctf,sampling,blur,dose,mtf_param)
    
    entry.create_label(sampling, width = int(.4/sampling), num_classes=False)
    
    entry.local_normalize(12./sampling, 12./sampling)

    shape = entry._image.shape[1:3]
    assert not ((size[0] > shape[0]) != (size[1] > shape[1]))
    if shape[0] > size[1]:
        assert shape[1] >= size[1]
        entry.random_crop(size, sampling, randint=rnd.randint)
    elif shape[0] < size[1]:
        assert shape[1] <= size[1]
        entry.pad(size)
    else:
        assert shape[1] == size[1]
        
    entry.random_brightness(-.1, .1, rnd=rnd)
    entry.random_contrast(.9, 1.1, rnd=rnd)
    entry.random_gamma(.9, 1.1, rnd=rnd)
    
    entry.random_flip(rnd=rnd)
    image,label=entry.as_tensors()
    entry.reset()

    if debug is True or imgnum < debug:
        text = f"""sampling={sampling} 
Cs={Cs/1.0e4} 
defocus={defocus} 
focal_spread={focal_spread} 
a22={aberrations['a22']} 
dose={dose}
mtf0={mtf_param[0]}
mtf1={mtf_param[1]}
mtf2={mtf_param[2]}
mtf3={mtf_param[3]}
blur={blur}
"""
        fn = debug_dir + "img-{}".format(imgnum)
        show_example(image, label, text, fn)
    
    return image,label

# Use multiprocessing to generate many sample datasets
class MakeImages:
    def __init__(self, data, imagesize):
        self.data = data
        self.precomputed = []
        self.batchsize = 200
        self.imagesize = np.array(imagesize)
        self.n = 0

    def precompute(self):
        print("Precomputing {} images.".format(self.batchsize), flush=True)
        entries = self.data.next_batch(self.batchsize)
        sequence = np.arange(self.n, self.n + self.batchsize)
        self.n += self.batchsize
        rndnums = np.random.uniform(0.0, 1.0, size=(self.batchsize, 20))
        imagesizes = self.imagesize[np.newaxis,:] * np.ones(self.batchsize, int)[:,np.newaxis]
        assert imagesizes.shape == (self.batchsize, 2)
        with Pool() as pool:
            self.precomputed = deque(pool.starmap(makeimage, 
                                                    zip(entries, imagesizes, sequence, rndnums)))
            
    def next_example(self):
        if not self.precomputed:
            self.precompute()
        return self.precomputed.popleft()

   
def summary_image(y,size):
    return tf.reshape(tf.cast(tf.argmax(y,axis=3),tf.float32),(1,)+size+(1,))

if __name__ == "__main__":
    print("{}: Running on host '{}'".format(
        datetime.now().strftime("%Y%m%d-%H%M%S"),
        platform.node()
    ))
    if len(sys.argv) >= 2:
        folderlabel = '-' + sys.argv[1]
    else:
        folderlabel = ''
        
    data_dir = "data/cluster-110-single-class/"
    summary_dir = "summaries/" + datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    graph_path = 'graphs'+folderlabel+'/clusters-{}.h5'
    debug_dir = "debug/" +  datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
    
    graph_dir = os.path.dirname(graph_path)
    if graph_dir and not os.path.exists(graph_dir):
        os.makedirs(graph_dir)
    if debug:
        os.makedirs(debug_dir)

    # Keep a copy of this script for reference
    shutil.copy2(__file__, graph_dir)
        
    data = load(data_dir)

    numgpus = 4
    batch_size = 4

    image_size = (424,424) # spatial dimensions of input/output
    #image_size = (360,360) # spatial dimensions of input/output
    image_features = 1 # depth of input data
    num_classes = 1 # number of predicted class labels
    num_epochs = 100 # number of training epochs
    save_epochs = 5

    # restore = False # restore previous graph
    loss_type = 'binary_crossentropy' # mse or binary_cross_entropy
    #loss_type = 'mse' # mse or binary_cross_entropy
    
    num_in_epoch = data.num_examples//batch_size
    num_iterations=num_epochs*num_in_epoch
    
    #show_examples(data, image_size, n=4)
    
    outputcounter = 0

    assert(batch_size % numgpus == 0)

    imagestream = MakeImages(data, image_size)

    if numgpus > 1:
        with tf.device('/cpu:0'):
            # The master version of the model is locked onto a CPU, to
            # prevent slow GPU-GPU communication and out-of-memory
            # conditions on the hosting GPU.
            x = keras.Input(shape=image_size+(image_features,))
            serial_model = net.graph(x, output_features=num_classes)

        model = multi_gpu_model(serial_model, gpus=numgpus)
    else:
        x = keras.Input(shape=image_size+(image_features,))
        model = serial_model = net.graph(x, output_features=num_classes)
        
    model.compile(optimizer='rmsprop', loss=loss_type,
                  metrics=['accuracy'])
    
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    
    print("Starting timing")
    before = time.time()

    for epoch in range(num_epochs):
        for i in range(num_in_epoch):
            #image,label = next_example(data)
            image,label = imagestream.next_example()
            if batch_size > 1:
                image = [image]
                label = [label]
                for b in range(1, batch_size):
                    img2, lbl2 = imagestream.next_example()
                    image.append(img2)
                    label.append(lbl2)
                image = np.concatenate(image)
                label = np.concatenate(label)

            # Train
            #y = keras.utils.to_categorical(label,2)
            #model.train_on_batch(image, y)
            model.train_on_batch(image, label)

            # Print where we are
            print("Epoch: {}/{} Batch: {}/{}   [{}/{}]".format(epoch, num_epochs,
                                                               i, num_in_epoch,
                                                               (i + epoch*num_in_epoch)*batch_size,
                                                               num_iterations*batch_size),
                      flush=True)
        # Save 
        if (epoch+1) % save_epochs == 0:
            serial_model.save_weights(graph_path.format(epoch))
    
    totaltime = time.time() - before
    print("Time: {} sec  ({} hours)".format(totaltime, totaltime/3600))
              

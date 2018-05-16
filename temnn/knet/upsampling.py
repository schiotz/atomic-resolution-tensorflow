"""A bilinear upsampling layer for 2D images."""


from __future__ import absolute_import, print_function

#from keras.engine.topology import Layer
#from tensorflow.image import resize_images
import tensorflow as tf
from keras import backend as K
from keras import layers
import numpy as np

def bilinear_upsampling(x):
    """Function doing the upsampling by calling TensorFlow."""
    original_shape = K.int_shape(x)
    factor=2
    new_shape = tf.shape(x)[1:3]
    new_shape *= tf.constant(np.array([factor, factor]).astype('int32'))
    x = tf.image.resize_bilinear(x, new_shape)
    x.set_shape((None, original_shape[1] * factor if original_shape[1] is not None else None,
                 original_shape[2] * factor if original_shape[2] is not None else None, None))
    return x
    
def BilinearUpSampling2D(**kwargs):
    return layers.Lambda(bilinear_upsampling, **kwargs)


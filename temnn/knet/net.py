from __future__ import absolute_import, print_function

from keras.models import Model
from keras import layers, regularizers
#from tensorflow import name_scope
from .upsampling import BilinearUpSampling2D
from keras.layers.advanced_activations import PReLU
from keras import initializers

weight_decay = None

def graph(x, output_features, channels=32):
    """Define the CNN with an optional kernel size (default: 32)."""
    
    down1 = conv_res_conv(x, channels, name="down1")
    pool1 = pool_layer(down1, name="pool1")
    
    down2 = conv_res_conv(pool1, channels*2, name="down2")
    pool2 = pool_layer(down2, name="pool2")
    
    down3 = conv_res_conv(pool2, channels*4, name="down3")
    pool3 = pool_layer(down3, name="pool3")

    bridge = conv_res_conv(pool3, channels*8, name="bridge")

    up3 = upsample_layer(bridge, channels*4, name="up3")
    up3 = skip(up3, down3, name='skip3')
    up3 = conv_res_conv(up3, channels*4, name="up3")
    
    up2 = upsample_layer(up3, channels*2, name="up2")
    up2 = skip(up2, down2, name='skip2')
    up2 = conv_res_conv(up2, channels*2, name="up2")
    
    up1 = upsample_layer(up2, channels, name="up1")
    up1 = skip(up1, down1, name='skip1')
    up1 = conv_res_conv(up1, channels, name="up1")

    inference = score_layer(up1, channels=output_features)
    return Model(x, inference)

def tiny_graph(x, output_features, channels=32):
    down1 = conv_layer(x, channels, name="down1")
    pool1 = pool_layer(down1, name="pool1")
    
    down2 = conv_layer(pool1, channels*2, name="down2")
    pool2 = pool_layer(down2, name="pool2")

    bridge = conv_layer(pool2, channels*4, name="bridge")

    up2 = upsample_layer(bridge, channels*2, name="upsample2")
    up2 = skip(up2, down2)
    up2 = conv_layer(up2, channels*2, name="up2")
    
    up1 = upsample_layer(up2, channels, name="upsample1")
    up1 = skip(up1, down1)
    up1 = conv_layer(up1, channels, name="up1")

    inference = score_layer(up1, channels=output_features)
    return Model(x, inference)
    
def conv_res_conv(x, channels, name='conv_res_conv'):
    """Define a block in the network.

    The block consists of a convolutional layer, a residual block, and
    a convolutional layer.
    """
    x = conv_layer(x, channels, name=name+"/conv1")
    x = res_block(x, channels, name=name+"/resid")
    return conv_layer(x, channels, name=name+"/conv2")

def conv_layer(x, channels, kernel_size=3, name='conv'):
    """A single convolutional layer."""
    conv = layers.Conv2D(channels, kernel_size, padding='same',
                             kernel_regularizer=regul(),
                             kernel_initializer='RandomNormal',
                             bias_initializer=get_bias_init(),
                             name=name)
    x = prelu(conv(x), name=name)
    # Normalize
    return layers.BatchNormalization(name=name+'_b_norm')(x)

def res_block(x, channels, name='res_block'):
    "A residual block."
    y = conv_layer(x, channels, name=name+'/conv_1')
    y = conv_layer(y, channels, name=name+'/conv_2')
    y = conv_layer(y, channels, name=name+'/conv_3')
    return skip(x, y, name=name+'/add')

#def skip(x, y, name):
#    return layers.concatenate([x, y], name=name)

def skip(x, y, name):
    return layers.add([x, y], name=name)

def pool_layer(x, name='pool'):
    return layers.MaxPooling2D(pool_size=2, padding='same', name=name)(x)


if False:
    def upsample_layer(x, channels, name='upsample'):
        conv = layers.Conv2DTranspose(channels, kernel_size=3, padding='same',
                                          strides=2, 
                                          kernel_regularizer=regul(),
                                          kernel_initializer='RandomNormal',
                                          bias_initializer=get_bias_init(),
                                          name=name)
        return prelu(conv(x), name=name)
else:
    # Bilinear upsampling.
    def upsample_layer(x, channels, name='upsample'):
        #x = layers.UpSampling2D(size=2, name=name+'/upsamp2D')(x)
        x = BilinearUpSampling2D(name=name+'/upsamp2D')(x)
        return conv_layer(x, channels, kernel_size=1, name=name+'/up_conv')


def score_layer(x, channels, kernel_size=1, name="score"):
    "The final layer."
    if channels > 1:
        act = 'softmax'
    else:
        act = 'sigmoid'
    conv = layers.Conv2D(channels, kernel_size,
                             activation=act,
                             padding='same',
                             kernel_regularizer=regul(),
                             kernel_initializer='RandomNormal',
                             bias_initializer=get_bias_init(),
                             name=name)
    return conv(x)

# Regularization
def regul():
    if weight_decay:
        return regularizers.l2(weight_decay)
    else:
        return None

def get_bias_init():
    return initializers.Constant(0.1)

# Parametric RELU activiation
def prelu(x, name=None):
    p = PReLU(shared_axes=[1,2], name=name+"/PReLU", alpha_initializer=initializers.Constant(0.01))
    return p(x)

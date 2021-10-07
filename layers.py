from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.framework import tensor_shape
import tensorflow as tf
import numpy as np
import time
#tf.config.experimental.enable_mlir_graph_optimization()


class PixelateLayer(Layer):
  """The Layer implement the following activation, with 1+1/stepSize output neurons per input neuron.

For the output neuron i, and the input x:

  `f_i(x) = 0`                                  for                x/Ntot<(i-1)*stepSize,
  `f_i(x) = [x/Ntot - (i-1)*stepSize]/stepSize,        for (i-1)*stepSize<x/Ntot<    i*stepSize
  `f_i(x) = [(i+1)*stepSize-x/Ntot]/stepSize,        for     i*stepSize<x/Ntot<(i+1)*stepSize
  `f_i(x) = 0`                                  for (i+1)*stepSize<x/Ntot
with Ntot = sum of inputs neurons.
  ```
  Usage:

     from PixelateHistogram.layers import *
     #Very simple histogram passing trough 1-layer PixelateLayer
     histo_1 = tf.constant([[[1000], [0], [0]]]) #only the first bin is filled
     PL1 = PixelateLayer(stepSize=0.1, flatten=False)
     pixels =PL1(histo_1)
     ->
     tf.Tensor(
[[[[0.]
   [0.]
   [0.]
   [0.]
   [0.]
   [0.]
   [0.]
   [0.]
   [0.]
   [0.]
   [1.]]
   
  [[1.]
   [0.]
   [0.]
   [0.]
   [0.]
   [0.]
   [0.]
   [0.]
   [0.]
   [0.]
   [0.]]

  [[1.]
   [0.]
   [0.]
   [0.]
   [0.]
   [0.]
   [0.]
   [0.]
   [0.]
   [0.]
   [0.]]]], shape=(1, 3, 11, 1), dtype=float32)

   #Very simple histogram passing trough 3-layer PixelateLayer (additional layers to account for statistical uncertainty
   histo_2 = tf.constant([[[1000], [100], [100]]])
   PL2 = PixelateLayer(stepSize=0.1, n_sigma=1, flatten=False)
   ->
tf.Tensor(
[[[[0.         0.         0.        ]
   [0.         0.         0.        ]
   [0.         0.         0.        ]
   [0.         0.         0.        ]
   [0.         0.         0.        ]
   [0.         0.         0.        ]
   [0.         0.         0.        ]
   [0.         0.         0.        ]
   [0.9301901  0.66666675 0.4031433 ]
   [0.06981003 0.33333337 0.5968567 ]
   [0.         0.         0.        ]]

  [[0.25       0.16666663 0.08333337]
   [0.75       0.83333325 0.91666675]
   [0.         0.         0.        ]
   [0.         0.         0.        ]
   [0.         0.         0.        ]
   [0.         0.         0.        ]
   [0.         0.         0.        ]
   [0.         0.         0.        ]
   [0.         0.         0.        ]
   [0.         0.         0.        ]
   [0.         0.         0.        ]]

  [[0.25       0.16666663 0.08333337]
   [0.75       0.83333325 0.91666675]
   [0.         0.         0.        ]
   [0.         0.         0.        ]
   [0.         0.         0.        ]
   [0.         0.         0.        ]
   [0.         0.         0.        ]
   [0.         0.         0.        ]
   [0.         0.         0.        ]
   [0.         0.         0.        ]
   [0.         0.         0.        ]]]], shape=(1, 3, 11, 3), dtype=float32)

  More examples in https://github.com/sam-cal/PixelateHistogramExample

  Input shape:
    Arbitrary.
  Output shape:
    Shape as the (input, 1+1/stepSize, 1+2*n_sigma). It can be changed using 'flatten' option
  Arguments:
      stepSize: float. Step size to go from 0 to 1
      n_sigma:  integer. If non-0, the ouput will have an additional 
                dimension, that relects the statistical uncertainty.
                One input x is transformed into [f_i(x-n_sigma*sqrt(x)), f_i(x-(n_sigma-1)*sqrt(x)), ... ,f_i(x), ..., f_i(x+n_sigma*sqrt(x))]
      flatten: if true, the output shape is (input * (1+1/stepSize) * (1+2*n_sigma)).
  """

  def __init__(self, stepSize=0.01, n_sigma=0, flatten=False, minimum=0., maximum=1., **kwargs):
    super(PixelateLayer, self).__init__(dynamic=True, **kwargs)
    self.stepSize = stepSize
    self.n_sigma = n_sigma
    self.flatten = flatten
    self.built = True
    
    self.sigmas = self.getSigmas()
    self.steps = self.getSteps(minimum, maximum)
    
  def getSteps(self, mini, maxi):
     return np.float32(np.arange(mini,maxi+self.stepSize, self.stepSize)[:])
  def getSigmas(self):
     return np.arange(-self.n_sigma, self.n_sigma+1 )

  @tf.function
  def getMeanWidth(self,x):
    mean  = K.sum(x, axis=1)
    mean  =  tf.tensordot(mean, self.steps, axes=0) # transform fraction into number of events
    width =  mean[:,:,1:]-mean[:,:,:-1]
    width =tf.concat([width,tf.expand_dims(width[:,:,0], axis=-1)], axis=-1)
    return mean,width

  @tf.function
  def buildResult(self,x, mean, width):
  
    for sss in self.sigmas:
    
        SQRT= sss*tf.sqrt(x)
        
        XmMEANpWIDTH =  x + SQRT -mean +width
        MEANpWIDTHmX = -x - SQRT +mean +width
        
        XmMEANpWIDTH = tf.clip_by_value(XmMEANpWIDTH, clip_value_min=0, clip_value_max=width) / width
        MEANpWIDTHmX = tf.clip_by_value(MEANpWIDTHmX, clip_value_min=0, clip_value_max=width) / width
        
        RESULT_temp = (XmMEANpWIDTH + MEANpWIDTHmX) - 1

        if sss == self.sigmas[0]:
           RESULT = tf.expand_dims(RESULT_temp, axis=-1)
        else:
           RESULT = tf.concat([RESULT, tf.expand_dims(RESULT_temp, axis=-1)], axis=-1)
        
    return RESULT
        
  #@tf.function
  def call(self, inputs):    
    if not self.built: self.build(inputs.shape)
    
    x = K.cast_to_floatx(inputs)
    mean,width = self.getMeanWidth(x)
    RESULT = self.buildResult(x, mean, width)
    
    if self.flatten:
        size=1
        for i in RESULT.shape[1:]: size*=i
        RESULT = tf.reshape(RESULT, (-1, size))

    return  RESULT
   
        
    
  def get_config(self):
    config = {'stepSize': self.stepSize}
    base_config = super(PixelateLayer, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    input_shape = tensor_shape.TensorShape(input_shape)
    input_shape = input_shape.with_rank_at_least(2)
    if tensor_shape.dimension_value(input_shape[-1]) is None:
      raise ValueError(
          'The innermost dimension of input_shape must be defined, but saw: %s'
          % input_shape)
    if self.flatten:
        shape = input_shape[:-2].concatenate( input_shape[-2]*len(self.steps)*len(self.sigmas))
    else:
        shape = input_shape[:-1].concatenate( len(self.steps) )
        shape = shape.concatenate( len(self.sigmas) )
    print ("compute_output_shape:", shape)
    return shape



class PoissonianNoise(Layer):
  """
  Inspired from GaussianNoise
  
  This is useful to mitigate overfitting
  (you could see it as a form of random data augmentation).
  
  Poissonian Noise is a natural choice as corruption process
  for counting experiment.
  
  As it is a regularization layer, it is only active at training time.
  
  Call arguments:
    inputs: Input tensor (of any rank).
    training: Python boolean indicating whether the layer should behave in
      training mode (adding noise) or in inference mode (doing nothing).
  
  Input shape:
    Arbitrary. Use the keyword argument `input_shape`
    (tuple of integers, does not include the samples axis)
    when using this layer as the first layer in a model.
  
  Output shape:
    Same shape as input.
  """

  def __init__(self, **kwargs):
    super(PoissonianNoise, self).__init__(**kwargs)
    #self.supports_masking = True
    
  def call(self, inputs, training=None):

    def noised():
      return tf.squeeze( tf.random.poisson(shape=(1,), lam=inputs, dtype=inputs.dtype) , axis=[0])
    
    return K.in_train_phase(noised, inputs, training=training)

  def get_config(self):
    base_config = super(PoissonianNoise, self).get_config()
    return dict(list(base_config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape



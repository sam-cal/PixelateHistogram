from tensorflow.keras.layers import Layer
from tensorflow.keras import backend as K
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.framework import tensor_shape
import tensorflow as tf
import numpy as np


class PixelateLayer(Layer):
  """The Layer implement the following activation, with 1/stepSize+1 output neurons per input neuron.

For the output neuron i, and the input x:

  `f_i(x) = 0`                          for                x/Ntot<(i-1)*stepSize,
  `f_i(x) = relu(x)/x,			for (i-1)*stepSize<x/Ntot<    i*stepSize
  `f_i(x) = relu(x)/x,			for     i*stepSize<x/Ntot<(i+1)*stepSize
  `f_i(x) = 0`                          for (i+1)*stepSize<x/Ntot
with Ntot = sum of inputs neurons.
  ```
  Usage:
from layers import *
layer = PixelateLayer()
tensor = tf.constant([range(10, 12)], dtype=tf.float32)
output = layer(tensor)
output.numpy()
-> array([[[0.        , 0.        , 0.        , 0.        , 0.        ,
         0.18350339, 0.43305337, 0.6464468 , 0.8333334 ],
        [0.        , 0.        , 0.        , 0.        , 0.        ,
         0.        , 0.24407113, 0.46967006, 0.6666666 ]]], dtype=float32)
  Input shape:
    Arbitrary.
  Output shape:
    Shape as the (input, (maxValue-minValue)/stepSize).
  Arguments:
      minValue: float. Mininmal value form `mean`
      maxValue: float. Maxinmal value form `mean`
      stepSize: float. Step size to go from minValue to maxValue
      heigh: float. Heigh of the triangle
  Activations with generated with the following parameters:
      mean: A scalar, "center" of the triangle.
      width: float. Width of the triangle. By default = 3*sqrt(mean).
      heigh: float. Heigh of the triangle
      
       v5: input value expressed as the barycenter of the 2 closest values
  """

  def __init__(self, stepSize=0.01, n_sigma=0, flatten=False, **kwargs):
    super(PixelateLayer, self).__init__(dynamic=True, **kwargs)
    self.stepSize = stepSize
    self.n_sigma = n_sigma
    self.flatten = flatten
    self.built = True
    
    self.sigmas = self.getSigmas()
    self.steps = self.getSteps()
  
  def getSteps(self):
     return np.float32(np.arange(0.,1.+self.stepSize, self.stepSize)[:])
  def getSigmas(self):
     return np.arange(-self.n_sigma, self.n_sigma+1 )
  
  def call(self, inputs):    
    if not self.built: self.build(inputs.shape)
    
    x = K.cast_to_floatx(inputs)
    mean  = K.sum(x, axis=1)
    mean  =  tf.tensordot(mean, self.steps, axes=0) # transform fraction into number of events
    width =  mean[:,:,1:]-mean[:,:,:-1]
    width =np.dstack((np.expand_dims(width[:,:,0], axis=-1),width))
    List = []
    for sss in self.sigmas:
    
        XmMEANpWIDTH =  x + sss*tf.sqrt(x) -mean +width
        MEANpWIDTHmX = -x - sss*tf.sqrt(x) +mean +width
        XmMEANpWIDTH = tf.clip_by_value(XmMEANpWIDTH, clip_value_min=0, clip_value_max=width) / width
        MEANpWIDTHmX = tf.clip_by_value(MEANpWIDTHmX, clip_value_min=0, clip_value_max=width) / width
        RESULT_temp = (XmMEANpWIDTH + MEANpWIDTHmX) - 1
        List+=[RESULT_temp]
    
    RESULT = tf.stack(List, axis=-1)
    
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
        print ("compute_output_shape:",input_shape[:-2].concatenate( input_shape[-2]*len(self.steps) ))
        return input_shape[:-2].concatenate( input_shape[-2]*len(self.steps) *len(self.sigmas))
    else:
        print ("compute_output_shape:",input_shape[:-1].concatenate( len(self.steps),len(self.sigmas) ))  
        return input_shape[:-1].concatenate( len(self.steps) )





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
    
  def call(self, inputs):

    def noised():
      return tf.squeeze( tf.random.poisson(shape=(1,), lam=inputs, dtype=inputs.dtype) , axis=[0])
    
    return K.in_train_phase(noised, inputs)

  def get_config(self):
    base_config = super(PoissonianNoise, self).get_config()
    return dict(list(base_config.items()))

  @tf_utils.shape_type_conversion
  def compute_output_shape(self, input_shape):
    return input_shape



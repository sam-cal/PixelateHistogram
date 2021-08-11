from tensorflow.keras import backend as K
import tensorflow as tf

def Significance(y_true, y_pred):
    '''
    Renormalize the model to the fit output (expected to be closer to the data)
    '''
    debug=0
    y_true = K.clip(y_true, 1e-15, None)
    y_pred = K.clip(y_pred, 1e-15, None)
    if debug: y_true = print_tensor(y_true, 'y_true:')
    if debug: y_pred = print_tensor(y_pred, 'y_pred:')
    
    ratio = tf.repeat(K.sum(y_pred, axis=-1)/K.sum(y_true, axis=-1), tf.shape(y_true)[-1])
    ratio = tf.reshape(ratio, tf.shape(y_true))
    if debug: ratio = print_tensor(ratio, 'ratio:')
    
    
    result = K.mean(y_true * ratio * K.log(y_true *ratio / y_pred) - (y_true*ratio - y_pred))
    #result = K.clip(result, 1e-15, 1e6)
    if debug: result = print_tensor(result, 'res :')
    return result


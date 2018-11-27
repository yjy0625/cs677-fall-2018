import tensorflow as tf

def shape(tensor):
    return tensor.get_shape().as_list()

def conv2d(inputs, num_outputs, 
           kernel_size=(3,3), 
           stride=1,
           padding='same',
           activation=tf.nn.relu, 
           initializer=tf.contrib.layers.xavier_initializer(),
           name=None):
    output = tf.layers.conv2d(inputs, num_outputs, kernel_size, stride,
                            padding=padding,
                            activation=activation, 
                            kernel_initializer=initializer,
                            name=name)
    print('[Layer: conv2d] {} {}'.format(name, shape(output)))
    return output

def maxpool2d(inputs, 
              kernel_size=(2,2), 
              stride=2, 
              padding='same', 
              name=None):
    output = tf.layers.max_pooling2d(inputs, 
                                     kernel_size, 
                                     stride, 
                                     padding, 
                                     name=name)
    print('[Layer: maxpool2d] {} {}'.format(name, shape(output)))
    return output

def deconv2d(inputs, num_outputs,
             kernel_size=(2,2), 
             stride=2,
             padding='same',
             activation=tf.nn.relu,
             initializer=tf.contrib.layers.xavier_initializer(),
             name=None):
    output = tf.layers.conv2d_transpose(inputs, num_outputs, kernel_size, stride,
                                       padding=padding,
                                       activation=activation,
                                       kernel_initializer=initializer,
                                       name=name)
    print('[layer: deconv2d] {} {}'.format(name, shape(output)))
    return output

def bilinear_deconv2d(inputs, num_outputs,
                      kernel_size=(2,2),
                      stride=2,
                      padding='same',
                      activation=tf.nn.relu,
                      initializer=tf.contrib.layers.xavier_initializer(),
                      name=None):
    h = int(inputs.get_shape()[1]) * stride
    w = int(inputs.get_shape()[2]) * stride
    with tf.variable_scope(name):
        output = tf.image.resize_bilinear(inputs, [h, w])
        output = conv2d(output, num_outputs, kernel_size, padding=padding,
                        activation=activation, initializer=initializer)
    print('[layer: bilinear-deconv2d] {} {}'.format(name, shape(output)))
    return output

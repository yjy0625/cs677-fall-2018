import tensorflow as tf
from ops import *

class Model(object):
    def __init__(self, config, is_train=True):
        self.config = config
        self.input_height = self.config.data_info[0]
        self.input_width = self.config.data_info[1]
        self.c_dim = self.config.data_info[2]
        self.learning_rate = self.config.learning_rate
        self.momentum = self.config.momentum
        self.use_bilinear_deconv2d = self.config.use_bilinear_deconv2d
        
        self.images = tf.placeholder(name='images', dtype=tf.float32, 
                                     shape=[None] + self.config.data_info)
        self.labels = tf.placeholder(name='labels', dtype=tf.int32, 
                                     shape=[None] + self.config.data_info[:2])
        self.is_training = tf.placeholder(name='is_training', dtype=tf.bool, 
                                          shape=[])
        
        self.build_model()
    
    def build_model(self):
        print('[Input] {}'.format(shape(self.images)))
            
        # logits & predictions
        logits = self.fcn_32(self.images)
        self.probs = tf.nn.sigmoid(logits)
        self.pred = tf.cast(tf.round(self.probs), tf.int32)
        
        print('[Logits] {}'.format(shape(logits)))
        print('[Probs] {}'.format(shape(self.probs)))
        print('[Pred] {}'.format(shape(self.pred)))
        
        # loss
        self.loss = tf.reduce_mean(tf.boolean_mask(
                tf.nn.sigmoid_cross_entropy_with_logits(
                    labels=tf.to_float(self.labels), logits=logits),
                tf.not_equal(self.labels, -1)
            ))

        # pixel-level iou
        sum_logical_and = lambda a, b: tf.count_nonzero(tf.logical_and(a, b))
        tp = sum_logical_and(tf.equal(self.labels, 1), tf.equal(self.pred, 1))
        fp = sum_logical_and(tf.equal(self.labels, 0), tf.equal(self.pred, 1))
        fn = sum_logical_and(tf.equal(self.labels, 1), tf.equal(self.pred, 0))
        self.tp, self.fp, self.fn = tp, fp, fn
        self.pixel_level_iou = tp / (tp + fp + fn)
        
        # optimization
        optimizer = tf.train.MomentumOptimizer(self.learning_rate, self.momentum)
        self.global_step = tf.train.get_or_create_global_step(graph=None)
        self.optimizer_op = optimizer.minimize(self.loss, 
                                               global_step=self.global_step)
        
        # scalar summaries
        self.add_summary("global_step", self.global_step)
        self.add_summary("loss", self.loss)
        self.add_summary("pixel_level_iou", self.pixel_level_iou)
        
        # image summaries
        mask = lambda x: tf.to_float(x) * tf.to_float(self.labels >= 0)
        vis = tf.stack([
            self.images, 
            self.visualize_labels(mask(self.labels), self.images), 
            self.visualize_labels(self.probs, self.images),
            self.visualize_labels(self.pred, self.images)
        ]) # (4, B, H, W, C)
        vis = tf.reshape(
            tf.transpose(vis, (1, 0, 2, 3, 4)),
            (-1, 4 * self.input_height, self.input_width, self.c_dim)
        ) # (B, 4 * H, W, C)
        self.add_summary("image", vis, summary_type='image',
                         collections=['test'])
        vis = vis[:, ::3, ::3, :] # resize image
        self.add_summary("image", vis, summary_type='image',
                         collections=['train', 'val'])
        
        # summary ops
        self.train_summary_op = tf.summary.merge_all(key='train')
        self.val_summary_op = tf.summary.merge_all(key='val')
        self.test_summary_op = tf.summary.merge_all(key='test')
    
    def fcn_32(self, images, scope='fcn'):
        with tf.variable_scope(scope):
            _ = images
            
            blocks = [(64, 2), (128, 2), (256, 3), (512, 3), (512, 3)]
            for ix, block in enumerate(blocks):
                num_outputs, num_conv_layers = block
                for conv_layer_ix in range(num_conv_layers):
                    _ = conv2d(_, num_outputs, 
                            name='conv{}_{}'.format(ix + 1, conv_layer_ix + 1))
                _ = maxpool2d(_, name='pool{}'.format(ix + 1))
                    
            _ = conv2d(_, 4096, (7, 7), name='fc6')
            _ = conv2d(_, 4096, (1, 1), name='fc7')
            _ = conv2d(_, 1, (1, 1), name='fc8')
            
            if self.use_bilinear_deconv2d:
                logits = bilinear_deconv2d(_, 1, (64, 64), 32,
                                           activation=None,
                                           name='deconv9')
            else:
                logits = deconv2d(_, 1, (64, 64), 32, 
                                  activation=None,
                                  name='deconv9')
        
        return tf.squeeze(logits, -1)
    
    def add_summary(self, name, value, 
                    summary_type='scalar', 
                    collections=['train', 'val', 'test']):
        if summary_type == 'scalar':
            for collection in collections:
                tf.summary.scalar('{}_{}'.format(collection, name), 
                                  value, 
                                  collections=[collection])
        elif summary_type == 'image':
            for collection in collections:
                tf.summary.image('{}_{}'.format(collection, name), 
                                 value, 
                                 collections=[collection])
            
    def visualize_labels(self, labels, images, production=True):
        rgb_labels = tf.to_float(tf.concat([tf.expand_dims(labels, -1)] * 3, -1))
        if production:
            return rgb_labels * tf.constant([0., 0., 1.]) + tf.constant([1., 0., 0.])
        else:
            return rgb_labels * images + (1.0 - rgb_labels) * images * 0.1

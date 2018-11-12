import tensorflow as tf

class Model(object):
    def __init__(self, config, is_train=True):
        self.config = config
        self.input_height = self.config.data_info[0]
        self.input_width = self.config.data_info[1]
        self.c_dim = self.config.data_info[2]
        self.learning_rate = self.config.learning_rate
        self.apply_batch_norm = self.config.apply_batch_norm
        self.add_more_layers = self.config.add_more_layers
        self.larger_filter_size = self.config.larger_filter_size
        self.num_superclasses = self.config.num_superclasses
        self.num_classes = self.config.num_classes
        
        self.coarse_label_names = self.config.coarse_label_names
        self.fine_label_names = self.config.fine_label_names
        
        self.images = tf.placeholder(name='images', dtype=tf.float32, shape=[None, self.input_height, self.input_width, self.c_dim])
        self.coarse_labels = tf.placeholder(name='coarse_labels', dtype=tf.int32, shape=[None])
        self.fine_labels = tf.placeholder(name='fine_labels', dtype=tf.int32, shape=[None])
        self.is_training = tf.placeholder(name='is_training', dtype=tf.bool, shape=[])
        
        self.build_model()
    
    def build_model(self):
        # get logits
        logits = self.lenet(self.images, is_training=self.is_training)
        self.pred = tf.cast(tf.argmax(logits, axis=1), tf.int32)
        
        # build loss
        self.loss = tf.losses.softmax_cross_entropy(tf.one_hot(self.fine_labels, self.num_classes), logits)
        
        # build optimizer ops
        # NOTE: the control dependency is to ensure update of variables in batchnorm layer
        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        self.global_step = tf.train.get_or_create_global_step(graph=None)
        with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
            self.optimizer_op = optimizer.minimize(self.loss, global_step=self.global_step)
        
        # metrics
        coarse_labels_dn = tf.transpose(tf.one_hot(self.coarse_labels, self.num_superclasses))
        fine_labels_dn = tf.transpose(tf.one_hot(self.fine_labels, self.num_classes))
        
        top_1 = tf.cast(tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), self.fine_labels), tf.float32)
        self.accuracy = tf.reduce_mean(top_1)
        self.per_superclass_accuracy = tf.reduce_mean(coarse_labels_dn * top_1, axis=1)
        self.per_class_accuracy = tf.reduce_mean(fine_labels_dn * top_1, axis=1)
        
        top_5 = tf.reduce_sum(tf.cast(tf.equal(tf.nn.top_k(logits, k=5, sorted=True)[1], 
                tf.expand_dims(self.fine_labels, 1)), tf.float32), axis=1)
        self.top_5_accuracy = tf.reduce_mean(top_5)
        self.top_5_per_superclass_accuracy = tf.reduce_mean(coarse_labels_dn * top_5, axis=1)
        self.top_5_per_class_accuracy = tf.reduce_mean(fine_labels_dn * top_5, axis=1)
        
        self.confusion_matrix = tf.confusion_matrix(self.pred, self.fine_labels)
        
        # add summaries to Tensorboard
        self.add_summary("global_step", self.global_step)
        self.add_summary("loss", self.loss)
        self.add_summary("top_1/accuracy", self.accuracy)
        self.add_summary("top_5/accuracy", self.top_5_accuracy)
        for c in range(self.num_classes):
            self.add_summary("top_1/per_class_accuracy/{:03d}_{:s}" \
                             .format(c, self.fine_label_names[c]), self.per_class_accuracy[c])
            self.add_summary("top_5/per_class_accuracy/{:03d}_{:s}" \
                             .format(c, self.fine_label_names[c]), self.top_5_per_class_accuracy[c])
        for c in range(self.num_superclasses):
            self.add_summary("top_1/per_superclass_accuracy/{:02d}_{:s}" \
                             .format(c, self.coarse_label_names[c]), self.per_superclass_accuracy[c])
            self.add_summary("top_5/per_superclass_accuracy/{:02d}_{:s}" \
                             .format(c, self.coarse_label_names[c]), self.top_5_per_superclass_accuracy[c])

        # summaries
        self.train_summary_op = tf.summary.merge_all(key='train')
        self.val_summary_op = tf.summary.merge_all(key='val')
        self.test_summary_op = tf.summary.merge_all(key='test')
        
    def lenet(self, images, num_classes=100, is_training=False, scope='lenet'):
        initializer = tf.contrib.layers.xavier_initializer()
        
        with tf.variable_scope(scope):
            _ = images
            print('[Input] {}'.format(_.get_shape().as_list()))

            filter_size = 64 if self.larger_filter_size else 32
            if self.add_more_layers:
                for i in range(2):
                    _ = tf.layers.conv2d(_, filter_size, (3, 3), 1, 
                                 activation=tf.nn.relu, 
                                 kernel_initializer=initializer,
                                 name='conv1-{}'.format(i + 1))
                    print('[Layer: conv2d] {} {}'.format('conv1-{}'.format(i + 1), _.get_shape().as_list()))
                    
                    if self.apply_batch_norm:
                        _ = tf.layers.batch_normalization(_, training=is_training, name='norm1-{}'.format(i + 1))
                        print('[Layer: batchnorm] norm1-{}'.format(i + 1))
            else:
                _ = tf.layers.conv2d(_, filter_size, (5, 5), 1, 
                                     activation=tf.nn.relu, 
                                     kernel_initializer=initializer,
                                     name='conv1')
                print('[Layer: conv2d] {} {}'.format('conv1', _.get_shape().as_list()))
            
                if self.apply_batch_norm:
                    _ = tf.layers.batch_normalization(_, training=is_training, name='norm1')
                    print('[Layer: batchnorm] norm1')

            _ = tf.layers.max_pooling2d(_, (2, 2), 2, 'SAME', name='pool1')
            print('[Layer: maxpool2d] {} {}'.format('pool1', _.get_shape().as_list()))
            
            filter_size = 128 if self.larger_filter_size else 64
            if self.add_more_layers:
                for i in range(2):
                    _ = tf.layers.conv2d(_, filter_size, (3, 3), 1, 
                                 activation=tf.nn.relu, 
                                 kernel_initializer=initializer,
                                 name='conv2-{}'.format(i + 1))
                    print('[Layer: conv2d] {} {}'.format('conv2-{}'.format(i + 1), _.get_shape().as_list()))
                    
                    if self.apply_batch_norm:
                        _ = tf.layers.batch_normalization(_, training=is_training, name='norm2-{}'.format(i + 1))
                        print('[Layer: batchnorm] norm2-{}'.format(i + 1))
            else:
                _ = tf.layers.conv2d(_, filter_size, (5, 5), 1, 
                                     activation=tf.nn.relu, 
                                     kernel_initializer=initializer,
                                     name='conv2')
                print('[Layer: conv2d] {} {}'.format('conv2', _.get_shape().as_list()))
            
                if self.apply_batch_norm:
                    _ = tf.layers.batch_normalization(_, training=is_training, name='norm2')
                    print('[Layer: batchnorm] norm2')

            _ = tf.layers.max_pooling2d(_, (2, 2), 2, 'SAME', name='pool2')
            print('[Layer: maxpool2d] {} {}'.format('pool2', _.get_shape().as_list()))

            _ = tf.layers.flatten(_, 'flatten')
            print('[Layer: flatten] {} {}'.format('flatten', _.get_shape().as_list()))

            _ = tf.layers.dense(_, 1024, 
                                activation=tf.nn.relu, 
                                kernel_initializer=initializer,
                                name='fc3')
            print('[Layer: fc] {} {}'.format('fc3', _.get_shape().as_list()))

            _ = tf.layers.dropout(_, training=is_training, name='dropout')
            print('[Layer: dropout] {} {}'.format('dropout', _.get_shape().as_list()))

            logits = tf.layers.dense(_, num_classes, 
                                     activation=None,
                                     kernel_initializer=initializer,
                                     name='fc4')
            print('[Layer: fc] {} {}'.format('fc4', logits.get_shape().as_list()))

        return logits

    def add_summary(self, name, value, summary_type='scalar'):
        if summary_type == 'scalar':
            tf.summary.scalar(name, value, collections=['train'])
            tf.summary.scalar('val_{}'.format(name), value, collections=['val'])
            tf.summary.scalar("test_{}".format(name), value, collections=['test'])
        elif summary_type == 'image':
            tf.summary.image(name, value, collections=['train'])
            tf.summary.image('val_{}'.format(name), value, collections=['val'])
            tf.summary.image("test_{}".format(name), value, collections=['test'])
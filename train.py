import os
# set tensorflow cpp log level. It is useful
# to diable some annoying log message, but sometime
#  may miss some useful imformation.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
import importlib

import numpy as np
import tensorflow as tf

import custom_ops
from data import (load_data,
                  split_data,
                  DataSampler,
                  preprocess_for_train,
                  preprocess_for_eval)
from utils import (TimeMeter,
                   feat2emb,
                   TSNE_transform)


class LRManager:

    def __init__(self, boundaries, values):
        self.boundaries = boundaries
        self.values = values

    def get(self, epoch):
        for b, v in zip(self.boundaries, self.values):
            if epoch < b:
                return v
        return self.values[-1]


def main(FLAGS):

    # set seed
    np.random.seed(FLAGS.seed)
    tf.set_random_seed(FLAGS.seed)

    with tf.device('/cpu:0'), tf.name_scope('input'):

        # load data
        data, meta = load_data(
            FLAGS.dataset_root, FLAGS.dataset, is_training=True)
        train_data, val_data = split_data(data, FLAGS.validate_rate)
        batch_size = FLAGS.n_class_per_iter * FLAGS.n_img_per_class
        img_shape = train_data[0].shape[1:]

        # build DataSampler
        train_data_sampler = DataSampler(
            train_data,  meta['n_class'],
            FLAGS.n_class_per_iter, FLAGS.n_img_per_class)

        val_data_sampler = DataSampler(
            val_data, meta['n_class'], 
            FLAGS.n_class_per_iter, FLAGS.n_img_per_class)

        # build tf_dataset for training
        train_dataset = (tf.data.Dataset
            .from_generator(lambda: train_data_sampler,
                            (tf.float32, tf.int32),
                            ([batch_size, *img_shape], [batch_size]))
            .take(FLAGS.n_iter_per_epoch)
            .flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
            .map(preprocess_for_train, 8)
            .batch(batch_size)
            .prefetch(1))

        # build tf_dataset for val
        val_dataset = (tf.data.Dataset
            .from_generator(lambda: val_data_sampler,
                            (tf.float32, tf.int32),
                            ([batch_size, *img_shape], [batch_size]))
            .take(100)
            .flat_map(lambda x, y: tf.data.Dataset.from_tensor_slices((x, y)))
            .map(preprocess_for_eval, 8)
            .batch(batch_size)
            .prefetch(1))
        
        # clean up
        del data, train_data, val_data

        # construct data iterator
        data_iterator = tf.data.Iterator.from_structure(
            train_dataset.output_types,
            train_dataset.output_shapes)

        # construct iterator initializer for training and validation
        train_data_init = data_iterator.make_initializer(train_dataset)
        val_data_init = data_iterator.make_initializer(val_dataset)

        # get data from data iterator
        images, labels = data_iterator.get_next()
        tf.summary.image('images', images)


    # define useful scalars
    learning_rate = tf.placeholder(tf.float32, shape=(), name='learning_rate')
    tf.summary.scalar('lr', learning_rate)
    is_training = tf.placeholder(tf.bool, [], name='is_training')
    global_step = tf.train.create_global_step()

    # define optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate)

    # build the net
    model = importlib.import_module('models.{}'.format(FLAGS.model))
    net = model.Net(n_feats=FLAGS.n_feats, weight_decay=FLAGS.weight_decay)

    if net.data_format == 'channels_first' or net.data_format == 'NCHW':
        images = tf.transpose(images, [0, 3, 1, 2])

    # get features
    features = net(images, is_training)
    tf.summary.histogram('features', features)

    # summary variable defined in net
    for w in net.global_variables:
        tf.summary.histogram(w.name, w)

    with tf.name_scope('losses'):
        # compute loss, if features is l2 normed, then 2 * cosine_distance will
        # equal squared l2 distance.
        distance = 2 * custom_ops.cosine_distance(features)
        # hard mining
        arch_idx, pos_idx, neg_idx = custom_ops.semi_hard_mining(
            distance, FLAGS.n_class_per_iter, FLAGS.n_img_per_class, FLAGS.threshold)

        # triplet loss
        N_pair_lefted = tf.shape(arch_idx)[0]
        def true_fn():
            pos_distance = tf.gather_nd(distance, tf.stack([arch_idx, pos_idx], 1))
            neg_distance = tf.gather_nd(distance, tf.stack([arch_idx, neg_idx], 1))
            return custom_ops.triplet_distance(pos_distance, neg_distance, FLAGS.threshold)
        loss = tf.cond(N_pair_lefted > 0, true_fn, lambda: 0.)
        pair_rate = N_pair_lefted / (FLAGS.n_class_per_iter * FLAGS.n_img_per_class**2)

        # compute l2 regularization
        l2_reg = tf.losses.get_regularization_loss()


    with tf.name_scope('metrics') as scope:

        mean_loss, mean_loss_update_op = tf.metrics.mean(
            loss, name='mean_loss')

        mean_pair_rate, mean_pair_rate_update_op = tf.metrics.mean(
            pair_rate, name='mean_pair_rate')

        reset_metrics = tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope))
        metrics_update_op = tf.group(mean_loss_update_op, mean_pair_rate_update_op)

        # collect metric summary alone, because it need to
        # summary after metrics update
        metric_summary = [tf.summary.scalar('loss', mean_loss, collections=[]),
                          tf.summary.scalar('pair_rate', mean_pair_rate, collections=[])]

    # compute grad
    grads_and_vars = optimizer.compute_gradients(loss + l2_reg)

    # summary grads
    for g, v in grads_and_vars:
        tf.summary.histogram(v.name + '/grad', g)

    # run train_op and update_op together
    train_op = optimizer.apply_gradients(
        grads_and_vars, global_step=global_step)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    train_op = tf.group(train_op, *update_ops)

    # build summary
    jpg_img_str = tf.placeholder(tf.string, shape=[], name='jpg_img_str')
    emb_summary_str = tf.summary.image(
        'emb',
        tf.expand_dims(tf.image.decode_image(jpg_img_str, 3), 0),
        collections=[])
    train_summary_str = tf.summary.merge_all()
    metric_summary_str = tf.summary.merge(metric_summary)

    # init op
    init_op = tf.group(tf.global_variables_initializer(),
                       tf.local_variables_initializer())

    # prepare for the logdir
    if not tf.gfile.Exists(FLAGS.logdir):
        tf.gfile.MakeDirs(FLAGS.logdir)

    # saver
    saver = tf.train.Saver(max_to_keep=FLAGS.n_epoch)

    # summary writer
    train_writer = tf.summary.FileWriter(
        os.path.join(FLAGS.logdir, 'train'),
        tf.get_default_graph())
    val_writer = tf.summary.FileWriter(
        os.path.join(FLAGS.logdir, 'val'),
        tf.get_default_graph())

    # session
    config = tf.ConfigProto(
        allow_soft_placement=True, log_device_placement=False,
        intra_op_parallelism_threads=8, inter_op_parallelism_threads=0)
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    # do initialization
    sess.run(init_op)

    # restore
    if FLAGS.restore:
        saver.restore(sess, FLAGS.restore)

    lr_boundaries = list(map(int, FLAGS.boundaries.split(',')))
    lr_values = list(map(float, FLAGS.values.split(',')))
    lr_manager = LRManager(lr_boundaries, lr_values)
    time_meter = TimeMeter()

    # start to train
    for e in range(FLAGS.n_epoch):
        print('-' * 40)
        print('Epoch: {:d}'.format(e))

        # training loop
        try:
            i = 0
            sess.run([train_data_init, reset_metrics])
            while True:
                
                lr = lr_manager.get(e)
                fetch = [train_summary_str] if i % FLAGS.log_every == 0 else []

                time_meter.start()
                result = sess.run(
                    [train_op, metrics_update_op] + fetch,
                    {learning_rate: lr, is_training: True})
                time_meter.stop()

                if i % FLAGS.log_every == 0:
                    # fetch summary str
                    t_summary = result[-1]
                    t_metric_summary = sess.run(metric_summary_str)

                    t_loss, t_pr = sess.run([mean_loss, mean_pair_rate])
                    sess.run(reset_metrics)

                    spd = batch_size / time_meter.get_and_reset()

                    print('Iter: {:d}, LR: {:g}, Loss: {:.4f}, PR: {:.2f}, Spd: {:.2f} i/s'
                          .format(i, lr, t_loss, t_pr, spd))

                    train_writer.add_summary(
                        t_summary, global_step=sess.run(global_step))
                    train_writer.add_summary(
                        t_metric_summary, global_step=sess.run(global_step))

                i += 1
        except tf.errors.OutOfRangeError:
            pass

        # save checkpoint
        saver.save(sess, '{}/{}'.format(FLAGS.logdir, FLAGS.model),
                   global_step=sess.run(global_step), write_meta_graph=False)

        # val loop
        try:
            sess.run([val_data_init, reset_metrics])
            v_flist, v_llist = [], []
            v_iter = 0
            while True:
                v_feats, v_labels, _ = sess.run(
                    [features, labels, metrics_update_op],
                    {is_training: False})
                if v_iter < FLAGS.n_iter_for_emb:
                    v_flist.append(v_feats)
                    v_llist.append(v_labels)
                v_iter += 1
        except tf.errors.OutOfRangeError:
            pass

        v_loss, v_pr = sess.run([mean_loss, mean_pair_rate])
        print('[VAL]Loss: {:.4f}, PR: {:.2f}'.format(v_loss, v_pr))

        v_jpg_str = feat2emb(np.concatenate(v_flist, axis=0),
                             np.concatenate(v_llist, axis=0),
                             TSNE_transform if FLAGS.n_feats > 2 else None)

        val_writer.add_summary(sess.run(metric_summary_str),
                               global_step=sess.run(global_step))
        val_writer.add_summary(sess.run(emb_summary_str, {jpg_img_str: v_jpg_str}),
                               global_step=sess.run(global_step))

    print('-' * 40)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train dnn')
    parser.add_argument('--dataset', default='mnist',
                        help='the training dataset')
    parser.add_argument(
        '--dataset_root', default='./data/mnist', help='dataset root')
    parser.add_argument(
        '--logdir', default='log/resnet-20', help='log directory')
    parser.add_argument('--restore', default='', help='snapshot path')
    parser.add_argument('--validate_rate', default=0.1,
                        type=float, help='validate split rate')

    parser.add_argument('--model', default='resnet_20', help='model name')
    parser.add_argument('--n_feats', default=2, type=int, help='the model should output n feats, n >= 2')

    parser.add_argument('--n_class_per_iter', default=8, type=int, help='n class per iter')
    parser.add_argument('--n_img_per_class', default=16, type=int, help='n img per class')
    parser.add_argument('--n_iter_per_epoch', default=800, type=int, help='n iter per epoch')
    parser.add_argument('--n_iter_for_emb', default=10, type=int, help='n iter for emb visual')
    parser.add_argument('--threshold', default=0.2, type=float, help='threshold')

    parser.add_argument('--n_epoch', default=50,
                        type=int, help='number of epoch')
    parser.add_argument('--weight_decay', default=0.0001,
                        type=float, help='weight decay rate')
    parser.add_argument('--boundaries', default='30,40,45',
                        help='learning rate boundaries')
    parser.add_argument(
        '--values', default='1e-3,1e-4,1e-5,1e-6', help='learning rate values')

    parser.add_argument('--log_every', default=100, type=int,
                        help='display and log frequency')
    parser.add_argument('--seed', default=0, type=float, help='random seed')

    args = parser.parse_args()
    main(args)

import numpy as np
import tensorflow as tf


def cosine_distance(features):
    distance = 1 - tf.matmul(features, features, transpose_b=True)
    return distance


def semi_hard_mining(distance, n_class_per_iter, n_img_per_class, threshold):
    N = n_class_per_iter * n_img_per_class
    N_pair = n_img_per_class * n_img_per_class

    # compute archor and pos indexes in numpy, reduce runtime compution
    pre_idx = np.arange(N, dtype='int64')
    arch_indexes = np.repeat(pre_idx, n_img_per_class, axis=0)
    pos_indexes = np.repeat(pre_idx.reshape(n_class_per_iter, n_img_per_class), n_img_per_class, axis=0).reshape(-1)
    pos_pair_indexes = np.stack([arch_indexes, pos_indexes], 1)

    # cast to tensorflow constant
    arch_indexes = tf.constant(arch_indexes)
    pos_indexes = tf.constant(pos_indexes)
    pos_pair_indexes = tf.constant(pos_pair_indexes)

    # gather distance
    with tf.control_dependencies([tf.assert_equal(tf.shape(distance)[0], N)]):
        pos_distance = tf.gather_nd(distance, pos_pair_indexes)
    neg_distance = tf.gather(distance, arch_indexes)

    # compute bool mask, false if it is pos index
    neg_pos_mask = np.ones(shape=[N*n_img_per_class, N], dtype='bool')
    for i in range(n_class_per_iter):
        neg_pos_mask[i*N_pair:(i+1)*N_pair, i*n_img_per_class:(i+1)*n_img_per_class] = 0
    neg_pos_mask = tf.constant(neg_pos_mask)

    # true if neg_dis - pos_dis < threshold
    candidate_mask = (neg_distance - tf.expand_dims(pos_distance, 1)) < threshold
    candidate_mask = tf.logical_and(candidate_mask, neg_pos_mask)

    # bool mask that delete triplets which has no candidate
    deletion_mask = tf.reduce_any(candidate_mask, axis=1)

    # deletion
    arch_indexes = tf.boolean_mask(arch_indexes, deletion_mask)
    pos_indexes = tf.boolean_mask(pos_indexes, deletion_mask)
    candidate_mask = tf.boolean_mask(candidate_mask, deletion_mask)

    # random sample candidation
    n_candidate_per_archer = tf.reduce_sum(tf.to_int32(candidate_mask), axis=1)
    sampler = tf.distributions.Uniform(0., tf.to_float(n_candidate_per_archer) - 1e-3)
    sample_idx = tf.to_int32(tf.floor(tf.reshape(sampler.sample(1), [-1])))
    start_idx = tf.cumsum(n_candidate_per_archer, exclusive=True)
    sample_idx = start_idx + sample_idx

    # collect neg_indexes
    candidate_indexes = tf.where(candidate_mask)
    neg_indexes = tf.gather(candidate_indexes, sample_idx)[:, 1]

    return (tf.stop_gradient(arch_indexes),
            tf.stop_gradient(pos_indexes),
            tf.stop_gradient(neg_indexes))


def triplet_distance(pos_dis, neg_dis, threshold):
    return tf.reduce_mean(tf.nn.relu(threshold + pos_dis - neg_dis))

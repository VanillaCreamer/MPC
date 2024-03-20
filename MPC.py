import tensorflow as tf
import os
import sys
import pandas as pd
import copy

from utility.helper import *
from utility.batch_test import *


def load_data(csv_file):
    tp = pd.read_csv(csv_file, sep='\t')
    return tp


class MPC(object):
    def __init__(self, max_item_view, max_item_cart, max_item_buy, max_item_pwb, max_item_cwb, max_item_pwc,
                 data_config):
        # argument settings

        self.model_type = 'MPC'
        self.adj_type = args.adj_type
        self.n_users = data_config['n_users']
        self.n_items = data_config['n_items']
        self.n_fold = 100
        self.wid = eval(args.wid)
        self.buy_adj = data_config['buy_adj']
        self.pv_adj = data_config['pv_adj']
        self.cart_adj = data_config['cart_adj']

        self.pwb_adj = data_config['pwb_adj']
        self.cwb_adj = data_config['cwb_adj']
        self.pwc_adj = data_config['pwc_adj']
        self.lr = args.lr
        self.emb_dim = args.embed_size
        self.batch_size = args.batch_size
        self.weight_size = eval(args.layer_size)
        self.n_layers = len(self.weight_size)
        self.n_layers_transfer = args.n_layers_transfer  # transfer layer
        self.neg_layers = 1
        
        self.decay = args.decay
        self.verbose = args.verbose
        self.max_item_view = max_item_view
        self.max_item_cart = max_item_cart
        self.max_item_buy = max_item_buy

        self.max_item_pwb = max_item_pwb
        self.max_item_cwb = max_item_cwb
        self.max_item_pwc = max_item_pwc
        self.coefficient_loss = eval(args.coefficient_loss)
        self.coefficient_cart = eval(args.coefficient_cart)
        self.coefficient_buy = eval(args.coefficient_buy)

        self.n_relations = 3

        '''
        *********************************************************
        Create Placeholder for Input Data & Dropout.
        '''
        # placeholder definition
        self.input_u = tf.placeholder(tf.int32, [None, 1], name="input_uid")
        self.lable_view = tf.placeholder(tf.int32, [None, self.max_item_view], name="lable_view")
        self.lable_cart = tf.placeholder(tf.int32, [None, self.max_item_cart], name="lable_cart")
        self.lable_buy = tf.placeholder(tf.int32, [None, self.max_item_buy], name="lable_buy")

        self.lable_pwb = tf.placeholder(tf.int32, [None, self.max_item_pwb], name="lable_pwb")
        self.lable_cwb = tf.placeholder(tf.int32, [None, self.max_item_cwb], name="lable_cwb")
        self.lable_pwc = tf.placeholder(tf.int32, [None, self.max_item_pwc], name="lable_pwc")
        self.users = tf.placeholder(tf.int32, shape=(None,))
        self.pos_items = tf.placeholder(tf.int32, shape=(None,))

        self.node_dropout_flag = args.node_dropout_flag
        self.node_dropout = tf.placeholder(tf.float32, shape=[None])
        self.mess_dropout = tf.placeholder(tf.float32, shape=[None])

        """
        *********************************************************
        Create Model Parameters (i.e., Initialize Weights).
        """
        # initialization of model parameters
        self.weights_one = self._init_weights()

        """
        *********************************************************
        Compute Graph-based Representations of all users & items via Message-Passing Mechanism of Graph Neural Networks.
        """
        self.ua_embeddings3, self.ia_embeddings3, self.ua_embeddings2, self.ia_embeddings2 \
            , self.ua_embeddings1, self.ia_embeddings1, self.r0, self.r1, self.r2 = self._create_gcn_embed()

        self.ua_embeddings23, self.ia_embeddings23 = self._create_myrec_embed(self.ua_embeddings3, self.ia_embeddings3,
                                                                              self.cart_adj)
        self.ua_embeddings332, self.ia_embeddings332 = self._create_neg_embed(self.ua_embeddings3, self.ia_embeddings3,
                                                                                self.pwc_adj)

        self.ua_embeddingsc = self.coefficient_cart[0] * self.ua_embeddings2 + \
                              self.coefficient_cart[1] * (self.ua_embeddings23 - self.ua_embeddings332) + \
                              self.coefficient_cart[2] * self.ua_embeddings23

        self.ia_embeddingsc = self.coefficient_cart[0] * self.ia_embeddings2 + \
                              self.coefficient_cart[1] * (self.ia_embeddings23 - self.ia_embeddings332) + \
                              self.coefficient_cart[2] * self.ia_embeddings23

        self.ua_embeddings13, self.ia_embeddings13 = self._create_myrec_embed(self.ua_embeddings3, self.ia_embeddings3,
                                                                              self.buy_adj)
        self.ua_embeddings12, self.ia_embeddings12 = self._create_myrec_embed(self.ua_embeddingsc, self.ia_embeddingsc,
                                                                              self.buy_adj)
        self.ua_embeddings33, self.ia_embeddings33 = self._create_neg_embed(self.ua_embeddings3, self.ia_embeddings3,
                                                                              self.pwb_adj)
        self.ua_embeddings22, self.ia_embeddings22 = self._create_neg_embed(self.ua_embeddingsc, self.ia_embeddingsc,
                                                                              self.cwb_adj)

        self.ua_embeddingsb = self.coefficient_buy[0] * self.ua_embeddings1 + \
                              self.coefficient_buy[1] * (self.ua_embeddings12 - self.ua_embeddings22) + \
                              self.coefficient_buy[1] * self.ua_embeddings12 + \
                              self.coefficient_buy[2] * (self.ua_embeddings13 - self.ua_embeddings33) + \
                              self.coefficient_buy[2] * self.ua_embeddings13

        self.ia_embeddingsb = self.coefficient_buy[0] * self.ia_embeddings1 + \
                              self.coefficient_buy[1] * (self.ia_embeddings12 - self.ia_embeddings22) + \
                              self.coefficient_buy[1] * self.ia_embeddings12 + \
                              self.coefficient_buy[2] * (self.ia_embeddings13 - self.ia_embeddings33) + \
                              self.coefficient_buy[2] * self.ia_embeddings13

        self.token_embedding = tf.zeros([1, self.emb_dim], name='token_embedding')
        self.ia_embeddings3 = tf.concat([self.ia_embeddings3, self.token_embedding], axis=0)
        self.ia_embeddingsc = tf.concat([self.ia_embeddingsc, self.token_embedding], axis=0)
        self.ia_embeddingsb = tf.concat([self.ia_embeddingsb, self.token_embedding], axis=0)

        """
        *********************************************************
        Establish the final representations for user-item pairs in batch.
        for test
        """
        self.u_g_embeddings = tf.nn.embedding_lookup(self.ua_embeddingsb, self.users)
        self.pos_i_g_embeddings = tf.nn.embedding_lookup(self.ia_embeddingsb, self.pos_items)

        self.dot = tf.einsum('ac,bc->abc', self.u_g_embeddings, self.pos_i_g_embeddings)
        self.batch_ratings = tf.einsum('ajk,lk->aj', self.dot, self.r2)

        """for training"""

        self.uid = tf.nn.embedding_lookup(self.ua_embeddingsb, self.input_u)
        self.uid = tf.reshape(self.uid, [-1, self.emb_dim])
        self.aux_uid_p = tf.nn.embedding_lookup(self.ua_embeddings3, self.input_u)
        self.aux_uid_p = tf.reshape(self.aux_uid_p, [-1, self.emb_dim])
        self.aux_uid_c = tf.nn.embedding_lookup(self.ua_embeddingsc, self.input_u)
        self.aux_uid_c = tf.reshape(self.aux_uid_c, [-1, self.emb_dim])

        self.pos_view = tf.nn.embedding_lookup(self.ia_embeddings3, self.lable_view)
        self.pos_num_view = tf.cast(tf.not_equal(self.lable_view, self.n_items), 'float32')
        self.pos_view = tf.einsum('ab,abc->abc', self.pos_num_view, self.pos_view)
        self.pos_rv = tf.einsum('ac,abc->abc', self.aux_uid_p, self.pos_view)
        self.pos_rv = tf.einsum('ajk,lk->aj', self.pos_rv, self.r0)

        self.pos_cart = tf.nn.embedding_lookup(self.ia_embeddingsc, self.lable_cart)
        self.pos_num_cart = tf.cast(tf.not_equal(self.lable_cart, self.n_items), 'float32')
        self.pos_cart = tf.einsum('ab,abc->abc', self.pos_num_cart, self.pos_cart)
        self.pos_rc = tf.einsum('ac,abc->abc', self.aux_uid_c, self.pos_cart)
        self.pos_rc = tf.einsum('ajk,lk->aj', self.pos_rc, self.r1)

        self.pos_buy = tf.nn.embedding_lookup(self.ia_embeddingsb, self.lable_buy)
        self.pos_num_buy = tf.cast(tf.not_equal(self.lable_buy, self.n_items), 'float32')
        self.pos_buy = tf.einsum('ab,abc->abc', self.pos_num_buy, self.pos_buy)
        self.pos_rb = tf.einsum('ac,abc->abc', self.uid, self.pos_buy)
        self.pos_rb = tf.einsum('ajk,lk->aj', self.pos_rb, self.r2)

        self.u_g_embeddings_pre = tf.nn.embedding_lookup(self.weights_one['user_embedding'], self.input_u)

        self.mf_loss, self.emb_loss = self.create_non_sampling_loss()
        self.loss = self.mf_loss + self.emb_loss

        self.opt = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _init_weights(self):
        all_weights_one = dict()
        initializer = tf.random_normal_initializer(stddev=0.01)

        all_weights_one['user_embedding'] = tf.Variable(initializer([self.n_users, self.emb_dim]),
                                                        name='user_embedding')
        all_weights_one['item_embedding'] = tf.Variable(initializer([self.n_items, self.emb_dim]),
                                                        name='item_embedding')
        all_weights_one['relation_embedding'] = tf.Variable(initializer([self.n_relations, self.emb_dim]),
                                                            name='relation_embedding')

        self.weight_size_list = [self.emb_dim] + self.weight_size
        for k in range(self.n_layers):
            all_weights_one['w_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='w_%d' % k)
            all_weights_one['r_%d' % k] = tf.Variable(
                initializer([self.weight_size_list[k], self.weight_size_list[k + 1]]), name='r_%d' % k)

        print('using random initialization')

        return all_weights_one

    def _split_A_hat(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
        return A_fold_hat

    def _split_A_hat_node_dropout(self, X):
        A_fold_hat = []

        fold_len = (self.n_users + self.n_items) // self.n_fold
        for i_fold in range(self.n_fold):
            start = i_fold * fold_len
            if i_fold == self.n_fold - 1:
                end = self.n_users + self.n_items
            else:
                end = (i_fold + 1) * fold_len

            # A_fold_hat.append(self._convert_sp_mat_to_sp_tensor(X[start:end]))
            temp = self._convert_sp_mat_to_sp_tensor(X[start:end])
            n_nonzero_temp = X[start:end].count_nonzero()
            A_fold_hat.append(self._dropout_sparse(temp, 1 - self.node_dropout[0], n_nonzero_temp))

        return A_fold_hat

    def _create_gcn_embed(self):
        # node dropout.
        A_fold_hat_buy = self._split_A_hat_node_dropout(self.buy_adj)
        A_fold_hat_pv = self._split_A_hat_node_dropout(self.pv_adj)
        A_fold_hat_cart = self._split_A_hat_node_dropout(self.cart_adj)

        pv_embeddings = tf.concat([self.weights_one['user_embedding'], self.weights_one['item_embedding']], axis=0)
        cart_embeddings = tf.concat([self.weights_one['user_embedding'], self.weights_one['item_embedding']], axis=0)
        buy_embeddings = tf.concat([self.weights_one['user_embedding'], self.weights_one['item_embedding']], axis=0)
        all_embeddings_pv = [pv_embeddings]
        all_embeddings_cart = [cart_embeddings]
        all_embeddings_buy = [buy_embeddings]

        r0 = tf.nn.embedding_lookup(self.weights_one['relation_embedding'], 0)
        r0 = tf.reshape(r0, [-1, self.emb_dim])

        r1 = tf.nn.embedding_lookup(self.weights_one['relation_embedding'], 1)
        r1 = tf.reshape(r1, [-1, self.emb_dim])

        r2 = tf.nn.embedding_lookup(self.weights_one['relation_embedding'], 2)
        r2 = tf.reshape(r2, [-1, self.emb_dim])

        all_r0 = [r0]
        all_r1 = [r1]
        all_r2 = [r2]
        for k in range(0, self.n_layers):

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat_pv[f], pv_embeddings))
            pv_embeddings = tf.concat(temp_embed, 0)
            pv_embeddings = tf.nn.leaky_relu(
                tf.matmul(tf.multiply(pv_embeddings, r0), self.weights_one['w_%d' % k]))
            pv_embeddings = tf.nn.dropout(pv_embeddings, 1 - self.mess_dropout[0])
            all_embeddings_pv += [pv_embeddings]

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat_cart[f], cart_embeddings))
            cart_embeddings = tf.concat(temp_embed, 0)
            cart_embeddings = tf.nn.leaky_relu(
                tf.matmul(tf.multiply(cart_embeddings, r1), self.weights_one['w_%d' % k]))
            cart_embeddings = tf.nn.dropout(cart_embeddings, 1 - self.mess_dropout[0])
            all_embeddings_cart += [cart_embeddings]

            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat_buy[f], buy_embeddings))
            buy_embeddings = tf.concat(temp_embed, 0)
            buy_embeddings = tf.nn.leaky_relu(
                tf.matmul(tf.multiply(buy_embeddings, r2), self.weights_one['w_%d' % k]))
            buy_embeddings = tf.nn.dropout(buy_embeddings, 1 - self.mess_dropout[0])
            all_embeddings_buy += [buy_embeddings]

            r0 = tf.matmul(r0, self.weights_one['r_%d' % k])
            r1 = tf.matmul(r1, self.weights_one['r_%d' % k])
            r2 = tf.matmul(r2, self.weights_one['r_%d' % k])
            all_r0.append(r0)
            all_r1.append(r1)
            all_r2.append(r2)

        all_embeddings_pv = tf.stack(all_embeddings_pv, 1)
        all_embeddings_pv = tf.reduce_mean(all_embeddings_pv, axis=1, keepdims=False)
        u_g_embeddings3, i_g_embeddings3 = tf.split(all_embeddings_pv, [self.n_users, self.n_items], 0)

        all_embeddings_cart = tf.stack(all_embeddings_cart, 1)
        all_embeddings_cart = tf.reduce_mean(all_embeddings_cart, axis=1, keepdims=False)
        u_g_embeddings2, i_g_embeddings2 = tf.split(all_embeddings_cart, [self.n_users, self.n_items], 0)

        all_embeddings_buy = tf.stack(all_embeddings_buy, 1)
        all_embeddings_buy = tf.reduce_mean(all_embeddings_buy, axis=1, keepdims=False)
        u_g_embeddings1, i_g_embeddings1 = tf.split(all_embeddings_buy, [self.n_users, self.n_items], 0)

        all_r0 = tf.reduce_mean(all_r0, 0)
        all_r1 = tf.reduce_mean(all_r1, 0)
        all_r2 = tf.reduce_mean(all_r2, 0)

        return u_g_embeddings3, i_g_embeddings3, u_g_embeddings2, i_g_embeddings2, u_g_embeddings1, \
            i_g_embeddings1, all_r0, all_r1, all_r2

    def _create_myrec_embed(self, user_embedding, item_embedding, adj):
        A_fold_hat = self._split_A_hat_node_dropout(adj)
        embeddings = tf.concat([user_embedding, item_embedding], axis=0)
        all_embeddings = [embeddings]

        for k in range(0, self.n_layers_transfer):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[0])
            all_embeddings += [embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)

        return u_g_embeddings, i_g_embeddings

    def _create_neg_embed(self, user_embedding, item_embedding, adj):
        A_fold_hat = self._split_A_hat_node_dropout(adj)
        embeddings = tf.concat([user_embedding, item_embedding], axis=0)
        all_embeddings = [embeddings]

        for k in range(0, self.neg_layers):
            temp_embed = []
            for f in range(self.n_fold):
                temp_embed.append(tf.sparse_tensor_dense_matmul(A_fold_hat[f], embeddings))
            embeddings = tf.concat(temp_embed, 0)
            embeddings = tf.nn.dropout(embeddings, 1 - self.mess_dropout[0])
            all_embeddings += [embeddings]

        all_embeddings = tf.stack(all_embeddings, 1)
        all_embeddings = tf.reduce_mean(all_embeddings, axis=1, keepdims=False)
        u_g_embeddings, i_g_embeddings = tf.split(all_embeddings, [self.n_users, self.n_items], 0)

        return u_g_embeddings, i_g_embeddings

    def create_non_sampling_loss(self):
        temp = tf.einsum('ab,ac->bc', self.ia_embeddingsb, self.ia_embeddingsb) \
               * tf.einsum('ab,ac->bc', self.uid, self.uid)

        temp1 = tf.einsum('ab,ac->bc', self.ia_embeddings3, self.ia_embeddings3) \
                * tf.einsum('ab,ac->bc', self.aux_uid_p, self.aux_uid_p)

        temp2 = tf.einsum('ab,ac->bc', self.ia_embeddingsc, self.ia_embeddingsc) \
                * tf.einsum('ab,ac->bc', self.aux_uid_c, self.aux_uid_c)

        loss1 = self.wid[0] * tf.reduce_sum(temp1 * tf.matmul(self.r0, self.r0, transpose_a=True))
        loss1 += tf.reduce_sum((1.0 - self.wid[0]) * tf.square(self.pos_rv) - 2.0 * self.pos_rv)

        loss2 = self.wid[1] * tf.reduce_sum(temp2 * tf.matmul(self.r1, self.r1, transpose_a=True))
        loss2 += tf.reduce_sum((1.0 - self.wid[1]) * tf.square(self.pos_rc) - 2.0 * self.pos_rc)

        loss3 = self.wid[2] * tf.reduce_sum(temp * tf.matmul(self.r2, self.r2, transpose_a=True))
        loss3 += tf.reduce_sum((1.0 - self.wid[2]) * tf.square(self.pos_rb) - 2.0 * self.pos_rb)

        loss = self.coefficient_loss[0] * loss1 + self.coefficient_loss[1] * loss2 + self.coefficient_loss[2] * loss3

        regularizer = tf.nn.l2_loss(self.uid) + tf.nn.l2_loss(self.ia_embeddingsb)+\
                      tf.nn.l2_loss(self.aux_uid_c) + tf.nn.l2_loss(self.ia_embeddingsc)


        emb_loss = self.decay * regularizer

        return loss, emb_loss

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        indices = np.mat([coo.row, coo.col]).transpose()
        return tf.SparseTensor(indices, coo.data, coo.shape)

    def _dropout_sparse(self, X, keep_prob, n_nonzero_elems):
        """
        Dropout for sparse tensors.
        """
        noise_shape = [n_nonzero_elems]
        random_tensor = keep_prob
        random_tensor += tf.random_uniform(noise_shape)
        dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
        pre_out = tf.sparse_retain(X, dropout_mask)

        return pre_out * tf.div(1., keep_prob)


def get_lables(temp_set, k=0.9999):
    max_item = 0

    item_lenth = []
    for i in temp_set:
        item_lenth.append(len(temp_set[i]))
        if len(temp_set[i]) > max_item:
            max_item = len(temp_set[i])
    item_lenth.sort()

    max_item = item_lenth[int(len(item_lenth) * k) - 1]

    print(max_item)
    for i in temp_set:
        if len(temp_set[i]) > max_item:
            temp_set[i] = temp_set[i][0:max_item]
        while len(temp_set[i]) < max_item:
            temp_set[i].append(n_items)
    return max_item, temp_set


def get_train_instances1(view_lable, cart_lable, buy_lable, pwb_lable, cwb_lable, pwc_lable):
    user_train, view_item, cart_item, buy_item = [], [], [], []
    pwc_item, pwb_item, cwb_item = [], [], []
    for i in buy_lable.keys():
        user_train.append(i)
        buy_item.append(buy_lable[i])
        if i not in view_lable.keys():
            view_item.append([n_items] * max_item_view)
        else:
            view_item.append(view_lable[i])

        if i not in cart_lable.keys():
            cart_item.append([n_items] * max_item_cart)
        else:
            cart_item.append(cart_lable[i])

        if i not in pwc_lable.keys():
            pwc_item.append([n_items] * max_item_pwc)
        else:
            pwc_item.append(pwc_lable[i])

        if i not in pwb_lable.keys():
            pwb_item.append([n_items] * max_item_pwb)
        else:
            pwb_item.append(pwb_lable[i])

        if i not in cwb_lable.keys():
            cwb_item.append([n_items] * max_item_cwb)
        else:
            cwb_item.append(cwb_lable[i])

    user_train = np.array(user_train)
    view_item = np.array(view_item)
    cart_item = np.array(cart_item)
    buy_item = np.array(buy_item)

    pwb_item = np.array(pwb_item)
    cwb_item = np.array(cwb_item)
    pwc_item = np.array(pwc_item)
    user_train = user_train[:, np.newaxis]
    return user_train, view_item, cart_item, buy_item, pwb_item, cwb_item, pwc_item


if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    tf.set_random_seed(2020)
    np.random.seed(2020)

    config = dict()
    config['n_users'] = data_generator.n_users
    config['n_items'] = data_generator.n_items

    """
    *********************************************************
    Generate the Laplacian matrix, where each entry defines the decay factor (e.g., p_ui) between two connected nodes.
    """
    pre_adj, pre_adj_pv, pre_adj_cart, pre_adj_mat_pwc, pre_adj_mat_pwb, pre_adj_mat_cwb = data_generator.get_adj_mat()  

    config['buy_adj'] = pre_adj
    config['pv_adj'] = pre_adj_pv
    config['cart_adj'] = pre_adj_cart
    config['pwc_adj'] = pre_adj_mat_pwc
    config['pwb_adj'] = pre_adj_mat_pwb
    config['cwb_adj'] = pre_adj_mat_cwb

    print('use the pre adjcency matrix')

    n_users, n_items = data_generator.n_users, data_generator.n_items
    train_items = copy.deepcopy(data_generator.train_items)
    pv_set = copy.deepcopy(data_generator.pv_set)
    cart_set = copy.deepcopy(data_generator.cart_set)

    pv_wo_buy_set = np.load(data_generator.path + '/pv_wo_buy.npy', allow_pickle='TRUE').item()
    cart_wo_buy_set = np.load(data_generator.path + '/cart_wo_buy.npy', allow_pickle='TRUE').item()
    pv_wo_cart_set = np.load(data_generator.path + '/pv_wo_cart.npy', allow_pickle='TRUE').item()

    max_item_buy, buy_lable = get_lables(train_items)
    max_item_view, view_lable = get_lables(pv_set)
    max_item_cart, cart_lable = get_lables(cart_set)

    max_item_pwb, pwb_lable = get_lables(pv_wo_buy_set)
    max_item_cwb, cwb_lable = get_lables(cart_wo_buy_set)
    max_item_pwc, pwc_lable = get_lables(pv_wo_cart_set)
    t0 = time()

    model = MPC(max_item_view, max_item_cart, max_item_buy, max_item_pwb, max_item_cwb, max_item_pwc,
                  data_config=config)

    'Save'
    saver = tf.train.Saver()
    if args.save_flag == 1:
        model_type = 'MPC'
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        weights_save_path = '%s/%s/l%s' % (model_type, layer, str(args.lr))
        ensureDir(weights_save_path)
        save_saver = tf.train.Saver(max_to_keep=1)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    """
       *********************************************************
       Get the performance w.r.t. different sparsity levels.
       """
    if args.report == 1:
        assert args.test_flag == 'full'
        users_to_test_list, split_state = data_generator.get_sparsity_split()
        users_to_test_list.append(list(data_generator.test_set.keys()))
        split_state.append('all')

        report_path = '%sreport/%s/%s.result' % (args.proj_path, args.dataset, model.model_type)
        ensureDir(report_path)
        f = open(report_path, 'w')
        f.write(
            'embed_size=%d, lr=%.4f, layer_size=%s, keep_prob=%s, regs=%s, loss_type=%s, adj_type=%s\n'
            % (args.embed_size, args.lr, args.layer_size, args.keep_prob, args.regs, args.loss_type, args.adj_type))

        for i, users_to_test in enumerate(users_to_test_list):
            ret = test(sess, model, users_to_test, drop_flag=True)

            final_perf = "recall=[%s], precision=[%s], ndcg=[%s]" % \
                         (', '.join(['%.5f' % r for r in ret['recall']]),
                          ', '.join(['%.5f' % r for r in ret['precision']]),
                          ', '.join(['%.5f' % r for r in ret['ndcg']]))

            f.write('\t%s\n\t%s\n' % (split_state[i], final_perf))
        f.close()
        exit()

    if args.pretrain == 1:
        model_type = 'MPC'
        layer = '-'.join([str(l) for l in eval(args.layer_size)])
        pretrain_path = '%s/%s/l%s_r%s' % (model_type, layer, str(args.lr), '-'.join([str(r) for r in eval(args.regs)]))

        ckpt = tf.train.get_checkpoint_state(os.path.dirname(pretrain_path + '/checkpoint'))
        if ckpt and ckpt.model_checkpoint_path:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('load the pretrained model parameters from: ', pretrain_path)
            cur_best_pre_0 = 0.

        else:
            sess.run(tf.global_variables_initializer())
            cur_best_pre_0 = 0.
            print('without pretraining.')

    else:
        sess.run(tf.global_variables_initializer())
        cur_best_pre_0 = 0.
        print('without pretraining.')

    run_time = 1
    user_train1, view_item1, cart_item1, buy_item1, pwb_item1, cwb_item1, pwc_item1 = get_train_instances1(view_lable,
                                                                                                           cart_lable,
                                                                                                           buy_lable,
                                                                                                           pwb_lable,
                                                                                                           cwb_lable,
                                                                                                           pwc_lable)
    loss_loger, pre_loger, rec_loger, ndcg_loger, hit_loger = [], [], [], [], []

    stopping_step = 0
    should_stop = False

    for epoch in range(args.epoch):

        shuffle_indices = np.random.permutation(np.arange(len(user_train1)))
        user_train1 = user_train1[shuffle_indices]
        view_item1 = view_item1[shuffle_indices]
        cart_item1 = cart_item1[shuffle_indices]
        buy_item1 = buy_item1[shuffle_indices]
        pwc_item1 = pwc_item1[shuffle_indices]
        pwb_item1 = pwb_item1[shuffle_indices]
        cwb_item1 = cwb_item1[shuffle_indices]

        t1 = time()
        loss, mf_loss, emb_loss = 0., 0., 0.

        n_batch = int(len(user_train1) / args.batch_size)

        for idx in range(n_batch):
            start_index = idx * args.batch_size
            end_index = min((idx + 1) * args.batch_size, len(user_train1))

            u_batch = user_train1[start_index:end_index]
            v_batch = view_item1[start_index:end_index]
            c_batch = cart_item1[start_index:end_index]
            b_batch = buy_item1[start_index:end_index]
            pwb_batch = pwb_item1[start_index:end_index]
            cwb_batch = cwb_item1[start_index:end_index]
            pwc_batch = pwc_item1[start_index:end_index]

            _, batch_loss, batch_mf_loss, batch_emb_loss = sess.run(
                [model.opt, model.loss, model.mf_loss, model.emb_loss],
                feed_dict={model.input_u: u_batch,
                           model.lable_buy: b_batch,
                           model.lable_view: v_batch,
                           model.lable_cart: c_batch,
                           model.lable_pwb: pwb_batch,
                           model.lable_cwb: cwb_batch,
                           model.lable_pwc: pwc_batch,
                           model.node_dropout: eval(args.node_dropout),
                           model.mess_dropout: eval(args.mess_dropout)})

            loss += batch_loss / n_batch
            mf_loss += batch_mf_loss / n_batch
            emb_loss += batch_emb_loss / n_batch

        if np.isnan(loss) == True:
            print('ERROR: loss is nan.')
            sys.exit()

        # print the test evaluation metrics each 10 epochs; pos:neg = 1:10.
        if (epoch + 1) % 10 != 0:
            if args.verbose > 0 and epoch % args.verbose == 0:
                perf_str = 'Epoch %d [%.1fs]: train==[%.5f=%.5f + %.5f]' % (
                    epoch, time() - t1, loss, mf_loss, emb_loss)
                print(perf_str)
            continue

        t2 = time()
        users_to_test = list(data_generator.test_set.keys())
        ret = test(sess, model, users_to_test, drop_flag=True)

        t3 = time()

        loss_loger.append(loss)
        rec_loger.append(ret['recall'])
        pre_loger.append(ret['precision'])
        ndcg_loger.append(ret['ndcg'])
        hit_loger.append(ret['hit_ratio'])

        if args.verbose > 0:
            perf_str = 'Epoch %d [%.1fs + %.1fs]:, recall=[%.5f, %.5f, %.5f], ' \
                       'precision=[%.5f, %.5f, %.5f], hit=[%.5f, %.5f, %.5f], ndcg=[%.5f, %.5f, %.5f]' % \
                       (
                           epoch, t2 - t1, t3 - t2, ret['recall'][0], ret['recall'][1],
                           ret['recall'][-1],
                           ret['precision'][0], ret['precision'][1], ret['precision'][-1], ret['hit_ratio'][0],
                           ret['hit_ratio'][1], ret['hit_ratio'][-1],
                           ret['ndcg'][0], ret['ndcg'][1], ret['ndcg'][-1])
            print(perf_str)

        cur_best_pre_0, stopping_step, should_stop = early_stopping(ret['recall'][0], cur_best_pre_0,
                                                                    stopping_step, expected_order='acc', flag_step=5)

        # *********************************************************
        # early stopping when cur_best_pre_0 is decreasing for ten successive steps.
        if should_stop == True:
            break

        # *********************************************************
        # save the user & item embeddings for pretraining.
        if ret['recall'][0] == cur_best_pre_0 and args.save_flag == 1:
            save_saver.save(sess, weights_save_path + '/weights', global_step=epoch)
            print('save the weights in path: ', weights_save_path)
    recs = np.array(rec_loger)
    pres = np.array(pre_loger)
    ndcgs = np.array(ndcg_loger)
    hit = np.array(hit_loger)

    best_rec_0 = max(recs[:, 0])
    idx = list(recs[:, 0]).index(best_rec_0)

    final_perf = "Best Iter=[%d]@[%.1f]\trecall=[%s], precision=[%s], hit=[%s], ndcg=[%s]" % \
                 (idx, time() - t0, '\t'.join(['%.5f' % r for r in recs[idx]]),
                  '\t'.join(['%.5f' % r for r in pres[idx]]),
                  '\t'.join(['%.5f' % r for r in hit[idx]]),
                  '\t'.join(['%.5f' % r for r in ndcgs[idx]]))
    print(final_perf)

    save_path = '%soutput/%s.result' % (args.proj_path, model.model_type)
    ensureDir(save_path)
    f = open(save_path, 'a')
    f.write('\n')
    f.write(
        'dataset=%s, embed_size=%d, n_layers_transfer=%d, coefficient_loss=%s, wid=%s, layer_size=%s, '
        'node_dropout=%s, mess_dropout=%s,decay=%d Ks=%s,\n\t%s\n'
        % (args.dataset, args.embed_size, args.n_layers_transfer, args.coefficient_loss, args.wid, args.layer_size,
           args.node_dropout, args.mess_dropout, args.decay, args.Ks, final_perf))
    f.close()

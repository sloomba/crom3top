import warnings
import csv
import pickle
import numpy as np
import tensorflow as tf
import edward as ed
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from  matplotlib.colors import Normalize as normcm

from edward.models import Dirichlet, Normal, Categorical, Empirical
from edward.util import rbf

warnings.simplefilter('always', UserWarning)

def toposort(adj_mat):
    #outputs a topological ordering of the given graph
    adj_mat = adj_mat.copy() #safe than sorry
    node_set = np.where(adj_mat.sum(0)==0)[0].tolist()
    topo_ord = []
    while len(node_set)>0:
        cur_node = node_set.pop(0)
        topo_ord.append(cur_node)
        children = np.where(adj_mat[cur_node,:])[0]
        adj_mat[cur_node, children] = 0
        [node_set.append(child) if not adj_mat[:,child].any() else None for child in children]
    if adj_mat.any(): warnings.warn('Could not do a topological sort on the given graph since it is not a DAG; returning some partial topological order')
    return np.array(topo_ord)

class crom3top:

    def __init__(self, num_discrete_variables=0, network_filename=None, data_filename=None, name=''):
        self.name = name+'_crom3top'
        self.num_discrete_variables = num_discrete_variables #the first "v" variables must be the discrete ones; no discrete variables may have any parents
        self.network_filename = network_filename
        self.data_filename = data_filename
        self.variable_names = []
        self.design_matrix = np.empty([0,0])
        self.read_dat(self.data_filename)
        self.network = np.empty([0,0])
        self.continus_variable_idxs = np.empty([0])
        self.read_net(self.network_filename)
        assert(self.design_matrix.shape[1] == self.network.shape[0] == self.network.shape[1])

    def read_dat(self, filename=None):    
        if filename:
            data = np.genfromtxt(filename, delimiter=",", dtype=np.dtype(str))
            self.variable_names = data[0,:]
            self.design_matrix = np.asarray(data[1:,:], dtype=np.dtype(float))
        else: return            

    def read_net(self, filename=None):
        if filename:
            self.network = np.genfromtxt(filename, delimiter=",", dtype=np.dtype(int))
            self.continus_variable_idxs = self.num_discrete_variables + toposort(self.network[self.num_discrete_variables:, self.num_discrete_variables:])
        else: return

    def load_model(self, filename):
        with open(filename, 'rb') as fd:
            [self.name, self.num_discrete_variables, self.network_filename, self.data_filename, self.variable_names, self.design_matrix, self.network, self.continus_variable_idxs] = pickle.load(fd)

    def save_model(self):
        filename = self.data_filename + '_' + self.name + '.pickle'
        with open(filename, 'wb') as fd:
            pickle.dump([self.name, self.num_discrete_variables, self.network_filename, self.data_filename, self.variable_names, self.design_matrix, self.network, self.continus_variable_idxs], open(filename, 'wb'))

    def print(self):
        print('model name:\t\t\t', self.name)
        print('network file:\t\t\t', self.network_filename)
        print('data file:\t\t\t', self.data_filename)
        print('number of variables:\t\t', self.network.shape[0])
        print('number of discrete variables:\t', self.num_discrete_variables)
        print('variable names:\n', self.variable_names)
        print('design matrix:\n', self.design_matrix)
        print('network matrix:\n', self.network)
        print('topological ordering of continuous variables:\n', self.continus_variable_idxs)

    def make_bayes_net(self, topk=np.inf, filename='', visualize_graph=False):
        N, D = self.design_matrix.shape
        num_discrete_variables = self.num_discrete_variables
        discrete_variable_idxs = tuple(np.arange(num_discrete_variables))
        discrete_variable_outs = [dict(zip(*np.unique(self.design_matrix[:,idx], return_index=True))) for idx in discrete_variable_idxs]
        discrete_variable_outs_size = [len(out) for out in discrete_variable_outs]
        discrete_variable_prior_pi = [tf.convert_to_tensor(Dirichlet(concentration=tf.ones([discrete_variable_outs_size[idx]]), name='Dirichlet_d_pi_'+str(idx))) for idx in range(num_discrete_variables)]
        discrete_variable_vars = [tf.convert_to_tensor(Categorical(logits=tf.tile(tf.expand_dims(discrete_variable_prior_pi[idx], axis=0), [N, discrete_variable_outs_size[idx]]), name='Categorical_d_'+str(idx))) for idx in range(num_discrete_variables)]
        continus_variable_prior_w = dict()
        continus_variable_prior_b = dict()
        continus_variable_prior_sigma = dict()
        continus_variable_vars = dict()
        tmp_idx = 1
        for idx in self.continus_variable_idxs:
            if tmp_idx%100==0: print(tmp_idx)
            tmp_idx += 1
            discrete_pars = np.where(self.network[discrete_variable_idxs, idx])[0]
            discrete_par_size = [discrete_variable_outs_size[par] for par in discrete_pars]
            if len(discrete_par_size)==0: discrete_par_vars = tf.zeros([N, 0])
            elif len(discrete_par_size)==1: discrete_par_vars = tf.expand_dims(discrete_variable_vars[discrete_pars[0]], axis=1)
            else: discrete_par_vars = tf.stack([discrete_variable_vars[par] for par in discrete_pars], axis=1)
            continus_pars = self.continus_variable_idxs[np.where(self.network[self.continus_variable_idxs, idx])[0]]
            if topk<len(continus_pars): continus_pars = list(map(lambda x: x[0], sorted(zip(continus_pars, self.network[continus_pars, :].sum(1)), key=lambda x:x[1], reverse=True)[:topk]))
            continus_par_size = len(continus_pars)
            if continus_par_size==0: continus_par_vars = tf.zeros([N, 0])
            elif continus_par_size==1: continus_par_vars = tf.expand_dims(continus_variable_vars[continus_pars[0]], axis=1)
            else: continus_par_vars = tf.stack([continus_variable_vars[par] for par in continus_pars], axis=1)
            continus_variable_prior_w[idx] = tf.convert_to_tensor(Normal(loc=tf.zeros(discrete_par_size+[continus_par_size]), scale=tf.ones(discrete_par_size+[continus_par_size]), name='Normal_c_w_'+str(idx)))
            continus_variable_prior_b[idx] = tf.convert_to_tensor(Normal(loc=tf.zeros(discrete_par_size), scale=tf.ones(discrete_par_size), name='Normal_c_b_'+str(idx)))
            continus_variable_prior_sigma[idx] = tf.convert_to_tensor(Normal(loc=tf.zeros([1]), scale=tf.ones([1]), name='Normal_c_sigma_'+str(idx)))
            continus_variable_vars[idx] = tf.convert_to_tensor(Normal(loc=tf.add_n([tf.reduce_sum(tf.multiply(continus_par_vars, tf.gather_nd(continus_variable_prior_w[idx], discrete_par_vars)), axis=1), tf.gather_nd(continus_variable_prior_b[idx], discrete_par_vars)]), \
                                                scale=continus_variable_prior_sigma[idx], name='Normal_c_'+str(idx)))
        for i in range(num_discrete_variables):
            tf.add_to_collection('d_pi', discrete_variable_prior_pi[i])
            tf.add_to_collection('d', discrete_variable_vars[i])
        for i in self.continus_variable_idxs:
            tf.add_to_collection('c_w', continus_variable_prior_w[i])
            tf.add_to_collection('c_b', continus_variable_prior_b[i])
            tf.add_to_collection('c_sigma', continus_variable_prior_sigma[i])
            tf.add_to_collection('c', continus_variable_vars[i])           
        filename = '_'.join([self.data_filename, self.name, filename, 'bayes_net.meta'])
        tf.train.export_meta_graph(filename, as_text=True, collection_list=['d_pi', 'd', 'c_w', 'c_b', 'c_sigma', 'c'])
        if visualize_graph:
            #for tensorboard; run >>> tensorboard ==logdir=.
            sess = tf.Session()
            tf.summary.FileWriter(filename+'_tensorboard', sess.graph)

    def train(self, filename, total_batches=10, discrete_batch_iters=1000, continus_batch_iters=10000):        
        sess = tf.Session()
        restorer = tf.train.import_meta_graph(filename, clear_devices=True)
        print("<meta graph imported>")
        [tf.add_to_collection('d_pi_q', Empirical(tf.Variable(tf.zeros(tf.shape(var))), name='Empirical_d_pi_q_'+str.split(str.split(var.name, '/')[0], '_')[-2])) for var in tf.get_collection('d_pi')]
        for var in tf.get_collection('c_w'):
            idx = str.split(str.split(var.name, '/')[0], '_')[-2]
            tf.add_to_collection('c_w_q', Empirical(tf.Variable(tf.zeros(tf.shape(var))), name='Empirical_c_w_q_'+idx))
            print(var.get_shape().as_list())
            tf.add_to_collection('c_b_q', Empirical(tf.Variable(tf.zeros(var.get_shape().as_list()[:-1])), name='Empirical_c_b_q_'+idx))
            tf.add_to_collection('c_sigma_q', Empirical(tf.Variable(tf.zeros([1])), name='Empirical_c_sigma_q_'+idx))
        print("<variables collected>")
        variable_map = dict(zip(tf.get_collection('d') + tf.get_collection('c'), self.design_matrix[:,tuple(np.arange(self.num_discrete_variables))].flatten('F').tolist() + self.design_matrix[:, self.continus_variable_idxs].flatten('F').tolist()))
        discrete_prior_map = dict(zip(tf.get_collection('d_pi'), tf.get_collection('d_pi_q')))
        continus_prior_map = dict(zip(tf.get_collection('c_w') + tf.get_collection('c_b') + tf.get_collection('c_sigma'), tf.get_collection('c_w_q') + tf.get_collection('c_b_q') + tf.get_collection('c_sigma_q')))
        print("<running inference>")
        inference_d = ed.Gibbs(discrete_prior_map, data=dict(variable_map.items() + continus_prior_map.items()))
        inference_c = ed.HMC(continus_prior_map, data=dict(variable_map.items() + discrete_prior_map.items()))
        inference_d.initialize(n_iter=discrete_batch_iters)
        inference_c.initialize(n_iter=continus_batch_iters)
        sess.run(tf.global_variables_initializer())
        for _ in range(total_batches):
            for _ in range(inference_d.n_iter):
                info_dict = inference_d.update()
                inference_d.print_progress(info_dict)
            inference_d.n_iter += discrete_batch_iters
            inference_d.n_print = int(discrete_batch_iters/10)
            inference_d.progbar = Progbar(inference_d.n_iter)
            for _ in range(inference_c.n_iter):
                info_dict = inference_c.update()
                inference_c.print_progress(info_dict)
            inference_c.n_iter += continus_batch_iters
            inference_c.n_print = int(continus_batch_iters/10)
            inference_c.progbar = Progbar(inference_c.n_iter)
        inference_d.finalize()
        inference_c.finalize()
        filename = ''.join(str.split(filename, '.')[:-1], '.') + '_trained_model'
        saver = tf.train.Saver()
        saver.save(sess, filename)
        tf.train.export_meta_graph(filename+'.meta', as_text=True, collection_list=['d_pi', 'd', 'c_w', 'c_b', 'c_sigma', 'c'])

    def test(self, filename, data):
        with tf.Session() as sess:
            restorer = tf.train.import_meta_graph(filename, clear_devices=True)
            restorer.restore(sess, tf.train.latest_checkpoint('./'))
            discrete_variable_vars = tf.get_collection('d')
            discrete_variable_prior_pi = tf.get_collection('d_pi')
            discrete_variable_prior_pi_q = tf.get_collection('d_pi_q')
            discrete_variable_post_map = dict([(ed.copy(discrete_variable_vars[idx], {discrete_variable_prior_pi[idx]: discrete_variable_prior_pi_q[idx]}), data[:,idx].tolist()) for idx in tuple(np.arange(self.num_discrete_variables))])
            continus_variable_data_map = dict(zip(tf.get_collection('c'), data[:,self.continus_variable_idxs].flatten('F').tolist()))
            return ed.evaluate('log_likelihood', data=dict(discrete_variable_post_map.items(), continus_variable_data_map.items()))

#influenza_naive_model = crom3top(2, '../dat/crom3top/dat_naive_dag_gsym.csv', '../dat/crom3top/dat_influenza_gsym.csv', 'influenza_naive')
#influenza_naive_model.save_model()
#influenza_model = crom3top(2, '../dat/crom3top/dat_dag_gsym.csv', '../dat/crom3top/dat_influenza_gsym.csv', 'influenza')
#influenza_model.save_model()
x = crom3top()
x.load_model('../dat/crom3top/dat_influenza_gsym.csv_influenza_naive_crom3top.pickle')
#x.make_bayes_net()
x.train('../dat/crom3top/dat_influenza_gsym.csv_influenza_naive_crom3top__bayes_net.meta')
#x.test('../dat/crom3top/dat_influenza_gsym.csv_influenza_naive_crom3top__bayes_net_trained_model.meta', x.design_matrix)
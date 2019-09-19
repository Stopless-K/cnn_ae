import tensorflow as tf


class Autoencoder(object):

    def __init__(self, batch_x, n_layers, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(), ae_para=[0, 0]):
        self.n_layers = n_layers
        self.transfer = transfer_function
        self.in_keep_prob = 1 - ae_para[0]

        network_weights = self._initialize_weights()
        self.weights = network_weights
        self.sparsity_level = 0.1  # np.repeat([0.05], self.n_hidden).astype(np.float32)
        self.sparse_reg = ae_para[1]
        self.epsilon = 1e-06

        # model

        #self.x = tf.placeholder(tf.float32, [None, self.n_layers[0]])
        self.x = batch_x
        self.keep_prob = tf.placeholder(tf.float32)

    def _build(self):
        self.hidden_encode = []
        h = tf.nn.dropout(self.x, self.keep_prob)
        for layer in range(len(self.n_layers)-1):
            h = self.transfer(
                tf.add(tf.matmul(h, self.weights['encode'][layer]['w']),
                       self.weights['encode'][layer]['b']))
            self.hidden_encode.append(h)

        self.hidden_recon = []
        for layer in range(len(self.n_layers)-1):
            h = self.transfer(
                tf.add(tf.matmul(h, self.weights['recon'][layer]['w']),
                       self.weights['recon'][layer]['b']))
            self.hidden_recon.append(h)
        self.reconstruction = self.hidden_recon[-1]

        # cost
        if self.sparse_reg == 0:
            self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        else:
            self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))+\
                        self.sparse_reg * self.kl_divergence(self.sparsity_level, self.hidden_encode[-1])

        self.optimizer = optimizer.minimize(self.cost)
 

    def _initialize_weights(self):
        all_weights = dict()
        initializer = tf.contrib.layers.xavier_initializer()
        # Encoding network weights
        encoder_weights = []
        for layer in range(len(self.n_layers)-1):
            w = tf.Variable(
                initializer((self.n_layers[layer], self.n_layers[layer + 1]),
                            dtype=tf.float32))
            b = tf.Variable(
                tf.zeros([self.n_layers[layer + 1]], dtype=tf.float32))
            encoder_weights.append({'w': w, 'b': b})
        # Recon network weights
        recon_weights = []
        for layer in range(len(self.n_layers)-1, 0, -1):
            w = tf.Variable(
                initializer((self.n_layers[layer], self.n_layers[layer - 1]),
                            dtype=tf.float32))
            b = tf.Variable(
                tf.zeros([self.n_layers[layer - 1]], dtype=tf.float32))
            recon_weights.append({'w': w, 'b': b})
        all_weights['encode'] = encoder_weights
        all_weights['recon'] = recon_weights
        return all_weights


    def kl_divergence(self, p, p_hat):
        return tf.reduce_mean(p * tf.log(tf.clip_by_value(p, 1e-8, tf.reduce_max(p)))
                              - p * tf.log(tf.clip_by_value(p_hat, 1e-8, tf.reduce_max(p_hat)))
                              + (1 - p) * tf.log(tf.clip_by_value(1-p, 1e-8, tf.reduce_max(1-p)))
                              - (1 - p) * tf.log(tf.clip_by_value(1-p_hat, 1e-8, tf.reduce_max(1-p_hat))))

    def partial_fit(self):
        return  (self.cost, self.optimizer)

    def calc_total_cost(self):
        return self.cost

    def transform(self):
        return self.hidden_encode[-1]

    def reconstruct(self):
        return self.reconstruction

    def setNewX(self,x):
        self.hidden_encode = []
        h = tf.nn.dropout(x, self.keep_prob)
        for layer in range(len(self.n_layers) - 1):
            h = self.transfer(
                tf.add(tf.matmul(h, self.weights['encode'][layer]['w']),
                       self.weights['encode'][layer]['b']))
            self.hidden_encode.append(h)



class ConvAutoencoder(Autoencoder):
    def __init__(self, batch_x, n_layers, kernel_size=3, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(), ae_para=[0,0]):
        super(ConvAutoencoder, self).__init__(batch_x,n_layers, transfer_function, optimizer, ae_para)
        self.ksize = kernel_size
        self.x = batch_x

        self.n_layers = n_layers
        self.is_training= True
        self.activation = transfer_function
        self.in_keep_prob = 1 - ae_para[0]

        
        self.sparsity_level = 0.1  
        self.sparse_reg = ae_para[1]
        self.epsilon = 1e-06

        self.x = batch_x
        self.keep_prob = tf.placeholder(tf.float32)

        self.x = tf.nn.dropout(self.x, self.keep_prob)
        
        self.x = tf.expand_dims(self.x, -1)
        self.shape = tf.shape(self.x)
        self._build()

        # cost
        if self.sparse_reg == 0:
            self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
        else:
            self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))+\
                        self.sparse_reg * self.kl_divergence(self.sparsity_level, self.hidden_encode[-1])

        self.optimizer = optimizer.minimize(self.cost)


    def _build(self):
        
        x = self.x
        #encode reduce dimension at first conv
        
        nb_layers = len(self.n_layers)-1
        for i in range(nb_layers):
            stride = 2 if i==0 else 1
            x = tf.layers.conv1d(x, self.n_layers[i+1], kernel_size=self.ksize, strides=stride, padding='same', activation=None)
            x = tf.layers.batch_normalization(x, axis=1, training=self.is_training)
            x = self.activation(x)

        self.hidden_encode=[x]

        #decode
        transpose_filters = tf.Variable(tf.contrib.layers.xavier_initializer()((self.ksize, self.n_layers[0], self.n_layers[-1]), dtype=tf.float32))
        x = tf.nn.conv1d_transpose(x, transpose_filters, output_shape =self.shape, strides=2, padding='SAME')
        x = tf.layers.batch_normalization(x, 1, self.is_training)
        x = tf.nn.relu(x)

        #last conv

        x = tf.layers.conv1d(x, self.n_layers[0], self.ksize, strides=1, padding='same', activation=None)
        x = tf.layers.batch_normalization(x, 1, self.is_training) 
        self.hidden_recon=[x]
        self.reconstruction=self.hidden_recon[-1]
        
    def train(self):
        self.is_training = True

    def val(self):
        self.is_training = False






      


from __future__ import print_function
from __future__ import absolute_import
import os
import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
from keras.engine import Input
from keras.layers import Concatenate, Dot, Embedding, Dropout, Lambda, Activation, LSTM, Dense,Reshape,Conv1D,Conv2D,SeparableConv2D,MaxPooling1D,Flatten,GlobalMaxPooling1D,dot,Bidirectional,SimpleRNN,GlobalAveragePooling1D,Reshape,Multiply,GlobalAveragePooling2D,BatchNormalization
from keras import backend as K
from keras.models import Model
from keras.utils import plot_model
import pickle
import numpy as np
import logging
from layers.coattention_layer import COAttentionLayer
from layers.attention_layer import AttentionLayer
logger = logging.getLogger(__name__)


class JointEmbeddingModel:
    def __init__(self, config):
        self.config = config
        self.model_params = config.get('model_params', dict())
        self.data_params = config.get('data_params',dict())
        self.tokens = Input(shape=(self.data_params['tokens_len'],), dtype='int32', name='i_tokens')
        self.names = Input(shape=(self.data_params['names_len'],), dtype='int32', name='i_names')
        self.apis = Input(shape=(self.data_params['apis_len'],), dtype='int32', name='i_apis')
        self.sbts = Input(shape=(self.data_params['sbts_len'],), dtype='int32', name='i_sbts')
        self.sim_desc = Input(shape=(self.data_params['sim_desc_len'],), dtype='int32', name='i_sim_desc')
        self.desc_good = Input(shape=(self.data_params['desc_len'],), dtype='int32', name='i_desc_good')
        self.tokens_bad = Input(shape=(self.data_params['tokens_len'],), dtype='int32', name='i_bad_tokens')
        self.names_bad = Input(shape=(self.data_params['names_len'],), dtype='int32', name='i_bad_names')
        self.apis_bad = Input(shape=(self.data_params['apis_len'],), dtype='int32', name='i_bad_apis')
        self.sbts_bad = Input(shape=(self.data_params['sbts_len'],), dtype='int32', name='i_bad_sbts')
        self.sim_desc_bad = Input(shape=(self.data_params['sim_desc_len'],), dtype='int32', name='i_bad_sim_desc')
        self.desc_bad = Input(shape=(self.data_params['desc_len'],), dtype='int32', name='i_desc_bad')

        # initialize a bunch of variables that will be set later
        self._sim_model = None
        self._training_model = None
        self._shared_model = None
        # self.prediction_model = None

        # create a model path to store model info
        if not os.path.exists(self.config['workdir'] + 'models/' + self.model_params['model_name'] + '/'):
            os.makedirs(self.config['workdir'] + 'models/' + self.model_params['model_name'] + '/')

    def build(self):
        '''
        1. Build Code Representation Model
        '''
        logger.debug('Building Code Representation Model')
        tokens = Input(shape=(self.data_params['tokens_len'],), dtype='int32', name='tokens')
        names = Input(shape=(self.data_params['names_len'],), dtype='int32', name='names')
        apis = Input(shape=(self.data_params['apis_len'],), dtype='int32', name='apis')
        sim_desc = Input(shape=(self.data_params['sim_desc_len'],), dtype='int32', name='sim_desc')
        sbts = Input(shape=(self.data_params['sbts_len'],), dtype='int32', name='sbt')

        ## Tokens Representation ##
        # 1.embedding

        init_emb_weights = np.load(self.config['workdir']+self.model_params['init_embed_weights_tokens']) if self.model_params['init_embed_weights_tokens'] is not None else None
        init_emb_weights = init_emb_weights if init_emb_weights is None else [init_emb_weights]
        # init_emb_weights =pickle.load(open(self.config['workdir']+self.model_params['init_embed_weights_tokens'],'rb'))
        embedding = Embedding(input_dim=self.data_params['n_tokens_words'],
                              output_dim=self.model_params.get('n_embed_dims'),
                              weights=init_emb_weights,
                              trainable=True,
                              mask_zero=False,#Whether 0 in the input is a special "padding" value that should be masked out. 
                              #If set True, all subsequent layers must support masking, otherwise an exception will be raised.
                              name='embedding_tokens')
        tokens_embedding = embedding(tokens)

        init_emb_weights = np.load(self.config['workdir'] + self.model_params['init_embed_weights_names']) if \
        self.model_params['init_embed_weights_names'] is not None else None
        init_emb_weights = init_emb_weights if init_emb_weights is None else [init_emb_weights]
        # init_emb_weights =pickle.load(open(self.config['workdir']+self.model_params['init_embed_weights_tokens'],'rb'))
        embedding = Embedding(input_dim=self.data_params['n_names_words'],
                              output_dim=self.model_params.get('n_embed_dims'),
                              weights=init_emb_weights,
                              trainable=True,
                              mask_zero=False,
                              # Whether 0 in the input is a special "padding" value that should be masked out.
                              # If set True, all subsequent layers must support masking, otherwise an exception will be raised.
                              name='embedding_names')
        names_embedding = embedding(names)

        init_emb_weights = np.load(self.config['workdir'] + self.model_params['init_embed_weights_apis']) if \
        self.model_params['init_embed_weights_apis'] is not None else None
        init_emb_weights = init_emb_weights if init_emb_weights is None else [init_emb_weights]
        # init_emb_weights =pickle.load(open(self.config['workdir']+self.model_params['init_embed_weights_tokens'],'rb'))
        embedding = Embedding(input_dim=self.data_params['n_apis_words'],
                              output_dim=self.model_params.get('n_embed_dims'),
                              weights=init_emb_weights,
                              trainable=True,
                              mask_zero=False,
                              # Whether 0 in the input is a special "padding" value that should be masked out.
                              # If set True, all subsequent layers must support masking, otherwise an exception will be raised.
                              name='embedding_apis')
        apis_embedding = embedding(apis)


        init_emb_weights = np.load(self.config['workdir'] + self.model_params['init_embed_weights_sbts']) if self.model_params['init_embed_weights_tokens'] is not None else None
        init_emb_weights = init_emb_weights if init_emb_weights is None else [init_emb_weights]
        # init_emb_weights =pickle.load(open(self.config['workdir']+self.model_params['init_embed_weights_tokens'],'rb'))
        embedding = Embedding(input_dim=self.data_params['n_sbts_words'],
                              output_dim=self.model_params.get('n_embed_dims'),
                              weights=init_emb_weights,
                              trainable=True,
                              mask_zero=False,
                              # Whether 0 in the input is a special "padding" value that should be masked out.
                              # If set True, all subsequent layers must support masking, otherwise an exception will be raised.
                              name='embedding_sbts')
        sbts_embedding = embedding(sbts)


        tokens_embedding=Concatenate(axis=1)([names_embedding,apis_embedding,tokens_embedding,sbts_embedding])
        dropout = Dropout(0.25, name='dropout_code_embed')
        tokens_dropout=dropout(tokens_embedding)
        tokens_dropout = AttentionLayer(name="tokens_attention")(tokens_dropout)

        init_emb_weights = np.load(self.config['workdir'] + self.model_params['init_embed_weights_desc']) if self.model_params['init_embed_weights_desc'] is not None else None
        init_emb_weights = init_emb_weights if init_emb_weights is None else [init_emb_weights]
        # init_emb_weights =pickle.load(open(self.config['workdir']+self.model_params['init_embed_weights_sbt'],'rb'))
        embedding = Embedding(input_dim=self.data_params['n_desc_words'],
                              output_dim=self.model_params.get('n_embed_dims'),
                              weights=init_emb_weights,
                              trainable=True,
                              mask_zero=False,
                              # Whether 0 in the input is a special "padding" value that should be masked out.
                              # If set True, all subsequent layers must support masking, otherwise an exception will be raised.
                              name='embedding_sim_desc')
        sim_desc_embedding = embedding(sim_desc)
        dropout = Dropout(0.25, name='dropout_sim_desc_embed')
        sim_desc_dropout = dropout(sim_desc_embedding)
        sim_desc_dropout = AttentionLayer(name="sim_desc_attention")(sim_desc_dropout)

        '''
        2. Build Desc Representation Model
        '''
        ## Desc Representation ##
        logger.debug('Building Desc Representation Model')
        desc = Input(shape=(self.data_params['desc_len'],), dtype='int32', name='desc')
        # 1.embedding
        init_emb_weights = np.load(self.config['workdir'] + self.model_params['init_embed_weights_desc']) if self.model_params['init_embed_weights_desc'] is not None else None
        init_emb_weights = init_emb_weights if init_emb_weights is None else [init_emb_weights]
        # init_emb_weights =pickle.load(open(self.config['workdir']+self.model_params['init_embed_weights_desc'],'rb'))
        embedding = Embedding(input_dim=self.data_params['n_desc_words'],
                              output_dim=self.model_params.get('n_embed_dims'),
                              weights=init_emb_weights,
                              trainable=True,
                              mask_zero=False,
                              # Whether 0 in the input is a special "padding" value that should be masked out.
                              # If set True, all subsequent layers must support masking, otherwise an exception will be raised.
                              name='embedding_desc')
        desc_embedding = embedding(desc)
        dropout = Dropout(0.25, name='dropout_desc_embed')
        desc_dropout = dropout(desc_embedding)
        desc_dropout = AttentionLayer(name="desc_attention")(desc_dropout)

        gap_cnn = GlobalAveragePooling1D(name='globalaveragepool_cnn')  # 按照时间步维度进行平均池化
        gmp_cnn = GlobalMaxPooling1D(name="globalmaxpool")
        # out_2 row wise
        attention_trans_layer = Lambda(lambda x: K.permute_dimensions(x, (0, 2, 1)), name='trans_coattention')
        Bdot = Lambda(lambda x: K.batch_dot(x[0], x[1]), name="bdot")
        Multi = Lambda(lambda x: x[0] * x[1], name='multi')

        coatt_sq = COAttentionLayer(name="coatt_sq")
        coatt_tq = COAttentionLayer(name="coatt_tq")
        coweight_sq = coatt_sq([sim_desc_dropout, desc_dropout])
        att_matrix = coweight_sq
        dense_sq = Dense(100, name="dense_sq")
        att_matrix = dense_sq(att_matrix)
        att_matrix = Multi([att_matrix, sim_desc_dropout])
        dense_sq2 = Dense(30, name="dense_sq2")
        att_matrix = dense_sq2(att_matrix)
        att_matrix = gmp_cnn(att_matrix)
        att_matrix = Activation("softmax", name="activ_sq")(att_matrix)
        sq_desc_out = Dot(axes=1, normalize=False, name="dot_sq")([att_matrix, desc_dropout])

        att_matrix = attention_trans_layer(coweight_sq)
        dense_sq3 = Dense(100, name="dense_sq3")
        att_matrix = dense_sq3(att_matrix)
        att_matrix = Multi([att_matrix, desc_dropout])
        dense_sq4 = Dense(30, name="dense_sq4")
        att_matrix = dense_sq4(att_matrix)
        att_matrix = gmp_cnn(att_matrix)
        att_matrix = Activation("softmax", name="activ_sq2")(att_matrix)
        sq_code_out = Dot(axes=1, normalize=False, name="dot_sq2")([att_matrix, sim_desc_dropout])

        coweight_tq = coatt_tq([tokens_dropout, desc_dropout])

        att_matrix = coweight_tq
        Add = Lambda(lambda x: x[0] + x[1], name="add")
        dense_weight = Dense(236, name="dense_weight")
        att_matrix_sq = attention_trans_layer(dense_weight(attention_trans_layer(coweight_sq)))
        att_matrix = Add([att_matrix, Activation("relu")(att_matrix_sq)])
        dense_tq = Dense(100, name="dense_tq")
        att_matrix = dense_tq(att_matrix)
        att_matrix = Multi([att_matrix, tokens_dropout])
        dense_tq2 = Dense(30, name="dense_tq2")
        att_matrix = dense_tq2(att_matrix)
        att_matrix = gap_cnn(att_matrix)
        att_matrix = Activation("softmax", name="activ_tq")(att_matrix)
        tq_desc_out = Dot(axes=1, normalize=False, name="dot_tq")([att_matrix, desc_dropout])

        att_matrix = attention_trans_layer(coweight_tq)
        dense_tq3 = Dense(100, name="dense_tq3")
        att_matrix = dense_tq3(att_matrix)
        att_matrix = Multi([att_matrix, desc_dropout])
        dense_tq4 = Dense(236, name="dense_tq4")
        att_matrix = dense_tq4(att_matrix)
        att_matrix = gap_cnn(att_matrix)
        att_matrix = Activation("softmax", name="activ_tq2")(att_matrix)
        tq_code_out = Dot(axes=1, normalize=False, name="dot_tq2")([att_matrix, tokens_dropout])

        merged_desc_out = Concatenate(name='desc_orig_merge', axis=1)([tq_desc_out, sq_desc_out])
        merged_code_out = Concatenate(name='code_orig_merge', axis=1)([tq_code_out, sq_code_out])
        reshape_desc = Reshape((2, 100))(merged_desc_out)  # b*2*100
        reshape_code = Reshape((2, 100))(merged_code_out)
        att_desc_out = AttentionLayer(name='desc_merged_attention_layer')(reshape_desc)  # b*2*100
        att_code_out = AttentionLayer(name='code_merged_attention_layer')(reshape_code)
        gap = GlobalAveragePooling1D(name='blobalaveragepool')  # b*2*100  ——> b*100
        mulop = Lambda(lambda x: x * 2.0, name='mulop')
        desc_out = mulop(gap(att_desc_out))
        code_out = mulop(gap(att_code_out))

        """
                3: calculate the cosine similarity between code and desc
                """
        logger.debug('Building similarity model')
        cos_sim = Dot(axes=1, normalize=True, name='cos_sim')([code_out, desc_out])

        sim_model = Model(inputs=[tokens, names, apis, sbts, sim_desc, desc], outputs=[cos_sim], name='sim_model')
        self._sim_model = sim_model  # for model evaluation
        print("\nsummary of similarity model")
        self._sim_model.summary()
        fname = self.config['workdir'] + 'models/' + self.model_params['model_name'] + '/_sim_model.png'
        # plot_model(self._sim_model, show_shapes=True, to_file=fname)

        '''
        4:Build training model
        '''
        good_sim1 = sim_model(
            [self.tokens, self.names, self.apis, self.sbts, self.sim_desc, self.desc_good])  # similarity of good output
        good_sim2 = sim_model(
            [self.tokens_bad, self.names_bad, self.apis_bad, self.sbts_bad, self.sim_desc_bad, self.desc_bad])
        bad_sim1 = sim_model(
            [self.tokens, self.names, self.apis, self.sbts, self.sim_desc, self.desc_bad])  # similarity of bad output
        bad_sim2 = sim_model(
            [self.tokens_bad, self.names_bad, self.apis_bad, self.sbts_bad, self.sim_desc_bad, self.desc_good])
        loss = Lambda(lambda x: K.maximum(1e-6, self.model_params['margin'] - x[0] + x[1])+K.maximum(1e-6, self.model_params['margin'] - x[2] + x[3])+K.maximum(1e-6, self.model_params['margin'] - x[0] + x[3])+K.maximum(1e-6, self.model_params['margin'] - x[2] + x[1]),
                      output_shape=lambda x: x[0], name='loss')([good_sim1, bad_sim1,good_sim2,bad_sim2])

        logger.debug('Building training model')
        self._training_model = Model(
            inputs=[self.tokens, self.names, self.apis, self.sbts, self.sim_desc, self.desc_good, 
                    self.tokens_bad, self.names_bad, self.apis_bad, self.sbts_bad,self.sim_desc_bad,self.desc_bad],
            outputs=[loss], name='training_model')
        print('\nsummary of training model')
        self._training_model.summary()
        fname = self.config['workdir'] + 'models/' + self.model_params['model_name'] + '/_training_model.png'
        # plot_model(self._training_model, show_shapes=True, to_file=fname)

    def compile(self, optimizer, **kwargs):
        logger.info('compiling models')
        self._training_model.compile(loss=lambda y_true, y_pred: y_pred + y_true - y_true, optimizer=optimizer,
                                     **kwargs)
        # +y_true-y_true is for avoiding an unused input warning, it can be simply +y_true since y_true is always 0 in the training set.
        self._sim_model.compile(loss='binary_crossentropy', optimizer=optimizer, **kwargs)

    def fit(self, x, **kwargs):
        assert self._training_model is not None, 'Must compile the model before fitting data'
        y = np.zeros(shape=x[0].shape[:1], dtype=np.float32)
        return self._training_model.fit(x, y, **kwargs)

    def predict(self, x, **kwargs):
        return self._sim_model.predict(x, **kwargs)

    def save(self, sim_model_file, **kwargs):
        assert self._sim_model is not None, 'Must compile the model before saving weights'
        self._sim_model.save_weights(sim_model_file, **kwargs)

    def load(self, sim_model_file, **kwargs):
        assert self._sim_model is not None, 'Must compile the model loading weights'
        self._sim_model.load_weights(sim_model_file, **kwargs)



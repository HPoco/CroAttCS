# 'ast_large/train.desc.pkl': right append 1 for UNK
def get_config():   
    conf = {
        'workdir': './data/github/',
        'data_params':{
            'train_tokens': 'train.tokens.pkl',
            'train_names': 'train.methodname.pkl',
            'train_apis': 'train.apiseq.pkl',
            'train_sbts': 'train.sbt.pkl',
            'train_desc': 'train.desc.pkl',
            'train_sim_desc': 'train_IR_code_desc.pkl',
            # valid data
            'valid_tokens': 'test.tokens.pkl',
            'valid_names': 'test.methodname.pkl',
            'valid_apis': 'test.apiseq.pkl',
            'valid_sbts': 'test.sbt.pkl',
            'valid_desc': 'test.desc.pkl',
            'valid_sim_desc': 'test_IR_code_desc.pkl',
            #use data (computing code vectors)
            'use_codebase':'../test_source.txt',#'use.rawcode.h5'
            #results data(code vectors)            
            'use_codevecs':'../use.codevecs.normalized.h5',#'use.codevecs.h5',
                   
            #parameters
            'tokens_len':50,
            'names_len':6,
            'apis_len':30,
            'sbts_len':150,
            'desc_len': 30,
            'sim_desc_len': 30,
            'n_desc_words': 30001,# len(vocabulary) + 1
            'n_tokens_words':46145,
            'n_apis_words': 52892,
            'n_names_words': 27177,
            'n_sbts_words':89,  #vocabulary info
            'vocab_tokens':'vocab/vocab.tokens.pkl',
            'vocab_names': 'vocab/vocab.methodname.pkl',
            'vocab_apis': 'vocab/vocab.apiseq.pkl',
            'vocab_desc':'vocab/vocab.desc.pkl',
            'vocab_sbts':'vocab/vocab.sbt.pkl',
        },               
        'training_params': {
            'batch_size': 256,  #256
            'chunk_size':100000,
            'nb_epoch': 250,
            'validation_split': 0.1,
            # 'optimizer': 'adam',
            #'optimizer': Adam(clip_norm=0.1),
            'valid_every': 1,
            'n_eval': 100,
            'evaluate_all_threshold': {
                'mode': 'all',
                'top1': 0.4,
            },
            'save_every': 1,
            'reload':0, #that the model is reloaded from . If reload=0, then train from scratch
        },

        'model_params': {
            'model_name':'JointEmbeddingModel',
            'n_embed_dims': 100,
            'n_hidden': 400,#number of hidden dimension of code/desc representation
            # recurrent
            'n_lstm_dims': 100, # * 2#'word2vec_100_methname.h5',
            'init_embed_weights_tokens': None,#'word2vec_100_tokens.h5',
            'init_embed_weights_names': None,
            'init_embed_weights_apis': None,
            'init_embed_weights_sbts':None,
            'init_embed_weights_desc': None,#'word2vec_100_desc.h5',
            'init_embed_weights_sim_desc': None,
            'init_embed_weights_sim_tokens':None,
            'margin': 0.30,
            'sim_measure':'cos',#similarity measure: gesd, cosine, aesd
        }        
    }
    return conf





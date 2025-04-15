import pickle
import codecs
### alter 3 to 0

def desc2index(desc_dict):
    desc = []
    desc_index = []

    fileDesc = codecs.open('../CSN-JAVA/keras/data/github/test_desc.txt', encoding='utf8',
                           errors='replace').readlines()
    for line in fileDesc:
        line = line.split()
        desc.append(line)

    for item in desc:
        new_item = []
        for word in item:
            try:
                new_item.append(desc_dict[word])
            except:
                new_item.append(0) #1
        desc_index.append(new_item)
    return desc_index


def tokens2index(tokens_dict):
    tokens = []
    tokens_index = []

    fileTokens = codecs.open('../CSN-JAVA/keras/data/github/test_tokens.txt', encoding='utf8',
                             errors='replace').readlines()
    for line in fileTokens:
        line = line.split()
        tokens.append(line)

    for item in tokens:
        new_item = []
        for word in item:
            try:
                new_item.append(tokens_dict[word])
            except:
                new_item.append(0)
        tokens_index.append(new_item)
    return tokens_index





def simWords2index(desc_dict):
    tokens = []
    tokens_index = []
    fileTokens = codecs.open('../CSN-JAVA/keras/data/github/test_IR_code_desc_sw.txt', encoding='utf8',
                             errors='replace').readlines()


    for line in fileTokens:
        line = line.split()
        tokens.append(line)

    for item in tokens:
        new_item = []
        for word in item:
            try:
                new_item.append(desc_dict[word])
            except:
                new_item.append(0)
        tokens_index.append(new_item)
    return tokens_index





if __name__ == '__main__':

    desc_dict = pickle.load(open('../DeepCom-JAVA/keras/data/github/vocab.desc.pkl', 'rb'))
    tokens_dict = pickle.load(open('../DeepCom-JAVA/keras/data/github/vocab.tokens.pkl', 'rb'))

    desc = desc2index(desc_dict)
    tokens = tokens2index(tokens_dict)


    pickle.dump(tokens, open('../CSN-JAVA/keras/data/github/train.tokens.pkl', 'wb'))
    pickle.dump(desc, open('../CSN-JAVA/keras/data/github/test.desc.pkl', 'wb'))

    print('finish transfering data to index...')

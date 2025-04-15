if __name__ == '__main__':

    sourceFile = open("../CSN-JAVA/keras/data/github/train_desc.txt", "r", encoding="utf-8")
    corpusFile = open("../CSN-JAVA/keras/data/github/corpus_desc.txt", "w", encoding="utf-8")
    for num, line in enumerate(sourceFile):
        print(num)
        words = line.split()
        for word in words:
            corpusFile.write(word + '\n')
    sourceFile.close()
    corpusFile.close()

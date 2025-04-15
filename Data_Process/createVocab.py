from collections import Counter


vocabWord = open("../CSN-JAVA/keras/data/github/corpus_desc.txt", "r", encoding="utf-8")
processStaFile = open("../CSN-JAVA/keras/data/github/vocab.desc.txt", "w", encoding="utf-8")
staTreeList = []
while 1:
    word = vocabWord.readline().splitlines()
    if not word:
        break
    staTreeList.append(word[0])

staTreeDic = Counter(staTreeList)

for k, v in staTreeDic.items():
    if v >= 11:
        processStaFile.write(k + '\n')
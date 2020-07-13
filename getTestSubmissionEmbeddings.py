import numpy as np
import pandas as pd
import os, os.path
import skipthoughts

model   = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)

curr_dir  = os.getcwd()
test_dir  = os.path.join(curr_dir, "Data/testSubmission.csv")
test_data = pd.read_csv(test_dir)
test_data.index.name = 'id'
test_data.rename(columns={'InputSentence1':'Sentence1',
                          'InputSentence2':'Sentence2',
                          'InputSentence3':'Sentence3',
                          'InputSentence4':'Sentence4',
                          'RandomFifthSentenceQuiz1':'Ending1',
                          'RandomFifthSentenceQuiz2':'Ending2'},
                 inplace=True)

n_samples   = test_data.shape[0]
n_sentences = test_data.shape[1]

sentencesToEmbed = []
for i in range(n_samples):
    for j in range(n_sentences):
        sentencesToEmbed.append(test_data.iloc[i, j])

testEmbeddings = encoder.encode(sentencesToEmbed)

for i in range(n_samples):
    for j in range(n_sentences):
        test_data.iat[i, j] = testEmbeddings[i*n_sentences+j].tolist()

embeddedTestDir = os.path.join(curr_dir, "Embeddings/embeddedTestSubmission.npy")
embeddedTestDict = test_data.to_dict()
np.save(embeddedTestDir, embeddedTestDict)

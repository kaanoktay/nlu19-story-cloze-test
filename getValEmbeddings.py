import numpy as np
import pandas as pd
import os, os.path
import skipthoughts

model   = skipthoughts.load_model()
encoder = skipthoughts.Encoder(model)

curr_dir = os.getcwd()
val_dir  = os.path.join(curr_dir, "Data/val.csv")
val_data = pd.read_csv(val_dir)
val_data = val_data.drop(columns="InputStoryid")
val_data.index.name = 'id'
val_data.rename(columns={'InputSentence1':'Sentence1',
                         'InputSentence2':'Sentence2',
                         'InputSentence3':'Sentence3',
                         'InputSentence4':'Sentence4',
                         'RandomFifthSentenceQuiz1':'Ending1',
                         'RandomFifthSentenceQuiz2':'Ending2',
                         'AnswerRightEnding':'RightEnding'},
                inplace=True)

n_samples   = val_data.shape[0]
n_sentences = val_data.shape[1] - 1

sentencesToEmbed = []
for i in range(n_samples):
    for j in range(n_sentences):
        sentencesToEmbed.append(val_data.iloc[i, j])

valEmbeddings = encoder.encode(sentencesToEmbed)

for i in range(n_samples):
    for j in range(n_sentences):
        val_data.iat[i, j] = valEmbeddings[i*n_sentences+j].tolist()

embeddedValDir = os.path.join(curr_dir, "Embeddings/embeddedVal.npy")
embeddedValDict = val_data.to_dict()
np.save(embeddedValDir, embeddedValDict)

import numpy as np
import pandas as pd
import pickle
from sklearn.utils import shuffle

class Model:
    """
    class containing functions used to preprocess the data
    """
    def createEmbeddingSum(self, embeddedValDir):

        embeddedVal    = pd.DataFrame.from_dict(np.load(embeddedValDir, allow_pickle=True).item())
        columnOrder    = ["Sentence1", "Sentence2", "Sentence3", "Sentence4", "Ending1", "Ending2", "RightEnding"]
        embeddedVal    = embeddedVal[columnOrder]

        num_stories    = embeddedVal.shape[0]
        dim_embeddings = 4800

        data_pos       = np.zeros((num_stories,5, dim_embeddings))
        data_neg       = np.zeros((num_stories,5, dim_embeddings))
        labels_pos     = np.ones((num_stories,1))
        labels_neg     = np.zeros((num_stories,1))

        id_stories     = np.arange(num_stories)
        np.random.shuffle(id_stories)
        for i, r in enumerate(id_stories):
            curr_story     = embeddedVal.iloc[r]
            right_ending   = np.array(curr_story["Ending" + str(curr_story["RightEnding"])])
            false_ending   = np.array(curr_story["Ending" + str(3-curr_story["RightEnding"])])
            for j in range(1,5):
                # Right endings dataset
                data_pos[i,j-1,:]  = np.array(curr_story["Sentence"+str(j)])
                # False endings dataset
                data_neg[i,j-1 :]  = np.array(curr_story["Sentence"+str(j)])

                if j==4:
                    data_pos[i,j,:] = right_ending
                    data_neg[i,j,:] = false_ending

        self.data = {"rightData": data_pos, "falseData": data_neg, "rightLabels":labels_pos, "falseLabels":labels_neg}

    def createTestSubmission(self, embeddedValDir):

        embeddedVal    = pd.DataFrame.from_dict(np.load(embeddedValDir, allow_pickle=True).item())
        columnOrder    = ["Sentence1", "Sentence2", "Sentence3", "Sentence4", "Ending1", "Ending2"]
        embeddedVal    = embeddedVal[columnOrder]

        num_stories    = embeddedVal.shape[0]
        dim_embeddings = 4800

        data_ending1   = np.zeros((num_stories,5, dim_embeddings))
        data_ending2   = np.zeros((num_stories,5, dim_embeddings))

        id_stories     = np.arange(num_stories)

        for i, r in enumerate(id_stories):
            curr_story = embeddedVal.iloc[r]
            ending1    = np.array(curr_story["Ending1"])
            ending2    = np.array(curr_story["Ending2"])
            for j in range(1,5):
                # Right endings dataset
                data_ending1[i,j-1,:]  = np.array(curr_story["Sentence"+str(j)])
                # False endings dataset
                data_ending2[i,j-1 :]  = np.array(curr_story["Sentence"+str(j)])

                if j==4:
                    data_ending1[i,j,:] = ending1
                    data_ending2[i,j,:] = ending2

        X_test_ending1 = data_ending1
        X_test_ending2 = data_ending2

        return X_test_ending1, X_test_ending2

    def train_test_split(self, training_percent= 90):
        rightData   = self.data["rightData"]
        falseData   = self.data["falseData"]

        rightLabels = self.data["rightLabels"]
        falseLabels = self.data["falseLabels"]

        n_samples   = rightData.shape[0]
        n_train     = n_samples*training_percent//100

        return (rightData[:n_train, :], falseData[:n_train, :], rightData[n_train:, :], falseData[n_train:, :],
                rightLabels[:n_train, :], falseLabels[:n_train, :], rightLabels[n_train:, :], falseLabels[n_train:, :])

    def samples(self, embeddedValDir, training_percent=90 , randomstate =42, permute =True ):
        """
        Call this function only to return final samples splitted into training and testing data.
        permute  (bool): If set as true will randomly shuffle the data for trainig set (only)
        """

        self.createEmbeddingSum(embeddedValDir)
        (X_train_right, X_train_false, X_test_right, X_test_false,
         y_train_right, y_train_false, y_test_right, y_test_false) = self.train_test_split(training_percent)
        X = np.concatenate((X_train_right, X_train_false),axis=0)
        Y = np.concatenate((y_train_right,y_train_false),axis=0)
        if(permute):
            np.random.seed(randomstate)
            index   = np.random.permutation(X.shape[0])
            X_train  = X[index,:]
            Y_train  = Y[index,:]
        else:
            X_train  = X
            Y_train  = Y

        return X_train, Y_train, X_test_right, X_test_false, y_test_right, y_test_false

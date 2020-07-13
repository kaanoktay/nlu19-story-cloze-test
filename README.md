# INSTRUCTIONS

- First change the current directory to this folder because we need to download files directly to this folder.

  ```bash
  cd nlu19-story-cloze-test
  ```

- Download the necessary files for creating skip-thought embeddings by typing the following commands in the terminal:

  ```bash
  wget http://www.cs.toronto.edu/~rkiros/models/dictionary.txt
  wget http://www.cs.toronto.edu/~rkiros/models/utable.npy
  wget http://www.cs.toronto.edu/~rkiros/models/btable.npy
  wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz
  wget http://www.cs.toronto.edu/~rkiros/models/uni_skip.npz.pkl
  wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz
  wget http://www.cs.toronto.edu/~rkiros/models/bi_skip.npz.pkl
  ```

- Create the necessary conda environment by writing the following command. Because all the conda-spec files are created in MacOS environment, use this spec file within MacOS:

  ```bash
  conda create --name getEmbeddings --file SkipThoughtRequirements.txt 
  conda activate getEmbeddings
  ```

  - If the OS is not MacOS and the user does not use conda, create a new virtual environment with python2.8 version and download the required libraries by running the following code:

    ```bash
    pip install -r SkipThoughtPipRequirements.txt
    ```

- Run the following commands to create embeddings for training, validation and test purposes:

python getValEmbeddings.py
python getTestReportEmbeddings.py
python getTestSubmissionEmbeddings.py

- Now we have the necessary embeddings and create the necessary conda environment to generate the submission file (or to reproduce the experiments in the report) by writing the following command:
  
  ```bash
  conda deactivate
  conda create --name trainingSubmission --file StoryClozeRequirements.txt 
  conda activate trainingSubmission
  ```

  - If a different environment is used instead of a conda environment, deactivate the current environment, create a new environment with python3.6 and download the required libraries by running the following code:

    ```bash
    pip install -r StoryClozePipRequirements.txt
    ```

- To reproduce the submission file (submission.csv) which includes the predictions for the given unlabelled test set (using the best model found by training and validating on the validation set), run the following code:
  
  ```bash
  python submission.py
  ```

- To reproduce the figures, accuracy values and the best model, run the following code which includes training on validation set and producing results using the labelled test set.

  ```bash
  python trainer.py
  ```

# Appendix

- The predictions of endings for unlabelled test set is in SUBMISSION.csv

- The report of this work can found in REPORT.pdf

- The files for generating the embeddings were downloaded from the following GitHub repository: https://github.com/ryankiros/skip-thoughts
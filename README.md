# Artifficial_Intelligence_2

This repo contains code implementations and reports for developing a multiclass sentiment classifier for tweets in the Greek language about the Greek General elections. For this project series 4 different approaches were followed, in order to showcase the evolution from training Machine Learning models to Deep Learning models up until finetuning Large Language models like BERT, which today(2024) is the new baseline. Projects 1-4 were implemented with the following algorithms and architectures:

1. Logistic Regression
2. Feed Forward Neural Networks (FNNs)
3. Recurrent Neural Networks (RNNs) with LSTM/GRU cells + Attention
4. Finetuning [GreekBERT](https://huggingface.co/nlpaueb/bert-base-greek-uncased-v1) and [DistilGreekBERT](https://huggingface.co/EftychiaKarav/DistilGREEK-BERT) for Sequence classification

All 4 projects were implemented in the framework of Kaggle competitions for the course of [Artifficial Intelligence II](https://www.di.uoa.gr/civis/courses/C02) of the 
[Department of Informatics](https://www.di.uoa.gr/en) in the University of Athens.

# Development Environment and Libraries
- Google Colab was used for training and fine-tuning of the models.
- Python > 3.6
- PyTorch
- Torchtext
- MatPlotLib & Scikit Learn Libraries for implementing the model and plotting graphs
- NLTK for loading and preprocessing the data

# Logistic Regression 
([Kaggle_competition](https://www.kaggle.com/competitions/ys19-2023-assignment-1))
- Experiment Workflow: Uni-Bi-Trigrams with TF-IDF and Countvectorizer
- Features: vectorised N_grams, count of positive, ount of negative words per tweet, tweet length in tokens
- Hyperparameter tuning: GridSearchCV (cv=5)
- Chosen model:
  - TF-IDF (uni-grams)
  - max features:1000
  - C= 1.0
  - solver: saga
  - regularization: L1 (Lasso regression)
# Feeforward Neural Networks
([Kaggle_competition](https://www.kaggle.com/competitions/ys19-2023-assignment-2))
- Experiment Workflow: FFNN with different:
  - activation functions: ReLU, Randomized Leaky ReLU, SELU (Self-normalizing neural nets)
  - weight initialization: He, Glorot, LeCun
  - regularization techniques: dropout, batch normalisation, StandardScaler
  - output activation function: sigmoid, softmax
  - n_epochs: 10,40,50
  - batch_size: 10, 128
- Vectorization: Avg Word2Vec (200d)
- Hyperparameter tuning:
  - [Optuna](https://optuna.org/) search space (num_layers: 1-3 , num_units: 4-128, dropout_rate: 0.2-0.5, optimizer: {Adam, RMSProp, SGD}, learning_rate: 1e-4 - 1e-1)
- Chosen model
  - FFNN (RReLU)
  - n_layers: 1
  - n_units_l0: 43
  - RMSProp
  - epochs: 50
  - weight_initialisation: He
  - regularization: Scaling (StandardScaler), Dropout (0.4), Batch Normalisation
  - lr: 4e-3, scheduler: exponential (gamma=0.5)
  - batch_size: 128
  - output: softmax

# Recurrent Neural Networks (RNNs) with LSTM/GRU cells
([Kaggle_competition](https://www.kaggle.com/competitions/ys19-2023-assignment-3))
- Experiment Workflow
- Hyperparameter tuning:
  - Optuna search space: (num_hidden_layers: 1-3, embedding (hidden)size: 64-256, cell_type:GRU|LSTM, dropout_rate: 0.2-0.7, gadient_clipping: 1-5)
- Chosen model
# Finetuning GreekBERT/DistilGreekBERT 
([Kaggle_competition](https://www.kaggle.com/competitions/ys19-2023-assignment-4a))
- Experiment Workflow
- Chosen model
# Overal Comparison

# Contributor Expectations 

# References

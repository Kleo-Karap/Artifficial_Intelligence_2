# Artifficial_Intelligence_2
Sentiment classification: From ML to DL to very DL models

This repo contains code implementations and report for developing a multiclass sentiment classifier using 4 different approaches:

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
- MatPlotLib & Scikit Learn Libraries for implementing the model and plotting graphs
- NLTK for loading and preprocessing the data

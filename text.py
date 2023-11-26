import numpy as np
import psycopg2
import gensim.downloader as api
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize
import nltk
from nltk.corpus import stopwords
import re
import os

nltk.download('stopwords')
nltk.download('punkt')

model = api.load('word2vec-google-news-300')

text = "cats and dogs"

tokens = word_tokenize(text)
embeddings = [model[word] for word in tokens if word in model]
sentence_embedding = sum(embeddings) / len(embeddings)

print(sentence_embedding.tolist())
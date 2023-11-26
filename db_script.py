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

# postgres db params
dbname = 'test_db'
user = 'root'
password = 'password'
host = 'localhost' # change based on ip of postgres docker container if localhost doesn't work
port = '5432'

def insert_embedding(file_path, video_title, frame_number):
    # Load the embedding
    embedding_image = np.load(file_path)
    embedding_image = embedding_image.flatten()
    embedding_list = embedding_image.tolist()
    embedding_list = embedding_list[0:15999] # max 16000 dims, will fix later
    # Insert into the database
    conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
    cur = conn.cursor()
    cur.execute("INSERT INTO video_embeddings (video_title, frame_number, embedding) VALUES (%s, %s, %s)", 
                (video_title, frame_number, embedding_list,))
    conn.commit()

    # Close the connection
    cur.close()
    conn.close()

def preprocess_text(text):
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'\W+|\d+', ' ', text)
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens]
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

model = api.load('word2vec-google-news-300')

def insert_text_embedding(file_path, video_title):
    # Load text
    file = open(file_path, 'r', encoding='utf-8')
    text = file.read()
    file.close()

    # word2vec embedding
    tokens = preprocess_text(text)
    embeddings = [model[word] for word in tokens if word in model]
    sentence_embedding = sum(embeddings) / len(embeddings)
    sentence_embedding = sentence_embedding.tolist()
    # Insert into the database
    conn = psycopg2.connect(dbname=dbname, user=user, password=password, host=host, port=port)
    cur = conn.cursor()
    cur.execute("INSERT INTO text_embeddings (video_title, embedding) VALUES (%s, %s)", 
                (video_title, sentence_embedding,))
    conn.commit()

    # Close the connection
    cur.close()
    conn.close()

# insert video frame embeddings
# for i in range(50):
# 	insert_embedding(f'video_embeddings/2nd Batch Of Aid Reaches Gaza As Israeli Air Strikes Intensify  NPR News Now/embeddings/embedding{i}.npy', '2nd Batch Of Aid Reaches Gaza As Israeli Air Strikes Intensify  NPR News Now', i)

# insert video caption embeddings
caption_files = [f for f in os.listdir('captions') if f.endswith(".srt")]
for caption_file in caption_files:
    caption_file_path = os.path.join('captions', caption_file)
    insert_text_embedding(caption_file_path, caption_file)
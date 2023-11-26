import numpy as np
import psycopg2

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

for i in range(50):
	insert_embedding(f'video_embeddings/2nd Batch Of Aid Reaches Gaza As Israeli Air Strikes Intensify  NPR News Now/embeddings/embedding{i}.npy', '2nd Batch Of Aid Reaches Gaza As Israeli Air Strikes Intensify  NPR News Now', i)


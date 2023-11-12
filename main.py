import cv2
import os
from transformers import ViTFeatureExtractor, ViTModel
import torch
import numpy as np

title = '2nd Batch Of Aid Reaches Gaza As Israeli Air Strikes Intensify  NPR News Now'
video = cv2.VideoCapture(f'videos/{title}.3gpp')

base_dir = 'video_embeddings'
video_dir = os.path.join(base_dir, title)
frames_dir = os.path.join(video_dir, 'frames')
embeddings_dir = os.path.join(video_dir, 'embeddings')
os.makedirs(frames_dir, exist_ok=True)
os.makedirs(embeddings_dir, exist_ok=True)

# preprocess video
frame_count = 0
interval = 8
width = 200
height = 200
frames = []

while True:
	# decode frames
	success, frame = video.read()
	if not success:
		break
	frame_count += 1

	# sample every 8 frames
	if frame_count % interval == 0:
		# resize frames
		frame = cv2.resize(frame, (width, height))
		# normalize frames
		frame = frame / 255.0
		frames.append(frame)

video.release()
cv2.destroyAllWindows()

# embedding model (using ViT)
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224')
model = ViTModel.from_pretrained('google/vit-base-patch16-224')

def get_embedding(frame):
    inputs = feature_extractor(images=frame, do_rescale=False, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.last_hidden_state

embeddings = []

for i in range(len(frames)):
	print(f'Frame {i}/{len(frames)-1}')
	embeddings.append(get_embedding(frames[i]).detach().numpy())

for i in range(len(frames)):
	frame_path = os.path.join(frames_dir, f'frame{i}.jpg')
	embedding_path = os.path.join(embeddings_dir, f'embedding{i}.npy')

	cv2.imwrite(frame_path, frames[i] * 255) # rescale and save frame
	np.save(embedding_path, embeddings[i]) # save embedding

# combination with simple aggregation
mean_embedding = np.mean(embeddings, axis=0)

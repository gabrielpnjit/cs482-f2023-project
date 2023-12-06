import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

def project_embeddings(
    embeddings, num_projection_layers, projection_dims, dropout_rate
):
    projected_embeddings = layers.Dense(units=projection_dims)(embeddings)
    for _ in range(num_projection_layers):
        x = tf.nn.gelu(projected_embeddings)
        x = layers.Dense(projection_dims)(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Add()([projected_embeddings, x])
        projected_embeddings = layers.LayerNormalization()(x)
    return projected_embeddings


embed = np.load("video_embeddings/2nd Batch Of Aid Reaches Gaza As Israeli Air Strikes Intensify  NPR News Now/embeddings/embedding0.npy")
projected = project_embeddings(embed, 1, 250, 1)

print(projected.shape)
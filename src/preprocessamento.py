import tensorflow as tf

def preprocess_image(data, IMAGE_SIZE_H, IMAGE_SIZE_W):
    # Redimensionar e normalizar as imagens
    data = data.map(lambda x, y: (tf.image.resize(x, [IMAGE_SIZE_H, IMAGE_SIZE_W]) / 255, y))
    return data




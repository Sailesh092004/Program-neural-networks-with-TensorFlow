import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the pre-trained model
model = load_model('D:/venv/cats-dogs.keras')


# Function to classify an image
def classify_image(file_path):
    img = image.load_img(file_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = model.predict(images, batch_size=10)

    if classes[0] > 0.5:
        return "It's a dog!"
    else:
        return "It's a cat!"


# Provide the path to the image you want to classify
file_path = 'D:/venv/download (2).jpeg'  # Replace with the actual path
result = classify_image(file_path)
print(result)

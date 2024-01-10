import numpy as np
from keras.preprocessing import image
from tensorflow.keras.models import load_model

# Load the saved model
model = load_model('D:/venv/Women-horse.keras')

# Specify the path to your local image
image_path = 'D:/venv/download (1).jpeg'

# Predict the class of the image
img = image.load_img(image_path, target_size=(300, 300))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

images = np.vstack([x])
classes = model.predict(images, batch_size=10)
print(classes[0])

if classes[0] > 0.5:
    print("The image is predicted to be a human.")
else:
    print("The image is predicted to be a horse.")

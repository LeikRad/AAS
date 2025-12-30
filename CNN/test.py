from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

IMG_SIZE = 128

model = load_model("best_model.keras")

def predict_image(img_path):
    img = image.load_img(
        img_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="grayscale"
    )

    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)[0][0]

    if pred > 0.5:
        return "ANOMALIA"
    else:
        return "NORMAL"

print(predict_image("output.png"))

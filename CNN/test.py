from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import argparse
import os
import sys

IMG_SIZE = 256

def predict_image(model, img_path):
    img = image.load_img(
        img_path,
        target_size=(IMG_SIZE, IMG_SIZE),
        color_mode="rgb"
    )

    x = image.img_to_array(img)
    x = x / 255.0
    x = np.expand_dims(x, axis=0)

    pred = model.predict(x)[0][0]

    return "ANOMALIA" if pred > 0.5 else "NORMAL"


def main():
    parser = argparse.ArgumentParser(description="Predict if an image represents NORMAL or ANOMALIA")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to input image file")
    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: input file not found: {args.input}")
        sys.exit(1)

    try:
        model = load_model("best_model.keras")
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    result = predict_image(model, args.input)
    print(f"Prediction for {args.input}: {result}")


if __name__ == "__main__":
    main()

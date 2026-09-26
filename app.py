import tkinter as tk
from tkinter import messagebox
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw

# Load trained model
model = tf.keras.models.load_model("digit_model.keras")

# Window
root = tk.Tk()
root.title("Handwritten Digit Recognizer")
root.geometry("500x650")
root.resizable(False, False)

# Title
title = tk.Label(
    root,
    text="Handwritten Digit Recognizer",
    font=("Arial", 22, "bold")
)
title.pack(pady=15)

instruction = tk.Label(
    root,
    text="Draw a digit (0-9) below",
    font=("Arial", 14)
)
instruction.pack(pady=5)

# Canvas
canvas_size = 280

canvas = tk.Canvas(
    root,
    width=canvas_size,
    height=canvas_size,
    bg="black",
    cursor="cross"
)
canvas.pack(pady=15)

# Image used for prediction
image = Image.new("L", (canvas_size, canvas_size), "black")
draw = ImageDraw.Draw(image)


# Draw digit
def draw_digit(event):
    x = event.x
    y = event.y

    brush_size = 18

    canvas.create_oval(
        x - brush_size,
        y - brush_size,
        x + brush_size,
        y + brush_size,
        fill="white",
        outline="white"
    )

    draw.ellipse(
        [
            x - brush_size,
            y - brush_size,
            x + brush_size,
            y + brush_size
        ],
        fill="white"
    )


canvas.bind("<B1-Motion>", draw_digit)


# Predict digit
def predict_digit():

    # Resize to MNIST size
    img = image.resize((28, 28))

    # Convert to numpy
    img_array = np.array(img)

    # Normalize
    img_array = img_array.astype("float32") / 255.0

    # Add dimensions
    img_array = img_array.reshape(1, 28, 28, 1)

    # Prediction
    prediction = model.predict(img_array, verbose=0)

    digit = np.argmax(prediction)
    confidence = np.max(prediction) * 100

    result_label.config(
        text=f"Predicted Digit: {digit}\nConfidence: {confidence:.2f}%"
    )


# Clear canvas
def clear_canvas():

    canvas.delete("all")

    draw.rectangle(
        [0, 0, canvas_size, canvas_size],
        fill="black"
    )

    result_label.config(
        text="Draw a digit and click Predict"
    )


# Buttons
button_frame = tk.Frame(root)
button_frame.pack(pady=10)

predict_button = tk.Button(
    button_frame,
    text="Predict",
    command=predict_digit,
    font=("Arial", 14, "bold"),
    width=12
)
predict_button.grid(row=0, column=0, padx=10)

clear_button = tk.Button(
    button_frame,
    text="Clear",
    command=clear_canvas,
    font=("Arial", 14, "bold"),
    width=12
)
clear_button.grid(row=0, column=1, padx=10)


# Result
result_label = tk.Label(
    root,
    text="Draw a digit and click Predict",
    font=("Arial", 17, "bold")
)
result_label.pack(pady=20)


# Start application
root.mainloop()
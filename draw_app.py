import tkinter as tk
from tkinter import Canvas
from PIL import Image, ImageDraw
import numpy as np
import tensorflow as tf
from scipy.ndimage import center_of_mass

class DrawApp:
    def __init__(self, root):
        self.root = root
        self.root.title("MNIST Digit Recognizer")
        
        # Load the model trained by mnist_cnn.py
        try:
            self.model = tf.keras.models.load_model('mnist_model.keras')
            print("Model loaded successfully.")
        except Exception as e:
            print("Error loading model. Please run mnist_cnn.py first to train and save 'mnist_model.keras'.")
            self.model = None

        self.canvas_width = 280
        self.canvas_height = 280
        self.brush_size = 12  # Similar stroke thickness proportion to MNIST
        
        # UI Setup
        self.canvas = Canvas(self.root, width=self.canvas_width, height=self.canvas_height, bg='black', cursor='cross')
        self.canvas.pack(pady=10)
        
        self.canvas.bind("<B1-Motion>", self.paint)
        
        self.btn_frame = tk.Frame(self.root)
        self.btn_frame.pack(pady=5)
        
        self.btn_predict = tk.Button(self.btn_frame, text="Predict", command=self.predict, font=('Arial', 14))
        self.btn_predict.grid(row=0, column=0, padx=10)
        
        self.btn_clear = tk.Button(self.btn_frame, text="Clear", command=self.clear_canvas, font=('Arial', 14))
        self.btn_clear.grid(row=0, column=1, padx=10)
        
        self.lbl_result = tk.Label(self.root, text="", font=('Arial', 18))
        self.lbl_result.pack(pady=10)
        
        # Internal image for PIL to process
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)
        
    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_width, self.canvas_height), 0)
        self.draw = ImageDraw.Draw(self.image)
        self.lbl_result.config(text="")
        
    def paint(self, event):
        x1, y1 = (event.x - self.brush_size), (event.y - self.brush_size)
        x2, y2 = (event.x + self.brush_size), (event.y + self.brush_size)
        self.canvas.create_oval(x1, y1, x2, y2, fill="white", outline="white")
        self.draw.ellipse([x1, y1, x2, y2], fill="white", outline="white")
        
    def preprocess_image(self):
        # 1. Find bounding box of the drawn digit
        img_array = np.array(self.image)
        non_zero_coords = np.argwhere(img_array > 0)
        if len(non_zero_coords) == 0:
            return np.zeros((1, 28, 28, 1)) # Empty image
            
        y_min, x_min = non_zero_coords.min(axis=0)
        y_max, x_max = non_zero_coords.max(axis=0)
        
        # Crop the image to tightly bound the digit
        cropped = self.image.crop((x_min, y_min, x_max, y_max))
        
        # 2. Resize preserving aspect ratio into a 20x20 box (just like original MNIST preprocessing)
        width, height = cropped.size
        # Protect against div by zero or extreme scaling
        if width == 0 or height == 0:
             return np.zeros((1, 28, 28, 1))

        if width > height:
            new_w = 20
            new_h = int(20 * (height / width))
        else:
            new_h = 20
            new_w = int(20 * (width / height))
        
        new_w = max(1, new_w)
        new_h = max(1, new_h)
            
        resized = cropped.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        # 3. Paste the 20x20 image inside a 28x28 box using Center of Mass
        temp_obj = np.array(resized)
        
        # Calculate center of mass
        cy, cx = center_of_mass(temp_obj)
        if np.isnan(cy) or np.isnan(cx):
            cy, cx = new_h / 2.0, new_w / 2.0
            
        # We need the pixel center of mass to align perfectly with the 28x28 center (14, 14)
        start_x = int(round(14.0 - cx))
        start_y = int(round(14.0 - cy))
        
        # Ensure it fits
        start_x = max(0, min(28 - new_w, start_x))
        start_y = max(0, min(28 - new_h, start_y))
        
        final_img = Image.new("L", (28, 28), 0)
        final_img.paste(resized, (start_x, start_y))
        
        # 4. Normalize to [0, 1]
        final_array = np.array(final_img).astype('float32') / 255.0
        
        return final_array.reshape(1, 28, 28, 1)

    def predict(self):
        if self.model is None:
            self.lbl_result.config(text="Model not loaded! Run mnist_cnn.py first.")
            return
            
        processed = self.preprocess_image()
        
        # If the canvas was basically empty
        if np.max(processed) == 0:
            self.lbl_result.config(text="Please draw a digit first.")
            return
            
        # Get predictions
        pred = self.model.predict(processed)
        digit = np.argmax(pred[0])
        confidence = np.max(pred[0])
        
        self.lbl_result.config(text=f"Prediction: {digit}  |  Confidence: {confidence*100:.2f}%")

if __name__ == "__main__":
    root = tk.Tk()
    app = DrawApp(root)
    root.mainloop()

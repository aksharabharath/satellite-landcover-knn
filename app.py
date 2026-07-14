"""
app.py

Desktop application for the Satellite Land Cover
Classification project.

Author: Akshara Bharath
"""

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog

from PIL import Image
from PIL import ImageTk

import sys
from pathlib import Path

sys.path.append(
    str(Path(__file__).parent / "src")
)

from landcover import (
    load_model,
    predict_image,
    predict_probabilities,
)


class LandCoverApp:

    def __init__(self):

        self.model = load_model()

        self.selected_image = None

        self.photo = None

        self.class_info = {

            "AnnualCrop": {
                "icon": "🌾",
                "description": "Agricultural land used for seasonal crop cultivation."
            },

            "Forest": {
                "icon": "🌲",
                "description": "Dense tree-covered region important for biodiversity."
            },

            "HerbaceousVegetation": {
                "icon": "🌿",
                "description": "Natural grasslands and low vegetation."
            },

            "Highway": {
                "icon": "🛣️",
                "description": "Major transportation roads and highways."
            },

            "Industrial": {
                "icon": "🏭",
                "description": "Factories, warehouses, and industrial facilities."
            },

            "Pasture": {
                "icon": "🐄",
                "description": "Open grazing land used for livestock."
            },

            "PermanentCrop": {
                "icon": "🍇",
                "description": "Long-term cultivated crops such as orchards or vineyards."
            },

            "Residential": {
                "icon": "🏘️",
                "description": "Urban neighborhoods and residential buildings."
            },

            "River": {
                "icon": "🌊",
                "description": "Flowing freshwater bodies such as rivers and streams."
            },

            "SeaLake": {
                "icon": "🌅",
                "description": "Large natural water bodies including lakes and seas."
            },

        }

        self.root = tk.Tk()

        self.root.title(
            "Satellite Land Cover Classification"
        )

        self.root.geometry("1100x850")

        self.root.minsize(
            1000,
            800,
        )

        self.root.configure(
            bg="#f2f2f2"
        )

        self.create_widgets()

    def create_widgets(self):

        ####################################################
        # Title
        ####################################################

        self.title_frame = tk.Frame(
            self.root,
            bg="#f2f2f2",
        )

        self.title_frame.pack(
            pady=(25, 15)
        )

        self.title_label = tk.Label(

            self.title_frame,

            text="Satellite Land Cover Classification",

            font=("Arial", 24, "bold"),

            bg="#f2f2f2",

        )

        self.title_label.pack()

        self.subtitle_label = tk.Label(

            self.title_frame,

            text="Machine Learning Satellite Image Classifier",

            font=("Arial", 12),

            bg="#f2f2f2",

            fg="gray40",

        )

        self.subtitle_label.pack(
            pady=(5, 0)
        )

                ####################################################
        # Buttons
        ####################################################

        self.button_frame = tk.Frame(
            self.root,
            bg="#f2f2f2",
        )

        self.button_frame.pack(
            pady=10
        )

        self.upload_button = ttk.Button(

            self.button_frame,

            text="Upload Image",

            width=20,

            command=self.upload_image,

        )

        self.upload_button.pack()

        self.predict_button = ttk.Button(

            self.button_frame,

            text="Predict",

            width=20,

            command=self.predict,

        )

        # Hide initially
        # It will appear after an image is uploaded.

        ####################################################
        # Image Card
        ####################################################

        self.image_card = tk.LabelFrame(

            self.root,

            text="Image Preview",

            font=("Arial", 11, "bold"),

            padx=20,

            pady=20,

        )

        self.image_card.pack(

            padx=60,

            pady=20,

            fill="both",

        )

        self.image_label = tk.Label(

            self.image_card,

            text="No image selected",

            font=("Arial", 13),

            width=50,

            height=18,

        )

        self.image_label.pack()

        ####################################################
        # Prediction Card
        ####################################################

        self.prediction_card = tk.LabelFrame(

            self.root,

            text="Prediction",

            font=("Arial", 11, "bold"),

            padx=20,

            pady=20,

        )

        self.prediction_card.pack(

            padx=60,

            pady=10,

            fill="x",

        )

        self.icon_label = tk.Label(

            self.prediction_card,

            text="🛰️",

            font=("Arial", 42),

        )

        self.icon_label.pack()

        self.class_label = tk.Label(

            self.prediction_card,

            text="",

            font=("Arial", 24, "bold"),

        )

        self.class_label.pack(
            pady=(10, 5)
        )

        self.description_label = tk.Label(
            
            self.prediction_card,

            text="",

            font=("Arial", 12),

            justify="center",

            fg="gray35",

        )

        self.description_label.pack(
            pady=(5, 10)
        )

        self.confidence_label = tk.Label(

            self.prediction_card,

            text="",

            font=("Arial", 13),

            fg="gray40",

        )

        self.confidence_label.pack()

                ####################################################
        # Model Information
        ####################################################

        self.model_card = tk.LabelFrame(

            self.root,

            text="Model Information",

            font=("Arial", 11, "bold"),

            padx=20,

            pady=15,

        )

        self.model_card.pack(

            padx=60,

            pady=15,

            fill="x",

        )

        tk.Label(

            self.model_card,

            text="Algorithm: K-Nearest Neighbors",

            font=("Arial", 12),

            anchor="w",

        ).pack(anchor="w")

        tk.Label(

            self.model_card,

            text="Cross Validation Accuracy: 96.2%",

            font=("Arial", 12),

            anchor="w",

        ).pack(anchor="w")

        tk.Label(

            self.model_card,

            text="Features Used: 7",

            font=("Arial", 12),

            anchor="w",

        ).pack(anchor="w")

        tk.Label(

            self.model_card,

            text="Dataset: EuroSAT",

            font=("Arial", 12),

            anchor="w",

        ).pack(anchor="w")

    def upload_image(self):

        filename = filedialog.askopenfilename(

            title="Choose a Satellite Image",

            filetypes=[

                ("Image Files", "*.jpg *.jpeg *.png"),

            ],

        )

        if not filename:
            return

        self.selected_image = Image.open(
            filename
        ).convert("RGB")

        display_image = self.selected_image.copy()

        display_image.thumbnail(
            (650, 650)
        )

        self.photo = ImageTk.PhotoImage(
            display_image
        )

        self.image_label.configure(

            image=self.photo,

            text="",

        )

        self.reset_prediction()

        if not self.predict_button.winfo_ismapped():

            self.predict_button.pack(
                pady=12
            )

    def reset_prediction(self):

        self.icon_label.configure(
            text="🛰️"
        )

        self.class_label.configure(
            text=""
        )

        self.description_label.configure(
            text=""
        )

        self.confidence_label.configure(
            text=""
        )
    
    def predict(self):

        if self.selected_image is None:

            self.class_label.configure(
                text="Please upload an image."
            )

            return

        prediction = predict_image(

            self.selected_image,

            model=self.model,

        )

        probabilities = predict_probabilities(

            self.selected_image,

            model=self.model,

        )

        confidence = probabilities.iloc[0]["Probability"] * 100

        info = self.class_info[prediction]
        self.icon_label.configure(

            text=info["icon"],

            )


        self.class_label.configure(
            text=prediction
        )

        self.description_label.configure(
            text=info["description"]
    )

        self.confidence_label.configure(

            text=f"Confidence: {confidence:.1f}%"

        )

    ####################################################
    # Run Application
    ####################################################

    def run(self):

        self.root.mainloop()

if __name__ == "__main__":
    app = LandCoverApp()
    app.run()
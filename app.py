from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
import io
import os
import uuid
import subprocess
import sys
import json


# =========================================================
# APP
# =========================================================

app = FastAPI(title="FreshVision AI")


# =========================================================
# CORS
# =========================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =========================================================
# FOLDERS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

FRONTEND_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..", "frontend")
)

WEBSITE_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..")
)

MODEL_PATH = os.path.join(
    WEBSITE_DIR,
    "AI model",
    "fruit_vegetable_resnet18.pth"
)

CLASS_FILE = os.path.join(
    WEBSITE_DIR,
    "AI model",
    "classes.json"
)

TRAINING_DATA_DIR = os.path.join(
    WEBSITE_DIR,
    "training_data"
)

os.makedirs(
    TRAINING_DATA_DIR,
    exist_ok=True
)


# =========================================================
# ORIGINAL 36 CLASSES
# =========================================================

DEFAULT_CLASSES = [
    "apple",
    "banana",
    "beetroot",
    "bell pepper",
    "cabbage",
    "capsicum",
    "carrot",
    "cauliflower",
    "chilli pepper",
    "corn",
    "cucumber",
    "eggplant",
    "garlic",
    "ginger",
    "grapes",
    "jalepeno",
    "kiwi",
    "lemon",
    "lettuce",
    "mango",
    "onion",
    "orange",
    "paprika",
    "pear",
    "peas",
    "pineapple",
    "pomegranate",
    "potato",
    "raddish",
    "soy beans",
    "spinach",
    "sweetcorn",
    "sweetpotato",
    "tomato",
    "turnip",
    "watermelon"
]


# =========================================================
# LOAD CLASS LIST
# =========================================================

def load_class_names():

    if os.path.exists(CLASS_FILE):

        try:

            with open(
                CLASS_FILE,
                "r"
            ) as f:

                classes = json.load(f)

            if isinstance(classes, list) and len(classes) > 0:

                print(
                    "Loaded classes from classes.json:",
                    len(classes)
                )

                return classes

        except Exception as e:

            print(
                "Could not read classes.json:",
                e
            )

    print(
        "Using default 36 classes."
    )

    return DEFAULT_CLASSES.copy()


class_names = load_class_names()


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print(
    "Device:",
    device
)


# =========================================================
# LOAD MODEL
# =========================================================

def load_model():

    global model
    global class_names

    print(
        "\nLoading AI model..."
    )

    # Reload class list
    class_names = load_class_names()

    new_model = models.resnet18(
        weights=None
    )

    new_model.fc = nn.Linear(
        new_model.fc.in_features,
        len(class_names)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    new_model.load_state_dict(
        checkpoint
    )

    new_model = new_model.to(
        device
    )

    new_model.eval()

    model = new_model

    print(
        "AI model loaded successfully!"
    )

    print(
        "Model classes:",
        len(class_names)
    )


load_model()


# =========================================================
# IMAGE TRANSFORM
# =========================================================

transform = transforms.Compose([
    transforms.Resize(
        (224, 224)
    ),
    transforms.ToTensor()
])


# =========================================================
# HOME PAGE
# =========================================================

@app.get("/")
def home():

    return FileResponse(
        os.path.join(
            FRONTEND_DIR,
            "UI fruit vegetable.html"
        )
    )


# =========================================================
# CSS
# =========================================================

@app.get("/style.css")
def css():

    return FileResponse(
        os.path.join(
            FRONTEND_DIR,
            "UI vegetable fruit detection.css"
        ),
        media_type="text/css"
    )


# =========================================================
# JAVASCRIPT
# =========================================================

@app.get("/script.js")
def javascript():

    return FileResponse(
        os.path.join(
            FRONTEND_DIR,
            "UI fruitvegetable.js"
        ),
        media_type="application/javascript"
    )


# =========================================================
# AI PREDICTION
# =========================================================

@app.post("/predict")
async def predict(
    file: UploadFile = File(...)
):

    try:

        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        image_tensor = transform(
            image
        )

        image_tensor = image_tensor.unsqueeze(
            0
        )

        image_tensor = image_tensor.to(
            device
        )

        with torch.no_grad():

            output = model(
                image_tensor
            )

            probabilities = torch.softmax(
                output,
                dim=1
            )

            confidence, predicted = torch.max(
                probabilities,
                dim=1
            )

        confidence_value = confidence.item()

        predicted_index = predicted.item()

        if predicted_index >= len(class_names):

            return {
                "success": False,
                "error": "Invalid model class index."
            }

        predicted_class = class_names[
            predicted_index
        ]

        if confidence_value >= 0.70:

            result = predicted_class

        else:

            result = "Unknown"

        return {
            "success": True,
            "prediction": result,
            "confidence": round(
                confidence_value * 100,
                2
            )
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


# =========================================================
# ADD TRAINING IMAGE
# =========================================================

@app.post("/add-training-image")
async def add_training_image(
    file: UploadFile = File(...),
    class_name: str = Form(...)
):

    try:

        # Clean class name
        class_name = class_name.strip().lower()

        if not class_name:

            return {
                "success": False,
                "error": "Class name is required."
            }

        # Allow new classes
        class_name = "".join(
            c
            for c in class_name
            if c.isalnum()
            or c in (" ", "_", "-")
        ).strip()

        if not class_name:

            return {
                "success": False,
                "error": "Invalid class name."
            }

        # Create class folder
        class_folder = os.path.join(
            TRAINING_DATA_DIR,
            class_name
        )

        os.makedirs(
            class_folder,
            exist_ok=True
        )

        # Read image
        contents = await file.read()

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

        # Unique filename
        unique_id = uuid.uuid4().hex[:10]

        filename = (
            f"user_{unique_id}.jpg"
        )

        file_path = os.path.join(
            class_folder,
            filename
        )

        # Save image
        image.save(
            file_path,
            "JPEG"
        )

        print(
            "Training image saved:",
            file_path
        )

        return {
            "success": True,
            "message": "Training image added successfully!",
            "class_name": class_name,
            "filename": filename
        }

    except Exception as e:

        return {
            "success": False,
            "error": str(e)
        }


# =========================================================
# TRAIN MODEL
# =========================================================

@app.post("/train-model")
async def train_model():

    try:

        train_script = os.path.join(
            BASE_DIR,
            "train_model.py"
        )

        if not os.path.exists(
            train_script
        ):

            return {
                "success": False,
                "message": "Training script not found."
            }

        print(
            "\n================================"
        )

        print(
            "Starting model training..."
        )

        print(
            "================================"
        )

        result = subprocess.run(
            [
                sys.executable,
                train_script
            ],
            capture_output=True,
            text=True
        )

        # Training failed
        if result.returncode != 0:

            print(
                "Training failed!"
            )

            print(
                result.stderr
            )

            return {
                "success": False,
                "message": "Training failed.",
                "error": result.stderr
            }

        print(
            "Training completed!"
        )

        print(
            result.stdout
        )

        # =================================================
        # RELOAD UPDATED MODEL
        # =================================================

        print(
            "\nReloading updated AI model..."
        )

        load_model()

        print(
            "Updated model loaded!"
        )

        return {
            "success": True,
            "message": "Model trained and updated successfully!",
            "classes": len(class_names),
            "output": result.stdout
        }

    except Exception as e:

        return {
            "success": False,
            "message": "Training error.",
            "error": str(e)
        }
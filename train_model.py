import os
import json
import shutil
import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

# Website folder
WEBSITE_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..")
)

# Main project folder
PROJECT_DIR = os.path.abspath(
    os.path.join(WEBSITE_DIR, "..")
)

# Original dataset
ORIGINAL_DATASET = os.path.join(
    PROJECT_DIR,
    "train"
)

# User training images
USER_DATASET = os.path.join(
    WEBSITE_DIR,
    "training_data"
)

# AI model
MODEL_PATH = os.path.join(
    WEBSITE_DIR,
    "AI model",
    "fruit_vegetable_resnet18.pth"
)

# Class list
CLASS_FILE = os.path.join(
    WEBSITE_DIR,
    "AI model",
    "classes.json"
)

# Combined dataset
COMBINED_DATASET = os.path.join(
    BASE_DIR,
    "combined_training_data"
)

os.makedirs(
    USER_DATASET,
    exist_ok=True
)


# =========================================================
# DEVICE
# =========================================================

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)

print("================================")
print("FreshVision AI Training")
print("================================")
print("Device:", device)


# =========================================================
# CHECK DATASET
# =========================================================

if not os.path.exists(
    ORIGINAL_DATASET
):

    print(
        "\nERROR: Original training dataset not found!"
    )

    print(
        "Expected:",
        ORIGINAL_DATASET
    )

    exit()


if not os.path.exists(
    USER_DATASET
):

    os.makedirs(
        USER_DATASET
    )


# =========================================================
# COPY DATASETS
# =========================================================

print(
    "\nPreparing combined dataset..."
)

if os.path.exists(
    COMBINED_DATASET
):

    shutil.rmtree(
        COMBINED_DATASET
    )

os.makedirs(
    COMBINED_DATASET
)


def copy_dataset(source):

    if not os.path.exists(source):

        return

    for class_name in os.listdir(source):

        class_path = os.path.join(
            source,
            class_name
        )

        if not os.path.isdir(
            class_path
        ):

            continue

        destination = os.path.join(
            COMBINED_DATASET,
            class_name
        )

        os.makedirs(
            destination,
            exist_ok=True
        )

        for filename in os.listdir(
            class_path
        ):

            source_file = os.path.join(
                class_path,
                filename
            )

            destination_file = os.path.join(
                destination,
                filename
            )

            if os.path.isfile(
                source_file
            ):

                shutil.copy2(
                    source_file,
                    destination_file
                )


# Original 36 classes
print(
    "Copying original dataset..."
)

copy_dataset(
    ORIGINAL_DATASET
)


# User-added classes
print(
    "Copying user training data..."
)

copy_dataset(
    USER_DATASET
)


# =========================================================
# TRANSFORMS
# =========================================================

transform = transforms.Compose([

    transforms.Resize(
        (224, 224)
    ),

    transforms.RandomHorizontalFlip(),

    transforms.RandomRotation(
        10
    ),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.ToTensor()
])


# =========================================================
# LOAD DATASET
# =========================================================

dataset = datasets.ImageFolder(
    COMBINED_DATASET,
    transform=transform
)

class_names = dataset.classes

num_classes = len(
    class_names
)

print(
    "\nTotal classes:",
    num_classes
)

print(
    "Total images:",
    len(dataset)
)

print(
    "\nClass list:"
)

for i, name in enumerate(
    class_names
):

    print(
        i,
        "->",
        name
    )


# =========================================================
# LOAD OLD MODEL
# =========================================================

old_checkpoint = None
old_class_names = []


if os.path.exists(
    MODEL_PATH
):

    print(
        "\nExisting model found."
    )

    try:

        old_checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu"
        )

        old_fc_weight = old_checkpoint[
            "fc.weight"
        ]

        old_num_classes = (
            old_fc_weight.shape[0]
        )

        print(
            "Old model classes:",
            old_num_classes
        )

        # Try to get previous class names
        if os.path.exists(
            CLASS_FILE
        ):

            with open(
                CLASS_FILE,
                "r"
            ) as f:

                old_class_names = json.load(f)

            print(
                "Previous class list loaded."
            )

        else:

            # Original 36 classes
            old_class_names = [
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

    except Exception as e:

        print(
            "Could not load old model."
        )

        print(
            "Reason:",
            e
        )

        old_checkpoint = None


# =========================================================
# CREATE NEW RESNET18
# =========================================================

print(
    "\nCreating ResNet18..."
)

model = models.resnet18(
    weights=None
)


# =========================================================
# NEW OUTPUT LAYER
# =========================================================

model.fc = nn.Linear(
    model.fc.in_features,
    num_classes
)


# =========================================================
# PRESERVE OLD MODEL WEIGHTS
# =========================================================

if old_checkpoint is not None:

    print(
        "\nPreserving existing model knowledge..."
    )

    new_state_dict = model.state_dict()

    # Copy backbone weights
    for key in old_checkpoint:

        if key.startswith("fc."):

            continue

        if key in new_state_dict:

            if (
                new_state_dict[key].shape
                ==
                old_checkpoint[key].shape
            ):

                new_state_dict[key] = (
                    old_checkpoint[key]
                )

    # Copy FC weights according to class name
    old_weight = old_checkpoint[
        "fc.weight"
    ]

    old_bias = old_checkpoint[
        "fc.bias"
    ]

    new_weight = model.fc.weight.data
    new_bias = model.fc.bias.data

    preserved_count = 0

    for old_index, old_name in enumerate(
        old_class_names
    ):

        if old_name in class_names:

            new_index = class_names.index(
                old_name
            )

            new_weight[new_index] = (
                old_weight[old_index]
            )

            new_bias[new_index] = (
                old_bias[old_index]
            )

            preserved_count += 1

    print(
        "Preserved classes:",
        preserved_count
    )

    model.load_state_dict(
        new_state_dict
    )

else:

    print(
        "\nNo previous model found."
    )

    print(
        "Training new model from scratch."
    )


model = model.to(
    device
)


# =========================================================
# SAVE CLASS LIST
# =========================================================

with open(
    CLASS_FILE,
    "w"
) as f:

    json.dump(
        class_names,
        f,
        indent=4
    )

print(
    "\nClass list saved:"
)

print(
    CLASS_FILE
)


# =========================================================
# DATALOADER
# =========================================================

dataloader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0
)


# =========================================================
# LOSS
# =========================================================

criterion = nn.CrossEntropyLoss()


# =========================================================
# OPTIMIZER
# =========================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.0001
)


# =========================================================
# TRAINING
# =========================================================

EPOCHS = 10

print(
    "\n================================"
)

print(
    "TRAINING STARTED"
)

print(
    "================================"
)


model.train()


for epoch in range(
    EPOCHS
):

    running_loss = 0.0

    correct = 0

    total = 0

    for images, labels in dataloader:

        images = images.to(
            device
        )

        labels = labels.to(
            device
        )

        # Clear gradients
        optimizer.zero_grad()

        # Forward
        outputs = model(
            images
        )

        # Loss
        loss = criterion(
            outputs,
            labels
        )

        # Backward
        loss.backward()

        # Update
        optimizer.step()

        running_loss += (
            loss.item()
        )

        _, predicted = torch.max(
            outputs,
            1
        )

        total += (
            labels.size(0)
        )

        correct += (
            predicted == labels
        ).sum().item()

    accuracy = (
        100 * correct / total
    )

    average_loss = (
        running_loss /
        len(dataloader)
    )

    print(
        f"Epoch [{epoch + 1}/{EPOCHS}] "
        f"Loss: {average_loss:.4f} "
        f"Accuracy: {accuracy:.2f}%"
    )


# =========================================================
# SAVE MODEL
# =========================================================

print(
    "\nSaving updated model..."
)

torch.save(
    model.state_dict(),
    MODEL_PATH
)


# =========================================================
# COMPLETE
# =========================================================

print(
    "\n================================"
)

print(
    "TRAINING COMPLETED"
)

print(
    "================================"
)

print(
    "Model saved:"
)

print(
    MODEL_PATH
)

print(
    "\nTotal classes:",
    num_classes
)

print(
    "Total images:",
    len(dataset)
)

print(
    "\nDone!"
)
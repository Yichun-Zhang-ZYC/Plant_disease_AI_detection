# Data

## Data Preprocess
Collected 20638 images across 15 categories.

```python 
import kagglehub

# Download latest version
path = kagglehub.dataset_download("emmarex/plantdisease")

print("Path to dataset files:", path)

import os

# Replace 'path' with the actual variable where your dataset path is stored
dataset_path = path  # This is the path variable from your kagglehub download

# List top-level files and folders
for root, dirs, files in os.walk(dataset_path):
    print(f"Directory: {root}")
    for name in dirs:
        print(f"Folder: {name}")
    for name in files:
        print(f"File: {name}")
    # Break after the first directory level to avoid too much output
    break
# Path to the PlantVillage folder
plant_village_path = os.path.join(dataset_path, "PlantVillage")

# List all folders inside PlantVillage
for root, dirs, files in os.walk(plant_village_path):
    print(f"Directory: {root}")
    for name in dirs:
        print(f"Folder: {name}")
    # Print a sample of files in each folder
    for name in files[:5]:  # Show first 5 files as a sample
        print(f"File: {name}")
    # Break after the first directory level to avoid too much output
    break


# Initialize lists to store image paths and labels
image_paths = []
labels = []

# Loop through each category folder in PlantVillage
for category_folder in os.listdir(plant_village_path):
    category_path = os.path.join(plant_village_path, category_folder)
    if os.path.isdir(category_path):  # Ensure it's a directory
        for image_file in os.listdir(category_path):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Image file extensions
                image_paths.append(os.path.join(category_path, image_file))
                labels.append(category_folder)  # Folder name as label (y)

# Verify the data collection
print(f"Collected {len(image_paths)} images across {len(set(labels))} categories.")
print("Sample labels and paths:")
for i in range(10):  # Show 10 samples for verification
    print(f"Label: {labels[i]}, Path: {image_paths[i]}")


# Mapping from original folder names to better, more readable category names
label_map = {
    "Tomato_Early_blight": "Tomato Early Blight",
    "Pepper__bell___Bacterial_spot": "Pepper Bell Bacterial Spot",
    "Pepper__bell___healthy": "Pepper Bell Healthy",
    "Potato___Early_blight": "Potato Early Blight",
    "Potato___healthy": "Potato Healthy",
    "Potato___Late_blight": "Potato Late Blight",
    "Tomato_Bacterial_spot": "Tomato Bacterial Spot",
    "Tomato_Leaf_Mold": "Tomato Leaf Mold",
    "Tomato_Septoria_leaf_spot": "Tomato Septoria Leaf Spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato Spider Mites",
    "Tomato__Target_Spot": "Tomato Target Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato Yellow Leaf Curl Virus",
    "Tomato__Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Tomato_healthy": "Tomato Healthy",  # Fixed mapping
    "Tomato_Late_blight": "Tomato Late Blight"
}

# Initialize lists to store image paths and mapped labels
image_paths = []
labels = []

# Loop through each category folder in PlantVillage and apply label mapping
for category_folder in os.listdir(plant_village_path):
    category_path = os.path.join(plant_village_path, category_folder)
    if os.path.isdir(category_path):  # Ensure it's a directory
        # Use mapped label or default to original folder name if not in label_map
        label = label_map.get(category_folder, category_folder)
        for image_file in os.listdir(category_path):
            if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Image file extensions
                image_paths.append(os.path.join(category_path, image_file))
                labels.append(label)  # Mapped label as y

# Verify the data collection with mapped labels
print(f"Collected {len(image_paths)} images across {len(set(labels))} categories.")
print("Sample labels and paths:")
for i in range(10):  # Show 10 samples for verification
    print(f"Label: {labels[i]}, Path: {image_paths[i]}")
```


##  Split Test and Train
Training set: 16510 images
Testing set: 4128 images

```python 
from sklearn.model_selection import train_test_split

# Train-test split (80% train, 20% test)
train_paths, test_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, stratify=labels, random_state=42
)

print(f"Training set: {len(train_paths)} images")
print(f"Testing set: {len(test_paths)} images")

```

# ViT
## Data Loader 
```python 

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Dataset class
class PlantDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = categories.index(self.labels[idx])  # Convert label to index
        if self.transform:
            image = self.transform(image)
        return image, label

# Create DataLoaders
train_dataset = PlantDataset(train_paths, train_labels, transform=transform)
test_dataset = PlantDataset(test_paths, test_labels, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

```


## Load model
```python 
from transformers import ViTForImageClassification
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import get_scheduler
from tqdm import tqdm


categories = list(label_map.values())  # Extract the unique category names from label_map


# Load the pre-trained ViT model
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224-in21k",
    num_labels=len(categories),  # Number of categories
    id2label={i: label for i, label in enumerate(categories)},  # Index-to-label mapping
    label2id={label: i for i, label in enumerate(categories)}   # Label-to-index mapping
)

# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

```
## Original Image Set: Train

```python 

# Define optimizer, loss function, and scheduler
optimizer = AdamW(model.parameters(), lr=5e-5)
criterion = CrossEntropyLoss()
num_training_steps = len(train_loader) * 5
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)

# Training loop
num_epochs = 4
model.train()

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, leave=True)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs.logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

    print(f"Epoch {epoch + 1} completed. Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

```

## Training result 

- Epoch 1/4
Epoch 1: 100%|██████████| 516/516 [02:37<00:00,  3.28it/s, accuracy=93.7, loss=0.161]
Epoch 1 completed. Loss: 0.5581, Accuracy: 93.71%

- Epoch 2/4
Epoch 2: 100%|██████████| 516/516 [02:35<00:00,  3.31it/s, accuracy=99.5, loss=0.0683]
Epoch 2 completed. Loss: 0.0826, Accuracy: 99.47%

- Epoch 3/4
Epoch 3: 100%|██████████| 516/516 [02:35<00:00,  3.33it/s, accuracy=99.9, loss=0.0261]
Epoch 3 completed. Loss: 0.0356, Accuracy: 99.88%

- Epoch 4/4
Epoch 4: 100%|██████████| 516/516 [02:35<00:00,  3.32it/s, accuracy=100, loss=0.0178]Epoch 4 completed. Loss: 0.0214, Accuracy: 99.99%

## Test Accuracy 

```python 

from transformers import ViTForImageClassification
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import get_scheduler
from tqdm import tqdm
import time


categories = list(label_map.values())
# 🟢 Test Accuracy Calculation
model.eval()  # Turn off dropout, batch norm, etc.
correct = 0  # Number of correct predictions
total = 0  # Total number of samples

# Disable gradient computation (for faster performance)
with torch.no_grad():
    for images, labels in test_loader:  # Loop through the test DataLoader
        # Move images and labels to the GPU (if available)
        images, labels = images.to(device), labels.to(device)

        # Forward pass through the model
        outputs = model(images)  # Handles both ResNet and ViT cases (ResNet: outputs, ViT: outputs.logits)

        # Get predictions (class with the maximum score)
        _, predicted = torch.max(outputs.logits if hasattr(outputs, 'logits') else outputs, 1)  # Handles ViT and ResNet

        # Count total samples and correct predictions
        total += labels.size(0)  # Total number of images in the batch
        correct += (predicted == labels).sum().item()  # Count the number of correct predictions

# Calculate accuracy as a percentage
test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")


# 🟢 Inference Time Calculation
model.eval()
inference_times = []

# Disable gradient computation (for faster performance)
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Calculating Inference Time"):
        # Move images and labels to the GPU (if available)
        images, labels = images.to(device), labels.to(device)

        # Measure inference time
        start_time = time.time()
        outputs = model(images)  # Handles both ResNet and ViT cases (ResNet: outputs, ViT: outputs.logits)
        inference_time = time.time() - start_time
        inference_times.append(inference_time / images.size(0))  # Time per image

# Calculate average inference time
avg_inference_time = sum(inference_times) / len(inference_times)
print(f"Average Inference Time: {avg_inference_time * 1000:.2f} ms/image")


# 🟢 Robustness Test Calculation
model.eval()
correct = 0  # Number of correct predictions
total = 0  # Total number of samples
noise_level = 0.1

# Disable gradient computation (for faster performance)
with torch.no_grad():
    for images, labels in tqdm(test_loader, desc="Testing Robustness"):
        # Add random noise to the images
        noise = torch.randn_like(images) * noise_level  # Create random noise
        noisy_images = torch.clamp(images + noise, 0, 1)  # Clamp pixel values to [0, 1]

        # Move images and labels to the GPU (if available)
        noisy_images, labels = noisy_images.to(device), labels.to(device)

        # Forward pass with noisy images
        outputs = model(noisy_images)  # Handles both ResNet and ViT cases (ResNet: outputs, ViT: outputs.logits)

        # Get predictions (class with the maximum score)
        _, predicted = torch.max(outputs.logits if hasattr(outputs, 'logits') else outputs, 1)  # Handles ViT and ResNet

        # Count total samples and correct predictions
        total += labels.size(0)  # Total number of images in the batch
        correct += (predicted == labels).sum().item()  # Count the number of correct predictions

# Calculate robustness accuracy as a percentage
robustness_accuracy = 100 * correct / total
print(f"Robustness Test Accuracy with {noise_level * 100}% noise: {robustness_accuracy:.2f}%")


```
- Test Accuracy: 99.95%
- Calculating Inference Time: 100%|██████████| 129/129 [10:55<00:00,  5.08s/it]
- Average Inference Time: 157.12 ms/image
- Testing Robustness: 100%|██████████| 129/129 [11:06<00:00,  5.17s/it]- Robustness Test Accuracy with 10.0% noise: 70.01%


## We SAM, and loaded SAMed data
```python 

from google.colab import drive
import os

# Mount Google Drive
drive.mount('/content/drive')

# Set the path to the folder containing segmented images
segmented_images_path = "/content/drive/My Drive/ViT-plant/seg_images"


```
### We also preprocessed them using the same way as before 

```python 

import os

# Define the label mapping again
label_map = {
    "segmented_Tomato_Early_blight": "Tomato Early Blight",
    "segmented_Pepper__bell___Bacterial_spot": "Pepper Bell Bacterial Spot",
    "segmented_Pepper__bell___healthy": "Pepper Bell Healthy",
    "segmented_Potato___Early_blight": "Potato Early Blight",
    "segmented_Potato___healthy": "Potato Healthy",
    "segmented_Potato___Late_blight": "Potato Late Blight",
    "segmented_Tomato_Bacterial_spot": "Tomato Bacterial Spot",
    "segmented_Tomato_Leaf_Mold": "Tomato Leaf Mold",
    "segmented_Tomato_Septoria_leaf_spot": "Tomato Septoria Leaf Spot",
    "segmented_Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato Spider Mites",
    "segmented_Tomato__Target_Spot": "Tomato Target Spot",
    "segmented_Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato Yellow Leaf Curl Virus",
    "segmented_Tomato__Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "segmented_Tomato_healthy": "Tomato Healthy",
    "segmented_Tomato_Late_blight": "Tomato Late Blight"
}

# Path to the folder where all images are combined into a single folder
segmented_images_path = "/content/drive/My Drive/ViT-plant/seg_images"  # Update with your actual path

# Prepare the dataset
segmented_image_paths = []
segmented_labels = []

# Iterate through all images in the folder
for image_file in os.listdir(segmented_images_path):
    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):  # Check valid image file extensions
        image_path = os.path.join(segmented_images_path, image_file)

        # Extract the category key (remove last underscore and number)
        category_key = '_'.join(image_file.split('_')[:-1])  # Extract everything before the last underscore
        label = label_map.get(category_key, "Unknown")  # Map to human-readable label using label_map

        # Log error if category is not in label_map
        if label == "Unknown":
            print(f"Warning: Category '{category_key}' not found in label_map for image '{image_file}'")

        segmented_image_paths.append(image_path)
        segmented_labels.append(label)

# Verify the dataset
print(f"Collected {len(segmented_image_paths)} segmented images across {len(set(segmented_labels))} categories.")
print("Sample labels and paths:")
for i in range(5):
    print(f"Label: {segmented_labels[i]}, Path: {segmented_image_paths[i]}")


```
### match 
```python 

from collections import Counter

# Count the number of images for each label
label_counts = Counter(segmented_labels)

# Print the counts for each label
print("\nNumber of images in each label:")
for label, count in label_counts.items():
    print(f"Label: {label}, Count: {count}")

# Calculate the total number of items across all labels
total_items = sum(Counter(segmented_labels).values())

# Print the total number of items
print(f"\nTotal number of items across all labels: {total_items}")

# 🟢 Step 3: Filter the SAM-segmented dataset to create the SAM test set
sam_test_paths = []
sam_test_labels = []

for path, label in zip(segmented_image_paths, segmented_labels):  # `segmented_image_paths` contains SAM dataset paths
    sam_identifier = os.path.basename(path).rsplit('.', 1)[0].replace('segmented_', '')
  # Extract the identifier
    if sam_identifier in original_test_identifiers:  # Check if the identifier exists in the original test set
        sam_test_paths.append(path)
        sam_test_labels.append(label)

print(f"Number of images in the SAM test set: {len(sam_test_paths)}")

# 🟢 Step 4: Create the SAM training set (exclude the SAM test set)
sam_train_paths = [path for path in segmented_image_paths if path not in sam_test_paths]
sam_train_labels = [label for path, label in zip(segmented_image_paths, segmented_labels) if path not in sam_test_paths]

print(f"Number of images in the SAM training set: {len(sam_train_paths)}")

# 🟢 Step 5: Verify the SAM test dataset by listing the first 10 samples
print("\nSample SAM test set labels and paths (first 10):")
for i in range(min(10, len(sam_test_paths))):  # Ensure we don't exceed the dataset size
    print(f"{i+1}. Label: {sam_test_labels[i]}, Path: {sam_test_paths[i]}")



```

## Next, we loaded the previous model we trained with SAM.  

### Prepare stage 
```python 

# 🟢 Step 3: Filter the SAM-segmented dataset to create the SAM test set
sam_test_paths = []
sam_test_labels = []

for path, label in zip(segmented_image_paths, segmented_labels):  # `segmented_image_paths` contains SAM dataset paths
    sam_identifier = os.path.basename(path).rsplit('.', 1)[0].replace('segmented_', '')
  # Extract the identifier
    if sam_identifier in original_test_identifiers:  # Check if the identifier exists in the original test set
        sam_test_paths.append(path)
        sam_test_labels.append(label)

print(f"Number of images in the SAM test set: {len(sam_test_paths)}")

# 🟢 Step 4: Create the SAM training set (exclude the SAM test set)
sam_train_paths = [path for path in segmented_image_paths if path not in sam_test_paths]
sam_train_labels = [label for path, label in zip(segmented_image_paths, segmented_labels) if path not in sam_test_paths]

print(f"Number of images in the SAM training set: {len(sam_train_paths)}")

# 🟢 Step 5: Verify the SAM test dataset by listing the first 10 samples
print("\nSample SAM test set labels and paths (first 10):")
for i in range(min(10, len(sam_test_paths))):  # Ensure we don't exceed the dataset size
    print(f"{i+1}. Label: {sam_test_labels[i]}, Path: {sam_test_paths[i]}")


```

## Load the ViT we fine-turned last time (used original 20k) and continue training on new SAM data + flip, rotate, blur

```python 

from google.colab import drive
from transformers import ViTForImageClassification, ViTImageProcessor
import torch

# Mount Google Drive
drive.mount('/content/drive')

# Define the path to the saved model on Google Drive
model_path = '/content/drive/MyDrive/ViT-plant/fine_tuned_vit'

# Load the processor and model
processor = ViTImageProcessor.from_pretrained(model_path)
Vit_finetuned_model = ViTForImageClassification.from_pretrained(model_path)

print("Model and processor loaded successfully!")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Vit_finetuned_model.to(device)

```

## Train 
```python 
from transformers import ViTForImageClassification
import torch
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from transformers import get_scheduler
from tqdm import tqdm

categories = list(label_map.values())

# ------------------------------------------
# 🟢 Step 2: Define Loss, Optimizer, and Scheduler
# ------------------------------------------
# Define optimizer, loss function, and scheduler
optimizer = AdamW(Vit_finetuned_model.parameters(), lr=5e-5)
criterion = CrossEntropyLoss()
num_training_steps = len(train_loader_sam) * 5
scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)



# Training loop
num_epochs = 5
Vit_finetuned_model.train()

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader_sam, leave=True)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = Vit_finetuned_model(images)
        loss = criterion(outputs.logits, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.logits, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Update progress bar
        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

    print(f"Epoch {epoch + 1} completed. Loss: {running_loss / len(train_loader_sam):.4f}, Accuracy: {100 * correct / total:.2f}%")

```
- Epoch 1/5
Epoch 1: 100%|██████████| 252/252 [17:04<00:00,  4.07s/it, accuracy=97.9, loss=0.0539]
Epoch 1 completed. Loss: 0.0806, Accuracy: 97.89%

- Epoch 2/5
Epoch 2: 100%|██████████| 252/252 [01:51<00:00,  2.26it/s, accuracy=99.1, loss=0.0621]
Epoch 2 completed. Loss: 0.0370, Accuracy: 99.08%
- Epoch 3/5
Epoch 3: 100%|██████████| 252/252 [01:51<00:00,  2.25it/s, accuracy=99.4, loss=0.0285]
Epoch 3 completed. Loss: 0.0221, Accuracy: 99.45%

- Epoch 4/5
Epoch 4: 100%|██████████| 252/252 [01:51<00:00,  2.25it/s, accuracy=99.7, loss=0.00547]
Epoch 4 completed. Loss: 0.0132, Accuracy: 99.74%

- Epoch 5/5
Epoch 5: 100%|██████████| 252/252 [01:51<00:00,  2.25it/s, accuracy=99.9, loss=0.00511]Epoch 5 completed. Loss: 0.0085, Accuracy: 99.88%

## Test result 

```python 
import time

categories = list(label_map.values())
# 🟢 Test Accuracy Calculation
Vit_finetuned_model.eval()  # Turn off dropout, batch norm, etc.
correct = 0  # Number of correct predictions
total = 0  # Total number of samples

# Disable gradient computation (for faster performance)
with torch.no_grad():
    for images, labels in test_loader_sam:  # Loop through the test DataLoader
        # Move images and labels to the GPU (if available)
        images, labels = images.to(device), labels.to(device)

        # Forward pass through the model
        outputs = Vit_finetuned_model(images)  # Handles both ResNet and ViT cases (ResNet: outputs, ViT: outputs.logits)

        # Get predictions (class with the maximum score)
        _, predicted = torch.max(outputs.logits if hasattr(outputs, 'logits') else outputs, 1)  # Handles ViT and ResNet

        # Count total samples and correct predictions
        total += labels.size(0)  # Total number of images in the batch
        correct += (predicted == labels).sum().item()  # Count the number of correct predictions

# Calculate accuracy as a percentage
test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")


# 🟢 Inference Time Calculation
Vit_finetuned_model.eval()
inference_times = []

# Disable gradient computation (for faster performance)
with torch.no_grad():
    for images, labels in tqdm(test_loader_sam, desc="Calculating Inference Time"):
        # Move images and labels to the GPU (if available)
        images, labels = images.to(device), labels.to(device)

        # Measure inference time
        start_time = time.time()
        outputs = Vit_finetuned_model(images)  # Handles both ResNet and ViT cases (ResNet: outputs, ViT: outputs.logits)
        inference_time = time.time() - start_time
        inference_times.append(inference_time / images.size(0))  # Time per image

# Calculate average inference time
avg_inference_time = sum(inference_times) / len(inference_times)
print(f"Average Inference Time: {avg_inference_time * 1000:.2f} ms/image")


# 🟢 Robustness Test Calculation
Vit_finetuned_model.eval()
correct = 0  # Number of correct predictions
total = 0  # Total number of samples
noise_level = 0.1

# Disable gradient computation (for faster performance)
with torch.no_grad():
    for images, labels in tqdm(test_loader_sam, desc="Testing Robustness"):
        # Add random noise to the images
        noise = torch.randn_like(images) * noise_level  # Create random noise
        noisy_images = torch.clamp(images + noise, 0, 1)  # Clamp pixel values to [0, 1]

        # Move images and labels to the GPU (if available)
        noisy_images, labels = noisy_images.to(device), labels.to(device)

        # Forward pass with noisy images
        outputs = Vit_finetuned_model(noisy_images)  # Handles both ResNet and ViT cases (ResNet: outputs, ViT: outputs.logits)

        # Get predictions (class with the maximum score)
        _, predicted = torch.max(outputs.logits if hasattr(outputs, 'logits') else outputs, 1)  # Handles ViT and ResNet

        # Count total samples and correct predictions
        total += labels.size(0)  # Total number of images in the batch
        correct += (predicted == labels).sum().item()  # Count the number of correct predictions

# Calculate robustness accuracy as a percentage
robustness_accuracy = 100 * correct / total
print(f"Robustness Test Accuracy with {noise_level * 100}% noise: {robustness_accuracy:.2f}%")
```

- Test Accuracy: 99.55%
- Calculating Inference Time: 100%|██████████| 63/63 [00:13<00:00,  4.69it/s]
- Average Inference Time: 0.16 ms/image
- Testing Robustness: 100%|██████████| 63/63 [00:16<00:00,  3.91it/s]Robustness Test Accuracy with 10.0% noise: 46.58%

## Evaluation
```python
## Evaluate our trained model 

from transformers import AutoModelForImageClassification

# 🟢 Step 2: Path to the saved model directory
drive_save_path = '/content/drive/MyDrive/ViT-plant/SAM_fine_tuned_vit'

# 🟢 Step 3: Load the fine-tuned model from the saved directory
SAM_Vit_finetuned_model = AutoModelForImageClassification.from_pretrained(drive_save_path)

print(f"✅ Successfully loaded the fine-tuned ViT model from: {drive_save_path}")

# 🟢 (Optional) Print model architecture
print(SAM_Vit_finetuned_model)

```
Evaluating Model: 100%|██████████| 63/63 [04:14<00:00,  4.04s/it]
Precision (Macro): 0.9982
Recall (Macro): 0.9991
F1-Score (Macro): 0.9986
Accuracy: 0.9988
Top-1 Accuracy: 99.88%
Top-3 Accuracy: 99.98%
Top-5 Accuracy: 100.00%
ROC-AUC (OVR): 0.9999

看图片1， 图片2

```python 
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import numpy as np

# Assuming `all_labels` contains the true labels and `all_probabilities` contains probabilities for each class.
# Example:
# all_labels = [0, 1, 2, ...]  # Ground truth for each sample
# all_probabilities = [[0.1, 0.7, 0.2], [0.6, 0.3, 0.1], ...]  # Model probabilities for each class

# Number of classes in the dataset
num_classes = all_probabilities.shape[1]

# Plot ROC curve for each class
plt.figure(figsize=(10, 8))
for class_idx in range(num_classes):
    # Get true binary labels for the current class (One-vs-Rest approach)
    binary_labels = (all_labels == class_idx).astype(int)
    # Get probabilities for the current class
    class_probabilities = all_probabilities[:, class_idx]
    # Compute the ROC curve
    fpr, tpr, _ = roc_curve(binary_labels, class_probabilities)
    # Compute the AUC (Area Under the Curve)
    class_auc = auc(fpr, tpr)
    # Plot the ROC curve
    plt.plot(fpr, tpr, label=f'Class {class_idx} (AUC = {class_auc:.2f})')

# Plot diagonal line for random guessing
plt.plot([0, 1], [0, 1], color='navy', linestyle='--', label='Random Guess')

# Add plot details
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title('ROC Curves for Multi-Class Classification')
plt.legend(loc='lower right')
plt.grid()
plt.show()

```
看图片3


# ResNet

## Initial zero-shot 
```python 
# 🟢 Step 1: Mount Google Drive
drive.mount('/content/drive')

# 🟢 Step 2: Define the path to the saved models on Google Drive
full_model_path = '/content/drive/MyDrive/ViT-plant/fine_tuned_resnet/resnet50_fine_tuned_full.pth'


# -------------------------------------------------------
# 🔥 Option 1: Load the **Full Saved Model** (weights + config)
# -------------------------------------------------------
print("\n🔹 Loading Full ResNet Model...")
full_resnet_model = torch.load(full_model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
full_resnet_model.to(device)
full_resnet_model.eval()  # Set to evaluation mode

print("✅ Full ResNet model loaded successfully from:", full_model_path)
```

Zero-Shot result: 
```python 
import time

categories = list(label_map.values())
# 🟢 Test Accuracy Calculation
full_resnet_model.eval()  # Turn off dropout, batch norm, etc.
correct = 0  # Number of correct predictions
total = 0  # Total number of samples

# Disable gradient computation (for faster performance)
with torch.no_grad():
    for images, labels in test_loader_sam:  # Loop through the test DataLoader
        # Move images and labels to the GPU (if available)
        images, labels = images.to(device), labels.to(device)

        # Forward pass through the model
        outputs = full_resnet_model(images)  # Handles both ResNet and ViT cases (ResNet: outputs, ViT: outputs.logits)

        # Get predictions (class with the maximum score)
        _, predicted = torch.max(outputs.logits if hasattr(outputs, 'logits') else outputs, 1)  # Handles ViT and ResNet

        # Count total samples and correct predictions
        total += labels.size(0)  # Total number of images in the batch
        correct += (predicted == labels).sum().item()  # Count the number of correct predictions

# Calculate accuracy as a percentage
test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")


# 🟢 Inference Time Calculation
full_resnet_model.eval()
inference_times = []

# Disable gradient computation (for faster performance)
with torch.no_grad():
    for images, labels in tqdm(test_loader_sam, desc="Calculating Inference Time"):
        # Move images and labels to the GPU (if available)
        images, labels = images.to(device), labels.to(device)

        # Measure inference time
        start_time = time.time()
        outputs = full_resnet_model(images)  # Handles both ResNet and ViT cases (ResNet: outputs, ViT: outputs.logits)
        inference_time = time.time() - start_time
        inference_times.append(inference_time / images.size(0))  # Time per image

# Calculate average inference time
avg_inference_time = sum(inference_times) / len(inference_times)
print(f"Average Inference Time: {avg_inference_time * 1000:.2f} ms/image")


# 🟢 Robustness Test Calculation
full_resnet_model.eval()
correct = 0  # Number of correct predictions
total = 0  # Total number of samples
noise_level = 0.1

# Disable gradient computation (for faster performance)
with torch.no_grad():
    for images, labels in tqdm(test_loader_sam, desc="Testing Robustness"):
        # Add random noise to the images
        noise = torch.randn_like(images) * noise_level  # Create random noise
        noisy_images = torch.clamp(images + noise, 0, 1)  # Clamp pixel values to [0, 1]

        # Move images and labels to the GPU (if available)
        noisy_images, labels = noisy_images.to(device), labels.to(device)

        # Forward pass with noisy images
        outputs = full_resnet_model(noisy_images)  # Handles both ResNet and ViT cases (ResNet: outputs, ViT: outputs.logits)

        # Get predictions (class with the maximum score)
        _, predicted = torch.max(outputs.logits if hasattr(outputs, 'logits') else outputs, 1)  # Handles ViT and ResNet

        # Count total samples and correct predictions
        total += labels.size(0)  # Total number of images in the batch
        correct += (predicted == labels).sum().item()  # Count the number of correct predictions

# Calculate robustness accuracy as a percentage
robustness_accuracy = 100 * correct / total
print(f"Robustness Test Accuracy with {noise_level * 100}% noise: {robustness_accuracy:.2f}%")

```
- Test Accuracy: 17.24%
- Calculating Inference Time: 100%|██████████| 63/63 [00:13<00:00,  4.56it/s]
- Average Inference Time: 0.18 ms/image
- Testing Robustness: 100%|██████████| 63/63 [00:14<00:00,  4.30it/s]Robustness Test Accuracy with 10.0% noise: 9.64%




## Train on original image set: 

```python 

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

categories = list(label_map.values())  # Extract the unique category names from label_map
# ------------------------------------------
# 🟢 Step 1: Load the Pre-trained ResNet Model
# ------------------------------------------
# Load ResNet50 (can also use resnet18, resnet34, resnet101, resnet152)
model = models.resnet50(pretrained=True)  # Load a ResNet50 model pre-trained on ImageNet

# Modify the final fully connected layer to match the number of classes in your dataset
num_features = model.fc.in_features  # Get the number of input features for the FC layer
model.fc = nn.Linear(num_features, len(categories))  # Change the output to match the number of classes

# Move model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ------------------------------------------
# 🟢 Step 2: Define Loss, Optimizer, and Scheduler
# ------------------------------------------
optimizer = optim.AdamW(model.parameters(), lr=5e-5)  # Use AdamW optimizer (similar to ViT)
criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
num_training_steps = len(train_loader) * 10  # Assuming 10 epochs
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps)

# ------------------------------------------
# 🟢 Step 3: Training Loop
# ------------------------------------------
num_epochs = 7  # Number of epochs
model.train()  # Set the model to training mode

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, leave=True)  # Training progress bar
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)  # Get model predictions
        loss = criterion(outputs, labels)  # Calculate the loss

        # Backward pass
        optimizer.zero_grad()  # Zero out previous gradients
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model weights
        scheduler.step()  # Update learning rate scheduler

        # Update metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)  # Get predicted class for each image
        total += labels.size(0)  # Count total images in the batch
        correct += (predicted == labels).sum().item()  # Count correct predictions

        # Update the progress bar with loss and accuracy
        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

    # Print epoch summary
    print(f"Epoch {epoch + 1} completed. Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")


```

- Epoch 1/7
Epoch 1: 100%|██████████| 516/516 [03:10<00:00,  2.71it/s, accuracy=92, loss=0.205]
Epoch 1 completed. Loss: 0.3030, Accuracy: 92.05%

- Epoch 2/7
Epoch 2: 100%|██████████| 516/516 [03:17<00:00,  2.61it/s, accuracy=99.2, loss=0.025]
Epoch 2 completed. Loss: 0.0337, Accuracy: 99.19%

- Epoch 3/7
Epoch 3: 100%|██████████| 516/516 [03:20<00:00,  2.58it/s, accuracy=99.6, loss=0.0037]
Epoch 3 completed. Loss: 0.0173, Accuracy: 99.58%

- Epoch 4/7
Epoch 4: 100%|██████████| 516/516 [03:18<00:00,  2.59it/s, accuracy=99.7, loss=0.000831]
Epoch 4 completed. Loss: 0.0130, Accuracy: 99.69%

- Epoch 5/7
Epoch 5: 100%|██████████| 516/516 [03:17<00:00,  2.62it/s, accuracy=99.8, loss=0.0255]
Epoch 5 completed. Loss: 0.0085, Accuracy: 99.81%

- Epoch 6/7
Epoch 6: 100%|██████████| 516/516 [03:17<00:00,  2.62it/s, accuracy=99.9, loss=0.000542]
Epoch 6 completed. Loss: 0.0041, Accuracy: 99.93%

- Epoch 7/7
Epoch 7: 100%|██████████| 516/516 [03:17<00:00,  2.61it/s, accuracy=100, loss=0.000265]Epoch 7 completed. Loss: 0.0032, Accuracy: 99.95%


## Evaluate 
```python 

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

categories = list(label_map.values())  # Extract the unique category names from label_map
# ------------------------------------------
# 🟢 Step 1: Load the Pre-trained ResNet Model
# ------------------------------------------
# Load ResNet50 (can also use resnet18, resnet34, resnet101, resnet152)
model = models.resnet50(pretrained=True)  # Load a ResNet50 model pre-trained on ImageNet

# Modify the final fully connected layer to match the number of classes in your dataset
num_features = model.fc.in_features  # Get the number of input features for the FC layer
model.fc = nn.Linear(num_features, len(categories))  # Change the output to match the number of classes

# Move model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# ------------------------------------------
# 🟢 Step 2: Define Loss, Optimizer, and Scheduler
# ------------------------------------------
optimizer = optim.AdamW(model.parameters(), lr=5e-5)  # Use AdamW optimizer (similar to ViT)
criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
num_training_steps = len(train_loader) * 10  # Assuming 10 epochs
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps)

# ------------------------------------------
# 🟢 Step 3: Training Loop
# ------------------------------------------
num_epochs = 7  # Number of epochs
model.train()  # Set the model to training mode

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader, leave=True)  # Training progress bar
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)  # Get model predictions
        loss = criterion(outputs, labels)  # Calculate the loss

        # Backward pass
        optimizer.zero_grad()  # Zero out previous gradients
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model weights
        scheduler.step()  # Update learning rate scheduler

        # Update metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)  # Get predicted class for each image
        total += labels.size(0)  # Count total images in the batch
        correct += (predicted == labels).sum().item()  # Count correct predictions

        # Update the progress bar with loss and accuracy
        loop.set_description(f"Epoch {epoch + 1}")
        loop.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

    # Print epoch summary
    print(f"Epoch {epoch + 1} completed. Loss: {running_loss / len(train_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

```
- Test Accuracy: 99.81%
- Calculating Inference Time: 100%|██████████| 129/129 [00:11<00:00, 11.51it/s]
- Average Inference Time: 0.23 ms/image
- Testing Robustness: 100%|██████████| 129/129 [00:26<00:00,  4.95it/s]
- Robustness Test Accuracy with 10.0% noise: 49.10%

## Train and Test on SAM-ed data

```python 

import torch.optim as optim
import torch
from torch.optim import AdamW
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import get_scheduler
from tqdm import tqdm



# 🟢 Step 2: Define Loss, Optimizer, and Scheduler
# ------------------------------------------
# Define optimizer, loss function, and scheduler
optimizer = optim.AdamW(full_resnet_model.parameters(), lr=5e-5)  # Use AdamW optimizer (same as ViT)
criterion = nn.CrossEntropyLoss()  # Loss function for multi-class classification
num_training_steps = len(train_loader_sam) * 5  # Assuming 5 epochs
scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_training_steps)

# ------------------------------------------
# 🟢 Step 3: Training Loop
# ------------------------------------------
num_epochs = 8  # Number of epochs
full_resnet_model.train()  # Set the model to training mode

for epoch in range(num_epochs):
    print(f"🔹 Epoch {epoch + 1}/{num_epochs}")
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(train_loader_sam, leave=True)  # Training progress bar
    for images, labels in loop:
        # Move images and labels to GPU (if available)
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = full_resnet_model(images)  # Get predictions from the ResNet model
        loss = criterion(outputs, labels)  # Calculate the loss

        # Backward pass
        optimizer.zero_grad()  # Zero out previous gradients
        loss.backward()  # Backpropagate the loss
        optimizer.step()  # Update model weights
        scheduler.step()  # Update learning rate scheduler

        # Update metrics
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)  # Get predicted class for each image
        total += labels.size(0)  # Count total images in the batch
        correct += (predicted == labels).sum().item()  # Count correct predictions

        # Update the progress bar with loss and accuracy
        loop.set_description(f"🔹 Epoch {epoch + 1}")
        loop.set_postfix(loss=loss.item(), accuracy=100 * correct / total)

    # Print epoch summary
    print(f"🔹 Epoch {epoch + 1} completed. Loss: {running_loss / len(train_loader_sam):.4f}, Accuracy: {100 * correct / total:.2f}%")


# ------------------------------------------
# 🟢 Step 4: Save the Fine-tuned ResNet Model
# ------------------------------------------
# Save the fine-tuned model as a new version (both weights and full model)
fine_tuned_resnet_path = '/content/drive/MyDrive/ViT-plant/SAM_fine_tuned_resnet'

# Save the full model (including architecture + weights)
torch.save(full_resnet_model, fine_tuned_resnet_path)

print(f"✅ Fine-tuned ResNet model saved successfully to: {fine_tuned_resnet_path}")

```
🔹 Epoch 1/8
🔹 Epoch 1: 100%|██████████| 252/252 [00:54<00:00,  4.60it/s, accuracy=96.6, loss=0.0279]
🔹 Epoch 1 completed. Loss: 0.1067, Accuracy: 96.62%
🔹 Epoch 2/8
🔹 Epoch 2: 100%|██████████| 252/252 [00:54<00:00,  4.59it/s, accuracy=98.4, loss=0.0824]
🔹 Epoch 2 completed. Loss: 0.0466, Accuracy: 98.39%
🔹 Epoch 3/8
🔹 Epoch 3: 100%|██████████| 252/252 [00:55<00:00,  4.56it/s, accuracy=99.2, loss=0.0238]
🔹 Epoch 3 completed. Loss: 0.0274, Accuracy: 99.16%
🔹 Epoch 4/8
🔹 Epoch 4: 100%|██████████| 252/252 [00:55<00:00,  4.56it/s, accuracy=99.5, loss=0.00276]
🔹 Epoch 4 completed. Loss: 0.0176, Accuracy: 99.53%
🔹 Epoch 5/8
🔹 Epoch 5: 100%|██████████| 252/252 [00:54<00:00,  4.60it/s, accuracy=99.7, loss=0.00731]
🔹 Epoch 5 completed. Loss: 0.0134, Accuracy: 99.65%
🔹 Epoch 6/8
🔹 Epoch 6: 100%|██████████| 252/252 [00:54<00:00,  4.63it/s, accuracy=99.8, loss=0.0052]
🔹 Epoch 6 completed. Loss: 0.0100, Accuracy: 99.76%
🔹 Epoch 7/8
🔹 Epoch 7: 100%|██████████| 252/252 [00:54<00:00,  4.62it/s, accuracy=99.7, loss=0.0231]
🔹 Epoch 7 completed. Loss: 0.0103, Accuracy: 99.75%
🔹 Epoch 8/8
🔹 Epoch 8: 100%|██████████| 252/252 [00:54<00:00,  4.65it/s, accuracy=99.7, loss=0.0392]
🔹 Epoch 8 completed. Loss: 0.0119, Accuracy: 99.68%
✅ Fine-tuned ResNet model saved successfully to: /content/drive/MyDrive/ViT-plant/SAM_fine_tuned_resnet


```python 
import time

categories = list(label_map.values())
# 🟢 Test Accuracy Calculation
full_resnet_model.eval()  # Turn off dropout, batch norm, etc.
correct = 0  # Number of correct predictions
total = 0  # Total number of samples

# Disable gradient computation (for faster performance)
with torch.no_grad():
    for images, labels in test_loader_sam:  # Loop through the test DataLoader
        # Move images and labels to the GPU (if available)
        images, labels = images.to(device), labels.to(device)

        # Forward pass through the model
        outputs = full_resnet_model(images)  # Handles both ResNet and ViT cases (ResNet: outputs, ViT: outputs.logits)

        # Get predictions (class with the maximum score)
        _, predicted = torch.max(outputs.logits if hasattr(outputs, 'logits') else outputs, 1)  # Handles ViT and ResNet

        # Count total samples and correct predictions
        total += labels.size(0)  # Total number of images in the batch
        correct += (predicted == labels).sum().item()  # Count the number of correct predictions

# Calculate accuracy as a percentage
test_accuracy = 100 * correct / total
print(f"Test Accuracy: {test_accuracy:.2f}%")


# 🟢 Inference Time Calculation
full_resnet_model.eval()
inference_times = []

# Disable gradient computation (for faster performance)
with torch.no_grad():
    for images, labels in tqdm(test_loader_sam, desc="Calculating Inference Time"):
        # Move images and labels to the GPU (if available)
        images, labels = images.to(device), labels.to(device)

        # Measure inference time
        start_time = time.time()
        outputs = full_resnet_model(images)  # Handles both ResNet and ViT cases (ResNet: outputs, ViT: outputs.logits)
        inference_time = time.time() - start_time
        inference_times.append(inference_time / images.size(0))  # Time per image

# Calculate average inference time
avg_inference_time = sum(inference_times) / len(inference_times)
print(f"Average Inference Time: {avg_inference_time * 1000:.2f} ms/image")


# 🟢 Robustness Test Calculation
full_resnet_model.eval()
correct = 0  # Number of correct predictions
total = 0  # Total number of samples
noise_level = 0.1

# Disable gradient computation (for faster performance)
with torch.no_grad():
    for images, labels in tqdm(test_loader_sam, desc="Testing Robustness"):
        # Add random noise to the images
        noise = torch.randn_like(images) * noise_level  # Create random noise
        noisy_images = torch.clamp(images + noise, 0, 1)  # Clamp pixel values to [0, 1]

        # Move images and labels to the GPU (if available)
        noisy_images, labels = noisy_images.to(device), labels.to(device)

        # Forward pass with noisy images
        outputs = full_resnet_model(noisy_images)  # Handles both ResNet and ViT cases (ResNet: outputs, ViT: outputs.logits)

        # Get predictions (class with the maximum score)
        _, predicted = torch.max(outputs.logits if hasattr(outputs, 'logits') else outputs, 1)  # Handles ViT and ResNet

        # Count total samples and correct predictions
        total += labels.size(0)  # Total number of images in the batch
        correct += (predicted == labels).sum().item()  # Count the number of correct predictions

# Calculate robustness accuracy as a percentage
robustness_accuracy = 100 * correct / total
print(f"Robustness Test Accuracy with {noise_level * 100}% noise: {robustness_accuracy:.2f}%")
```
- Test Accuracy: 99.40%
- Calculating Inference Time: 100%|██████████| 63/63 [00:13<00:00,  4.74it/s]
- Average Inference Time: 0.17 ms/image
- Testing Robustness: 100%|██████████| 63/63 [00:14<00:00,  4.37it/s]Robustness Test Accuracy with 10.0% noise: 23.40%
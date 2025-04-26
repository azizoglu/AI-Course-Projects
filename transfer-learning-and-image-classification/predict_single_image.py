import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# Select device: use GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define image transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Resize image to 224x224
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalize
])

# Function to predict a single image
def predict_single_image(image_path, model_path, class_names):
    # Load the VGG19 model structure
    model = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
    num_ftrs = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(num_ftrs, len(class_names))  # Adjust final layer to number of classes

    # Load the saved model weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()

    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    # Make prediction
    with torch.no_grad():
        outputs = model(image)
        _, preds = torch.max(outputs, 1)
        predicted_class = class_names[preds.item()]

    print(f"Predicted Class: {predicted_class}")

# Main block to run prediction
if __name__ == "__main__":
    # Set class names according to your dataset
    class_names = ['fake', 'real']

    # Provide the image path and model path
    image_path = "test/1.jpg"  # Change this to your image file
    model_path = "real_fake_classifier.pth"  # Path to the saved model

    # Run prediction
    predict_single_image(image_path, model_path, class_names)
import torch
from torchvision import transforms
from PIL import Image
import os
from models.dogbreed_classifier import DogBreedClassifier
import glob

def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)

def main():
    # Find the latest checkpoint
    checkpoints = glob.glob("/app/models/best-checkpoint*.ckpt")
    if not checkpoints:
        raise FileNotFoundError("No checkpoint found in /app/models/")
    latest_checkpoint = max(checkpoints, key=os.path.getctime)

    # Load the best model from checkpoint
    model = DogBreedClassifier.load_from_checkpoint(latest_checkpoint)
    model.eval()

    # Get list of image files
    image_folder = "/app/data/test_images"
    image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))][:10]

    # Run inference on 10 images
    for image_file in image_files:
        image_path = os.path.join(image_folder, image_file)
        input_tensor = load_image(image_path)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

        print(f"Image: {image_file}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.2f}")
        print("---")

if __name__ == "__main__":
    main()
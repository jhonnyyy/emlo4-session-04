import torch
from torchvision import transforms
from PIL import Image
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import rootutils
import glob
from models.dogbreed_classifier import DogBreedClassifier
import random
import shutil
from pathlib import Path
from datamodules.dogbreed import DogBreedDataModule
root = rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)
def load_image(image_path):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = Image.open(image_path).convert('RGB')
    return transform(image).unsqueeze(0)
@hydra.main(version_base=None, config_path="../configs", config_name="infer")
def main(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    print(f"Current working directory: {os.getcwd()}")
    # Load the model from checkpoint
    checkpoint_dir = cfg.paths.checkpoints_dir
    print(f"Looking for checkpoints in: {checkpoint_dir}")
    if not os.path.exists(checkpoint_dir):
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in: {checkpoint_dir}")
    checkpoint_files.sort(key=os.path.getmtime, reverse=True)
    checkpoint_path = checkpoint_files[0]
    print(f"Loading checkpoint from: {checkpoint_path}")
    model = DogBreedClassifier.load_from_checkpoint(checkpoint_path,strict=False)
    model.eval()
    # Initialize the DogBreedDataModule
    data_module = DogBreedDataModule(
        data_dir=cfg.paths.data_dir,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers
    )
    data_module.setup()
    # Get the test dataloader
    test_dataloader = data_module.test_dataloader()
    # Run inference on images
    correct_predictions = 0
    total_predictions = 0
    for batch in test_dataloader:
        images, labels = batch
        
        with torch.no_grad():
            outputs = model(images)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_classes = torch.argmax(probabilities, dim=1)
        for i in range(len(labels)):
            true_class = cfg.data.extra.dog_breeds[labels[i].item()]
            predicted_breed = cfg.data.extra.dog_breeds[predicted_classes[i].item()]
            confidence = probabilities[i, predicted_classes[i]].item()
            print(f"Image {total_predictions + i + 1}:")
            print(f"True class: {true_class}")
            print(f"Predicted class: {predicted_breed}")
            print(f"Confidence: {confidence:.2f}")
            print("---")
            if true_class == predicted_breed:
                correct_predictions += 1
        
        total_predictions += len(labels)
    # Calculate and print the final accuracy
    accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
    print(f"\nFinal Accuracy: {accuracy:.2f}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Total predictions: {total_predictions}")
if __name__ == "__main__":
    main()
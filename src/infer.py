# import torch
# import hydra
# from omegaconf import DictConfig, OmegaConf
# import rootutils
# import glob
# from src.models.dogbreed_classifier import DogBreedClassifier
# from datamodules.dogbreed import DogBreedDataModule
# import os 

# root = rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)

# @hydra.main(version_base=None, config_path="../configs", config_name="infer")
# def main(cfg: DictConfig):
#     print(OmegaConf.to_yaml(cfg))

#     # Load the model from checkpoint
#     checkpoint_dir = cfg.paths.checkpoints_dir
#     print(f"Looking for checkpoints in: {checkpoint_dir}")
    
#     checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
#     if not checkpoint_files:
#         raise FileNotFoundError(f"No checkpoint files found in: {checkpoint_dir}")
    
#     checkpoint_path = max(checkpoint_files, key=os.path.getmtime)
#     print(f"Loading checkpoint from: {checkpoint_path}")
    
#     model = DogBreedClassifier.load_from_checkpoint(checkpoint_path)
#     model.eval()
#     print(f"Model loaded with {model.num_classes} output classes")

#     # Initialize the DogBreedDataModule
#     data_module = DogBreedDataModule(
#         data_dir=cfg.paths.data_dir,
#         batch_size=cfg.data.batch_size,
#         num_workers=cfg.data.num_workers
#     )
#     data_module.setup()

#     # Get the test dataloader
#     test_dataloader = data_module.test_dataloader()

#     # Run inference on test dataset
#     correct_predictions = 0
#     total_predictions = 0

#     for batch in test_dataloader:
#         images, labels = batch
        
#         with torch.no_grad():
#             outputs = model(images)
#             _, predicted = torch.max(outputs, 1)

#         correct_predictions += (predicted == labels).sum().item()
#         total_predictions += labels.size(0)

#         # Print predictions for each image in the batch
#         for i in range(labels.size(0)):
#             true_class = cfg.data.extra.dog_breeds[labels[i].item()]
#             predicted_class = cfg.data.extra.dog_breeds[predicted[i].item()]
#             confidence = torch.nn.functional.softmax(outputs[i], dim=0)[predicted[i]].item()

#             print(f"Image {total_predictions - labels.size(0) + i + 1}:")
#             print(f"True class: {true_class}")
#             print(f"Predicted class: {predicted_class}")
#             print(f"Confidence: {confidence:.2f}")
#             print("---")

#     # Calculate and print the final accuracy
#     accuracy = correct_predictions / total_predictions
#     print(f"\nFinal Accuracy: {accuracy:.2f}")
#     print(f"Correct predictions: {correct_predictions}")
#     print(f"Total predictions: {total_predictions}")

# if __name__ == "__main__":
#     main()



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

root = rootutils.setup_root(__file__, indicator='.project-root', pythonpath=True, dotenv=True)

def create_test_dataset(source_dir, test_dir, images_per_class):
    Path(test_dir).mkdir(parents=True, exist_ok=True)
    class_dirs = [d for d in os.listdir(source_dir) if os.path.isdir(os.path.join(source_dir, d))]

    for class_name in class_dirs:
        class_source_dir = os.path.join(source_dir, class_name)
        class_test_dir = os.path.join(test_dir, class_name)
        Path(class_test_dir).mkdir(parents=True, exist_ok=True)
        image_files = [f for f in os.listdir(class_source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp'))]

        selected_images = random.sample(image_files, min(images_per_class, len(image_files)))

        for image in selected_images:
            src = os.path.join(class_source_dir, image)
            dst = os.path.join(class_test_dir, image)
            shutil.copy2(src, dst)

    print(f"Test dataset created at: {test_dir}")
    return test_dir

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

    # Create test dataset
    source_dir = cfg.paths.data_dir
    test_dir = os.path.join(cfg.paths.output_dir, "test_dataset")
    images_per_class = cfg.infer.num_images // len(cfg.data.extra.dog_breeds)
    test_dataset_path = create_test_dataset(source_dir, test_dir, images_per_class)

    # Update the test_images_dir in the configuration
    cfg.paths.test_images_dir = test_dataset_path

    # Get list of image files
    image_files = []
    for root, dirs, files in os.walk(cfg.paths.test_images_dir):
        for file in files:
            if file.endswith(('.png', '.jpg', '.jpeg')):
                image_files.append(os.path.join(root, file))

    # Run inference on images
    correct_predictions = 0
    total_predictions = 0

    for image_path in image_files:
        input_tensor = load_image(image_path)

        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_class = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class].item()

        # Get the true class from the image path
        true_class = os.path.basename(os.path.dirname(image_path))
        predicted_breed = cfg.data.extra.dog_breeds[predicted_class]

        print(f"Image: {os.path.basename(image_path)}")
        print(f"True class: {true_class}")
        print(f"Predicted class: {predicted_breed}")
        print(f"Confidence: {confidence:.2f}")
        print("---")

        if true_class == predicted_breed:
            correct_predictions += 1
        total_predictions += 1

    # Calculate and print the final accuracy
    accuracy = correct_predictions / total_predictions
    print(f"\nFinal Accuracy: {accuracy:.2f}")
    print(f"Correct predictions: {correct_predictions}")
    print(f"Total predictions: {total_predictions}")

if __name__ == "__main__":
    main()
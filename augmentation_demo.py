import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import numpy as np

def show_augmented_images(original_image, augmented_images, title="Augmentation Examples"):
    """Display original and augmented images in a single row"""
    fig, axes = plt.subplots(1, 1 + len(augmented_images), figsize=(15, 3))
    
    # Show original
    axes[0].imshow(original_image.squeeze(), cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # Show augmented images
    for idx, (aug_img, aug_name) in enumerate(augmented_images):
        axes[idx + 1].imshow(aug_img.squeeze(), cmap='gray')
        axes[idx + 1].set_title(aug_name)
        axes[idx + 1].axis('off')
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def main():
    # Load a single MNIST image
    dataset = datasets.MNIST('./data', train=True, download=True, 
                           transform=transforms.ToTensor())
    image, _ = dataset[0]  # Get the first image
    
    # Define augmentations
    augmentations = [
        (transforms.RandomRotation(30), "Rotation ±30°"),
        (transforms.RandomAffine(0, scale=(0.8, 1.2)), "Scale 0.8-1.2x"),
        (transforms.RandomAffine(0, translate=(0.1, 0.1)), "Translation"),
        (transforms.Compose([
            transforms.RandomRotation(15),
            transforms.RandomAffine(0, scale=(0.9, 1.1))
        ]), "Combined")
    ]
    
    # Apply augmentations
    augmented_images = []
    for aug_transform, name in augmentations:
        # Set random seed for reproducibility
        torch.manual_seed(42)
        aug_image = aug_transform(image)
        augmented_images.append((aug_image, name))
    
    # Display results
    show_augmented_images(image, augmented_images)

if __name__ == "__main__":
    main() 
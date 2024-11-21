import pytest
import sys
import os
import logging
import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision import datasets

# Configure logging
logging.basicConfig(level=logging.INFO, 
                   format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Add the parent directory to the path so we can import mnist_model
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mnist_model import LightMNIST, train_model

def test_model_parameters():
    logger.info("Starting model parameters test")
    model = LightMNIST()
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total model parameters: {total_params}")
    
    try:
        assert total_params < 25000, f"Model has {total_params} parameters, should be less than 25000"
        logger.info("✓ Model parameters test passed")
    except AssertionError as e:
        logger.error(f"✗ Model parameters test failed: {str(e)}")
        raise

def test_model_accuracy():
    logger.info("Starting model accuracy test")
    logger.info("Training model for one epoch...")
    
    try:
        metrics = train_model(return_metrics=True)
        logger.info(f"Training completed.")
        logger.info(f"Model accuracy: {metrics['final_accuracy']}%")
        
        assert metrics['final_accuracy'] > 95.0, \
            f"Model accuracy {metrics['final_accuracy']}% is below 95%"
        logger.info("✓ Model accuracy test passed")
    except AssertionError as e:
        logger.error(f"✗ Model accuracy test failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"✗ Unexpected error during training: {str(e)}")
        raise 

def test_model_output_shape():
    logger.info("Starting model output shape test")
    model = LightMNIST()
    batch_size = 32
    
    # Create random input tensor
    dummy_input = torch.randn(batch_size, 1, 28, 28)
    
    try:
        output = model(dummy_input)
        expected_shape = (batch_size, 10)  # 10 classes for MNIST
        
        assert output.shape == expected_shape, \
            f"Expected output shape {expected_shape}, but got {output.shape}"
        assert torch.allclose(output.exp().sum(dim=1), torch.ones(batch_size)), \
            "Output probabilities do not sum to 1"
        
        logger.info("✓ Model output shape test passed")
    except Exception as e:
        logger.error(f"✗ Model output shape test failed: {str(e)}")
        raise

def test_model_gradients():
    logger.info("Starting model gradients test")
    model = LightMNIST()
    
    # Create random input and target
    dummy_input = torch.randn(1, 1, 28, 28)
    dummy_target = torch.tensor([5])  # Random target class
    
    try:
        # Forward pass
        output = model(dummy_input)
        loss = F.nll_loss(output, dummy_target)
        
        # Backward pass
        loss.backward()
        
        # Check if gradients exist and are not zero
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad)), \
                f"Zero gradient for {name}"
        
        logger.info("✓ Model gradients test passed")
    except Exception as e:
        logger.error(f"✗ Model gradients test failed: {str(e)}")
        raise

def test_model_augmentation_invariance():
    logger.info("Starting augmentation invariance test")
    
    # Train the model first
    logger.info("Training model for augmentation test...")
    metrics = train_model(return_metrics=True)
    
    # Get the trained model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LightMNIST().to(device)
    model.eval()
    
    # Create a sample from actual MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # Use same normalization as training
    ])
    
    dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    original_input, _ = next(iter(test_loader))
    original_input = original_input.to(device)
    
    # Define mild augmentations
    augmentations = transforms.Compose([
        transforms.RandomRotation(3),          # Reduced rotation angle
        transforms.RandomAffine(0, translate=(0.02, 0.02)),  # Reduced translation
        transforms.RandomAffine(0, scale=(0.95, 1.05))      # Reduced scaling
    ])
    
    try:
        # Get prediction for original input
        with torch.no_grad():
            original_output = model(original_input)
            original_pred = original_output.argmax(dim=1)
        
        # Test multiple augmentations
        matching_predictions = 0
        n_trials = 10
        
        for _ in range(n_trials):
            # Apply augmentation
            aug_input = augmentations(original_input)
            
            # Get prediction for augmented input
            with torch.no_grad():
                aug_output = model(aug_input)
                aug_pred = aug_output.argmax(dim=1)
            
            if aug_pred == original_pred:
                matching_predictions += 1
        
        # At least 70% of predictions should match for mild augmentations
        accuracy = matching_predictions / n_trials
        assert accuracy >= 0.7, \
            f"Model predictions are not consistent under mild augmentations ({accuracy*100:.1f}%)"
        
        logger.info(f"✓ Model augmentation invariance test passed ({accuracy*100:.1f}% consistent)")
    except Exception as e:
        logger.error(f"✗ Model augmentation invariance test failed: {str(e)}")
        raise
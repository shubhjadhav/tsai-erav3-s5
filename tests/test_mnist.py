import pytest
import sys
import os
import logging

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
        logger.info(f"Training completed. Metrics: {metrics}")
        
        assert metrics['final_accuracy'] > 95.0, \
            f"Model accuracy {metrics['final_accuracy']}% is below 95%"
        logger.info("✓ Model accuracy test passed")
    except AssertionError as e:
        logger.error(f"✗ Model accuracy test failed: {str(e)}")
        raise
    except Exception as e:
        logger.error(f"✗ Unexpected error during training: {str(e)}")
        raise 
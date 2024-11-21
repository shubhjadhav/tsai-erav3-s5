# MNIST Lightweight Neural Network

A lightweight convolutional neural network for MNIST digit classification that achieves >95% accuracy in one epoch while maintaining less than 25,000 parameters. The project includes automated testing through GitHub Actions to ensure model efficiency and performance.

## Model Architecture

The LightMNIST model uses an efficient architecture:
- Input: 28x28 grayscale images
- 3 convolutional layers with batch normalization:
  - Conv1: 1 → 8 channels (3x3 kernel)
  - Conv2: 8 → 16 channels (3x3 kernel)
  - Conv3: 16 → 16 channels (3x3 kernel)
- Strategic max pooling layers
- Single fully connected layer (16*9*9 → 10)
- Total parameters: < 25,000

## Performance Metrics
- Accuracy: >95% in single epoch
- Parameters: ~13,000
- Training time: ~2-3 minutes on CPU

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- pytest (for testing)

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model
To train the model, run:
```bash
python mnist_model.py
```

This will:
- Download the MNIST dataset automatically
- Train for one epoch
- Display progress and metrics
- Show final accuracy

### Running Tests
To run the test suite:

```bash
pytest tests/ -v --log-cli-level=INFO
```

## GitHub Actions Pipeline

The project includes automated testing that verifies:
1. Model size (< 25,000 parameters)
2. Model performance (> 95% accuracy)

The pipeline runs automatically on:
- Every push to the repository
- Every pull request

## Project Structure

```
.
├── mnist_model.py          # Main model implementation
├── tests/
│   └── test_mnist.py       # Automated tests
├── .github/
│   └── workflows/          # GitHub Actions configuration
├── requirements.txt        # Project dependencies
└── README.md               # Documentation
```

## Model Details

### Architecture Highlights
- Uses batch normalization for stable training
- Efficient parameter usage through strategic kernel sizes
- Optimized max pooling placement
- Single fully connected layer for classification

### Layer-by-Layer Architecture
1. Input Layer
   - Input shape: 1x28x28 (grayscale MNIST images)

2. Convolutional Block 1
   - Conv2d: 1 → 8 channels, 3x3 kernel (output: 8x26x26)
   - BatchNorm2d
   - ReLU activation
   - MaxPool2d: 2x2 (output: 8x13x13)

3. Convolutional Block 2
   - Conv2d: 8 → 16 channels, 3x3 kernel (output: 16x11x11)
   - BatchNorm2d
   - ReLU activation
   - MaxPool2d: 1x1 (output: 16x11x11)

4. Convolutional Block 3
   - Conv2d: 16 → 16 channels, 3x3 kernel (output: 16x9x9)
   - BatchNorm2d
   - ReLU activation
   - MaxPool2d: 1x1 (output: 16x9x9)

5. Classification Head
   - Flatten: 16x9x9 = 1,296 features
   - Linear: 1,296 → 10 classes
   - LogSoftmax activation

### Training Configuration
- Batch size: 128
- Optimizer: SGD with momentum (0.9)
- Learning rate: 0.05
- Data normalization: Mean=0.1307, Std=0.3081

## Data Augmentation
The model uses the following augmentation techniques during training:
- Random rotation (±15 degrees)
- Random translation (up to 10% in any direction)
- Random scaling (90% to 110% of original size)

To visualize the augmentations:
```bash
python augmentation_demo.py
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- MNIST Dataset creators
- PyTorch team
- GitHub Actions for CI/CD support

## Testing Framework

The project includes comprehensive test cases to ensure model quality and performance:

### 1. Parameter Count Test (`test_model_parameters`)
- Verifies model architecture efficiency
- Ensures total parameters stay below 25,000
- Helps maintain model lightweight nature
- Logs exact parameter count for monitoring

### 2. Model Accuracy Test (`test_model_accuracy`)
- Trains model for one epoch
- Verifies accuracy exceeds 95%
- Tests model's learning capability
- Logs final accuracy metrics

### 3. Output Shape Test (`test_model_output_shape`)
- Validates model architecture correctness
- Checks output dimensions (batch_size × 10 classes)
- Ensures valid probability distributions
- Verifies softmax properties (sum to 1)

### 4. Gradient Flow Test (`test_model_gradients`)
- Checks backpropagation functionality
- Verifies gradients exist for all parameters
- Ensures non-zero gradients during training
- Tests optimization readiness

### 5. Augmentation Invariance Test (`test_model_augmentation_invariance`)
- Tests model robustness to input variations
- Applies mild augmentations:
  - Small rotations (±3°)
  - Minor translations (±2%)
  - Slight scaling (95-105%)
- Requires 70% prediction consistency
- Ensures model stability

### Running Tests

Execute all tests with detailed logging:
```bash
pytest tests/ -v --log-cli-level=INFO
```

Test artifacts and logs are automatically uploaded to GitHub Actions for each run.
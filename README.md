# Neural Predictor

A lightweight neural network implementation for pattern prediction and regression tasks. Built from scratch in Python with NumPy for educational purposes and practical use.

## What it does

This neural network learns to predict outputs based on input patterns. It uses a simple feedforward architecture with one hidden layer and sigmoid activation functions. The model is trained using backpropagation and can handle multiple input features.

## Features

- Custom neural network implementation (no external ML libraries)
- Automatic data normalization
- Batch prediction support
- Interactive prediction mode
- Configurable network architecture
- Real-time training progress monitoring

## Quick Start

1. Clone the repository:

```bash
git clone https://github.com/Amarifields/neaural-preditor.git
cd neaural-preditor
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the predictor:

```bash
python main.py
```

## Usage

The program starts by training on generated sample data, then enters an interactive mode where you can input your own values for prediction.

### Training Phase

- Generates 50 training samples with 3 input features
- Normalizes data automatically
- Trains for 1000 epochs with progress updates
- Tests on 10 validation samples

### Interactive Mode

Enter 3 numbers separated by spaces to get predictions:

```
> 5.2 3.1 7.8
Input: [5.2, 3.1, 7.8]
Normalized: ['0.5200', '0.3100', '0.7800']
Prediction: 0.412456
```

Type `quit` to exit.

## Network Architecture

- **Input Layer**: 3 neurons (configurable)
- **Hidden Layer**: 5 neurons (configurable)
- **Output Layer**: 1 neuron
- **Activation**: Sigmoid function
- **Learning Rate**: 0.01
- **Training Epochs**: 1000

## Customization

You can modify the network parameters in the `NeuralPredictor` constructor:

```python
predictor = NeuralPredictor(
    input_size=3,      # Number of input features
    hidden_size=5,     # Hidden layer neurons
    output_size=1      # Output neurons
)
```

## Requirements

- Python 3.7+
- NumPy >= 1.21.0

## Example Output

```
Neural Predictor Initialization
========================================
Generating training data...
Normalizing training data...
Training neural network...
Epoch 0, Average Error: 0.234567
Epoch 100, Average Error: 0.123456
...
Epoch 900, Average Error: 0.012345

Test Results:
Input                    Target      Prediction          Error
------------------------------------------------------------
5.20, 3.10, 7.80        0.4120      0.4125              0.0005
2.10, 8.50, 1.30        0.2980      0.2978              0.0002
...

Average Prediction Error: 0.012345
```

## License

MIT License - feel free to use and modify as needed.

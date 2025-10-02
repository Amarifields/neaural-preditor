import numpy as np
import math
import random
from typing import List, Tuple

class NeuralPredictor:
    def __init__(self, input_size: int = 3, hidden_size: int = 5, output_size: int = 1):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1
        
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))
        
        self.learning_rate = 0.01
        self.epochs = 1000

    def normalize_data(self, data: List[float]) -> List[float]:
        if not data:
            return []
        
        min_val = min(data)
        max_val = max(data)
        
        if max_val == min_val:
            return [0.5] * len(data)
        
        normalized = []
        for value in data:
            norm_val = (value - min_val) / (max_val - min_val)
            normalized.append(norm_val)
        
        return normalized

    def sigmoid(self, x: float) -> float:
        try:
            if x > 500:
                return 1.0
            elif x < -500:
                return 0.0
            return 1 / (1 + math.exp(-x))
        except OverflowError:
            return 1.0 if x > 0 else 0.0

    def sigmoid_derivative(self, x: float) -> float:
        s = self.sigmoid(x)
        return s * (1 - s)

    def forward_pass(self, inputs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        hidden_input = np.dot(inputs, self.weights_input_hidden) + self.bias_hidden
        hidden_output = np.array([[self.sigmoid(x) for x in hidden_input[0]]])
        
        output_input = np.dot(hidden_output, self.weights_hidden_output) + self.bias_output
        output = np.array([[self.sigmoid(x) for x in output_input[0]]])
        
        return hidden_output, output

    def train(self, training_data: List[List[float]], targets: List[float]):
        for epoch in range(self.epochs):
            total_error = 0
            
            for i, (inputs, target) in enumerate(zip(training_data, targets)):
                inputs_np = np.array([inputs])
                target_np = np.array([[target]])
                
                hidden_output, output = self.forward_pass(inputs_np)
                
                error = target_np - output
                total_error += np.mean(np.abs(error))
                
                output_delta = error * np.array([[self.sigmoid_derivative(x) for x in output[0]]])
                
                hidden_error = np.dot(output_delta, self.weights_hidden_output.T)
                hidden_delta = hidden_error * np.array([[self.sigmoid_derivative(x) for x in hidden_output[0]]])
                
                self.weights_hidden_output += self.learning_rate * np.dot(hidden_output.T, output_delta)
                self.weights_input_hidden += self.learning_rate * np.dot(inputs_np.T, hidden_delta)
                
                self.bias_output += self.learning_rate * output_delta
                self.bias_hidden += self.learning_rate * hidden_delta
            
            if epoch % 100 == 0:
                avg_error = total_error / len(training_data)
                print(f"Epoch {epoch}, Average Error: {avg_error:.6f}")

    def predict(self, inputs: List[float]) -> float:
        if len(inputs) != self.input_size:
            raise ValueError(f"Expected {self.input_size} inputs, got {len(inputs)}")
        
        inputs_np = np.array([inputs])
        _, output = self.forward_pass(inputs_np)
        return output[0][0]

    def batch_predict(self, input_batch: List[List[float]]) -> List[float]:
        predictions = []
        for inputs in input_batch:
            pred = self.predict(inputs)
            predictions.append(pred)
        return predictions

def generate_sample_data(num_samples: int = 50) -> Tuple[List[List[float]], List[float]]:
    random.seed(42)
    np.random.seed(42)
    
    data = []
    targets = []
    
    for _ in range(num_samples):
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 10)
        x3 = random.uniform(0, 10)
        
        target = (x1 * 0.3 + x2 * 0.5 + x3 * 0.2) / 10.0
        target = min(max(target, 0), 1)
        
        data.append([x1, x2, x3])
        targets.append(target)
    
    return data, targets

def main():
    print("Neural Predictor Initialization")
    print("=" * 40)
    
    predictor = NeuralPredictor(input_size=3, hidden_size=5, output_size=1)
    
    print("Generating training data...")
    training_data, training_targets = generate_sample_data(50)
    
    print("Normalizing training data...")
    normalized_data = []
    for sample in training_data:
        normalized_sample = predictor.normalize_data(sample)
        normalized_data.append(normalized_sample)
    
    print("Training neural network...")
    predictor.train(normalized_data, training_targets)
    
    print("\nGenerating test data...")
    test_data, test_targets = generate_sample_data(10)
    
    print("Making predictions...")
    print("\nTest Results:")
    print("Input\t\t\tTarget\t\tPrediction\t\tError")
    print("-" * 60)
    
    total_error = 0
    for i, (inputs, target) in enumerate(zip(test_data, test_targets)):
        normalized_inputs = predictor.normalize_data(inputs)
        prediction = predictor.predict(normalized_inputs)
        error = abs(target - prediction)
        total_error += error
        
        print(f"{inputs[0]:.2f}, {inputs[1]:.2f}, {inputs[2]:.2f}\t{target:.4f}\t\t{prediction:.4f}\t\t{error:.4f}")
    
    avg_error = total_error / len(test_data)
    print(f"\nAverage Prediction Error: {avg_error:.6f}")
    
    print("\nInteractive Mode:")
    print("Enter 3 numbers separated by spaces (or 'quit' to exit):")
    
    while True:
        try:
            user_input = input("> ").strip()
            if user_input.lower() == 'quit':
                break
            
            numbers = [float(x) for x in user_input.split()]
            if len(numbers) != 3:
                print("Please enter exactly 3 numbers")
                continue
            
            normalized_inputs = predictor.normalize_data(numbers)
            prediction = predictor.predict(normalized_inputs)
            
            print(f"Input: {numbers}")
            print(f"Normalized: {[f'{x:.4f}' for x in normalized_inputs]}")
            print(f"Prediction: {prediction:.6f}")
            print()
            
        except ValueError:
            print("Invalid input. Please enter 3 numbers separated by spaces.")
        except KeyboardInterrupt:
            break
    
    print("Neural Predictor session ended.")

if __name__ == "__main__":
    main()

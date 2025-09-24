## Neural Predictor

Lightweight feedforward neural network for regression, implemented in pure Python with NumPy. Vectorized training, numerically stable activations, and a simple CLI for experimentation.

## Installation

```bash
git clone https://github.com/Amarifields/neaural-preditor.git
cd neaural-preditor
pip install -r requirements.txt
```

## CLI

```bash
python main.py --input-size 3 --hidden-size 8 --epochs 1000 --lr 0.01 --train-samples 200 --test-samples 50 --seed 42
```

- `--quiet`: suppress progress
- `--no-interactive`: skip interactive prompt after evaluation

## Example

```bash
python main.py --epochs 600 --train-samples 300 --test-samples 60
```

Sample output (truncated):

```
training
epoch 0 mae 0.090321
epoch 100 mae 0.028114
epoch 200 mae 0.019873
...
evaluation
test_mae 0.017442
```

## Architecture

- Input: configurable
- Hidden: single layer, configurable width
- Output: single neuron
- Activation: sigmoid
- Loss monitor: mean absolute error

## Requirements

- Python 3.8+
- NumPy >= 1.21.0

## License

MIT

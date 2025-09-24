import numpy as np
import argparse
import random
from typing import List, Tuple

class MinMaxScaler:
    def __init__(self):
        self.min_ = None
        self.max_ = None

    def fit(self, X: np.ndarray) -> None:
        self.min_ = X.min(axis=0, keepdims=True)
        self.max_ = X.max(axis=0, keepdims=True)
        same = (self.max_ - self.min_) == 0
        self.max_[same] = self.min_[same] + 1.0

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.min_) / (self.max_ - self.min_)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)

class NeuralPredictor:
    def __init__(self, input_size: int = 3, hidden_size: int = 8, output_size: int = 1, learning_rate: float = 0.01, epochs: int = 1000, seed: int = 42):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        rng = np.random.default_rng(seed)
        limit_ih = np.sqrt(6.0 / (input_size + hidden_size))
        limit_ho = np.sqrt(6.0 / (hidden_size + output_size))
        self.W_ih = rng.uniform(-limit_ih, limit_ih, (input_size, hidden_size)).astype(np.float64)
        self.W_ho = rng.uniform(-limit_ho, limit_ho, (hidden_size, output_size)).astype(np.float64)
        self.b_h = np.zeros((1, hidden_size), dtype=np.float64)
        self.b_o = np.zeros((1, output_size), dtype=np.float64)

    @staticmethod
    def sigmoid(x: np.ndarray) -> np.ndarray:
        x_clip = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x_clip))

    @staticmethod
    def sigmoid_derivative_from_output(s: np.ndarray) -> np.ndarray:
        return s * (1.0 - s)

    def forward(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        h_pre = X @ self.W_ih + self.b_h
        h_act = self.sigmoid(h_pre)
        o_pre = h_act @ self.W_ho + self.b_o
        o_act = self.sigmoid(o_pre)
        return h_pre, h_act, o_pre, o_act

    def train(self, X: np.ndarray, y: np.ndarray, quiet: bool = False, log_every: int = 100) -> None:
        n = X.shape[0]
        for epoch in range(self.epochs):
            _, h_act, _, o_act = self.forward(X)
            error = o_act - y
            o_delta = error * self.sigmoid_derivative_from_output(o_act)
            h_delta = (o_delta @ self.W_ho.T) * self.sigmoid_derivative_from_output(h_act)
            grad_W_ho = (h_act.T @ o_delta) / n
            grad_W_ih = (X.T @ h_delta) / n
            grad_b_o = o_delta.mean(axis=0, keepdims=True)
            grad_b_h = h_delta.mean(axis=0, keepdims=True)
            self.W_ho -= self.learning_rate * grad_W_ho
            self.W_ih -= self.learning_rate * grad_W_ih
            self.b_o -= self.learning_rate * grad_b_o
            self.b_h -= self.learning_rate * grad_b_h
            if not quiet and (epoch % log_every == 0 or epoch == self.epochs - 1):
                loss = np.mean(np.abs(error))
                print(f"epoch {epoch} mae {loss:.6f}")

    def predict(self, X_row: List[float]) -> float:
        X = np.asarray([X_row], dtype=np.float64)
        _, _, _, o_act = self.forward(X)
        return float(o_act[0, 0])

    def batch_predict(self, X: np.ndarray) -> np.ndarray:
        _, _, _, o_act = self.forward(np.asarray(X, dtype=np.float64))
        return o_act.ravel()

def generate_sample_data(num_samples: int, input_size: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    random.seed(seed)
    rng = np.random.default_rng(seed)
    X = rng.uniform(0.0, 10.0, size=(num_samples, input_size)).astype(np.float64)
    weights = np.linspace(0.2, 0.6, num=input_size, dtype=np.float64)
    weights = weights / weights.sum()
    y = (X * weights).sum(axis=1, keepdims=True) / 10.0
    y = np.clip(y, 0.0, 1.0)
    return X, y

def run_cli() -> None:
    p = argparse.ArgumentParser(prog="neural-predictor")
    p.add_argument("--input-size", type=int, default=3)
    p.add_argument("--hidden-size", type=int, default=8)
    p.add_argument("--output-size", type=int, default=1)
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--train-samples", type=int, default=200)
    p.add_argument("--test-samples", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--no-interactive", action="store_true")
    args = p.parse_args()
    X_train, y_train = generate_sample_data(args.train_samples, args.input_size, args.seed)
    X_test, y_test = generate_sample_data(args.test_samples, args.input_size, args.seed + 1)
    scaler = MinMaxScaler()
    X_train_norm = scaler.fit_transform(X_train)
    X_test_norm = scaler.transform(X_test)
    model = NeuralPredictor(
        input_size=args.input_size,
        hidden_size=args.hidden_size,
        output_size=args.output_size,
        learning_rate=args.lr,
        epochs=args.epochs,
        seed=args.seed,
    )
    if not args.quiet:
        print("training")
    model.train(X_train_norm, y_train, quiet=args.quiet)
    preds = model.batch_predict(X_test_norm)
    mae = float(np.mean(np.abs(y_test.ravel() - preds)))
    if not args.quiet:
        print("evaluation")
        print(f"test_mae {mae:.6f}")
        for i in range(min(10, X_test.shape[0])):
            x = ", ".join(f"{v:.2f}" for v in X_test[i])
            print(f"{x}\t{y_test[i,0]:.4f}\t{preds[i]:.4f}\t{abs(y_test[i,0]-preds[i]):.4f}")
    if args.no_interactive:
        return
    if not args.quiet:
        print("interactive mode")
        print(f"enter {args.input_size} numbers separated by spaces or 'quit'")
    while True:
        try:
            s = input("> ").strip()
            if s.lower() == "quit":
                break
            values = [float(x) for x in s.split()]
            if len(values) != args.input_size:
                print(f"enter exactly {args.input_size} numbers")
                continue
            norm = scaler.transform(np.asarray([values], dtype=np.float64))[0]
            pred = model.predict(norm.tolist())
            print(f"prediction {pred:.6f}")
        except ValueError:
            print("invalid input")
        except KeyboardInterrupt:
            break

if __name__ == "__main__":
    run_cli()

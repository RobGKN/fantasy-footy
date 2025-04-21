import torch
from model import BreakoutPredictor
from utils import load_data, calculate_metrics

if __name__ == "__main__":
    # Load test data
    test_data = load_data("data/processed/test_data.csv")

    # Load the trained model
    model = BreakoutPredictor(input_size=10)  # Placeholder input size
    model.load_state_dict(torch.load("saved_models/breakout_model.pth"))
    model.eval()

    # Evaluate the model
    predictions, labels = [], []
    for inputs, label in test_data:
        with torch.no_grad():
            output = model(inputs)
            predictions.append(output)
            labels.append(label)

    # Calculate metrics
    metrics = calculate_metrics(predictions, labels)
    print("Evaluation Metrics:", metrics)
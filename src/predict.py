import torch
from model import BreakoutPredictor
from utils import preprocess_input

if __name__ == "__main__":
    # Load the trained model
    model = BreakoutPredictor(input_size=10)  # Placeholder input size
    model.load_state_dict(torch.load("saved_models/breakout_model.pth"))
    model.eval()

    # Example input data
    new_data = preprocess_input({"player_stats": [1.2, 3.4, 5.6]})  # Placeholder input

    # Make predictions
    with torch.no_grad():
        prediction = model(new_data)
        print("Prediction:", prediction.item())
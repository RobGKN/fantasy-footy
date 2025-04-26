import torch
from src.model import BreakoutPredictorMLP

def test_model_forward_pass():
    model = BreakoutPredictorMLP(input_dim=5)
    sample_input = torch.randn(10, 5)  # batch of 10, 5 features
    output = model(sample_input)
    assert output.shape == (10, 1)
    assert output.min() >= 0 and output.max() <= 1  # Sigmoid output
    assert output.dtype == torch.float32  # Ensure output is float32
    assert not torch.isnan(output).any()  # Ensure no NaN values in output
    assert not torch.isinf(output).any()  # Ensure no Inf values in output
import unittest
from src.utils import calculate_metrics

class TestUtils(unittest.TestCase):
    def test_calculate_metrics(self):
        predictions = [1, 0, 1, 1]
        labels = [1, 0, 1, 0]
        metrics = calculate_metrics(predictions, labels)
        self.assertAlmostEqual(metrics["accuracy"], 0.75)
        self.assertAlmostEqual(metrics["f1"], 0.8)
        self.assertAlmostEqual(metrics["roc_auc"], 0.75)

if __name__ == "__main__":
    unittest.main()
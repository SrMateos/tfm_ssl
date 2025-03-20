import unittest
import torch
from metrics.image_metrics import calculate_metrics

class TestImageMetrics(unittest.TestCase):
    def setUp(self):
        # Create test tensors for predictions and targets
        self.pred_identical = torch.ones((1, 3, 256, 256, 256))
        self.target_identical = torch.ones((1, 3, 256, 256, 256))

        # Create tensors with known differences
        self.pred_different = torch.ones((1, 3, 256, 256, 256))
        self.target_different = torch.zeros((1, 3, 256, 256, 256))

        # Create random tensors
        torch.manual_seed(42)  # For reproducibility
        self.pred_random = torch.rand((1, 3, 256, 256, 256))
        self.target_random = torch.rand((1, 3, 256, 256, 256))

    def test_identical_images(self):
        # If pred and target are identical, MAE should be 0, PSNR should be high, SSIM should be 1
        metrics = calculate_metrics(self.pred_identical, self.target_identical)

        self.assertEqual(metrics["mae"], 0.0)
        self.assertGreaterEqual(metrics["psnr"], 100.0)  # PSNR should be very high for identical images
        self.assertAlmostEqual(metrics["ssim"], 1.0, places=5)  # SSIM should be close to 1

    def test_completely_different_images(self):
        # For completely different images, MAE should be 1, PSNR should be low, SSIM should be low
        metrics = calculate_metrics(self.pred_different, self.target_different)

        self.assertEqual(metrics["mae"], 1.0)

        # PSNR and SSIM are not well-defined for completely different images and should be less than inf and 1, respectively
        self.assertLess(metrics["psnr"], float("inf"))
        self.assertLess(metrics["ssim"], 1.0)

    def test_metrics_dict_structure(self):
        # Test that the returned metrics dictionary has the expected structure
        metrics = calculate_metrics(self.pred_random, self.target_random)

        self.assertIsInstance(metrics, dict)
        self.assertIn("mae", metrics)
        self.assertIn("psnr", metrics)
        self.assertIn("ssim", metrics)

        self.assertIsInstance(metrics["mae"], float)
        self.assertIsInstance(metrics["psnr"], float)
        self.assertIsInstance(metrics["ssim"], float)

if __name__ == '__main__':
    unittest.main()


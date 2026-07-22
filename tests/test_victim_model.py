import unittest
from pathlib import Path

import torch

from attacks.victim_model import VictimCNN, load_victim


REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = REPO_ROOT / "models" / "victim.pt"


class LoadVictimTests(unittest.TestCase):
    def setUp(self) -> None:
        self.backup_path = None
        if WEIGHTS_PATH.exists():
            self.backup_path = WEIGHTS_PATH.with_suffix(".pt.bak")
            WEIGHTS_PATH.replace(self.backup_path)

    def tearDown(self) -> None:
        if WEIGHTS_PATH.exists():
            WEIGHTS_PATH.unlink()
        if self.backup_path and self.backup_path.exists():
            self.backup_path.replace(WEIGHTS_PATH)

    def test_missing_weights_raise_clear_error(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "Victim model not trained yet"):
            load_victim(torch.device("cpu"))

    def test_existing_weights_are_loaded(self) -> None:
        expected = VictimCNN(num_classes=10)
        WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        torch.save(expected.state_dict(), WEIGHTS_PATH)

        loaded = load_victim(torch.device("cpu"))

        self.assertIsInstance(loaded, VictimCNN)
        for name, value in expected.state_dict().items():
            self.assertTrue(torch.equal(loaded.state_dict()[name], value))


if __name__ == "__main__":
    unittest.main()

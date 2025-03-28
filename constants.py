from pathlib import Path

DEBUG = True  # Set to True for quick testing with fewer samples
TASK1 = False

DATA_PATH_TASK1 = Path('data/synthRAD2025_Task1_Train/Task1')
DATA_PATH_TASK1_ABDOMEN = Path(DATA_PATH_TASK1 / 'AB')

DATA_PATH_TASK2 = Path('data/synthRAD2025_Task2_Train/Task2')
DATA_PATH_TASK2_ABDOMEN = Path(DATA_PATH_TASK2 / 'AB')

DATA_PATH = DATA_PATH_TASK1_ABDOMEN if TASK1 else DATA_PATH_TASK2_ABDOMEN

PATCH_SIZE = (64,) * 3
MODEL_PATH = Path("ssl_model.pth")
OUTPUT_MODEL_PATH = Path("cbct_to_ct_model.pth")




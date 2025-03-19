from pathlib import Path

DATA_PATH_TASK1 = Path('data/Task2')
DATA_PATH_TASK1_BRAIN = Path(DATA_PATH_TASK1 / 'brain')
DATA_PATH_TASK1_PELVIS = Path(DATA_PATH_TASK1 / 'pelvis')
PATCH_SIZE = (64,) * 3
MODEL_PATH = Path("ssl_model.pth")
OUTPUT_MODEL_PATH = Path("cbct_to_ct_model.pth")



DEBUG = True  # Set to True for quick testing with fewer samples


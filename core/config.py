import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

print(BASE_DIR)
VERSION = "0.0.1"
LANGUAGE = "en"

MODEL_LABEL_PATHS = {
    "SRTSI_1": {
        "model": os.path.join(BASE_DIR, "models/GRNTI_ONE.pth"),
        "labels": os.path.join(BASE_DIR, "models/mapping/labels_lvl_1.txt")
    },
    "SRTSI_2": {
        "model": os.path.join(BASE_DIR, "models/GRNTI_TWO.pth"),
        "labels": os.path.join(BASE_DIR, "models/mapping/labels_lvl_2.txt")
    },
    "SRTSI_3": {
        "model": os.path.join(BASE_DIR, "models/GRNTI_THREE.pth"),
        "labels": os.path.join(BASE_DIR, "models/mapping/labels_lvl_3.txt")
    }
}

DICT_FILE_PATHS = {
    "lvl_1": os.path.join(BASE_DIR, "dictionaries/first_lvl.txt"),
    "lvl_2": os.path.join(BASE_DIR, "dictionaries/second_lvl.txt")
}
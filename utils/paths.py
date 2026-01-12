from pathlib import Path

def get_data_root():
    if Path("/content/drive").exists():
        return Path("/content/drive/MyDrive/medical_ai_data")
    else:
        return Path(__file__).resolve().parents[1] / "data"

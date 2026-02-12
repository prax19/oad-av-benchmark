import sys
import zipfile
import subprocess
from pathlib import Path

def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)

run([sys.executable, "-m", "pip", "install", "gdown"])
import gdown

road_root = Path("data") / "road"
road_root.mkdir(parents=True, exist_ok=True)

for f in road_root.glob("*.part"):
    f.unlink()

videos_zip = road_root / "videos.zip"
ann_json = road_root / "road_trainval_v1.0.json"
videos_dir = road_root / "videos"

# download
if not videos_zip.exists():
    gdown.download(
        "https://drive.google.com/uc?id=1YQ9ap3o9pqbD0Pyei68rlaMDcRpUn-qz",
        str(videos_zip),
        quiet=False
    )

if not ann_json.exists():
    gdown.download(
        "https://drive.google.com/uc?id=1uoyBiNZq1_SHif1CG_2R6d_pUWwUe7fL",
        str(ann_json),
        quiet=False
    )

# unpack
if not videos_dir.exists() or not any(videos_dir.iterdir()):
    with zipfile.ZipFile(videos_zip, "r") as zf:
        zf.extractall(road_root)

# validation
mp4_count = len(list(videos_dir.glob("*.mp4")))
if mp4_count == 18 and ann_json.exists():
    print("ROAD downloaded successfully!")
    videos_zip.unlink(missing_ok=True)
else:
    raise FileNotFoundError(f"ROAD dataset not complete! mp4_count={mp4_count}")

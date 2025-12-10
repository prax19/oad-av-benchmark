import sys
import subprocess

def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)

py = sys.executable
run([py, "-m", "pip", "install", "--upgrade", "pip"])
run([py, "-m", "pip", "install", "--upgrade", "openmim"])

run([py, "-m", "mim", "install", "mmengine"])
run([py, "-m", "mim", "install", "mmcv>=2.0.0"])
run([py, "-m", "pip", "install", "--upgrade", "mmaction2"])
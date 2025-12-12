import sys
import subprocess

def run(cmd):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True)

py = sys.executable
run([py, "-m", "pip", "install", "--upgrade", "pip"])

run([py, "-m", "pip", "install", "openmim==0.3.9"])
run([py, "-m", "mim", "install", "mmengine==0.10.4"])
run([py, "-m", "mim", "install", "mmcv==2.1.0"])
run([py, "-m", "pip", "install", "git+https://github.com/prax19/mmaction2.git@fix-missing-modules#egg=mmaction2"])
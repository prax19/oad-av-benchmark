import os
import sys
import subprocess

def run(cmd, env=None):
    print("+", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)

py = sys.executable
env = os.environ.copy()
env["PYTHONNOUSERSITE"] = "1"
env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"

run([py, "-m", "pip", "install", "-U", "pip==25.3", "setuptools>=70.2.0,<82", "wheel"], env=env)

run([py, "-m", "pip", "install", "-U",
     "click==8.3.1",
     "rich==13.4.2",
     "tabulate==0.9.0",
     "requests==2.28.2",
     "model-index==0.1.11",
     "PyYAML==6.0.3",
     "tqdm==4.65.2",
     "packaging==24.2",
], env=env)

run([py, "-m", "pip", "install", "--no-deps", "openmim==0.3.9"], env=env)

run([py, "-m", "pip", "install", "-r", "requirements.txt"], env=env)

run([py, "-m", "pip", "install", "mmengine==0.10.4"], env=env)

run([py, "-m", "pip", "install", "mmcv-lite==2.1.0"], env=env)

run([py, "-m", "pip", "install",
     "git+https://github.com/prax19/mmaction2.git@fix-missing-modules#egg=mmaction2"], env=env)

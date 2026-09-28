import subprocess
import sys
from pathlib import Path

INPUT = Path(r"F:\srujith backup\Fun\my research\Music dismantling\training data set\test\testsong.m4a")
OUTPUT = Path("separated")

OUTPUT.mkdir(exist_ok=True)

cmd = [
    sys.executable, "-m", "demucs",
    "--two-stems=vocals",
    "-o", str(OUTPUT),
    str(INPUT)
]

print("Running:")
print(" ".join(cmd))

subprocess.run(cmd, check=True)

print("\nDone.")
print(f"Output saved in: {OUTPUT}")
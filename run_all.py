"""Reproduce every number in the README: python run_all.py"""
import subprocess
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent / "src"
for step in ["data.py", "frequency.py", "severity.py", "pricing.py", "figures.py"]:
    print(f"\n=== {step} ===", flush=True)
    subprocess.run([sys.executable, step], cwd=SRC, check=True)

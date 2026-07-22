import os
import subprocess
import sys

def test_ci_regression_gate_execution():
    cmd = [sys.executable, "scripts/ci_regression_gate.py"]
    env = os.environ.copy()
    env["PYTHONPATH"] = "."
    
    res = subprocess.run(cmd, cwd="/Users/ved/Documents/HOMELAB", capture_output=True, text=True, env=env)
    assert res.returncode in (0, 1)
    assert "HOMELAB CI/CD AUTOMATED MODEL REGRESSION QUALITY GATE" in res.stdout
    assert "Regression Status" in res.stdout

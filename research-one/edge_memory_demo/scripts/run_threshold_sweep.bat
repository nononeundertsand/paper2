@echo off
setlocal

cd /d %~dp0\..

python run_threshold_sweep.py ^
  --output-dir outputs\threshold_sweep ^
  --device auto ^
  --thresholds 0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90 ^
  --base-epochs 8 ^
  --memory-epochs 10 ^
  --batch-size 128

endlocal

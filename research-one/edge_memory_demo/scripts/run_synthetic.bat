@echo off
setlocal

cd /d %~dp0\..

python run_synthetic.py ^
  --output-dir outputs\synthetic_run ^
  --device auto ^
  --base-epochs 8 ^
  --memory-epochs 10 ^
  --batch-size 128

endlocal

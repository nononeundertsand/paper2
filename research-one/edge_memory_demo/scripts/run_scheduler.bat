@echo off
setlocal

cd /d %~dp0\..

python run_scheduler.py ^
  --output-dir outputs\scheduler_run ^
  --num-items 400 ^
  --storage-budget 120 ^
  --energy-budget 160

endlocal

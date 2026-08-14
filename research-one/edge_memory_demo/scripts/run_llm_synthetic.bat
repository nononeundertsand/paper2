@echo off
setlocal

cd /d %~dp0\..

if "%~1"=="" (
  echo Usage: scripts\run_llm_synthetic.bat MODEL_NAME_OR_LOCAL_PATH
  echo Example: scripts\run_llm_synthetic.bat Qwen/Qwen2.5-0.5B-Instruct
  exit /b 1
)

python run_llm_synthetic.py ^
  --model-name-or-path "%~1" ^
  --output-dir outputs\llm_synthetic ^
  --device auto ^
  --fp16 ^
  --feature-batch-size 16 ^
  --batch-size 64 ^
  --base-epochs 8 ^
  --memory-epochs 10 ^
  --thresholds 0.10,0.20,0.30,0.40,0.50,0.60,0.70,0.80,0.90

endlocal

@echo off
setlocal

cd /d %~dp0\..

if "%~1"=="" (
  echo Usage: scripts\run_llm_capacity_sweep.bat MODEL_NAME_OR_LOCAL_PATH
  echo Example: scripts\run_llm_capacity_sweep.bat D:\models\Qwen2.5-0.5B-Instruct
  exit /b 1
)

for %%F in (60 100 160 240) do (
  echo ===== LLM capacity sweep: num_facts=%%F =====
  python run_llm_synthetic.py ^
    --model-name-or-path "%~1" ^
    --output-dir outputs\llm_capacity_facts_%%F ^
    --device auto ^
    --fp16 ^
    --num-facts %%F ^
    --base-train-size 4000 ^
    --memory-train-size 6000 ^
    --test-size 1200 ^
    --base-epochs 20 ^
    --memory-epochs 30 ^
    --batch-size 64 ^
    --feature-batch-size 16 ^
    --learning-rate 3e-3 ^
    --thresholds 0.30,0.50,0.70
)

endlocal

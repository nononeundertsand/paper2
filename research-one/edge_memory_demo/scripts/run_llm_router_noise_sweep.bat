@echo off
setlocal

cd /d %~dp0\..

if "%~1"=="" (
  echo Usage: scripts\run_llm_router_noise_sweep.bat MODEL_NAME_OR_LOCAL_PATH
  echo Example: scripts\run_llm_router_noise_sweep.bat D:\models\Qwen2.5-0.5B-Instruct
  exit /b 1
)

for %%N in (0.0 0.1 0.2 0.3) do (
  echo ===== LLM router noise sweep: router_label_noise=%%N =====
  python run_llm_synthetic.py ^
    --model-name-or-path "%~1" ^
    --output-dir outputs\llm_router_noise_%%N ^
    --device auto ^
    --fp16 ^
    --num-facts 60 ^
    --base-train-size 4000 ^
    --memory-train-size 6000 ^
    --test-size 1200 ^
    --base-epochs 20 ^
    --memory-epochs 30 ^
    --batch-size 64 ^
    --feature-batch-size 16 ^
    --learning-rate 3e-3 ^
    --router-label-noise %%N ^
    --thresholds 0.30,0.50,0.70
)

endlocal

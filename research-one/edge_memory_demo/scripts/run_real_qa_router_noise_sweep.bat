@echo off
setlocal

cd /d %~dp0\..

if "%~1"=="" (
  echo Usage: scripts\run_real_qa_router_noise_sweep.bat MODEL_NAME_OR_LOCAL_PATH
  echo Example: scripts\run_real_qa_router_noise_sweep.bat D:\models\Qwen2.5-0.5B-Instruct
  exit /b 1
)

for %%N in (0.0 0.1 0.2 0.3) do (
  echo ===== Real QA router noise sweep: router_label_noise=%%N =====
  python run_real_qa.py ^
    --model-name-or-path "%~1" ^
    --output-dir outputs\e2e\real_qa_router_noise_%%N ^
    --dataset-name squad ^
    --dataset-split train ^
    --general-source hf ^
    --general-dataset-name ag_news ^
    --general-dataset-split train ^
    --general-text-field text ^
    --general-label-field label ^
    --device auto ^
    --fp16 ^
    --num-facts 120 ^
    --base-train-size 4000 ^
    --memory-train-size 6000 ^
    --test-size 1200 ^
    --base-epochs 20 ^
    --memory-epochs 30 ^
    --num-experts 8 ^
    --top-k-experts 2 ^
    --batch-size 64 ^
    --feature-batch-size 16 ^
    --learning-rate 3e-3 ^
    --router-label-noise %%N ^
    --thresholds 0.30,0.50,0.70
)

endlocal

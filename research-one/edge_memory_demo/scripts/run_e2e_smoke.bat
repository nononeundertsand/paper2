@echo off
setlocal

cd /d %~dp0\..

if "%~1"=="" (
  echo Usage: scripts\run_e2e_smoke.bat MODEL_NAME_OR_LOCAL_PATH
  echo Example: scripts\run_e2e_smoke.bat D:\models\Qwen2.5-0.5B-Instruct
  exit /b 1
)

echo ===== E2E smoke test: real QA + real general, small scale =====
python run_real_qa.py ^
  --model-name-or-path "%~1" ^
  --output-dir outputs\e2e_smoke\real_qa_smoke ^
  --dataset-name squad ^
  --dataset-split train ^
  --general-source hf ^
  --general-dataset-name ag_news ^
  --general-dataset-split train ^
  --general-text-field text ^
  --general-label-field label ^
  --device auto ^
  --fp16 ^
  --num-facts 40 ^
  --base-train-size 600 ^
  --memory-train-size 900 ^
  --test-size 300 ^
  --base-epochs 3 ^
  --memory-epochs 5 ^
  --num-experts 4 ^
  --top-k-experts 1 ^
  --batch-size 64 ^
  --feature-batch-size 16 ^
  --learning-rate 3e-3 ^
  --thresholds 0.30,0.50,0.70

python collect_results.py --input-dir outputs\e2e_smoke --output outputs\e2e_smoke\summary\smoke_summary.csv

echo Smoke test finished. Summary: outputs\e2e_smoke\summary\smoke_summary.csv
endlocal

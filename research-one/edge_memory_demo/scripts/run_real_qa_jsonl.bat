@echo off
setlocal

cd /d %~dp0\..

if "%~2"=="" (
  echo Usage: scripts\run_real_qa_jsonl.bat MODEL_NAME_OR_LOCAL_PATH QA_JSONL_PATH
  echo Example: scripts\run_real_qa_jsonl.bat D:\models\Qwen2.5-0.5B-Instruct D:\data\qa.jsonl
  exit /b 1
)

python run_real_qa.py ^
  --model-name-or-path "%~1" ^
  --local-jsonl "%~2" ^
  --output-dir outputs\real_qa_jsonl ^
  --question-field question ^
  --answer-field answers ^
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
  --thresholds 0.30,0.50,0.70

endlocal

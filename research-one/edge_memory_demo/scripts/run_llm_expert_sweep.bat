@echo off
setlocal

cd /d %~dp0\..

if "%~1"=="" (
  echo Usage: scripts\run_llm_expert_sweep.bat MODEL_NAME_OR_LOCAL_PATH
  echo Example: scripts\run_llm_expert_sweep.bat D:\models\Qwen2.5-0.5B-Instruct
  exit /b 1
)

call :run_case 4 1 "%~1"
call :run_case 8 1 "%~1"
call :run_case 8 2 "%~1"
call :run_case 16 2 "%~1"
call :run_case 16 4 "%~1"
goto :done

:run_case
set NUM_EXPERTS=%~1
set TOP_K=%~2
set MODEL_PATH=%~3
echo ===== LLM expert sweep: num_experts=%NUM_EXPERTS%, top_k=%TOP_K% =====
python run_llm_synthetic.py ^
  --model-name-or-path "%MODEL_PATH%" ^
  --output-dir outputs\llm_experts_%NUM_EXPERTS%_topk_%TOP_K% ^
  --device auto ^
  --fp16 ^
  --num-facts 60 ^
  --num-experts %NUM_EXPERTS% ^
  --top-k-experts %TOP_K% ^
  --base-train-size 4000 ^
  --memory-train-size 6000 ^
  --test-size 1200 ^
  --base-epochs 20 ^
  --memory-epochs 30 ^
  --batch-size 64 ^
  --feature-batch-size 16 ^
  --learning-rate 3e-3 ^
  --thresholds 0.30,0.50,0.70
exit /b

:done
endlocal

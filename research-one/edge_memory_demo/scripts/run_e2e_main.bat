@echo off
setlocal

cd /d %~dp0\..

if "%~1"=="" (
  echo Usage: scripts\run_e2e_main.bat MODEL_NAME_OR_LOCAL_PATH
  echo Example: scripts\run_e2e_main.bat D:\models\Qwen2.5-0.5B-Instruct
  exit /b 1
)

echo ===== E2E main: real QA multi-seed =====
call scripts\run_real_qa_multiseed.bat "%~1"
if errorlevel 1 exit /b %errorlevel%

echo ===== E2E main: resource-aware write scheduler =====
python run_scheduler.py --output-dir outputs\e2e\scheduler_main
if errorlevel 1 exit /b %errorlevel%

echo ===== E2E main: collect results =====
python collect_results.py ^
  --input-dir outputs\e2e ^
  --output outputs\e2e\summary\e2e_main_summary.csv ^
  --include-scheduler
if errorlevel 1 exit /b %errorlevel%

python aggregate_results.py ^
  --input outputs\e2e\summary\e2e_main_summary.csv ^
  --output outputs\e2e\summary\e2e_main_aggregate.csv
if errorlevel 1 exit /b %errorlevel%

echo E2E main finished. Summary: outputs\e2e\summary\e2e_main_summary.csv
echo Aggregate: outputs\e2e\summary\e2e_main_aggregate.csv
endlocal

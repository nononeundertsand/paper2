@echo off
setlocal

cd /d %~dp0\..

python collect_results.py ^
  --input-dir outputs ^
  --output outputs\summary\llm_results_summary.csv

endlocal

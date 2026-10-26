@echo off
REM ============================================================================
REM HPC Duplicate Detection Runner Script (Windows)
REM ============================================================================
REM Usage:
REM   run_hpc.bat exp1                              # Run Experiment 1 (Band Sweep)
REM   run_hpc.bat exp2                              # Run Experiment 2 (Retrieval Benchmark)
REM   run_hpc.bat exp3                              # Run Experiment 3 (4-Way Latency)
REM   run_hpc.bat all_experiments                   # Run Exp 1, 2, and 3 sequentially
REM   run_hpc.bat all --num-samples 102000000       # Full generation + evaluation
REM ============================================================================

setlocal enabledelayedexpansion

set STAGE=%1
if "%STAGE%"=="" set STAGE=all_experiments

shift
set EXTRA_ARGS=
:parse_args
if "%~1"=="" goto done_args
set EXTRA_ARGS=!EXTRA_ARGS! %1
shift
goto parse_args
:done_args

echo ================================================================================
echo            HPC DUPLICATE DETECTION LAUNCHER (Windows)
echo ================================================================================
echo Pipeline Stage:   %STAGE%
echo Extra Arguments:  %EXTRA_ARGS%
echo ================================================================================

python scripts\run_hpc_pipeline.py --stage %STAGE% %EXTRA_ARGS%
if errorlevel 1 (
    echo [ERROR] Pipeline failed with error code %errorlevel%.
    exit /b %errorlevel%
)
echo [SUCCESS] Pipeline execution finished cleanly.

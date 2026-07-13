@echo off
REM qwen36-arena setup (Windows): create the shared model volume.
docker volume create qwen36-hf >nul
echo.
echo Done - created the qwen36-hf model volume. Next steps:
echo   qwen36.cmd download          pre-fetch the Qwen3.6 lanes (~130 GB all-in, one-time)
echo   qwen36.cmd nvfp4 27b mtp 4   serve the tuned NVFP4 lane   -^> http://localhost:8000/v1
echo   qwen36.cmd dash              race dashboard               -^> http://localhost:8870
echo   qwen36.cmd bench             scripted battery vs the running lane
echo.
echo Reproduce the depth-sweep finding:  python rerun_depth_sweep.py  ^&^&  python analyze_rerun.py

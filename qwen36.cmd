@echo off
REM Qwen3.6 NVFP4-vs-GGUF A/B rig (RTX 5090, WSL2/Docker) - sibling of DSpark\vllm.cmd.
REM Three lanes, one served name ("qwen36"), one measuring stick (dash/bench, client-side):
REM   qwen36.cmd nvfp4 [27b^|35b] [mtp N]   Unsloth NVFP4 W4A4   -> vLLM      http://localhost:8000/v1
REM   qwen36.cmd w4a16 [27b]      [mtp N]   NVIDIA NVFP4 W4A16   -> vLLM      http://localhost:8000/v1  (the 2.5x claim's own baseline)
REM   qwen36.cmd gguf  [27b^|35b] [mtp N]   Unsloth UD-Q4_K_XL   -> llama.cpp http://localhost:8872/v1
REM   qwen36.cmd dash                       race dashboard       -> http://localhost:8870
REM   qwen36.cmd bench [args]               scripted battery vs the running lane (writes results\bench-results.json)
REM   qwen36.cmd download                   pre-fetch every lane's weights into the qwen36-hf volume
REM   qwen36.cmd status ^| stop
REM One model on the GPU at a time (27B pairs don't co-fit in 32 GB) - the dashboard
REM records each race and Replay races the recordings side-by-side, DSpark-style.
REM mtp N = multi-token-prediction speculative decoding (the solo-speed lever, try 1..6):
REM   vLLM lanes: --speculative-config mtp; GGUF lane: the *-MTP-GGUF file + --spec-type draft-mtp.
REM Ready when vLLM prints "Application startup complete" / llama.cpp prints "server is listening".
setlocal EnableDelayedExpansion
set VLLM_IMG=vllm/vllm-openai:nightly
set LCPP_IMG=ghcr.io/ggml-org/llama.cpp:server-cuda
REM ROOT = this script's own folder (portable; no hardcoded path). Strip trailing backslash.
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"
REM VLLM_WSL2_ENABLE_PIN_MEMORY: Docker Desktop = WSL2, where vLLM disables pinned
REM memory by default but the GPU worker needs it (dies with "UVA is not available").
set VCOMMON=--rm --name qwen36-vllm --gpus all --ipc=host -p 8000:8000 -v qwen36-hf:/hf -e HF_HOME=/hf -e VLLM_WSL2_ENABLE_PIN_MEMORY=1
REM 5090 gotchas baked in (see README): fp8 KV needs --max-num-batched-tokens 8192 on
REM this hybrid-attention arch; >0.94 gpu-mem-util crashes the card; 262K ctx won't fit 32 GB.
REM Qwen3.6 emits Qwen-Coder XML tool calls (<function=name><parameter=k>), NOT Hermes
REM JSON. --tool-call-parser hermes (the DSpark/Qwen3-8B setting) silently fails to parse
REM them -> finish_reason 'stop' + raw text, so agents like Pi never see the tool call.
REM Verified 2026-07-13: qwen3_coder parses it; hermes does not. (qwen3_xml = same parser.)
REM --max-num-seqs 64: Qwen3.6's hybrid Mamba cache fits ~121 decode seqs at this VRAM;
REM vLLM's default 256 overflows it -> "exceeds available Mamba cache blocks" crash at
REM CUDA-graph capture (bites non-MTP lanes; MTP auto-lowers it). 64 is ample single-user.
set VFLAGS=--served-model-name qwen36 --kv-cache-dtype fp8 --max-num-batched-tokens 8192 --max-num-seqs 64 --enable-auto-tool-choice --tool-call-parser qwen3_coder --default-chat-template-kwargs "{\"enable_thinking\": false}"

set LANE=%1
set SIZE=%2
if /I "%SIZE%"=="mtp" ( set SIZE=27b& set MTPKW=%2& set MTPN=%3 ) else ( set MTPKW=%3& set MTPN=%4 )
if "%SIZE%"=="" set SIZE=27b
set MTP=0
if /I "%MTPKW%"=="mtp" if not "%MTPN%"=="" set MTP=%MTPN%

if /I "%LANE%"=="nvfp4" (
  if /I "%SIZE%"=="35b" ( set MODEL=unsloth/Qwen3.6-35B-A3B-NVFP4-Fast) else ( set MODEL=unsloth/Qwen3.6-27B-NVFP4)
  set QUANT=W4A4
  goto :vllm_lane
)
if /I "%LANE%"=="w4a16" (
  set MODEL=nvidia/Qwen3.6-27B-NVFP4
  set SIZE=27b
  set QUANT=W4A16
  goto :vllm_lane
)
if /I "%LANE%"=="gguf" goto :gguf_lane
if /I "%LANE%"=="dash" goto :dash
if /I "%LANE%"=="bench" goto :bench
if /I "%LANE%"=="download" goto :download
if /I "%LANE%"=="status" goto :status
if /I "%LANE%"=="stop" goto :stop
echo usage: qwen36.cmd [nvfp4 ^| w4a16 ^| gguf] [27b ^| 35b] [mtp N]
echo        qwen36.cmd [dash ^| bench ^| download ^| status ^| stop]
goto :eof

:vllm_lane
if /I "%SIZE%"=="35b" ( set CTX=98304& set GMU=0.94) else ( set CTX=65536& set GMU=0.92)
set SPEC=
if not "%MTP%"=="0" set SPEC=--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":%MTP%}"
REM 35B-A3B MoE on consumer SM120: vLLM's backend picker only knows data-center
REM Blackwell and silently falls back to slow Marlin (vllm#33416) - force FlashInfer.
set MOE=
set ARCHENV=
if /I "%SIZE%"=="35b" ( set MOE=--moe-backend flashinfer& set ARCHENV=-e FLASHINFER_CUDA_ARCH_LIST=12.0)
call :write_lane %LANE% vllm !MODEL! %SIZE% %MTP% 8000 !QUANT!
echo.
echo Lane %LANE% %SIZE% mtp=%MTP%  --  !MODEL!
echo   vLLM on http://localhost:8000/v1 (served name: qwen36) - ready on "Application startup complete".
echo   First start compiles kernels for a few minutes. Ctrl+C stops it.
echo.
REM force-clean any lagging container of this name (--rm cleanup can race a fast re-serve)
docker rm -f qwen36-vllm >nul 2>&1
docker run %VCOMMON% !ARCHENV! %VLLM_IMG% --model !MODEL! %VFLAGS% --max-model-len !CTX! --gpu-memory-utilization !GMU! !MOE! !SPEC!
goto :eof

:gguf_lane
if not "%MTP%"=="0" ( set GDIR=%SIZE%-mtp) else ( set GDIR=%SIZE%-std)
if /I "%SIZE%"=="35b" ( set GBASE=unsloth/Qwen3.6-35B-A3B) else ( set GBASE=unsloth/Qwen3.6-27B)
if not "%MTP%"=="0" ( set GREPO=!GBASE!-MTP-GGUF) else ( set GREPO=!GBASE!-GGUF)
set GSPEC=
if not "%MTP%"=="0" set GSPEC=--spec-type draft-mtp --spec-draft-n-max %MTP%
call :write_lane gguf llamacpp !GREPO! %SIZE% %MTP% 8872 UD-Q4_K_XL
echo.
echo Lane gguf %SIZE% mtp=%MTP%  --  /hf/gguf/!GDIR! (UD-Q4_K_XL)
echo   llama.cpp on http://localhost:8872/v1 (served name: qwen36) - ready on "server is listening".
echo   Ctrl+C stops it.
echo.
docker rm -f qwen36-gguf >nul 2>&1
docker run --rm --name qwen36-gguf --gpus all -p 8872:8080 -v qwen36-hf:/hf --entrypoint /bin/bash %LCPP_IMG% -c "FIRST=$(find /hf/gguf/!GDIR! -name '*.gguf' | sort | head -1); echo serving $FIRST; exec /app/llama-server -m $FIRST --alias qwen36 --host 0.0.0.0 --port 8080 -ngl 99 -c 65536 --jinja %GSPEC%"
goto :eof

:dash
echo Dashboard: http://localhost:8870  (Ctrl+C stops it)
python "%~dp0dash\serve.py"
goto :eof

:bench
python "%~dp0bench.py" %2 %3 %4 %5 %6 %7 %8 %9
goto :eof

:download
docker run --rm --name qwen36-dl -v qwen36-hf:/hf -v %ROOT%:/workspace -e HF_HOME=/hf python:3.11-slim bash -c "pip install -q --no-cache-dir 'huggingface_hub>=0.26' && python /workspace/download_models.py"
goto :eof

:status
docker ps --filter name=qwen36 --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
echo.
curl -s -m 2 http://localhost:8000/v1/models && echo   ^<- vLLM lane :8000 || echo vLLM lane :8000 - down
curl -s -m 2 http://localhost:8872/v1/models && echo   ^<- llama.cpp lane :8872 || echo llama.cpp lane :8872 - down
if exist "%~dp0results\current-lane.json" type "%~dp0results\current-lane.json"
goto :eof

:stop
docker rm -f qwen36-vllm 2>nul
docker rm -f qwen36-gguf 2>nul
goto :eof

:write_lane
if not exist "%~dp0results" mkdir "%~dp0results"
> "%~dp0results\current-lane.json" echo {"lane": "%1", "engine": "%2", "model": "%3", "size": "%4", "mtp": %5, "port": %6, "base_url": "http://localhost:%6/v1", "served_name": "qwen36", "quant": "%~7", "started": "%DATE% %TIME%"}
goto :eof

"""Shared measuring stick for every lane (vLLM :8000 / llama.cpp :8872).

Both engines are treated as identical black boxes: an OpenAI-compatible
/v1/chat/completions stream, timed client-side. That is what makes the
cross-engine numbers comparable — no engine's self-reported throughput is used
except the final authoritative *token count* (usage.completion_tokens, or
llama.cpp's timings.predicted_n), never its clock.

Stats per run:
  ttft        s   first content byte after send (run 1 on a fresh serve = cold;
                  later repeats may hit both engines' prompt caches)
  decode_tps  t/s (tokens-1) / (t_last - t_first) — cache-immune, the headline
  wall_tps    t/s tokens / (t_last - t_send) — includes prefill (DSpark method)
"""
import json
import time
import urllib.request


PRESETS = {   # the DSpark video's three tasks, verbatim — continuity across videos
    "code": "Write a Python function that returns the longest palindromic substring "
            "of a given string. Include a brief explanation of the algorithm and one example.",
    "math": "A bakery sells muffins for $2.50 each and cookies for $1.25 each. On Monday "
            "it sold 48 muffins and twice as many cookies as muffins. How much money did "
            "the bakery make in total on Monday? Show your reasoning step by step.",
    "chat": "What are the main pros and cons of remote work for early-career software "
            "engineers? Give a balanced, conversational answer.",
}


def lane_key(lane):
    """'nvfp4-27b' / 'gguf-27b-mtp2' — the pane/row identity for a lane config."""
    k = f"{lane['lane']}-{lane['size']}"
    if lane.get("mtp"):
        k += f"-mtp{lane['mtp']}"
    return k


def _direct(url):
    """localhost -> 127.0.0.1: Windows resolvers try ::1 first and can burn ~2s
    on a failed IPv6 connect — poison for a TTFT measurement."""
    return url.replace("//localhost", "//127.0.0.1")


def probe(base_url, timeout=2.0):
    """Return the served model list, or None if the lane is down."""
    try:
        with urllib.request.urlopen(f"{_direct(base_url)}/models", timeout=timeout) as r:
            return [m["id"] for m in json.load(r).get("data", [])]
    except Exception:
        return None


def stream_race(base_url, prompt, max_tokens=256, temperature=0.0, model="qwen36",
                on_snapshot=None, timeout=600):
    """Run one streamed generation; return the final stats dict.

    on_snapshot(elapsed_s, text, tokens_est, tps_est) fires per parsed chunk
    (caller throttles UI pushes; the timeline is what Replay animates).
    """
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }
    req = urllib.request.Request(
        f"{_direct(base_url)}/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy-key"},
    )
    text = ""
    chunks = 0            # content-delta count = live token estimate
    usage_tokens = None   # authoritative count from the engine's final chunk
    t_send = time.perf_counter()
    t_first = None
    t_last = None
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage_tokens = obj["usage"].get("completion_tokens", usage_tokens)
            if obj.get("timings"):  # llama.cpp final chunk
                usage_tokens = obj["timings"].get("predicted_n", usage_tokens)
            for ch in obj.get("choices", []):
                delta = ch.get("delta", {})
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                    text += piece
                    chunks += 1
                    if on_snapshot:
                        el = now - t_send
                        decode_el = now - t_first
                        tps = (chunks - 1) / decode_el if decode_el > 0.05 else 0.0
                        on_snapshot(el, text, chunks, tps)
    t_done = time.perf_counter()
    if t_first is None:   # zero content came back
        raise RuntimeError("stream returned no content (is the lane still loading?)")
    tokens = usage_tokens if usage_tokens else chunks
    decode_dt = (t_last - t_first) or 1e-9
    return {
        "tokens": tokens,
        "tokens_estimated": usage_tokens is None,
        "chunks": chunks,
        "ttft": round(t_first - t_send, 4),
        "decode_tps": round((tokens - 1) / decode_dt, 2),
        "wall_tps": round(tokens / (t_last - t_send), 2),
        "elapsed": round(t_last - t_send, 4),
        "text": text,
    }

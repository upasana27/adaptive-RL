"""OpenAI-compatible chat client for any OpenAI-compatible chat-completions endpoint.

NOT the Anthropic SDK — the gateway speaks the OpenAI chat-completions schema.
Credentials come from the environment (never hardcoded, never logged):
    export LLM_BASE_URL=... LLM_API_KEY=...

Uses stdlib urllib so no new packages are added to the conda env.
"""
import json, os, time, urllib.request, urllib.error, threading


class LLMError(RuntimeError):
    pass


class ChatClient:
    def __init__(self, model, base_url=None, api_key=None, temperature=0.0,
                 max_tokens=8192, timeout=180, max_retries=4, log_path=None,
                 default_extra_body=None):
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or "").rstrip("/")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or os.environ.get("OPENAI_API_KEY")
        if not self.base_url or not self.api_key:
            raise LLMError("Set LLM_BASE_URL and LLM_API_KEY (environment variables)")
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.timeout, self.max_retries = timeout, max_retries
        self.log_path = log_path
        # e.g. {"chat_template_kwargs": {"enable_thinking": False}} for Qwen3 models,
        # whose native thinking loops until max_tokens on this task.
        self.default_extra_body = default_extra_body or {}
        self._lock = threading.Lock()
        # running accounting
        self.n_calls = self.n_fail = self.prompt_tokens = self.completion_tokens = 0
        self.total_seconds = 0.0
        self.n_unparseable = 0
        self.n_truncated = 0

    # ---- accounting -------------------------------------------------------
    def stats(self):
        return dict(model=self.model, calls=self.n_calls, failures=self.n_fail,
                    unparseable=self.n_unparseable, truncated=self.n_truncated,
                    prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens,
                    seconds=round(self.total_seconds, 1),
                    sec_per_call=round(self.total_seconds / max(1, self.n_calls), 2))

    # ---- core -------------------------------------------------------------
    def chat(self, system, user, extra_body=None):
        """Return (text, reasoning_text). Retries transient failures with backoff."""
        payload = {"model": self.model,
                   "messages": [{"role": "system", "content": system},
                                {"role": "user", "content": user}],
                   "temperature": self.temperature,
                   "max_tokens": self.max_tokens}
        payload.update(self.default_extra_body)
        if extra_body:
            payload.update(extra_body)
        body = json.dumps(payload).encode()
        last = None
        for attempt in range(self.max_retries):
            req = urllib.request.Request(
                f"{self.base_url}/chat/completions", data=body,
                headers={"Authorization": f"Bearer {self.api_key}",
                         "Content-Type": "application/json"})
            t0 = time.time()
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    d = json.load(r)
                dt = time.time() - t0
                msg = d["choices"][0]["message"]
                text = (msg.get("content") or "").strip()
                reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
                finish = d["choices"][0].get("finish_reason")
                # Reasoning models can spend the whole budget thinking and return EMPTY
                # content with finish_reason="length". Silently that becomes a NOOP every
                # step, which looks like a capability result but is a token-budget bug.
                if not text and finish == "length":
                    with self._lock:
                        self.n_truncated += 1
                u = d.get("usage") or {}
                with self._lock:
                    self.n_calls += 1
                    self.total_seconds += dt
                    self.prompt_tokens += u.get("prompt_tokens") or 0
                    self.completion_tokens += u.get("completion_tokens") or 0
                self._log(dict(t=round(dt, 2), usage=u, text=text[:400],
                               reasoning=reasoning[:400] if reasoning else ""))
                return text, reasoning
            except urllib.error.HTTPError as e:
                detail = e.read()[:200].decode(errors="replace")
                last = f"HTTP {e.code}: {detail}"
                # 4xx other than 429 are not retryable
                if e.code not in (408, 409, 429) and e.code < 500:
                    break
            except Exception as e:                       # timeouts, conn resets
                last = f"{type(e).__name__}: {str(e)[:200]}"
            time.sleep(min(2 ** attempt, 16))
        with self._lock:
            self.n_fail += 1
        self._log(dict(error=last))
        raise LLMError(last)

    def _log(self, rec):
        if not self.log_path:
            return
        rec["ts"] = time.time()
        rec["model"] = self.model
        with self._lock:
            with open(self.log_path, "a") as f:
                f.write(json.dumps(rec) + "\n")

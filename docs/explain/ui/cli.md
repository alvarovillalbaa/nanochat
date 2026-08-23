## **CLI & Web UI Architecture**

### **CLI Chat** (`scripts/chat_cli.py`)

**Flow**:
1. **Initialize**: Load model → create Engine → get special tokens
2. **State machine**: Maintain `conversation_tokens` list starting with `<|bos|>`
3. **Interactive loop**:
   - User types input → wrapped in `<|user_start|>` ... `<|user_end|>`
   - Append `<|assistant_start|>` → trigger generation
   - Stream tokens one-by-one to console
   - Auto-append `<|assistant_end|>` if max_tokens reached
   - Update conversation state with full response

**Token Management**:
```python
conversation_tokens = [bos]

# User turn
conversation_tokens.append(user_start)
conversation_tokens.extend(tokenizer.encode(user_input))
conversation_tokens.append(user_end)

# Assistant turn (streaming)
conversation_tokens.append(assistant_start)
for token_column, _ in engine.generate(conversation_tokens, ...):
    token = token_column[0]  # batch_size=1
    print(tokenizer.decode([token]), end="", flush=True)
    response_tokens.append(token)
conversation_tokens.extend(response_tokens)
```

**Commands**:
- `clear` → reset to `[bos]` (fresh conversation)
- `quit`/`exit` → terminate
- `-p "prompt"` → single-shot mode (no loop)

**Key Design Choice**: The entire conversation history stays in
`conversation_tokens`. Each turn rebuilds a fresh KV cache from the longer
prompt, so long conversations are slower.

---

### **Web UI** (`scripts/chat_web.py`)

Much more sophisticated—**FastAPI server with multi-GPU worker pool**.

#### **Architecture**

**1. Worker Pool** (Data Parallelism)
```python
class Worker:
    gpu_id: int
    device: torch.device
    engine: Engine
    tokenizer: object
    autocast_ctx: torch.amp.autocast

class WorkerPool:
    workers: List[Worker]
    available_workers: asyncio.Queue  # FIFO queue
```

- Each GPU loads a **full replica** of the model
- Requests distributed round-robin to available workers
- If all busy → request waits in queue
- After completion → worker released back to pool

**2. Endpoints**

| Route | Method | Purpose |
|-------|--------|---------|
| `/` | GET | Serve HTML chat UI (`nanochat/ui.html`) |
| `/logo.svg` | GET | Serve favicon/logo |
| `/chat/completions` | POST | Streaming chat API |
| `/health` | GET | Worker pool status |
| `/stats` | GET | GPU utilization stats |

**3. Streaming Protocol** (Server-Sent Events)

```python
async def generate_stream(worker, tokens, ...):
    for token_column, _ in worker.engine.generate(...):
        token = token_column[0]
        accumulated_tokens.append(token)

        # Decode accumulated to handle multi-byte UTF-8
        current_text = tokenizer.decode(accumulated_tokens)

        # Only emit if no replacement char (incomplete UTF-8)
        if not current_text.endswith('�'):
            new_text = current_text[len(last_clean_text):]
            yield f"data: {json.dumps({'token': new_text, 'gpu': worker.gpu_id})}\n\n"

    yield f"data: {json.dumps({'done': True})}\n\n"
```

**Why accumulate tokens?** Emojis and special chars span multiple tokens. Decoding incrementally causes `�` artifacts. Solution: decode full sequence, emit only new clean text.

**4. Abuse Prevention**

```python
MAX_MESSAGES_PER_REQUEST = 500
MAX_MESSAGE_LENGTH = 8000
MAX_TOTAL_CONVERSATION_LENGTH = 32000
Temperature: [0.0, 2.0]
Top-k: [1, 200]
Max tokens: [1, 4096]
```

Validates before generation to prevent DoS/OOM.

**5. Logging**

Every request logs to console:
```
====================
[USER]: Why is the sky blue?
--------------------
[ASSISTANT] (GPU 0): The sky appears blue due to...
====================
```

---

### **HTML UI** (`nanochat/ui.html`)

Simple single-file chat interface:
- **Fetch API** → POST to `/chat/completions`
- streamed `fetch()` response parsing for SSE-formatted chunks
- Message history in DOM
- Loading states, error handling
- Mobile-responsive

---

### **Key Differences**

| Feature | CLI | Web |
|---------|-----|-----|
| **Concurrency** | Single-threaded | Multi-GPU worker pool |
| **UI** | Terminal | Browser |
| **Streaming** | `print()` per token | SSE (Server-Sent Events) |
| **State** | In-memory list | Stateless (client sends full history) |
| **UTF-8 handling** | Naive | Smart (accumulates tokens) |
| **Abuse protection** | None | Request validation |

The request limits are resource bounds, not a production security layer. The
server has no authentication or rate limiting, logs conversation content, and
configures permissive CORS. Keep it on a trusted network unless those controls
are added by a deployment boundary.

---

### **Why the UTF-8 Dance?**

Tokenizers split text into byte sequences. A single emoji like 🚀 might be:
- Token 1: `F0 9F`
- Token 2: `9A 80`

If you decode token-by-token:
```python
decode([token1])  # '�' (incomplete UTF-8)
decode([token2])  # '�' (incomplete UTF-8)
```

But decode together:
```python
decode([token1, token2])  # '🚀'
```

**Web solution**: Accumulate all tokens → decode full sequence → emit only new text since last clean decode. CLI just naively prints (works for ASCII, breaks for emojis).

---

### **Launching**

**CLI**:
```bash
python -m scripts.chat_cli -i sft -t 0.6 -k 50
# -i: checkpoint (sft/mid/rl/dpo)
# -t: temperature
# -k: top-k
```

**Web** (single GPU):
```bash
python -m scripts.chat_web -i sft -p 8000
# Visit http://<IP>:8000
```

**Web** (4 GPUs for load balancing):
```bash
python -m scripts.chat_web --num-gpus 4 -p 8000
# Up to four concurrent model workers; measure throughput and latency locally.
```

---

**Bottom Line**: CLI is a quick REPL for debugging. Web is a research server
with multi-GPU workers, streaming, and request-size validation; it is not
production-ready without deployment security and operational controls. Both
use the same `Engine` under the hood.

# KV Cache Implementation for MLX Models (2025/09 Update)

## Benefits
This implementation dramatically improves chat response times by reusing computation results from previous interactions. Key benefits:

* **Reduce Prompt Evaluation**
  - MLX has a challenge of taking longer for Prompt Evaluation compared to Nvidia GPUs. This becomes more pronounced as prompts get longer. In chat scenarios, response speed gradually slows down as the conversation with the LLM lengthens.
  - This is because the prompt includes the entire message history of the chat. In essence, we're having a "new" chat with the LLM each time while passing the previous chat history.
  - By using KV Cache, we reuse calculation results from previous text generations. This reduces the LLM's necessary calculations to only the newly received message. Consequently, the time for Prompt Evaluation is always proportional to the length of the latest message received from the user, unaffected by the volume of chat message history.

  KV Cache has been implemented in [`mlx-lm`](https://github.com/ml-explore/mlx-lm/). I'm deeply grateful to the developers of [`mlx`](https://github.com/ml-explore/mlx) and [`mlx-lm`](https://github.com/ml-explore/mlx-lm). 

## How It Works
My previous KV-Cache search implementation uses message_id as key.
Unlike the previous message_id-based approach, the new implementation uses **token sequence comparison** to find the best cache match:

1. When a request arrives, the full prompt is tokenized
2. The system scans all cache files to find:
   - The cache with the longest matching token prefix
   - Where the current prompt is longer than the cached sequence
3. Only tokens after the matching prefix are processed
4. The complete token sequence (prompt + generated tokens) is saved for future comparisons

This approach:
- Doesn't depend on client-provided message IDs (previous my implemantation rely on the message ID)
- This implementation follows the almost same principles as mlx_lm.server's. Thanks to all mlx_lm developers.

## Core Workflow

### Cache Lookup Process
1. **Tokenize Current Prompt**:
   - Convert prompt (or messages) to token sequence

2. **Scan Cache Files**:
   - Scan all safetensor files in the worker/kv_cache directory
   - find longuest matched kv_cache (safetensor) file.

3. **Process Only New Tokens**:
   - Only tokens after the common prefix are sent to the model
   - KV cache from the matching file is reused

4. **Save Complete Sequence**:
   - Full token sequence (prompt + generated) is saved with cache

### Token Comparison Logic
The common prefix calculation works like this:

```
Current prompt tokens:  [A, B, C, D, E, F, G, H]
Cached tokens:          [A, B, C, X, Y, Z]

Common prefix length: 3 (A, B, C)
Tokens to process:     [D, E, F, G, H]
```

This ensures maximum cache utilization even when conversations diverge.

## Prerequisites
* **Model Format**: Only MLX models support KV Cache
* **Parameter**: Include `"use_kv_cache": true` in your request
* **Tokenizers**: Must be compatible with your model's tokenization scheme

## Example Workflow

### Step 1: Initial Request (No Cache)

**Request**:
```json
{
  "messages": [
    {"role": "user", "content": "Hello! I read the book 'Harry Potter and the Prisoner of Azkaban' today."},
    {"role": "assistant", "content": "Hi there! That's great. Was it fun?"}
  ],
  "use_kv_cache": true
}
```

**Processing**:
1. Full prompt tokenized: `[123, 456, 789, ...]` (Assume length: 4500)
2. No cache match found (first request)
3. All tokens processed normally
4. Generate tokens. (Assume length: 100)
4. Complete token sequence saved to cache  (length: 4600)

### Step 2: Subsequent Request (With Cache)

**Request**:
```json
{
  "messages": [
    {"role": "user", "content": "Hello! I read the book 'Harry Potter and the Prisoner of Azkaban' today."},
    {"role": "assistant", "content": "Hi there! That's great. Was it fun?"},
    {"role": "user", "content": "Which of the characters is your favourite?"}
  ],
  "use_kv_cache": true
}
```

**Processing**:
1. Full prompt tokenized: `[123, 456, 789, ...]` (Asume length: 4750. this means new message is 150 tokens.)
2. Cache scan finds match with common prefix length: 4600
3. Only 150 new tokens processed (0.32% of original processing)
4. Response generated 30x faster than without cache

**Response includes**:
```json
{
  "usage": {
    "prompt_tokens": 4500,
    "completion_tokens": 21,
    "total_tokens": 4521,
    "kv_cache": {
      "prefix_length": 4500,
      "cache_hit": true
    }
  }
}
```

## Cache Management

### Automatic Cleanup
- Caches are automatically deleted when exceeding `MAX_KV_SIZE_GB` (default: 10GB)
- Oldest caches are removed first
- No manual cleanup required during normal operation


## Important Notes

**Token Sequence Storage**:
   - Token sequences are stored in cache metadata
   - This enables accurate token-level comparison


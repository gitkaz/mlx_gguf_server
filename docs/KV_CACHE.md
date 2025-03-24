## About KV Cache
In mlx_gguf_server, KV Cache can be utilized for Chat Completion in MLX models. Using KV Cache offers the following benefits:

* Fast Prompt Evaluation
  - MLX has a challenge of taking longer for Prompt Evaluation compared to Nvidia GPUs. This becomes more pronounced as prompts get longer. In chat scenarios, response speed gradually slows down as the conversation with the LLM lengthens.
  - This is because the prompt includes the entire message history of the chat. In essence, we're having a "new" chat with the LLM each time while passing the previous chat history.
  - By using KV Cache, we reuse calculation results from previous text generations. This reduces the LLM's necessary calculations to only the newly received message. Consequently, the time for Prompt Evaluation is always proportional to the length of the latest message received from the user, unaffected by the volume of chat message history.

  KV Cache has been implemented in [`mlx-lm`](https://github.com/ml-explore/mlx-lm/). I'm deeply grateful to the developers of [`mlx`](https://github.com/ml-explore/mlx) and [`mlx-lm`](https://github.com/ml-explore/mlx-lm). 


KV Cache accelerates chat interactions by reusing intermediate states from previous messages. The cache is **implicitly tied to the chat message history** and requires no external session IDs. Here's the precise workflow:

## Core Workflow
When use_kv_cache=True in a /v1/chat/completions request:

1. **Cache Search**:
  - The server scans the reverse of the provided messages list to find the latest message with a cached state.
  - For each message in reverse order (from newest to oldest):
    - Check if a cache file exists (e.g., {message_id}.safetensors).
  - If found, the cache is loaded, and only the new messages after this point are processed.

2. **Generation**:
  - Uses the loaded cache to skip re-processing older messages.
  - Only the new input message is added to the prompt.

3. **Cache Storage**:
  - After generation, a new cache is created using the latest message's message_id as the filename.

## Prerequisites
* Model Format: Only MLX models support KV Cache.
* API Endpoint: Use /v1/chat/completions with the `use_kv_cache` parameter.


## [Example Workflow]
### Step 1: Initial Request (No Cache)

* **Request**:
  ```
  curl -X POST -H "Content-Type: application/json" \
  -H "X-Model-Id: 0" \
  -d '{
      "messages": [
          {"role": "user", "content": "Hello! I read the book 'Harry Potter and the Prisoner of Azkaban' today.", "message_id": "msg_1_uuid"}
      ],
      "use_kv_cache": true,
      "temperature": 0.7
  }' \
  http://localhost:4000/v1/chat/completions
  ```

* **Internal Processing**:
  1. Scan KV cache file from bottom of messages list (reverse scan) by using "message_id". In this case, `msg_1_uuid.safetensors` is looking for, but the file doesn't find.
  2. Then eval all messages as a prompt and generates text.


* **Response**:
  ```
  {
    "choices": [{"message": {"content": "Hi there! That's great. Was it fun?", "message_id": "msg_2_uuid"}}],
    "usage": ...
  }
  ```

* **Cache Action**:
  - A new cache file `msg_2_uuid.safetensors` is stored in `llm_process/kv_cache` directory.

### Step 2: Subsequent Request (With Cache)

* **Request**:
  ```
  curl -X POST -H "Content-Type: application/json" \
  -H "X-Model-Id: 0" \
  -d '{
      "messages": [
          {"role": "user", "content": "Hello! I read the book 'Harry Potter and the Prisoner of Azkaban' today.", "message_id": "msg_1_uuid"},
          {"role": "assistant", "content": "That's great. Was it fun?", "message_id": "msg_2_uuid"},
          {"role": "user", "content": "Which of the characters is your favourite?", "message_id": "msg_3_uuid"}
      ],
      "use_kv_cache": true,
      "temperature": 0.7
  }' \
  http://localhost:4000/v1/chat/completions
  ```

* **Internal Processing**:
  1. Scan KV cache file from bottom of messages list (reverse scan) by using "message_id". In this case, `msg_2_uuid.safetensors` will be found.
  2. Then cut messages list at the found message line and above. In this acse, the new messages list only include single line (`{"role": "user", "content": "Which of the characters is your favourite?", "message_id": "msg_3"}`).
  3. KV cache file `msg2_uuid.safetensors` is loaded and evaluate prompt "Which of the characters is your favourite?". Then text will be generated.

* **Response**:
  ```
  {
    "choices": [{"message": {"content": "That's a great question! It's tough to choose just one ... but I'd say Remus Lupin. ", "message_id": "msg_4_uuid"}}],
    "usage": ...
  }
  ```

* **Cache Action**:
  + A new cache file `msg_4_uuid.safetensors` is stored in `llm_process/kv_cache` directory.

## Notes

1. **Message IDs**:
  - Messages must have message_id fields (UUIDs or unique strings) for cache lookup.

2. **Storage Limits**:
  - Caches are auto-deleted when exceeding MAX_KV_SIZE_GB (default: 10GB).
  - Oldest caches are removed first.


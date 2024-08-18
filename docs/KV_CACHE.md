Using KV Cache

## [Introduction]

### [About KV Cache]
In this program, KV Cache can be utilized for Chat Completion in MLX. Using KV Cache offers the following benefits:

* Fast Prompt Evaluation
  + MLX has a challenge of taking longer for Prompt Evaluation compared to Nvidia GPUs. This becomes more pronounced as prompts get longer. In chat scenarios, response speed gradually slows down as the conversation with the LLM lengthens.
  + This is because the prompt includes the entire message history of the chat. In essence, we're having a "new" chat with the LLM each time while passing the previous chat history.
  + By using KV Cache, we reuse calculation results from previous text generations. This reduces the LLM's necessary calculations to only the newly received message. Consequently, the time for Prompt Evaluation is always proportional to the length of the latest message received from the user, unaffected by the volume of chat message history.

While KV Cache has been implemented in MLX's [`mlx-examples`](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm) repository for some time, as far as I could tell, it was limited to a single prompt and not suitable for multiple message exchanges like in a chat. With just a slight addition to the code in [`mlx_lm.utils.generate_step`](https://github.com/ml-explore/mlx-examples/blob/main/llms/mlx_lm/utils.py), I was able to enable KV Cache for continuous use and updates in chat-like interactions. I'm deeply grateful to the developers of [`mlx`](https://github.com/ml-explore/mlx) and [`mlx-examples`]([url](https://github.com/ml-explore/mlx-examples/tree/main/llms/mlx_lm). 

Following illustration showing the difference in prompts with and without KV Cache

### [Difference in Text Generation Speed With and Without KV Cache]
I've created a video demonstrating how much the generation speed differs with and without KV Cache using `mlx-community/gemma-2-27b-it-8bit`.
1. Load a large article (this one)
2. In the next turn, request to "summarize the article"
3. Measure the time difference until the summary output begins


### [Measurement Results]
| | Without KV Cache | With KV Cache |
----|----|----
| |  https://github.com/user-attachments/assets/7fe31aa4-7f32-4c23-b11c-ede3acc152f5 | https://github.com/user-attachments/assets/221eebcb-c595-43b2-ba77-226a94e48f0f |
|1st turn| prompt_tokens: 4487, prompt_eval_time: 14.320998374954797 sec | **prompt_tokens: 4487, prompt_eval_time: 13.99744424992241 sec** |
|2nd turn| prompt_tokens: 4503, prompt_eval_time: 14.016152999945916 sec | **prompt_tokens: 14, prompt_eval_time: 0.8610775420675054 sec** |

In the first turn, there's no difference as KV Cache isn't available, but in the second turn, due to the effectiveness of KV Cache, there's a significant difference in prompt eval token count, resulting in 17.5 times faster response.

## [Note]
Outputs may not be identical when using KV Cache compared to not using it (even if temperature = 0).



Below, I explain how to use KV Cache in mlx_gguf_server.

## [Prerequisites]
* The target model should be in MLX format
* The API Endpoint should be `/v1/chat/completions`
* The POST request should include `kv_cache_session_id: number`
These are the conditions.

## [Operation]
For accesses meeting the above prerequisites, the following occurs during text generation:
1.  Determine if a cache already exists
  + Internally, KV Cache is managed using the number set in "kv_cache_session_id" as an identifier.
2. If no cache is found, generate text normally by including all chat messages in the prompt. Simultaneously, generate a KV Cache with the specified kv_cache_session_id and save it in memory.
3. If a cache is found in step 1, use that cache. In this case, the prompt for text generation only uses the latest input from the user, i.e., the end of the messages.

## [Managing KV Cache]
The following API endpoints can be used to check and delete KV Cache:

* /v1/internal/model/kv_cache/info
  + Method: GET
  + Responds with the current KV Cache ID of the target model and its capacity (Bytes), but currently, the capacity is (probably) not the correct value.

* /v1/internal/model/kv_cache/remove_cache
  + Method: POST
  + POST parameter: {session_id: Session ID (number)}
  + Deletes the KV Cache specified by session_id. As it's specified by session_id, only one KV Cache can be deleted at a time.

* /v1/internal/model/kv_cache/remove_old_caches
  + Method: POST
  + POST parameter: {seconds: Number of seconds (number)}
  + Deletes multiple KV Caches at once whose last update time is older than the specified number of seconds from the current time.

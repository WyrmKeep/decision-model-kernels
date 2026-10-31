# The model and the server

This document gives facts about the model and about the server. It gives the layers of the model and the "mix8" quantization. It also gives the work of `/v1/systemone`, the flags of the server in production, and the time that each part uses. The [method](method.md) gives the rules for each measurement. The [kernel document](kernels.md) gives the kernels on the request path.

## Layers of the model

The model is H2O Lightning 4B. Its layers are the same as the layers of Qwen3.5-4B. It is a hybrid model. The 32 layers are in 8 groups of four. Each group has three recurrent layers and then one attention layer. The recurrent layers have the type Gated DeltaNet.

| Item | Value | Unit |
|---|---:|---|
| Model layers | 32 | layers |
| Recurrent layers (Gated DeltaNet) | 24 | layers |
| Attention layers | 8 | layers |
| Vocabulary | 248,320 | tokens |
| Embedding | 2,560 | numbers for each token |

Source: the [README](../README.md), the [inspection of the upstream code](notebook/2026-10-30-phase0-upstream-survey.md), and the file [`config.json`](https://huggingface.co/h2oai/h2o-lightning-4b/blob/main/config.json) of the model on Hugging Face.

The output head has 248,320 rows of 2,560 values. It changes the data of one token to one logit for each token of the vocabulary. The file `config.json` sets the field `tie_word_embeddings` to true. The embedding and the output head thus use the same weights.

## Memory of a hybrid model

An attention layer keeps a KV cache. The KV cache has the data of all previous tokens. It has more data after each new token. The server can remove the data of the last tokens from it.

A recurrent layer keeps a layer state. The layer state has the same number of numbers for each length of the prompt. The layer changes the layer state at each token. It does not keep the data of each token. As a result, the server cannot cut a layer state back to a previous token. The server can go back only to a checkpoint.

| Layer type | Layers | KV cache for each token (KiB) | Layer state for each slot (MiB) |
|---|---:|---:|---:|
| Attention layer | 8 | 32 | 0 |
| Recurrent layer | 24 | 0 | 50 |

Source: we calculated the values from the file `config.json` of the model and from the code of llama.cpp at tag `b12004`. The KV cache uses 16 bits for each number. The layer state uses 32 bits for each number. [INFERENCE: we did not measure these values.]

Two rules follow from this difference. The first rule is that the server can use the first tokens of a previous prompt again only with a checkpoint. The second rule is that all slots in one ubatch must have the same number of tokens. The next sections show the effect of the two rules.

## The "mix8" quantization

Quantization changes the model weights to a small number of bits for each weight. The "mix8" model uses two types. It uses Q4_0 for the weights of the feed-forward layers. It uses Q8_0 for the weights of the attention layers and of the recurrent layers. It also uses Q8_0 for the embedding and for the output head.

| Weights | Type | Tensor names |
|---|---|---|
| Feed-forward layers | Q4_0 | `ffn_gate`, `ffn_up`, `ffn_down` |
| Attention layers | Q8_0 | `attn_q`, `attn_k`, `attn_v`, `attn_output` |
| Recurrent layers | Q8_0 | `attn_qkv`, `attn_gate`, `ssm_out`, `ssm_alpha`, `ssm_beta` |
| Embedding and output head | Q8_0 | `token_embd` |

Source: the quantize command in [bench/README.md](../bench/README.md).

The two types keep the model weights in groups of 32 weights. Each group has one scale. The scale uses 16 bits. The program multiplies the integer of a weight by the scale of its group to get the model weight.

Q8_0 keeps each integer in 8 bits. A group of Q8_0 uses 34 bytes. Q4_0 keeps each integer in 4 bits. A group of Q4_0 uses 18 bytes. For Q4_0, the integer is the 4-bit number in the file minus 8.

| Type | Weights in a group | Bits for each integer | Bytes in a group | Bits for each weight |
|---|---:|---:|---:|---:|
| Q4_0 | 32 | 4 | 18 | 4.5 |
| Q8_0 | 32 | 8 | 34 | 8.5 |

Source: the types `block_q4_0` and `block_q8_0` in `ggml/src/ggml-common.h` of llama.cpp.

The work before this project measured the speed of the types. MMQ runs Q4_0 faster than Q4_K_M. At 3,000 tokens, `llama bench` gave 4,650 to 4,690 tokens/s for Q4_0 and 4,400 to 4,420 tokens/s for Q4_K_M. Q8_0 for all weights gave 4,781 tokens/s.

The quality of the answers is the cause of the mixture of types. Q8_0 for all weights gave only 91 of 100 correct decisions on the private benchmark, which is below the guard. Q4_0 for all weights gave 93 of 100, which is the limit of the guard. The "mix8" model gave 94 of 100.

The file name of the "mix8" model is `h2o-lightning-4b-rq-Q4_0-mix8.gguf`. The [benchmark document](../bench/README.md) shows how to make it from the Q8_0 file `h2o-lightning-4b-Q8_0.gguf`. The Q8_0 file is in the repository `h2oai/h2o-lightning-4b-gguf` on Hugging Face. It has 4.48 GB. The model has the `Apache 2.0` license. The repository of the project has no model file, and the [publication gate](../tools/publish/README.md) blocks files of the type `.gguf`.

## How `/v1/systemone` works

A request to `POST /v1/systemone` has one state and a list of questions. The benchmark sends one to three questions in each request. The server gives one probability for each possible answer of each question.

### Prompts and slots

The server makes one prompt for each question. A prompt has the system prompt, the state, and the question. The prompts of one request thus start with the same tokens. These tokens are the shared prefix.

The server puts the questions in groups. A group has at most one question for each slot. The first question of a group is the parent task. It runs the model on the shared prefix. The other questions of the group are child tasks.

A child task waits for the parent task. The server then copies the data of the parent task to it. For a layer state, this copy moves no data in `b12004`. The graph gathers the layer states in each pass. The child task then runs the model only on the tokens after the shared prefix.

The flags `-np 3 --kv-unified` give three slots with a unified KV cache.

Source at tag `b12004`, as the inspection of the upstream code gives it: `tools/server/server-decision.cpp:1753-1783`, `tools/server/server-context.cpp:4360-4372`, `tools/server/server-context.cpp:4785-4789`, `tools/server/server-context.cpp:845-864`, `src/llama-memory-recurrent.cpp:262-297`, and `src/llama-graph.cpp:3937-3949`.

### Equal lengths in a ubatch

The hybrid memory of llama.cpp makes each ubatch with the function `split_equal`. In one ubatch, all slots have the same number of tokens. The shortest question sets this number. The cause is in the recurrent layers. They use a tensor with one row of tokens for each slot. All rows of the tensor have the same length.

The length of a question is its number of tokens after the shared prefix. The first ubatch has only as many tokens of each question as the shortest question has. The server must run one more pass for the longer questions. Each different length thus adds one pass. One more pass for some tokens uses about 11.5 ms. The work before this project measured this value.

Two questions with different lengths use three passes. The first pass is for the shared prefix. The two other passes are one for each length. Two questions with the same length use two passes.

Source at tag `b12004`: `src/llama-memory-hybrid.cpp:90` and `src/llama-batch.cpp:619-712`.

### The answer

At the end of each question, the model gives one logit for each token of the vocabulary. For H2O Lightning, the text of each possible answer is one token. This token is the answer token. The server reads the logits of the answer tokens only. It reads them with `llama_get_logits_ith`. The answer is the softmax of these logits.

A question can have more than one variant. The server calculates the softmax for each variant. It then calculates the mean of the probabilities. H2O Lightning has one variant for each question.

The output head continues to calculate all 248,320 rows. The server reads only the rows of the answer tokens. The answer does not use the other rows. In Q8_0, the output head has about 675 MB.

Source at tag `b12004`: `tools/server/server-context.cpp:2527-2535`, `tools/server/server-decision.cpp:178-193`, and `src/models/qwen35.cpp:233`. The function `format_answer` in `tools/server/server-decision.cpp` calculates the softmax. The function `n_variants` in the same file gives the number of variants.

## Flags of the server in production

The server in production is `llama-server` of tag `b12004` with the "mix8" model. The file `bench/launch.example.sh` uses the same flags.

| Flag | Value | Meaning |
|---|---:|---|
| `-ngl` | 999 | The largest number of layers that the server keeps in VRAM. The model has 32 layers. All layers are on the GPU. |
| `-c` | 16384 | The number of token positions of the KV cache. |
| `-b` | 2048 | The largest number of tokens in a batch. |
| `-ub` | 2048 | The largest number of tokens in a ubatch. The model reads one ubatch in each pass. |
| `-np` | 3 | The number of slots. |
| `--kv-unified` | None | The slots use one unified KV cache. They use the positions of `-c` together. |
| `--no-webui` | None | The server does not give the WebUI. |
| `--no-slots` | None | The server does not give the part of the API that shows the slots. |

Source: the flag texts in `common/arg.cpp` at tag `b12004`.

The baseline is the stock build. The private benchmark measured it before this project started. The benchmark has 48 requests. A short request has a state of 11 to 84 tokens and one or two questions. A medium request has a state of 985 to 1,307 tokens and three questions. A long request has a state of 2,899 to 3,414 tokens and three questions.

| Build | Short request (ms) | Medium request (ms) | Long request (ms) | Mean of all requests (ms) |
|---|---:|---:|---:|---:|
| Stock build `b12004` (baseline) | 71 | 357 | 779 | 236 |

Source: the private benchmark, which we ran before this project started, and the token counts in [bench/README.md](../bench/README.md). The [README](../README.md) has the results of the project for the private benchmark. The public benchmark has a different baseline.

## Where the time goes

The numbers in this part come from work that we did before this project started. That work used the private benchmark and a profiler. The [entry of phase 3](notebook/2026-10-31-phase3-where-the-time-goes.md) gives new measurements of the project. The last column shows which numbers we measured. Each other number has the mark [INFERENCE].

| Quantity | Value | Unit | Measured |
|---|---:|---|---|
| Time of a short request with one question | 36.7 | ms | Yes |
| Time to read the layer weights one time | about 6 | ms | No. [INFERENCE: calculated from 2.7 GB of layer weights and 448 GB/s] |
| Time of one more pass for each different question length | about 11.5 | ms | Yes |
| Time of the output head with 248,320 rows | 1.7 | ms | Yes |
| Kernel launches in one pass | about 1,280 | launches | Yes |
| GPU time in small kernels that wait for memory | about 25 | % | Yes |
| Speed of MMQ as a part of the highest int8 speed of the GPU | 36 | % | Yes. We calculated the highest speed |

Source: the work before this project.

A short request with one question uses 36.7 ms. The time to read the layer weights one time is about 6 ms. The work before this project did not find where the other time of about 30 ms goes. The possible causes are the work of the server for each request, the tokenizer, the copies of the layer states, and the time between launches. [INFERENCE: we did not measure these causes.]

The work in a request has three types. The first type is necessary work. The server reads the state one time, reads each question one time, and calculates one output row for each answer token. The second type is work that the answer does not use: the other output rows and one pass for each different question length. The third type is the time of the many launches and of the small kernels. Fusion and CUDA graphs can decrease this time [INFERENCE: we have no test of this].

## Related documents

| Document | Contents |
|---|---|
| [method.md](method.md) | The method of the benchmark and of the tests of correctness |
| [kernels.md](kernels.md) | The kernels on the request path |
| [glossary.md](glossary.md) | The terms of the project |
| [notebook/](notebook/README.md) | The tests of the project, each with a date and a prediction |
| [../bench/README.md](../bench/README.md) | The public benchmark: cases, labels, client |

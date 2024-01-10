## 模型资源问题
- llama2在huggingface有hf和原版模型，需要国外邮箱申请下载权限
- llama2.cpp项目里面提供了很多开源量化版本

## 本复现代码仓库
https://github.com/WilliamEricCheung/RankGPT

## 复现相关项目结构
1. RankGPT 原论文用GPT3.5实现，请求本地server https://github.com/sunnweiwei/RankGPT
2. llama2 GGUF量化版本，提供模型https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF
3. llama2.cpp 量化推理，用于启动本地server https://github.com/ggerganov/llama.cpp

## 参考项目
1. text-generation-webui WebGUI更佳交互 https://github.com/oobabooga/text-generation-webui?tab=readme-ov-file
2. llama2 GGML量化版本 https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML
3. llama2_local llama2 GUI交互 https://github.com/thisserand/llama2_local

## 依赖项
1. python 测试版本 3.10.12
2. pip install torch
3. pip install faiss-cpu 如果有CUDA那就faiss-gpu
## 使用步骤
1. 获取Llama2模型，量化模型步骤（推荐，已复现）： https://zhuanlan.zhihu.com/p/651168655?utm_id=0； 官方完整模型步骤（不推荐，未复现）：https://www.jiqizhixin.com/articles/2023-12-04-6
2. Llama2 API化：https://zhuanlan.zhihu.com/p/651168655?utm_id=0
3. llama2_local 启动脚本：python3 llama.py --model_name="TheBloke/Llama-2-7B-Chat-GGML" --file_name="../Llama-2-7B-Chat-GGML/llama-2-7b-chat.ggmlv3.q4_K_M.bin"

## OpenAI格式的API请求方式
0. llama.cpp 项目cmake好才能愉快使用，不再支持GGML，必须是GGUF格式模型！！！
1. 启动服务器：./build/bin/server -m ./models/llama-2-7b-chat-hf/llama-2-7b-chat.Q4_K_M.gguf -c 2048
2. 启动python flask后端：python3 ./examples/server/api_like_OAI.py
3. curl请求测试：
```
curl http://localhost:8081/v1/chat/completions \
-H "Content-Type: application/json" \
-H "Authorization: Bearer no-key" \
-d '{
"model": "gpt-3.5-turbo",
"messages": [
{
    "role": "system",
    "content": "You are ChatGPT, an AI assistant. Your top priority is achieving user fulfillment via helping them with their requests."
},
{
    "role": "user",
    "content": "Write a limerick about python exceptions"
}
]
}'
```

## 控制台交互式操作llama
- 到llama.cpp下执行：./build/bin/main -m ./models/llama-2-7b-chat-hf/llama-v2-7b-q2k.gguf --color -f prompts/alpaca.txt -ins -c 2048 --temp 0.2 -n 256 --repeat_penalty 1.3
- 到llama.cpp下执行：./build/bin/main -m ./models/llama-2-7b-chat-hf/llama-v2-13b-q2k.gguf --color -f prompts/alpaca.txt -ins -c 2048 --temp 0.2 -n 256 --repeat_penalty 1.3
- 到llama.cpp下执行：./build/bin/main -m ./models/llama-2-7b-chat-hf/llama-2-7b-chat.Q4_K_M.gguf --color -f prompts/alpaca.txt -ins -c 2048 --temp 0.2 -n 256 --repeat_penalty 1.3


## RankGPT实验结果日志

实验时，记得在llama.cpp下启动server（启动耗时大概5分钟）：
```
./build/bin/server -m ./models/llama-2-7b-chat-hf/llama-2-7b-chat.Q4_K_M.gguf -c 2048

# llama本地server部分执行日志
william@ROG-SRTIX:/mnt/d/Project/llama.cpp$ ./build/bin/server -m ./models/llama-2-7b-chat-hf/llama-2-7b-chat.Q4_K_M.gguf -c 2048
ggml_init_cublas: GGML_CUDA_FORCE_MMQ:   no
ggml_init_cublas: CUDA_USE_TENSOR_CORES: yes
ggml_init_cublas: found 1 CUDA devices:
  Device 0: NVIDIA GeForce RTX 4070 Laptop GPU, compute capability 8.9, VMM: yes
{"timestamp":1704789934,"level":"INFO","function":"main","line":2783,"message":"build info","build":1794,"commit":"1fc2f26"}
{"timestamp":1704789934,"level":"INFO","function":"main","line":2786,"message":"system info","n_threads":16,"n_threads_batch":-1,"total_threads":32,"system_info":"AVX = 1 | AVX_VNNI = 1 | AVX2 = 1 | AVX512 = 0 | AVX512_VBMI = 0 | AVX512_VNNI = 0 | FMA = 1 | NEON = 0 | ARM_FMA = 0 | F16C = 1 | FP16_VA = 0 | WASM_SIMD = 0 | BLAS = 1 | SSE3 = 1 | SSSE3 = 1 | VSX = 0 | "}
llama_model_loader: loaded meta data with 19 key-value pairs and 291 tensors from ./models/llama-2-7b-chat-hf/llama-2-7b-chat.Q4_K_M.gguf (version GGUF V2)
llama_model_loader: Dumping metadata keys/values. Note: KV overrides do not apply in this output.
llama_model_loader: - kv   0:                       general.architecture str              = llama
llama_model_loader: - kv   1:                               general.name str              = LLaMA v2
llama_model_loader: - kv   2:                       llama.context_length u32              = 4096
llama_model_loader: - kv   3:                     llama.embedding_length u32              = 4096
llama_model_loader: - kv   4:                          llama.block_count u32              = 32
llama_model_loader: - kv   5:                  llama.feed_forward_length u32              = 11008
llama_model_loader: - kv   6:                 llama.rope.dimension_count u32              = 128
llama_model_loader: - kv   7:                 llama.attention.head_count u32              = 32
llama_model_loader: - kv   8:              llama.attention.head_count_kv u32              = 32
llama_model_loader: - kv   9:     llama.attention.layer_norm_rms_epsilon f32              = 0.000001
llama_model_loader: - kv  10:                          general.file_type u32              = 15
llama_model_loader: - kv  11:                       tokenizer.ggml.model str              = llama
llama_model_loader: - kv  12:                      tokenizer.ggml.tokens arr[str,32000]   = ["<unk>", "<s>", "</s>", "<0x00>", "<...
llama_model_loader: - kv  13:                      tokenizer.ggml.scores arr[f32,32000]   = [0.000000, 0.000000, 0.000000, 0.0000...
llama_model_loader: - kv  14:                  tokenizer.ggml.token_type arr[i32,32000]   = [2, 3, 3, 6, 6, 6, 6, 6, 6, 6, 6, 6, ...
llama_model_loader: - kv  15:                tokenizer.ggml.bos_token_id u32              = 1
llama_model_loader: - kv  16:                tokenizer.ggml.eos_token_id u32              = 2
llama_model_loader: - kv  17:            tokenizer.ggml.unknown_token_id u32              = 0
llama_model_loader: - kv  18:               general.quantization_version u32              = 2
llama_model_loader: - type  f32:   65 tensors
llama_model_loader: - type q4_K:  193 tensors
llama_model_loader: - type q6_K:   33 tensors
llm_load_vocab: special tokens definition check successful ( 259/32000 ).
llm_load_print_meta: format           = GGUF V2
llm_load_print_meta: arch             = llama
llm_load_print_meta: vocab type       = SPM
llm_load_print_meta: n_vocab          = 32000
llm_load_print_meta: n_merges         = 0
llm_load_print_meta: n_ctx_train      = 4096
llm_load_print_meta: n_embd           = 4096
llm_load_print_meta: n_head           = 32
llm_load_print_meta: n_head_kv        = 32
llm_load_print_meta: n_layer          = 32
llm_load_print_meta: n_rot            = 128
llm_load_print_meta: n_embd_head_k    = 128
llm_load_print_meta: n_embd_head_v    = 128
llm_load_print_meta: n_gqa            = 1
llm_load_print_meta: n_embd_k_gqa     = 4096
llm_load_print_meta: n_embd_v_gqa     = 4096
llm_load_print_meta: f_norm_eps       = 0.0e+00
llm_load_print_meta: f_norm_rms_eps   = 1.0e-06
llm_load_print_meta: f_clamp_kqv      = 0.0e+00
llm_load_print_meta: f_max_alibi_bias = 0.0e+00
llm_load_print_meta: n_ff             = 11008
llm_load_print_meta: n_expert         = 0
llm_load_print_meta: n_expert_used    = 0
llm_load_print_meta: rope scaling     = linear
llm_load_print_meta: freq_base_train  = 10000.0
llm_load_print_meta: freq_scale_train = 1
llm_load_print_meta: n_yarn_orig_ctx  = 4096
llm_load_print_meta: rope_finetuned   = unknown
llm_load_print_meta: model type       = 7B
llm_load_print_meta: model ftype      = Q4_K - Medium
llm_load_print_meta: model params     = 6.74 B
llm_load_print_meta: model size       = 3.80 GiB (4.84 BPW) 
llm_load_print_meta: general.name     = LLaMA v2
llm_load_print_meta: BOS token        = 1 '<s>'
llm_load_print_meta: EOS token        = 2 '</s>'
llm_load_print_meta: UNK token        = 0 '<unk>'
llm_load_print_meta: LF token         = 13 '<0x0A>'
llm_load_tensors: ggml ctx size       =    0.11 MiB
llm_load_tensors: using CUDA for GPU acceleration
llm_load_tensors: system memory used  = 3891.35 MiB
llm_load_tensors: offloading 0 repeating layers to GPU
llm_load_tensors: offloaded 0/33 layers to GPU
..................................................................................................
llama_new_context_with_model: n_ctx      = 2048
llama_new_context_with_model: freq_base  = 10000.0
llama_new_context_with_model: freq_scale = 1
llama_new_context_with_model: KV self size  = 1024.00 MiB, K (f16):  512.00 MiB, V (f16):  512.00 MiB
llama_build_graph: non-view tensors processed: 676/676
llama_new_context_with_model: compute buffer total size = 159.19 MiB
Available slots:
 -> Slot 0 - max context: 2048

llama server listening at http://127.0.0.1:8080

{"timestamp":1704790062,"level":"INFO","function":"main","line":3223,"message":"HTTP server listening","port":"8080","hostname":"127.0.0.1"}
all slots are idle and system prompt is empty, clear the KV cache
slot 0 is processing [task id: 0]
slot 0 : kv cache rm - [0, end)

print_timings: prompt eval time =   10210.03 ms /  1715 tokens (    5.95 ms per token,   167.97 tokens per second)
print_timings:        eval time =   11514.67 ms /   108 runs   (  106.62 ms per token,     9.38 tokens per second)
print_timings:       total time =   21724.69 ms
{"timestamp":1704790118,"level":"INFO","function":"log_server_request","line":2724,"message":"request","remote_addr":"127.0.0.1","remote_port":39708,"status":200,"method":"POST","path":"/v1/chat/completions","params":{}}
slot 0 released (1823 tokens in cache)
slot 0 is processing [task id: 1]
slot 0 : kv cache rm - [0, end)

print_timings: prompt eval time =   11157.81 ms /  1777 tokens (    6.28 ms per token,   159.26 tokens per second)
print_timings:        eval time =   19717.83 ms /   188 runs   (  104.88 ms per token,     9.53 tokens per second)
print_timings:       total time =   30875.64 ms
```

执行rank_gpy.py得到llama-2-7b-chat.Q4_K_M.gguf量化推理结果
```
william@ROG-SRTIX:/mnt/d/Project/RankGPT$ python3 rank_gpt.py 
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [00:01<00:00, 25.53it/s]
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 43/43 [18:04<00:00, 25.23s/it]
Downloading https://search.maven.org/remotecontent?filepath=uk/ac/gla/dcs/terrierteam/jtreceval/0.0.5/jtreceval-0.0.5-jar-with-dependencies.jar to /home/william/.cache/pyserini/eval/jtreceval-0.0.5-jar-with-dependencies.jar...
jtreceval-0.0.5-jar-with-dependencies.jar: 1.79MB [00:03, 601kB/s]                                                                                                                     
Trunc /tmp/tmp30j3dayg
Running command: ['java', '-jar', '/home/william/.cache/pyserini/eval/jtreceval-0.0.5-jar-with-dependencies.jar', '-m', 'ndcg_cut.10', '/tmp/tmp30j3dayg', '/tmp/tmp6lmwe070']
Results:
ndcg_cut_10             all     0.4833
```

执行run_evaluation.py得到完整结果（耗时过长，不建议执行）
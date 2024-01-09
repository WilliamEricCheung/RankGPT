## 项目结构
1. RankGPT 原论文用GPT3.5实现 https://github.com/sunnweiwei/RankGPT
2. llama2_local llama2 GUI交互 https://github.com/thisserand/llama2_local
3. llama2.cpp 量化推理 https://github.com/ggerganov/llama.cpp
3. text-generation-webui WebGUI更佳交互 https://github.com/oobabooga/text-generation-webui?tab=readme-ov-file
4. llama2 GGML量化版本 https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML
## 依赖项
1. python 测试版本 3.10.12
2. pip install torch
3. pip install faiss-cpu 如果有CUDA那就faiss-gpu
## 使用步骤
1. 获取Llama2模型，量化模型步骤（推荐，已复现）： https://zhuanlan.zhihu.com/p/651168655?utm_id=0；官方完整模型步骤（不推荐，未复现）：https://www.jiqizhixin.com/articles/2023-12-04-6
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
4. 修改OpenAI测试脚本的base_url
```
openai = openai.OpenAI(
    base_url="http://localhost:8080/v1", # "http://<Your api-server IP>:port"
    api_key = "sk-no-key-required"
)
```

## 控制台交互式操作llama
- 到llama.cpp下执行：./build/bin/main -m ./models/llama-2-7b-chat-hf/llama-v2-7b-q2k.gguf --color -f prompts/alpaca.txt -ins -c 2048 --temp 0.2 -n 256 --repeat_penalty 1.3
- 到llama.cpp下执行：./build/bin/main -m ./models/llama-2-7b-chat-hf/llama-v2-13b-q2k.gguf --color -f prompts/alpaca.txt -ins -c 2048 --temp 0.2 -n 256 --repeat_penalty 1.3
- 到llama.cpp下执行：./build/bin/main -m ./models/llama-2-7b-chat-hf/llama-2-7b-chat.Q4_K_M.gguf --color -f prompts/alpaca.txt -ins -c 2048 --temp 0.2 -n 256 --repeat_penalty 1.3


## RankGPT实验结果日志

实验时，记得在llama.cpp下启动server（启动耗时大概5分钟）：
```
./build/bin/server -m ./models/llama-2-7b-chat-hf/llama-2-7b-chat.Q4_K_M.gguf -c 2048
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
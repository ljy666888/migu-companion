# migu-companion

2026 动感地带 AI+ 高校创智计划 · 主题二「数字人综合情感陪伴对话模型」的开源实现。

模型基于 Qwen2.5-1.5B-Instruct 做 LoRA 微调，输入一段对话历史，一次输出四样东西：
回复文本、情感标签、用户画像、记忆引用。情感标签在官方验证集上准确率 0.7067，
约为多数类基线的 2.49 倍；用户画像较弱，详见 `docs/技术方案与实验报告.pdf`。

## 模型下载

模型权重约 2.9 GB，超出 git 单文件 100 MB 的限制，因此拆成 2 个分片放在
本仓库的 **Releases** 页面。到 Releases 下载全部分片：

- `model.tar.part-01` — 1.56 GB
- `model.tar.part-02` — 1.32 GB

放在同一个目录下，然后按下面的命令合并。

## 合并与解压

**Windows（PowerShell 或 CMD）**

```cmd
copy /b model.tar.part-01+model.tar.part-02 model.tar
tar -xf model.tar
```

**Linux / macOS**

```bash
cat model.tar.part-* > model.tar
tar -xf model.tar
```

解压后得到 `model/` 目录，里面有 `model.safetensors`、`config.json`、`tokenizer.json` 等文件。
把整个 `model/` 放进本仓库根目录即可。

校验：合并后的 `model.tar` 应当是 3,098,910,720 字节。

## 环境

```
Windows 10/11，NVIDIA 显卡（显存 4 GB 以上）
Python 3.10+，推荐 3.12
```

```powershell
python -m venv .venv
.\.venv\Scripts\activate

# 必须带 --index-url，否则 pip 会装成 CPU 版，跑起来慢几十倍且不报错
pip install torch==2.6.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install transformers peft accelerate -i https://pypi.tuna.tsinghua.edu.cn/simple
```

验证 GPU 可用：

```powershell
python -c "import torch; print(torch.cuda.is_available())"
```

输出 `True` 才算装对。

## 开始对话

```powershell
.\.venv\Scripts\python.exe src\chat.py --model ./model
```

`--model` 指向解压出来的模型目录。看到 `ready.` 之后直接打字。

每轮会同时输出：

```
--- reply ---
模型的回复文本

emotion : anxiety
profile : traits=[] interests=['career_development'] style=[]
memory  : []
```

命令：

- `/sample 5` 加载第 5 条真实验证样本，并显示标准答案对照
- `/raw` 显示模型原始输出
- `/history` 查看当前对话历史
- `/reset` 清空重来
- `/quit` 退出

## 目录

```
src/chat.py                    交互式对话程序
src/common.py                  公共函数（提示词、样例迭代）
official/participant/          官方推理脚本与配置模板
data/eval_input.jsonl          验证样本
data/eval_reference.jsonl      参考标准答案
docs/                          技术报告、评测数据、多轮对话案例
model/                         解压后的模型（需自行下载）
```

## 批量评测

```powershell
$env:EVALUATION_BACKEND='transformers'
$env:EVALUATION_MODEL_PATH='C:\路径\model'
python official\participant\run_inference.py data\eval_input.jsonl out_eval
```

## 许可

本仓库的微调权重基于 **Qwen2.5-1.5B-Instruct**（Apache License 2.0），
分发与修改遵循同一许可。详见 `LICENSE-NOTICE.md`。

## 已知短板

- `style` 字段几乎总是输出空数组，而参考集中约四成非空，该字段 F1 接近零
- 画像不随轮次累积，每轮独立推断
- 回复偏向给出建议清单，源自训练数据的风格分布
- 偶尔会输出枚举之外的情感标签，被评测逻辑置空

这些都写在 `docs/技术方案与实验报告.pdf` 的第五节。

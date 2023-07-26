# やりとりxイミトル
> このリポジトリは[イミトル(semantleの日本語版)](https://github.com/sim-so/semantle-jp)をassistantと楽しめるものです。

## 遊び方 How to play

### 1. Hugging Face Space
こちらの[space](https://huggingface.co/spaces/sim-so/semantle-jp-with-chat)から設置なしでプレイすることができます。

### 2. Local

#### Create virtualenv
```bash
python3.10 -m venv semantle-jp-with-chat
source semantle-jp/bin/activate
```

#### Install requirements
```bash
pip install -r requirements.txt
```

#### Run gradio
```bash
gradio app.py
```

## ご了承のお願い
ゲームをするため、openaiのapiが必要です。答えによって少々tokenを使うようになります。

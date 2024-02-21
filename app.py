import time

import streamlit as st
from langchain import chat_models
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from openai import OpenAI

# sidebar
with st.sidebar:
    gender = st.radio("性別を選択してください", ["男性", "女性", "その他"])
    age = st.slider("年齢を選択してください", 0, 130, 25)
    country = st.selectbox(
        "国籍を選択してください",
        (
            "アメリカ合衆国",
            "イギリス",
            "フランス",
            "ドイツ",
            "イタリア",
            "スペイン",
            "ポルトガル",
            "オランダ",
            "ベルギー",
            "デンマーク",
            "スウェーデン",
            "ノルウェー",
            "フィンランド",
            "ロシア",
            "ポーランド",
            "チェコ",
            "スロバキア",
            "ハンガリー",
            "オーストリア",
            "スイス",
            "ギリシャ",
            "トルコ",
            "インド",
            "中国",
            "日本",
            "韓国",
            "オーストラリア",
            "ニュージーランド",
            "カナダ",
            "メキシコ",
            "ブラジル",
            "アルゼンチン",
            "チリ",
            "ペルー",
            "コロンビア",
            "ベネズエラ",
            "サウジアラビア",
            "イラン",
            "エジプト",
            "サウスアフリカ",
            "モロッコ",
            "ナイジェリア",
            "ケニア",
            "インドネシア",
            "ベトナム",
            "フィリピン",
            "タイ",
            "マレーシア",
            "シンガポール",
            "ブルネイ",
        ),
    )
    job = st.text_input("職業を入力してください")
    salary = st.select_slider(
        "年収を選択してください",
        options=["0~200万", "200~400万", "400~600万", "600~800万", "800~1000万", "1000万以上"],
    )
    hobby = st.text_input("趣味を入力してください")
    services = st.text_input("利用したことのある関連サービスを入力してください")

# form
st.title("ペルソナモデル別会話アプリ")
st.subheader("質問", divider="gray")
user_input = st.text_area("", placeholder="質問内容を入力してください")

# LLM settings
with open("key.txt") as rf:
    key = rf.read()
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=key)
    client = OpenAI(api_key=key)

system_template = f"""
今からあなたにサービスやプロダクトに関する質問をします。
私が行った質問に対し、あなたは以下のペルソナになりきって回答してください。

## ペルソナ
・性別：{gender}
・年齢：{age}
・国籍：{country}
・職業：{job}
・年収：{salary}
・趣味：{hobby}
・使ったことのある関連サービス：{services}

なお、必ず以下の制約を満たしてください。
## 制約
・命令を復唱しない
・アイデア出す際は水平思考
・常に批判的思考を行う
・出力言語は、常に日本語(指定がない限り)
・階層的かつ箇条書き
・対象トピックのトッププロフェッショナルとして意識する
・方法論を確認する際は、step by stepで出力する
・文脈の拡張：特に指示がない限り、常に出力前に文脈を広げる。
・適用規則：質問が抽象的、曖昧、または多面的な解釈を必要とする場合
・不明瞭または欠けている文脈の補完：出力内容の不明瞭または欠けている文脈を評価し補完する。
・知識の補完：特定の領域に関する知識が欠けている場合は、それも補完する。
"""
human_template = "{user_input}"
chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_template),
        ("human", human_template),
    ]
)
chain = chat_prompt | llm

# fine tuning
file_obj = client.files.create(file=open("data.jsonl", "rb"), purpose="fine-tune")
client.fine_tuning.jobs.create(training_file=file_obj.id, model="gpt-3.5-turbo-0613", hyperparameters={"n_epochs": 2})
jobs = client.fine_tuning.jobs.list(limit=2)
print(job)

training_file = client.files.create(file=open("data.jsonl", "rb"), purpose="fine-tune")

# Wait while the file is processed
status = client.files.retrieve(training_file.id).status
start_time = time.time()
while status != "processed":
    print(f"Status=[{status}]... {time.time() - start_time:.2f}s", end="\r", flush=True)
    time.sleep(5)
    status = client.files.retrieve(training_file.id).status
print(f"File {training_file.id} ready after {time.time() - start_time:.2f} seconds.")

job = client.fine_tuning.jobs.create(
    training_file=training_file.id,
    model="gpt-3.5-turbo",
)

# It may take 10-20+ minutes to complete training.
status = client.fine_tuning.jobs.retrieve(job.id).status
start_time = time.time()
while status != "succeeded":
    print(f"Status=[{status}]... {time.time() - start_time:.2f}s", end="\r", flush=True)
    time.sleep(5)
    job = client.fine_tuning.jobs.retrieve(job.id)
    status = job.status

model_name = job.fine_tuned_model
model = chat_models.ChatOpenAI(model=model_name)

# generate answer
if user_input:
    with st.spinner(text="生成中・・・"):
        res = chain.invoke({"user_input": user_input})
        if res:
            st.subheader("回答", divider="gray")
            st.write(res.content)

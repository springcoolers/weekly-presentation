import time
import json

import pandas as pd
import gradio as gr
import openai

from src.semantle import get_guess, get_secret, get_puzzle_num
from src.functions import get_functions

GPT_MODEL = "gpt-3.5-turbo"
TITLE = "やりとりxイミトル"

with open("data/rule.md", "r", encoding="utf-8") as f:
    RULEBOOK = "\n".join(f.readlines())

def get_rulebook():
    return RULEBOOK

def _execute_function(function_call, chat_messages, guess_result):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        available_functions = {
            "guess_word": get_guess,
            "lookup_answer": get_secret,
            "retrieve_puzzle_num": get_puzzle_num,
            "read_rule": get_rulebook,
        }
        function_name = function_call["name"]
        function_to_call = available_functions[function_name]
        function_args = json.loads(function_call["arguments"])
        function_response = function_to_call(
            **function_args
        )
        if function_call["name"] == "guess_word":
            guess_result = function_response
        print(function_response)
        # Step 4: send the info on the function call and function response to GPT
        chat_messages.append(
            {"role": "function",
             "name": function_name,
             "content": f"{function_response}"}
        )   # extend conversation with function response
        next_response = openai.ChatCompletion.create(
            model=GPT_MODEL,
            messages=chat_messages,
            functions=get_functions(),
            temperature=0
        )   # get a new response from GPT where it can se the function response
        chat_messages.append(next_response.choices[0].message.to_dict())
        return next_response.choices[0].message.to_dict(), chat_messages, guess_result

def create_chat(system_input, user_input, guess_result=dict()):
    chat_messages = []
    for s in system_input:
        chat_messages.append({"role": "system", "content": s})
    chat_messages.append({"role": "user", "content": user_input})
    response = openai.ChatCompletion.create(
        model=GPT_MODEL,
        messages=chat_messages,
        functions=get_functions(),
        temperature=0
    )
    response_message = response.choices[0].message.to_dict()
    chat_messages.append(response_message) # extend conversation with assistant's reply

    # Step 2: check if CPT wanted to call a function
    while response_message.get("function_call"):
        # Step 3: call the function
        # Note: the JSON response may not always be valid; be sure to handle errors
        response_message, chat_messages, guess_result = _execute_function(response_message["function_call"], chat_messages, guess_result)
    print(chat_messages)
    return response_message, chat_messages, guess_result

with gr.Blocks() as demo:
    with gr.Row():
        gr.Markdown(
            """
            # やりとりxイミトル
            「イミトル」は[semantle日本語版](https://semantoru.com/)の名前で、こちらはイミトルをassistantと楽しめるspaceです。
            #### ゲームのやり方
            - 正解は一つの単語で、これを答えるとゲームの勝利になります。
            - 推測した単語が正解じゃない場合、類似度スコアと順位が表示されます。それは正解を推測する大事なヒントになります。
            #### assistantの仕事
            - 単語のスコアとランク以外に他のヒントがもらえます。
            - ゲームに関して困っている時、何か質問してみてください。
            #### ご了承のお願い
            - ゲームをするため、openaiのapiが必要です。答えによって少々tokenを使うようになります。
            """
        )

    with gr.Row():
        with gr.Column():
            api_key = gr.Textbox(placeholder="sk-...", label="OPENAI_API_KEY", value=None, type="password")
            idx = gr.State(value=0)
            guessed = gr.State(value=set())
            guesses = gr.State(value=list())
            cur_guess = gr.JSON(visible=False)
            history = gr.State(list())
            guesses_table = gr.DataFrame(
                value=pd.DataFrame(columns=["i", "guess", "sim", "rank"]),
                headers=["i", "guess", "score", "rank"],
                datatype=["number", "str", "number", "str"],
                elem_id="guesses-table",
                interactive=False
            )
        with gr.Column(elem_id="chat_container"):
            msg = gr.Textbox(
                placeholder="ゲームをするため、まずはAPI KEYを入れてください。",
                label="答え",
                interactive=False,
                max_lines=1
            )
            chatbot = gr.Chatbot(elem_id="chatbot")

        def unfreeze():
            return msg.update(interactive=True, placeholder="正解と思う言葉を答えてください。")
        def reset_history():
            return list()
        def greet(key, gradio_messages):
            openai.api_key = key
            system_input = [get_rulebook(), "あなたは「イミトル」というゲームの進行役です。ユーザーが答えをするところです。よろしくお願いします。"]
            user_input = ""
            reply, _, _ = create_chat(system_input, user_input, guess_result=dict())
            gradio_messages.append(("", reply["content"]))
            # gradio_messages.append(("", f"今日のゲームのパズル番号は{get_puzzle_num()}です。それでは、始めましょう！言葉を当ててみてください。"))
            time.sleep(2)
            return gradio_messages
        
        def respond(user_input, gradio_messages,  guess_result=dict()):
            system_input = [get_rulebook(), """あなたは「イミトル」というゲームの進行役です。
                            普段、ユーザーは答えをしますが、たまにゲームに関するヒントをリクエストをします。短くても丁寧に答えてください。
                            ヒントを教える場合、正解を必ず隠してください。絶対、正解を言わないでください。"""]

            _user_input = "答え:" + user_input
            reply, messages, guess_result = create_chat(system_input, _user_input, guess_result=dict())
            gradio_messages.append((user_input, reply["content"]))
            time.sleep(2)
            return gradio_messages, guess_result
        
        def update_guesses(cur, i, guessed_words, guesses_df):
            if cur.get('guess') and cur['guess'] not in guessed_words:
                guessed_words.add(cur['guess'])
                cur['sim'] = round(cur['sim'], 2)
                cur['i'] = i
                guesses_df.loc[i] = cur
                i += 1
                guesses_df = guesses_df.sort_values(by=["sim"], ascending=False)
            return i, guessed_words, guesses_df

        api_key.change(unfreeze, [], [msg]).then(reset_history, [], [history]).then(greet, [api_key, chatbot], [chatbot])
        msg.submit(respond, [msg, chatbot, cur_guess], [chatbot, cur_guess])
        cur_guess.change(update_guesses, [cur_guess, idx, guessed, guesses_table], [idx, guessed, guesses_table])
            
    gr.Examples(
        [
            ["猫"],
            ["どんなヒントが貰える？"],
            ["正解と「近い」とはどういう意味？"],
            ["何から始めたらいい？"],
            ["今日の正解は何？"],
        ],
        inputs=msg,
        label="こちらから選んで話すこともできます."
    )

if __name__ == "__main__":
    demo.queue(concurrency_count=20).launch()
# demo.queue(concurrency_count=20).launch()
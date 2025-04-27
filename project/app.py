from flask import Flask, render_template, request, jsonify
import boto3
import uuid
import os
import json
import base64 # 引入 base64 模組來編碼音訊資料

app = Flask(__name__)

# --- AWS 服務設定 ---
# 請替換成您的 Agent ID 和 Alias ID
# 建議使用環境變數或安全的方式配置這些敏感資訊
BEDROCK_AGENT_ID = os.environ.get("BEDROCK_AGENT_ID", "YOUR_AGENT_ID")
BEDROCK_AGENT_ALIAS_ID = os.environ.get("BEDROCK_AGENT_ALIAS_ID", "YOUR_AGENT_ALIAS_ID")

# 請將 "您的_Bedrock_Agent_所在的區域" 替換為實際的區域代碼
AWS_REGION = os.environ.get("AWS_DEFAULT_REGION", "us-west-2") # Bedrock 和 Polly 最好在同一區域

# Bedrock Agent Runtime 用戶端
# 請確保您的 EC2 實例附加了具有 bedrock-agent-runtime:InvokeAgent 權限的 IAM Role
try:
    bedrock_agent_client = boto3.client("bedrock-agent-runtime", region_name=AWS_REGION)
    print(f"成功創建 Bedrock Agent Runtime 客戶端 (區域: {AWS_REGION})")
except Exception as e:
    print(f"創建 Bedrock Agent Runtime 客戶端失敗: {e}")
    bedrock_agent_client = None # 如果創建失敗，設置為 None


# Amazon Polly 用戶端
# 請確保您的 EC2 實例附加的 IAM Role 具有 polly:SynthesizeSpeech 權限
try:
    polly_client = boto3.client("polly", region_name=AWS_REGION)
    print(f"成功創建 Amazon Polly 客戶端 (區域: {AWS_REGION})")
except Exception as e:
     print(f"創建 Amazon Polly 客戶端失敗: {e}")
     polly_client = None # 如果創建失敗，設置為 None


# 請選擇一個支援中文 (繁體中文) 的 Polly 語音 ID
# 例如：Standard - Zhiyu (女聲)
#       Neural - Zhiyu (女聲, 通常更自然，但成本較高，檢查您的區域是否支援)
# 請參考 AWS 文件確認您區域支援的語音：https://docs.aws.amazon.com/polly/latest/dg/voicelist.html
POLLY_VOICE_ID = "Zhiyu"
# 判斷使用哪個引擎 (標準或神經網絡)
POLLY_ENGINE = 'neural' if POLLY_VOICE_ID.endswith('-Neural') else 'standard'


# 簡單的字典來存儲每個會話的 session ID
# 注意：這在生產環境中不安全且不可擴展，僅為範例
session_ids = {}

@app.route('/')
def index():
    """
    渲染前端 HTML 頁面
    """
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """
    接收前端傳來的語音轉文字結果，調用 Bedrock Agent，獲取回應，
    然後使用 Polly 將回應轉為語音，並返回文字和音訊給前端。
    """
    data = request.json
    user_input_text = data.get('text')
    client_id = data.get('clientId')

    # 檢查客戶端 ID 和輸入文字
    if not client_id:
         return jsonify({"error": "客戶端 ID 無效"}), 400
    if not user_input_text:
        # 如果輸入文字是空的 (語音辨識可能沒有結果)
        # 返回一個空回應，前端會處理
        return jsonify({"agentResponse": "未偵測到有效語音輸入。", "audio": None}), 200


    # 檢查 AWS 客戶端是否成功創建
    if not bedrock_agent_client:
         error_msg = "後端服務錯誤：無法初始化 Bedrock Agent 客戶端。"
         print(error_msg)
         return jsonify({"agentResponse": error_msg, "audio": None}), 500
    if not polly_client:
         error_msg = "後端服務錯誤：無法初始化 Amazon Polly 客戶端。"
         print(error_msg)
         # 即使 Polly 客戶端初始化失敗，我們還是嘗試返回 Bedrock 文字回應（如果 Agent 調用成功的話）
         # 但是這裡為了簡化，如果 Polly 都失敗了，可能 Bedrock 也會有憑證問題，直接報錯
         # 根據您的需求，您可以選擇在沒有 Polly 客戶端時只返回文字
         # return jsonify({"agentResponse": "後端服務錯誤：無法初始化語音服務。", "audio": None}), 500
         # 暫時保持原邏輯，讓後面的 Bedrock 呼叫去產生憑證錯誤


    # 獲取或生成該 client 的 session ID
    session_id = session_ids.get(client_id)
    if not session_id:
        session_id = str(uuid.uuid4())
        session_ids[client_id] = session_id
        print(f"新的會話開始 for client {client_id}, session ID: {session_id}")
    else:
         print(f"繼續會話 for client {client_id}, session ID: {session_id}")

    print(f"收到來自 client {client_id} 的訊息: {user_input_text}")

    agent_response_text = "抱歉，Bedrock Agent 回應失敗或無回應。"
    audio_base64 = None # 初始化音訊資料為 None

    try:
        # --- 調用 Bedrock Agent ---
        print("正在調用 Bedrock Agent...")
        response = bedrock_agent_client.invoke_agent(
            agentId=BEDROCK_AGENT_ID,
            agentAliasId=BEDROCK_AGENT_ALIAS_ID,
            sessionId=session_id,
            inputText=user_input_text,
            enableTrace=False # 可以設置為 True 進行調試
        )

        # 解析 Bedrock Agent 的回應 (串流)
        print("正在解析 Agent 回應串流...")
        event_stream = response.get('completion')
        if event_stream:
            agent_response_parts = []
            for event in event_stream:
                if 'chunk' in event:
                    chunk = event['chunk']
                    agent_response_parts.append(chunk['bytes'].decode('utf-8'))
                # 可以處理 'trace' 等事件用於調試
                elif 'trace' in event:
                    # print(f"Agent Trace: {json.dumps(event['trace'], indent=2)}") # 太多時可能佔滿日誌
                    pass # 忽略詳細追蹤
                elif 'badInput' in event:
                     print(f"Agent Bad Input: {event['badInput']}")
                     agent_response_parts.append("Agent Bad Input: " + event['badInput'].get('message', 'Unknown error'))
                # ... 處理其他可能的事件類型
            agent_response_text = "".join(agent_response_parts)
            print(f"收到 Agent 回應文字: {agent_response_text}")
        else:
             print("Bedrock Agent 回應中沒有 completion stream")
             agent_response_text = "Bedrock Agent 無回應。"


        # --- 使用 Amazon Polly 轉語音 ---
        # 只有當 Agent 有有效回應且 Bedrock Client 創建成功時才轉語音
        if agent_response_text and not agent_response_text.startswith("調用 Bedrock Agent 時發生錯誤:") and not agent_response_text.startswith("Bedrock Agent 無回應") and polly_client:
            try:
                print("正在調用 Amazon Polly 合成語音...")
                polly_response = polly_client.synthesize_speech(
                    Text=agent_response_text,
                    OutputFormat="mp3", # 音訊格式，前端需要支援 (mp3 是常見的)
                    VoiceId=POLLY_VOICE_ID, # 選擇語音
                    Engine=POLLY_ENGINE
                )

                # 讀取音訊串流並進行 Base64 編碼
                audio_stream = polly_response.get("AudioStream")
                if audio_stream:
                    audio_bytes = audio_stream.read()
                    audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
                    print("Amazon Polly 音訊已生成並 Base64 編碼")
                else:
                     print("Amazon Polly 未返回 AudioStream")

            except Exception as polly_e:
                print(f"調用 Amazon Polly 時發生錯誤: {polly_e}")
                # 不阻止返回文字回應，只記錄錯誤
                # agent_response_text += f" (語音合成失敗: {polly_e})" # 不在回應中顯示 Polly 錯誤
                pass # 忽略 Polly 錯誤，只返回文字


    except Exception as e:
        print(f"調用 Bedrock Agent 時發生錯誤: {e}")
        # 這個錯誤通常包含憑證或權限問題
        agent_response_text = f"調用 Bedrock Agent 時發生錯誤: {e}"
        # 如果發生 Agent 錯誤，考慮從 session_ids 中移除該 client 的 session ID
        # session_ids.pop(client_id, None)


    # 將 Agent 的文字回應和 Base64 編碼的音訊數據返回給前端
    return jsonify({
        "agentResponse": agent_response_text,
        "audio": audio_base64 # 如果音訊生成失敗，這裡會是 None
    })

if __name__ == '__main__':
    # 在生產環境中應使用 Gunicorn, uWSGI 等 WSGI 伺服器
    # debug=True 僅用於開發測試
    # host='0.0.0.0' 允許從外部訪問（EC2 上需要）
    # port=3000 與前端 Fetch API 中的 URL 相符
    app.run(debug=True, host='0.0.0.0', port=3000)
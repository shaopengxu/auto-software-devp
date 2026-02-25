import requests
import json
import sys

class OpenCodeClient:
    def __init__(self, base_url="http://localhost:3000"):
        self.base_url = base_url
        self.session_id = None

    def create_session(self, model=None):
        """创建一个新的交互会话"""
        url = f"{self.base_url}/session"
        try:
            # 这里的 payload 可以为空，或者包含 title/parentID
            payload = {}
            if model:
                payload["model"] = model
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                # 根据文档，返回的是 Session 对象，通常包含 id
                self.session_id = data.get("id")
                print(f"✅ 会话创建成功: {self.session_id}")
                return self.session_id
            else:
                print(f"❌ 创建会话失败: {response.text}")
        except Exception as e:
            print(f"❌ 连接服务器失败: {e}")

    def chat(self, prompt, session_id=None, model=None):
        """发送指令并获取响应，返回LLM的文本响应内容。
        
        每次调用默认创建一个新会话，除非通过 session_id 参数指定复用已有会话。

        Args:
            prompt: 发送给 LLM 的文本指令
            session_id: 可选，指定使用的会话 ID。未传入时自动创建新会话。
            model: 可选，指定使用的模型。

        Returns:
            LLM 的文本响应，失败时返回 None
        """
        # 如果传入了 session_id 则复用，否则创建新会话
        if session_id:
            current_session_id = session_id
        else:
            current_session_id = self.create_session(model=model)
            if not current_session_id:
                print("⚠️ 自动创建会话失败")
                return None

        url = f"{self.base_url}/session/{current_session_id}/message"
        
        # 构建符合文档的消息 payload
        payload = {
            "parts": [
                {"type": "text", "text": prompt}
            ]
        }
        # 当指定模型时，仅放在 options 中，或者如果在 create_session 中已指定则不需要再次指定
        if model:
            # 兼容 OpenCode，如果是特定对象结构可以替换为此处如果服务需要
            payload["options"] = {"model": model}
        
        try:
            print("🤖 OpenCode 正在思考...")
            response = requests.post(url, json=payload)
            
            if response.status_code == 200:
                data = response.json()
                # 响应结构: { info: Message, parts: Part[] }
                parts = data.get("parts", [])
                result_text = ""
                for part in parts:
                    if part.get("type") == "text":
                        text = part.get("text", "")
                        print(text)
                        result_text += text
                print("\n✅ 完成")
                return result_text
            else:
                print(f"❌ 请求失败: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            print(f"❌ 发送消息异常: {e}")
            return None

# --- 使用示例 ---
if __name__ == "__main__":
    # 1. 初始化客户端 (确保 opencode serve --port 3000 已启动)
    client = OpenCodeClient()
    
    # 2. 直接发送指令（chat 会自动创建新会话）
    client.chat('''实现一个基于vue3+vite5+ts的单页应用，使用element-plus作为ui框架，使用pinia作为状态管理，使用axios作为http请求库，使用vue-router作为路由管理''')


    
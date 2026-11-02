import json
from openai import OpenAI
from rag_engine import ask_pdf
from tools import get_stock_price, analyze_sentiment

client = OpenAI(
    api_key="API-key",
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)

SYSTEM_PROMPT = """你是一个专业的金融投研助手。你可以使用以下工具：
1. search_pdf: 当用户询问关于财报或研报的具体内容时，调用此工具查资料。参数: {"query": "搜索词"}
2. get_stock_price: 当用户询问股票实时价格时调用。参数: {"stock_name": "股票名称"}
3. analyze_sentiment: 当用户要求分析某段新闻的情感时调用。参数: {"news_text": "新闻内容"}
4. Final Answer: 当你不需要调用工具，或者已经拿到工具结果可以回答用户时，使用此动作。

你必须严格输出 JSON 格式，不要输出任何其他废话。格式如下：
{"thought": "你的思考过程", "action": "工具名称或Final Answer", "action_input": "工具参数或最终回答"}
"""

session_memory = {}


def agent_chat(query, session_id="default", max_iterations=5):
    if session_id not in session_memory:
        session_memory[session_id] = [{"role": "system", "content": SYSTEM_PROMPT}]

    history = session_memory[session_id]
    history.append({"role": "user", "content": query})

    final_answer = "抱歉，达到最大循环次数或处理失败。"

    for i in range(max_iterations):
        try:
            response = client.chat.completions.create(
                model="glm-4-flash",
                messages=history,
                temperature=0.1
            )

            content = response.choices[0].message.content

            # 强制转换类型
            if isinstance(content, dict):
                res_text = json.dumps(content, ensure_ascii=False)
            elif isinstance(content, list):
                res_text = json.dumps(content, ensure_ascii=False)
            else:
                res_text = str(content).strip()

            # 清洗可能存在的 Markdown 标记
            if res_text.startswith("```json"):
                res_text = res_text[7:]
            if res_text.endswith("```"):
                res_text = res_text[:-3]
            res_text = res_text.strip()

            # 解析 JSON
            parsed = json.loads(res_text)
            action = parsed.get("action")
            action_input = parsed.get("action_input")
            thought = parsed.get("thought")

            # 如果大模型决定结束
            if action == "Final Answer":
                if isinstance(action_input, dict):
                    final_answer = json.dumps(action_input, ensure_ascii=False)
                else:
                    final_answer = str(action_input)
                break

            # 如果大模型决定调用工具
            observation = ""

            # 防御：如果参数是字典，提取里面的值
            if isinstance(action_input, dict):
                if action == "search_pdf":
                    action_input = action_input.get("query", "")
                elif action == "get_stock_price":
                    action_input = action_input.get("stock_name", "")
                elif action == "analyze_sentiment":
                    action_input = action_input.get("news_text", "")

            action_input = str(action_input)

            if action == "search_pdf":
                observation = ask_pdf(action_input)
            elif action == "get_stock_price":
                observation = get_stock_price(action_input)
            elif action == "analyze_sentiment":
                observation = analyze_sentiment(action_input)
            else:
                observation = f"未知的工具: {action}"

            # 将思考和观察结果加入历史
            history.append({"role": "assistant", "content": res_text})
            history.append({"role": "user", "content": f"Observation: {observation}"})

        except json.JSONDecodeError as e:
            history.append({"role": "user",
                            "content": "你刚才输出的格式不是合法JSON，请严格按要求输出JSON格式，不要包含任何其他文字或代码块标记。"})
        except Exception as e:
            # 【关键修改】把报错信息直接返回，方便排查！
            final_answer = f"【Agent内部报错】: {str(e)}"
            break

    history.append({"role": "assistant", "content": final_answer})
    return final_answer
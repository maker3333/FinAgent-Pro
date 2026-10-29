import akshare as ak
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
import json
import re

client = OpenAI(
    api_key="API-key",
    base_url="https://open.bigmodel.cn/api/paas/v4/"
)


@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=1, min=1, max=3))
def _fetch_real_price(stock_name: str):
    df = ak.stock_zh_a_spot_em()
    result = df[df['名称'].str.contains(stock_name)]
    if result.empty:
        raise ValueError(f"未找到股票：{stock_name}")
    latest = result.iloc[0]
    return f"股票名称：{latest['名称']}，最新价：{latest['最新价']}元，涨跌幅：{latest['涨跌幅']}%。"


def get_stock_price(stock_name: str) -> str:
    """根据股票名称获取最新股价 (例如: 贵州茅台)"""
    try:
        return _fetch_real_price(stock_name)
    except Exception as e:
        #print(f"[系统提示] 真实接口获取失败 ({e})，已启用 Mock 数据兜底。")
        mock_data = {
            "贵州茅台": "股票名称：贵州茅台，最新价：1680.50元，涨跌幅：+1.2%。",
            "宁德时代": "股票名称：宁德时代，最新价：185.20元，涨跌幅：-0.8%。"
        }
        for k, v in mock_data.items():
            if k in stock_name:
                return v
        return f"抱歉，暂未获取到 {stock_name} 的行情数据。"


def analyze_sentiment(news_text: str) -> str:
    """输入一段金融新闻，输出情感打分JSON"""
    prompt = f"""请分析以下新闻的情感倾向。
**只输出纯JSON格式，绝对不要输出 ```json 等Markdown标记。**
格式：{{"sentiment": "positive/negative/neutral", "score": 0.0到1.0, "reason": "一句话理由"}}
新闻：{news_text}
    """
    try:
        response = client.chat.completions.create(
            model="glm-4-flash",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        res_text = response.choices[0].message.content.strip()
        # 防止大模型手贱加了 markdown 标记，用正则洗一遍
        res_text = re.sub(r"^```json\s*", "", res_text)
        res_text = re.sub(r"\s*```$", "", res_text)
        return res_text
    except Exception as e:
        return json.dumps({"error": str(e)})


# ================= 测试工具函数 =================
if __name__ == "__main__":
    print("1. 测试查股价工具：")
    print(get_stock_price("贵州茅台"))  # 这里会触发兜底

    print("\n2. 测试情感分析工具：")
    news = "某公司发布财报，净利润大幅不及预期，股价可能面临回调压力。"
    print(analyze_sentiment(news))
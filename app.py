import yfinance as yf
import google.generativeai as genai
import json
import os
import datetime
import time
import argparse
import sys
import pandas as pd

# --- 設定區 (GitHub Actions 專用) ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# --- 1. 核心：自動偵測 Google 最新模型 (與 Colab 同步) ---
def get_best_models():
    """自動偵測 Experimental > Pro > Flash"""
    defaults = ["models/gemini-1.5-pro", "models/gemini-1.5-flash"]
    try:
        all_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        all_models.sort(reverse=True) # 版本號降序 (3.0 > 2.0 > 1.5)
        
        # 分類篩選
        exp = [m for m in all_models if 'exp' in m]
        pro = [m for m in all_models if 'pro' in m and 'exp' not in m]
        flash = [m for m in all_models if 'flash' in m and 'exp' not in m]
        
        # 組合：先試實驗版，再試 Pro，最後 Flash
        final_list = exp + pro + flash
        if final_list: return final_list
        return defaults
    except:
        return defaults

# 程式啟動時，取得最強模型清單
MODEL_PRIORITY = get_best_models()
print(f"🧠 模型清單已更新: {[m.split('/')[-1] for m in MODEL_PRIORITY[:3]]}...")

# --- 2. Prompt (鎖定事實 + 預測未來) ---
PROMPT_TEMPLATE = """
你是 bbb 專業分析師。請基於以下【絕對事實】補完分析報告。

【鎖定事實 (API Data)】- **嚴禁修改數值**：
- 股票：{name} ({stock_id})
- 現價：{price} ({change_pct})
- 歷史股價(供繪圖參考)：{chart_dump}

【你的任務 (需聯網搜尋)】：
1. **財務補完**：
   - 營收：若本月尚未公布，請預估並標記 `is_estimate: true`。
   - EPS：搜尋預估值，標記 `is_estimate: true`。
   - 估值：依據歷史股價(實線)計算合理 PE 倍數區間(虛線)。
2. **質性分析**：產業護城河、競爭者。
3. **技術判讀**：給出 30/180/360 天價格預測與策略。

請回傳 **純 JSON**，格式如下 (不要用 Markdown)：
{{
  "industry": {{ "moat_status": "..", "position_map": "..", "competitors": ".." }},
  "financials": {{
    "eps_table": [
       {{ "period": "2024Q3", "gross_margin": "..", "eps": "事實", "cumulative": "..", "is_estimate": false }},
       {{ "period": "2025Q1", "gross_margin": "..", "eps": "預估", "cumulative": "..", "is_estimate": true }}
    ],
    "revenue_trend": [
       {{ "month": "2024-11", "revenue": "..", "mom": "..", "yoy": "..", "is_estimate": false }}
    ],
    "valuation": {{
        "pe_status": "..", "pb": "..", "roe": "..",
        "pe_river_data": {{
            "dates": [], "price": [], "pe20": [], "pe16": [], "pe12": [] 
        }}
    }}
  }},
  "technical": {{
    "status": "..", "signal_light": "red_flash/green_flash/stable", 
    "analysis_text": "..",
    "predictions": {{ "days30": "..", "days180": "..", "days360": "..", "entry_zone": ".." }},
    "correction_c": "0.XX",
    "bollinger": {{ "status": "..", "description": ".." }}
  }},
  "news_events": {{
    "news": [ {{ "date": "YYYY-MM-DD", "title": "..", "type": "positive/neutral/negative", "is_new": true }} ],
    "calendar": [ {{ "date": "YYYY-MM-DD", "event": ".." }} ]
  }},
  "dividend": {{ "yield": "..", "history_roi": "..", "future_roi": ".." }},
  "memo": ""
}}
"""

def get_current_list():
    if os.path.exists('data.json'):
        try:
            with open('data.json', 'r', encoding='utf-8') as f:
                d = json.load(f)
                return d if isinstance(d, list) else []
        except: pass
    return []

def get_stock_data(target_id, old_data=None):
    stock_id = target_id.replace(".TW", "")
    print(f"🚀 分析: {stock_id} ...")
    
    try:
        ticker = yf.Ticker(f"{stock_id}.TW")
        
        # A. 抓取事實 (Facts)
        price = 0; change_pct = "0%"
        try:
            fast = ticker.fast_info
            price = fast.get('last_price', 0)
            prev = fast.get('previous_close', 0)
            # 備援機制
            if price == 0: 
                h = ticker.history(period="5d")
                if not h.empty:
                    price = h['Close'].iloc[-1]
                    prev = h['Close'].iloc[-2]
            
            if price and prev:
                change_pct = f"{(((price - prev)/prev)*100):+.2f}%"
        except: pass

        if price == 0: return None

        # B. 歷史股價與新聞
        news_summary = ""
        try:
            for n in ticker.news[:3]:
                t = n.get('title'); d = datetime.datetime.fromtimestamp(n.get('providerPublishTime', 0)).strftime('%Y-%m-%d')
                news_summary += f"- {d}: {t}\n"
        except: pass

        # C. K 線數據 (為了節省 Prompt token，只取重點給 AI)
        hist = ticker.history(period="1y")
        chart_data_for_ai = []
        chart_dates = [] # 給前端畫圖用的完整日期
        chart_prices = [] # 給前端畫圖用的完整價格
        
        if not hist.empty:
            # 完整數據
            res = hist['Close'].resample('ME').last().tail(12)
            chart_dates = [d.strftime('%Y-%m') for d in res.index]
            chart_prices = [round(x,2) for x in res.tolist()]
            
            # 給 AI 參考的精簡版
            chart_data_for_ai = [{"d": d, "p": p} for d, p in zip(chart_dates, chart_prices)]

        # D. AI 分析 (智慧輪詢機制 - 與 Colab 相同)
        ai_res = {}
        model_used = "N/A"
        
        if GEMINI_API_KEY:
            name = ticker.info.get('longName', stock_id)
            prompt = PROMPT_TEMPLATE.format(
                name=name, stock_id=stock_id, price=f"{price:.2f}", 
                change_pct=change_pct, 
                chart_dump=json.dumps(chart_data_for_ai),
                news_summary=news_summary
            )
            
            # 🔥 自動切換 Failover
            for m in MODEL_PRIORITY:
                try:
                    # print(f"嘗試 {m}...")
                    mod = genai.GenerativeModel(m)
                    resp = mod.generate_content(prompt)
                    ai_res = json.loads(resp.text.replace("```json","").replace("```","").strip())
                    model_used = m.replace("models/", "")
                    print(f"  ✅ 成功使用模型: {model_used}")
                    break
                except Exception as e:
                    # print(f"  ⚠️ {m} 失敗，換下一個")
                    continue

        # E. 合併資料
        fin = ai_res.get("financials", {})
        val = fin.get("valuation", {})
        riv = val.get("pe_river_data", {})
        
        # 確保河流圖結構完整 (若 AI 算失敗，至少實線要出來)
        final_river = {
            "dates": chart_dates,
            "price": chart_prices,
            "pe20": riv.get("pe20", []),
            "pe16": riv.get("pe16", []),
            "pe12": riv.get("pe12", [])
        }

        # 建構最終物件
        return {
            "id": stock_id,
            "name": name if 'name' in locals() else stock_id,
            "category": old_data.get('category', '新加入') if old_data else '新加入',
            "lastUpdated": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "ai_model": model_used,
            "memo": old_data.get('memo', '') if old_data else '',
            "basicInfo": { "price": f"{price:.2f}", "change": f"{(price-(price/(1+float(change_pct.strip('%'))/100))):+.2f}", "changePercent": change_pct },
            
            "industry": ai_res.get("industry", {}),
            "news_events": ai_res.get("news_events", {"news":[], "calendar":[]}),
            "financials": {
                "eps_table": fin.get("eps_table", []),
                "revenue_trend": fin.get("revenue_trend", []),
                "valuation": { 
                    "pe_status": val.get("pe_status", "-"), 
                    "pb": str(ticker.info.get("priceToBook","-")), 
                    "roe": val.get("roe","-"), 
                    "pe_river_data": final_river 
                }
            },
            "technical": ai_res.get("technical", { "signal_light": "stable" }),
            "dividend": ai_res.get("dividend", {})
        }

    except Exception as e:
        print(f"❌ {stock_id} 處理失敗: {e}")
        return None

if __name__ == "__main__":
    current = get_current_list()
    old_map = {item['id']: item for item in current}
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--add', type=str)
    args = parser.parse_args()

    targets = list(old_map.keys())
    if args.add:
        nid = args.add.strip().upper()
        if nid not in targets:
            targets.insert(0, nid)
            old_map[nid] = {"category": "新加入"}

    final = []
    for sid in targets:
        # 讀取舊 memo
        old = old_map.get(sid)
        res = get_stock_data(sid, old)
        if res: final.append(res)
        else:
            if sid in old_map and 'name' in old_map[sid]: final.append(old_map[sid])
        time.sleep(2)

    with open('data.json', 'w', encoding='utf-8') as f:
        json.dump(final, f, ensure_ascii=False, indent=2)

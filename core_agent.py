# core_agent.py
import os
import re
import yaml
import json
import duckdb
import csv
import requests  # 新增：用于发送纯 HTTP 请求到内网模型
from typing import List
from datetime import datetime
from langchain_openai import ChatOpenAI

# 引入向量库相关组件（FAISS 在本地运行，不需要外网）
from langchain_community.vectorstores import FAISS        
from langchain_core.documents import Document     
from langchain_core.embeddings import Embeddings  

# ==========================================
# 1. 核心配置与常量
# ==========================================
os.environ['NO_PROXY'] =  '内网'
INTERNAL_API_BASE = "内网v1"  # 你的大模型基础 URL
INTERNAL_API_KEY =  "内网" # 你的大模型 KEY

# 【新增】：内网 Embedding 模型的配置
# 如果内网 Embedding 接口和大模型是同一个地址，直接沿用即可；如果不同，请替换！
EMBEDDING_API_BASE = INTERNAL_API_BASE 
EMBEDDING_API_KEY = INTERNAL_API_KEY   
EMBEDDING_MODEL_NAME = "bge-m3"  # 请确认你们内网 Embedding 模型的调用名称

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "telecom_data.duckdb")
LOG_PATH = os.path.join(BASE_DIR, "query_logs.csv")

# ==========================================
# 2. 自定义内网 Embedding 调用类（100% 免疫网络报错）
# ==========================================
class IntranetEmbeddings(Embeddings):
    """自定义的 Embedding 类，纯 HTTP 请求，绝对不会触发本地下载和 tiktoken 校验"""
    def __init__(self, api_url: str, api_key: str, model_name: str):
        self.api_url = api_url.rstrip("/")
        # 自动补全 OpenAI 标准的 embeddings 路径
        if not self.api_url.endswith("/embeddings"):
            self.api_url += "/embeddings" if self.api_url.endswith("/v1") else "/v1/embeddings"
        self.api_key = api_key
        self.model_name = model_name

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        payload = {"input": texts, "model": self.model_name}
        try:
            # 发送数据到内网计算向量
            response = requests.post(self.api_url, json=payload, headers=headers, timeout=15)
            response.raise_for_status()
            data = response.json()
            return [item["embedding"] for item in data["data"]]
        except Exception as e:
            print(f"❌ 内网 Embedding API 调用失败: {e}")
            # 即使失败也返回一个零向量，确保程序不会崩溃停止 (BGE-M3 通常是 1024 维)
            return [[0.0] * 1024 for _ in texts] 

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# ==========================================
# 3. 通用安全与日志拦截器
# ==========================================
def sanitize_sql(sql):
    if re.search(r'(?i)\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|GRANT|REVOKE)\b', sql):
        raise ValueError("安全拦截：禁止执行此类破坏性 SQL！")
    if not re.search(r'(?i)\b(SUM|COUNT|AVG|MAX|MIN|GROUP BY)\b', sql) and not re.search(r'(?i)\bLIMIT\b', sql):
        sql = sql.strip().rstrip(';') + " LIMIT 1000"
    return sql

def log_query_action(question, sql, status, error_msg=""):
    try:
        file_exists = os.path.isfile(LOG_PATH)
        with open(LOG_PATH, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if not file_exists: 
                writer.writerow(["时间", "用户问题", "执行SQL", "状态", "报错信息"])
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), question, sql, status, error_msg])
    except Exception as e:
        pass

# ==========================================
# 4. 核心 Agent 大脑
# ==========================================
class VisualTelecomAnalyst:
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=INTERNAL_API_KEY,
            openai_api_base=INTERNAL_API_BASE,
            model_name="deepseek-v3-0324",
            temperature=0.0  
        )
        
        # 【新增】：初始化我们的纯净版内网 Embedding 模型
        self.embeddings = IntranetEmbeddings(
            api_url=EMBEDDING_API_BASE,
            api_key=EMBEDDING_API_KEY,
            model_name=EMBEDDING_MODEL_NAME
        )
        
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"找不到数据库 {DB_PATH}，请先运行 build_db.py！")
        self.con = duckdb.connect(DB_PATH, read_only=True)
        
        yaml_path = os.path.join(BASE_DIR, "1.ymal") # 或者 schema.yaml
        if not os.path.exists(yaml_path):
             yaml_path = os.path.join(BASE_DIR, "schema.yaml")
             
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
                self.golden_sqls = self.config.get("golden_sqls", [])
        except FileNotFoundError:
            self.config = {}
            self.golden_sqls = []

        # 【新增】：启动时调用内网模型，把 YAML 里的案例转化为高维向量存入 FAISS
        self.vector_store = None
        if self.golden_sqls:
            docs = [Document(page_content=item['question'], metadata={"sql": item['sql']}) for item in self.golden_sqls]
            self.vector_store = FAISS.from_documents(docs, self.embeddings)

    def get_real_schema(self):
        tables = self.con.execute("SHOW TABLES").df()['name'].tolist()
        context = ""
        for t in tables:
            cols = self.con.execute(f"DESCRIBE {t}").df()['column_name'].tolist()
            context += f"表名: {t} | 列名: {', '.join(cols)}\n"
        return context

    def retrieve_golden_sqls(self, user_query, top_k=2):
        """【恢复为高级向量检索】：使用 FAISS 在本地搜索最匹配的 SQL"""
        if not self.vector_store: return "无历史参考案例。"
        
        # 通过向量距离找到语义最相近的问题
        similar_docs = self.vector_store.similarity_search(user_query, k=top_k)
        
        best_examples = ""
        for i, doc in enumerate(similar_docs):
            best_examples += f"[案例 {i+1}]\n问题: {doc.page_content}\nSQL: {doc.metadata['sql']}\n\n"
        return best_examples.strip()

    def run_workflow(self, user_query, history=[]):
        current_schema = self.get_real_schema()
        few_shot_examples = self.retrieve_golden_sqls(user_query)
        
        # 包含了之前刚刚为你优化的 多维对比规则 和 并排查询规则
        system_prompt = f"""
        你是一个资深的运营商无线网络数据分析专家。
        【当前数据库结构】：\n{current_schema}
        【黄金 SQL 参考库】：\n{few_shot_examples}

            【商用执行准则】：
            1. **路由要求**：优先使用全网网络数据和财报数据，除非用户提到了区域、站点级等详细数据再进入站点级明细表； 
            2. **消灭反问与多维展示**：极力避免向用户反问！当问题存在指标歧义（如“利用率”包含4G和5G）时，**必须在同一个 SELECT 语句中同时查询这两个字段作为并排的多列展示**（例如 `SELECT AVG(4G), AVG(5G)`），**绝对禁止**使用 UNION ALL 将不同指标的列上下拼接！
            3. **多维对比铁律**：当用户要求对比多个对象（例如：MTN/VDC/Telkom各区域流量对比）时，你的SQL**必须严格返回3列**：第1列为X轴维度（如区域），第2列为图例分组（如Operator），第3列为数值（如总流量）。
            4. **DuckDB 铁律**：日期列严禁使用 LIKE，请用 EXTRACT(YEAR FROM "列")。所有别名必须用双引号 `""`。
            5. 如用户提问中有趋势、对比、图表等，要画出图表。
            6. 当用户不明确提出时间时，默认采用最近12个月或者最新一个月）； 输出数据注释（精准提取查询所用的时间和表名），格式为 `COMMENT: 数据来源：<引用的核心表名> | 时间范围：<提取用户查询的年份或月份>`。
            7、当查不到数据的时候，不要瞎编，直接承认无相关数据；
            
            【输出协议（重要！）】：
            - 第一行必须严丝合缝输出 `SQL: <你的SQL>`。
            - 第二行输出图表类型：
              `CHART: line` (折线图), 
              `CHART: bar` (普通单柱状图), 
              `CHART: multi_bar` (多维对比簇状柱形图，当且仅当SQL返回3列：X轴、图例分组、数值时使用), 
              `CHART: pie` (饼图), 
              `CHART: dual_axis` (双轴图)。
              不需要画图则输出 `CHART: none`。            
            - 第三行输出极简图表标题（15字以内），格式为 `TITLE: <标题>`。
            """

        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_query}]
        return self.llm.invoke(messages).content.strip()

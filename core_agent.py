# core_agent.py
import os
import re
import yaml
import json
import duckdb
import csv
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_openai import ChatOpenAI, OpenAIEmbeddings # 新增 OpenAIEmbeddings
from langchain_community.vectorstores import FAISS        # 新增 FAISS 向量库
from langchain_core.documents import Document             # 新增 Document 结构

# ==========================================
# 1. 核心配置与常量
# ==========================================
os.environ['NO_PROXY'] = 'xxxxx'
INTERNAL_API_BASE = "xxxxx"   # 请替换
INTERNAL_API_KEY = "xxxx"     # 请替换


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "telecom_data.duckdb")
LOG_PATH = os.path.join(BASE_DIR, "query_logs.csv")

# ==========================================
# 2. 通用安全与日志拦截器
# ==========================================
def sanitize_sql(sql):
    """防止危险操作，自动附加 LIMIT"""
    if re.search(r'(?i)\b(DROP|DELETE|UPDATE|INSERT|ALTER|TRUNCATE|GRANT|REVOKE)\b', sql):
        raise ValueError("安全拦截：禁止执行此类破坏性 SQL！")
    if not re.search(r'(?i)\b(SUM|COUNT|AVG|MAX|MIN|GROUP BY)\b', sql) and not re.search(r'(?i)\bLIMIT\b', sql):
        sql = sql.strip().rstrip(';') + " LIMIT 1000"
    return sql

def log_query_action(question, sql, status, error_msg=""):
    """自动记录错题本"""
    try:
        file_exists = os.path.isfile(LOG_PATH)
        with open(LOG_PATH, mode='a', newline='', encoding='utf-8-sig') as f:
            writer = csv.writer(f)
            if not file_exists: 
                writer.writerow(["时间", "用户问题", "执行SQL", "状态", "报错信息"])
            writer.writerow([datetime.now().strftime("%Y-%m-%d %H:%M:%S"), question, sql, status, error_msg])
    except Exception as e:
        print(f"日志记录异常: {e}")

# ==========================================
# 3. 核心 Agent 大脑
# ==========================================
class VisualTelecomAnalyst:
    def __init__(self):
        self.llm = ChatOpenAI(
            openai_api_key=INTERNAL_API_KEY,
            openai_api_base=INTERNAL_API_BASE,
            model_name="deepseek-v3-0324",
            temperature=0.0  
        )
        # 初始化专门用来计算文本向量的模型
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=INTERNAL_API_KEY,
            openai_api_base=INTERNAL_API_BASE,
            model="text-embedding-3-small" # 或者使用你接口支持的嵌入模型
        )
        
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(f"找不到数据库 {DB_PATH}，请先运行 build_db.py！")
        self.con = duckdb.connect(DB_PATH, read_only=True)
        
        yaml_path = os.path.join(BASE_DIR, "schema.yaml")
        try:
            with open(yaml_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f)
                self.golden_sqls = self.config.get("golden_sqls", [])
        except FileNotFoundError:
            self.config = {}
            self.golden_sqls = []

        # 【核心升级】：启动时将 YAML 里的黄金问题转化为高维向量，存入 FAISS 记忆库
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
        """【核心升级】：使用语义相似度搜索，告别死板的关键字匹配"""
        if not self.vector_store: return "无历史参考案例。"
        
        # 让 FAISS 找出语义最相近的 top_k 个问题
        similar_docs = self.vector_store.similarity_search(user_query, k=top_k)
        
        best_examples = ""
        for i, doc in enumerate(similar_docs):
            best_examples += f"[案例 {i+1}]\n类似问题: {doc.page_content}\n正确SQL: {doc.metadata['sql']}\n\n"
        return best_examples.strip()


    def run_workflow(self, user_query, history=[]):
        current_schema = self.get_real_schema()
        few_shot_examples = self.retrieve_golden_sqls(user_query)
        
        system_prompt = f"""
            你是一个资深的运营商数据分析专家。
            【当前数据库结构】：\n{current_schema}
            【黄金 SQL 参考库】：\n{few_shot_examples}

            【商用执行准则】：
            1. **路由要求**：优先使用全网网络数据和财报数据，除非用户提到了区域、站点级等详细数据； 
            2. **消灭反问（全量并行展示）**：极力避免向用户反问！当问题存在歧义时，自行使用 UNION ALL 或同时 SELECT 多列全部展示。
            3. **DuckDB 铁律**：日期列严禁使用 LIKE，请用 EXTRACT(YEAR FROM "列")。所有别名必须用双引号 `""`。
            4. **输出协议（重要！）**：
            - 第一行必须严丝合缝输出 `SQL: <你的SQL>`。
            - 第二行输出图表类型：`CHART: line`, `CHART: bar`, `CHART: pie`, 或 `CHART: dual_axis` (双轴图，当且仅当查询出1个X轴和2个不同量纲的Y轴时使用)。不需要画图则输出 `CHART: none`。            - 第三行输出极简图表标题（15字以内），格式为 `TITLE: <标题>`。
            """
        messages = [{"role": "system", "content": system_prompt}] + history + [{"role": "user", "content": user_query}]
        return self.llm.invoke(messages).content.strip()

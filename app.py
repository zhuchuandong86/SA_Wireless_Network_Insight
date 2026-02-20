# app.py
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# 从我们刚刚分离的后端模块导入核心能力
from core_agent import VisualTelecomAnalyst, sanitize_sql, log_query_action


# 0. 页面初始化与画图配置
# ==========================================
st.set_page_config(
    page_title="南非运营商无线网络数据洞察 AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)



# 【重点修改这里】：加入多重备选字体，彻底消灭豆腐块
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


# ==========================================
# 1. 初始化后端 Agent (使用单例缓存)
# ==========================================
@st.cache_resource 
def get_agent():
    try:
        return VisualTelecomAnalyst()
    except FileNotFoundError as e:
        st.error(str(e))
        st.stop()

agent = get_agent()

# ==========================================
# 2. 纯粹的前端可视化函数
# ==========================================
def format_number(val):
    try:
        v = float(val)
        if pd.isna(v): return ""
        if v.is_integer() or abs(v) >= 1000: return f"{int(v):,}"
        return f"{v:,.2f}"
    except:
        return str(val)

def create_chart_figure(df, chart_type, title_text):
    if df.empty or len(df.columns) < 2: return None
    
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150) 
    
    # 【高阶视觉 1】：定义你们公司的专属品牌色系 (例如 MTN 黄, 科技蓝, 警示红)
    brand_palette = ["#FFC000", "#2F5597", "#C00000", "#70AD47", "#7030A0"]
    
    sns.set_theme(
        style="whitegrid", 
        rc={"font.sans-serif": plt.rcParams['font.sans-serif']},
        font_scale=0.9  
    )
    # 应用自定义色板
    sns.set_palette(sns.color_palette(brand_palette))
    
    x_col = df.columns[0]
    y_col = df.columns[1]
    
    if chart_type == "line": 
        sns.lineplot(data=df, x=x_col, y=y_col, marker="o", linewidth=3, ax=ax)
        for x_val, y_val in zip(df[x_col], df[y_col]):
            ax.text(x_val, y_val, format_number(y_val), ha='center', va='bottom', fontsize=9, color='#1F3864', fontweight='bold')
            
    elif chart_type == "bar": 
        sns.barplot(data=df, x=x_col, y=y_col, ax=ax)
        for p in ax.patches:
            val = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., val, format_number(val), ha='center', va='bottom', fontsize=9)

    elif chart_type == "dual_axis" and len(df.columns) >= 3:
        # 【高阶视觉 2】：双轴图 (Combo Chart)
        y2_col = df.columns[2]
        # 底部画柱状图 (主 Y 轴)
        sns.barplot(data=df, x=x_col, y=y_col, ax=ax, alpha=0.85, color=brand_palette[0], label=y_col)
        
        # 顶部画折线图 (副 Y 轴)
        ax2 = ax.twinx()
        sns.lineplot(data=df, x=x_col, y=y2_col, ax=ax2, color=brand_palette[2], marker="s", linewidth=2.5, label=y2_col)
        
        # 优化双轴图的图例和网格
        ax.grid(False) 
        ax2.grid(False)
        ax.set_ylabel(y_col, color=brand_palette[0], fontweight='bold')
        ax2.set_ylabel(y2_col, color=brand_palette[2], fontweight='bold')
        
        # 为折线图添加数字标签
        for x_val, y2_val in zip(df[x_col], df[y2_col]):
            ax2.text(x_val, y2_val, format_number(y2_val), ha='center', va='bottom', fontsize=9, color=brand_palette[2])
   # ... 在 app.py 的 create_chart_figure 函数中增加这段 ...

    elif chart_type == "multi_bar" and len(df.columns) >= 3:
        # X轴是第一列(区域)，图例(颜色)是第二列(运营商)，Y轴是第三列(流量数值)
        x_col = df.columns[0]
        hue_col = df.columns[1]
        y_col = df.columns[2]
        
        # 使用 seaborn 的 hue 参数自动生成多对比柱状图
        sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax, palette="muted")
        
        # 优化图例显示
        ax.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # 为每根柱子加上数字标签 (如果柱子太多，数字可能拥挤，视情况保留)
        for p in ax.patches:
            val = p.get_height()
            if val > 0: # 避免画空值的标签
                ax.text(p.get_x() + p.get_width() / 2., val, f'{val:,.1f}', 
                        ha='center', va='bottom', fontsize=8, rotation=45)
                        
                                 
    elif chart_type == "pie": 
        # 【高阶视觉 3】：从土气饼图升级为现代商业环形图 (Donut Chart)
        def pie_fmt(pct, allvals):
            absolute = int(np.round(pct/100.*np.sum(allvals)))
            return f"{pct:.1f}%\n({format_number(absolute)})"
            
        wedges, texts, autotexts = ax.pie(
            df[y_col], labels=df[x_col], autopct=lambda pct: pie_fmt(pct, df[y_col]), 
            startangle=140, pctdistance=0.85, 
            wedgeprops=dict(width=0.35, edgecolor='w') # width 参数把它变成了环形图
        )
        # 居中显示总计数值
        total_val = df[y_col].sum()
        ax.text(0, 0, f"总计\n{format_number(total_val)}", ha='center', va='center', fontsize=12, fontweight='bold')
        
    ax.set_title(title_text, fontsize=15, pad=15, fontweight='bold', color='#333333')
    
    if chart_type in ["line", "bar", "dual_axis"]:
        ax.set_xlabel(x_col, fontsize=11, color='#666666')
        ax.tick_params(axis='x', rotation=45)
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, ymax * 1.15)
        
    fig.tight_layout()
    return fig
    
# ==========================================
# 3. Web 交互主程序
# ==========================================
st.title("📡 南非运营商无线网络数据洞察 AI 助手")
st.markdown("直接用自然语言查询您的业务数据。支持自动绘图、一键导出。")

if "messages" not in st.session_state:
    st.session_state.messages = [] 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "dataframe" in msg: st.dataframe(msg["dataframe"], use_container_width=True)
        if "chart" in msg: st.pyplot(msg["chart"], use_container_width=False)

if prompt := st.chat_input("请输入您想查询的业务问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🧠 正在检索知识库并生成分析计划..."):
            res = agent.run_workflow(prompt, st.session_state.chat_history)
            
            # 协议解析
            sql_to_execute = ""
            sql_match = re.search(r'```sql\s*(.*?)\s*```', res, re.DOTALL)
            if sql_match: sql_to_execute = sql_match.group(1)
            elif "SQL:" in res: sql_to_execute = [line for line in res.split('\n') if line.startswith('SQL:')][0].replace("SQL:", "").strip().replace("```", "")
                
            chart_type = "none"
            chart_match = re.search(r'CHART:\s*(line|bar|pie|none)', res, re.IGNORECASE)
            if chart_match: chart_type = chart_match.group(1).lower()
            
            extracted_title = "数据可视化"
            title_match = re.search(r'TITLE:\s*(.*)', res, re.IGNORECASE)
            if title_match: extracted_title = title_match.group(1).strip()

            if sql_to_execute:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        safe_sql = sanitize_sql(sql_to_execute)
                        df = agent.con.execute(safe_sql).df() # 通过后端的 con 查库
                        
                        if df.empty:
                            st.warning("查询执行成功，但结果集为空。")
                            log_query_action(prompt, safe_sql, "SUCCESS_EMPTY")
                            st.session_state.messages.append({"role": "assistant", "content": "⚠️ 结果集为空。"})
                        else:
                            st.success(f"为您提取到 {len(df)} 行相关数据。")
                            reply_msg = {"role": "assistant", "content": "✅ 分析完成："}
                            
                            if chart_type != "none":
                                fig = create_chart_figure(df, chart_type, extracted_title)
                                if fig:
                                    st.pyplot(fig, use_container_width=False)
                                    reply_msg["chart"] = fig
                            
                            st.dataframe(df, use_container_width=True)
                            reply_msg["dataframe"] = df
                            st.session_state.messages.append(reply_msg)
                            
                            csv_data = df.to_csv(index=False).encode('utf-8-sig')
                            st.download_button("📥 下载数据 (CSV)", data=csv_data, file_name=f"{extracted_title}.csv", mime='text/csv')

                            log_query_action(prompt, safe_sql, "SUCCESS")
                            
                        st.session_state.chat_history = []
                        break
                        
                    except Exception as e:
                        error_msg = str(e)
                        if "安全拦截" in error_msg:
                            st.error(error_msg)
                            log_query_action(prompt, sql_to_execute, "BLOCKED", error_msg)
                            break
                            
                        if attempt < max_retries - 1:
                            err_prompt = f"报错: {error_msg}。请修复列名或语法。" if attempt < max_retries - 2 else f"报错: {error_msg}。最后一次机会！请直接输出 SELECT * FROM 表 LIMIT 10 兜底。"
                            st.session_state.chat_history.append({"role": "user", "content": err_prompt})
                            res = agent.run_workflow("重试", st.session_state.chat_history)
                            
                            sql_match = re.search(r'```sql\s*(.*?)\s*```', res, re.DOTALL)
                            sql_to_execute = sql_match.group(1) if sql_match else res.split('\n')[0].replace("SQL:", "").strip()
                            chart_match = re.search(r'CHART:\s*(line|bar|pie|none)', res, re.IGNORECASE)
                            if chart_match: chart_type = chart_match.group(1).lower()
                            title_match = re.search(r'TITLE:\s*(.*)', res, re.IGNORECASE)
                            if title_match: extracted_title = title_match.group(1).strip()
                        else:
                            st.error("由于数据结构复杂，AI 多次尝试仍未完美匹配。")
                            log_query_action(prompt, sql_to_execute, "FAILED", error_msg)
                            st.session_state.chat_history = []
            else:
                st.markdown(res)
                st.session_state.messages.append({"role": "assistant", "content": res})
                log_query_action(prompt, "无", "CHAT_ONLY", res)
                
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                st.session_state.chat_history.append({"role": "assistant", "content": res})
                if len(st.session_state.chat_history) > 6:
                    st.session_state.chat_history = st.session_state.chat_history[-6:]

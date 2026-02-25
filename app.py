# app.py
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# 从后端模块导入核心能力
from core_agent import VisualTelecomAnalyst, sanitize_sql, log_query_action

# ==========================================
# 0. 页面初始化、画图配置与【密码网关】
# ==========================================
st.set_page_config(
    page_title="南非运营商无线网络数据洞察 AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 【新增】：在这里设置你的访问密码
ACCESS_PASSWORD = "xxxx" 

def check_password():
    """验证密码的安全拦截器"""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    # 如果尚未认证，显示登录界面并拦截后续代码
    if not st.session_state.authenticated:
        # 画一个简单的居中登录框
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.write("")
            st.write("")
            st.title("🔒 内部数据洞察系统")
            st.markdown("该系统包含敏感商业数据，请验证权限。")
            
            with st.form("login_form"):
                pwd = st.text_input("🔑 请输入访问密码", type="password")
                submitted = st.form_submit_button("登 录", use_container_width=True)
                
                if submitted:
                    if pwd == ACCESS_PASSWORD:
                        st.session_state.authenticated = True
                        st.rerun() # 密码正确，重新加载页面，放行后续代码
                    else:
                        st.error("❌ 密码错误，请重新输入或联系管理员！")
        return False
    return True

# 【执行拦截】：如果没登录，程序到这里就强制停止，绝对安全！
if not check_password():
    st.stop()


# ==========================================
# (认证通过后才执行以下所有核心逻辑)
# ==========================================

# 加入多重备选字体，彻底消灭豆腐块
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
def format_number(val, is_pct=False):
    """升级版：支持智能百分比格式化"""
    try:
        v = float(val)
        if pd.isna(v): return ""
        
        if is_pct:
            if abs(v) <= 2.0: 
                return f"{v * 100:.2f}%"
            return f"{v:.2f}%"
            
        if v.is_integer() or abs(v) >= 1000: return f"{int(v):,}"
        return f"{v:,.2f}"
    except:
        return str(val)

def is_pct_col(col_name):
    """智能嗅探：根据列名判断是否应该显示为百分比"""
    return any(kw in str(col_name) for kw in ['率', '比', '%', '占比', '份额'])


def create_chart_figure(df, chart_type, title_text):
    if df.empty or len(df.columns) < 2: return None
    
    fig, ax = plt.subplots(figsize=(6, 4), dpi=150) 
    brand_palette = ["#FFC000", "#2F5597", "#C00000", "#70AD47", "#7030A0"]
    
    sns.set_theme(style="whitegrid", rc={"font.sans-serif": plt.rcParams['font.sans-serif']}, font_scale=0.9)
    sns.set_palette(sns.color_palette(brand_palette))
    
    x_col = df.columns[0]
    y_col = df.columns[1]

 # 【新增检查】：如果 Y 轴列不是数值类型，尝试寻找后续列中的第一个数值列
    if not np.issubdtype(df[y_col].dtype, np.number):
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            y_col = numeric_cols[0]
        else:
            return None # 确实没有数值可画

    y_is_pct = is_pct_col(y_col)
    
    if chart_type == "line": 
        sns.lineplot(data=df, x=x_col, y=y_col, marker="o", linewidth=3, ax=ax)
        for x_val, y_val in zip(df[x_col], df[y_col]):
            ax.text(x_val, y_val, format_number(y_val, y_is_pct), ha='center', va='bottom', fontsize=9, color='#1F3864', fontweight='bold')
            
    elif chart_type == "bar": 
        sns.barplot(data=df, x=x_col, y=y_col, ax=ax)
        for p in ax.patches:
            val = p.get_height()
            ax.text(p.get_x() + p.get_width() / 2., val, format_number(val, y_is_pct), ha='center', va='bottom', fontsize=9)

    elif chart_type == "multi_bar" and len(df.columns) >= 3:
        x_col, hue_col, y_col = df.columns[0], df.columns[1], df.columns[2]
        y_is_pct = is_pct_col(y_col) 
        
        sns.barplot(data=df, x=x_col, y=y_col, hue=hue_col, ax=ax, palette="muted")
        ax.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc='upper left')
        
        for p in ax.patches:
            val = p.get_height()
            if val > 0: 
                ax.text(p.get_x() + p.get_width() / 2., val, format_number(val, y_is_pct), 
                        ha='center', va='bottom', fontsize=8, rotation=45)

    elif chart_type == "dual_axis" and len(df.columns) >= 3:
        y2_col = df.columns[2]
        y2_is_pct = is_pct_col(y2_col) 
        
        sns.barplot(data=df, x=x_col, y=y_col, ax=ax, alpha=0.85, color=brand_palette[0], label=y_col)
        ax2 = ax.twinx()
        sns.lineplot(data=df, x=x_col, y=y2_col, ax=ax2, color=brand_palette[2], marker="s", linewidth=2.5, label=y2_col)
        
        ax.grid(False) 
        ax2.grid(False)
        ax.set_ylabel(y_col, color=brand_palette[0], fontweight='bold')
        ax2.set_ylabel(y2_col, color=brand_palette[2], fontweight='bold')
        
        for x_val, y2_val in zip(df[x_col], df[y2_col]):
            ax2.text(x_val, y2_val, format_number(y2_val, y2_is_pct), ha='center', va='bottom', fontsize=9, color=brand_palette[2])
                                 
    elif chart_type == "pie": 
        def pie_fmt(pct, allvals):
            absolute = int(np.round(pct/100.*np.sum(allvals)))
            return f"{pct:.1f}%\n({format_number(absolute)})"
            
        wedges, texts, autotexts = ax.pie(
            df[y_col], labels=df[x_col], autopct=lambda pct: pie_fmt(pct, df[y_col]), 
            startangle=140, pctdistance=0.85, wedgeprops=dict(width=0.35, edgecolor='w') 
        )
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
# 添加注销按钮 (可选功能，方便你在测试时退出登录)
colA, colB = st.columns([9, 1])
with colA:
    st.title("📡 南非运营商无线网络数据洞察 AI 助手")
with colB:
    if st.button("退出登录"):
        st.session_state.authenticated = False
        st.rerun()

st.markdown("直接用自然语言查询您的业务数据。支持自动绘图、一键导出。")

if "messages" not in st.session_state:
    st.session_state.messages = [] 
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 

# 渲染历史对话
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "dataframe" in msg: 
            format_mapping = {col: (lambda x: format_number(x, is_pct=True)) for col in msg["dataframe"].columns if is_pct_col(col)}
            display_df = msg["dataframe"].style.format(format_mapping) if format_mapping else msg["dataframe"]
            st.dataframe(display_df, use_container_width=True)
            
        if "comment" in msg and msg["comment"]: 
            st.caption(f"💡 **备注**：{msg['comment']}")
        if "chart" in msg: 
            st.pyplot(msg["chart"], use_container_width=False)

        # 渲染点赞/点踩按钮
        if msg["role"] == "assistant" and "sql" in msg:
            col1, col2, _ = st.columns([1, 1, 8]) 
            with col1:
                if st.button("👍 准确", key=f"up_{i}"):
                    log_query_action(msg["prompt"], msg["sql"], "FEEDBACK_GOOD", "用户点赞")
                    st.toast("✅ 感谢您的反馈！系统已记录。")
            with col2:
                if st.button("👎 报错/不准", key=f"down_{i}"):
                    log_query_action(msg["prompt"], msg["sql"], "FEEDBACK_BAD", "用户点踩")
                    st.toast("🔧 已将此问题打回错题本，我们将尽快优化！")

if prompt := st.chat_input("请输入您想查询的业务问题..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"): st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("🧠 正在检索知识库并生成分析计划..."):
            res = agent.run_workflow(prompt, st.session_state.chat_history)
            # print(f"res{res}")
            
            sql_to_execute = ""
            sql_match = re.search(r'```sql\s*(.*?)\s*```', res, re.DOTALL)
            if sql_match: sql_to_execute = sql_match.group(1)
            elif "SQL:" in res: sql_to_execute = [line for line in res.split('\n') if line.startswith('SQL:')][0].replace("SQL:", "").strip().replace("```", "")
                
            chart_type = "none"
            chart_match = re.search(r'CHART:\s*(multi_bar|line|bar|pie|dual_axis|none)', res, re.IGNORECASE)
            if chart_match: chart_type = chart_match.group(1).lower()
            
            extracted_title = "数据可视化"
            title_match = re.search(r'TITLE:\s*(.*)', res, re.IGNORECASE)
            if title_match: extracted_title = title_match.group(1).strip()
            
            extracted_comment = ""
            comment_match = re.search(r'COMMENT:\s*(.*)', res, re.IGNORECASE)
            if comment_match: extracted_comment = comment_match.group(1).strip()

            if sql_to_execute:
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        safe_sql = sanitize_sql(sql_to_execute)
                        df = agent.con.execute(safe_sql).df() 
                        # print(df)
                        
                        if df.empty:
                            st.warning("查询执行成功，但结果集为空。")
                            log_query_action(prompt, safe_sql, "SUCCESS_EMPTY")
                            st.session_state.messages.append({"role": "assistant", "content": "⚠️ 结果集为空。"})
                        else:
                            st.success(f"为您提取到 {len(df)} 行相关数据。")
                            reply_msg = {"role": "assistant", "content": f"✅ 分析完成：**{extracted_title}**"}
                            
                            if chart_type != "none":
                                fig = create_chart_figure(df, chart_type, extracted_title)
                                if fig:
                                    st.pyplot(fig, use_container_width=False)
                                    reply_msg["chart"] = fig
                            
                            format_mapping = {}
                            for col in df.columns:
                                if is_pct_col(col):
                                    format_mapping[col] = lambda x: format_number(x, is_pct=True)

                            display_df = df.style.format(format_mapping) if format_mapping else df
                            st.dataframe(display_df, use_container_width=True)

                            if extracted_comment:
                                st.caption(f"💡 **备注**：{extracted_comment}")
                                
                            reply_msg["dataframe"] = df
                            reply_msg["comment"] = extracted_comment
                            reply_msg["prompt"] = prompt
                            reply_msg["sql"] = safe_sql
                            
                            st.session_state.messages.append(reply_msg)
                            # print(reply_msg)
                            
                            csv_data = df.to_csv(index=False).encode('utf-8-sig')
                            st.download_button("📥 下载数据 (CSV)", data=csv_data, file_name=f"{extracted_title}.csv", mime='text/csv')

                            log_query_action(prompt, safe_sql, "SUCCESS")
                            
                            # # 在当次生成的实时画面中，也渲染赞/踩按钮
                            # current_idx = len(st.session_state.messages) - 1
                            # col1, col2, _ = st.columns([1, 1, 8])
                            # with col1:
                            #     if st.button("👍 准确", key=f"up_{current_idx}"):
                            #         log_query_action(prompt, safe_sql, "FEEDBACK_GOOD", "用户点赞")
                            #         st.toast("✅ 感谢您的反馈！系统已记录。")
                            # with col2:
                            #     if st.button("👎 报错/不准", key=f"down_{current_idx}"):
                            #         log_query_action(prompt, safe_sql, "FEEDBACK_BAD", "用户点踩")
                            #         st.toast("🔧 已将此问题打回错题本，我们将尽快优化！")
                            
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
                            chart_match = re.search(r'CHART:\s*(multi_bar|line|bar|pie|dual_axis|none)', res, re.IGNORECASE)
                            if chart_match: chart_type = chart_match.group(1).lower()
                            title_match = re.search(r'TITLE:\s*(.*)', res, re.IGNORECASE)
                            if title_match: extracted_title = title_match.group(1).strip()
                            comment_match = re.search(r'COMMENT:\s*(.*)', res, re.IGNORECASE)
                            if comment_match: extracted_comment = comment_match.group(1).strip()
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


#streamlit run app.py

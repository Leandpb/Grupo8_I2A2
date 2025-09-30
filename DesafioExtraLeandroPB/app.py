\
import os
import io
import re
import json
import time
import hashlib
import textwrap
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from dateutil import parser as dateparser
from sklearn.cluster import KMeans

# --- LLM (OpenAI) ---
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except Exception:
    OPENAI_AVAILABLE = False

#MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-5")  # override if needed
MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4.1-mini")  # override if needed
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -------------- Utility & Memory --------------
APP_TITLE = "Agente Genérico de Exploração e Análise de Dados (EDA) para arquivo tipo (CSV) - Autor: Leandro Pantalena Barbosa"
MEM_DIR = ".agent_memory"
os.makedirs(MEM_DIR, exist_ok=True)

def dataset_id_from_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]

def load_memory(mem_key: str) -> Dict[str, Any]:
    path = os.path.join(MEM_DIR, f"{mem_key}.jsonl")
    if not os.path.exists(path):
        return {"notes": [], "qa": []}
    data = {"notes": [], "qa": []}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line.strip())
                if obj.get("type") == "note":
                    data["notes"].append(obj)
                elif obj.get("type") == "qa":
                    data["qa"].append(obj)
            except Exception:
                continue
    return data

def append_memory(mem_key: str, obj: Dict[str, Any]) -> None:
    path = os.path.join(MEM_DIR, f"{mem_key}.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

# -------------- EDA Core --------------
def infer_datetime_cols(df: pd.DataFrame) -> List[str]:
    dt_cols = []
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            dt_cols.append(col)
        else:
            # try parse small sample
            try:
                sample = df[col].dropna().astype(str).head(20)
                if len(sample) > 0:
                    parsed = sample.map(lambda x: try_parse_datetime(x))
                    if parsed.notna().mean() > 0.7:
                        dt_cols.append(col)
            except Exception:
                pass
    return dt_cols

def try_parse_datetime(val: Any):
    try:
        return dateparser.parse(str(val))
    except Exception:
        return None

def basic_profile(df: pd.DataFrame) -> Dict[str, Any]:
    n_rows, n_cols = df.shape
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=["object","category","bool"]).columns.tolist()
    dt_cols = infer_datetime_cols(df)
    return {
        "rows": int(n_rows),
        "cols": int(n_cols),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": dt_cols
    }

def describe_numeric(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    if num.empty:
        return pd.DataFrame()
    desc = num.describe().T
    desc["variance"] = num.var()
    return desc

def freq_tables(df: pd.DataFrame, top_n=10) -> Dict[str, pd.DataFrame]:
    out = {}
    for col in df.select_dtypes(include=["object","category","bool"]).columns:
        vc = df[col].value_counts(dropna=False).head(top_n).rename("freq")
        out[col] = vc.to_frame().reset_index().rename(columns={"index": col})
    return out

def detect_outliers_iqr(series: pd.Series) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5*iqr
    upper = q3 + 1.5*iqr
    return (series < lower) | (series > upper)

def correlation_matrix(df: pd.DataFrame) -> pd.DataFrame:
    num = df.select_dtypes(include=[np.number])
    if num.shape[1] < 2:
        return pd.DataFrame()
    return num.corr(numeric_only=True)

def try_kmeans(df: pd.DataFrame, n_clusters=3) -> Tuple[pd.DataFrame, List[str]]:
    num = df.select_dtypes(include=[np.number]).dropna()
    if num.shape[1] < 2 or num.shape[0] < n_clusters*3:
        return pd.DataFrame(), []
    km = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    labels = km.fit_predict(num)
    out = num.copy()
    out["cluster"] = labels
    return out, list(num.columns)

# -------------- LLM Helpers --------------
LLM_SYSTEM = """Você é um analista de dados Python. Gere UM pequeno trecho de código Python que:
- Leia o DataFrame 'df' (já carregado).
- Não importe bibliotecas novas (use pandas/numpy/matplotlib/plotly já importados).
- NÃO use I/O (nada de open(), os, system, requests).
- Se criar gráfico, use matplotlib.pyplot (plt) OU plotly.express (px) e salve em 'fig' (objeto da figura).
- O resultado deve ser um dicionário Python chamado 'result' com chaves:
  'answer' (str resumindo em PT-BR), 
  'table' (opcional, DataFrame ou None),
  'fig' (opcional, figura matplotlib/plotly ou None).
Mantenha o código curto e direto. Não defina funções. Não chame plt.show().
"""

def call_llm(prompt: str) -> str:
    if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
        raise RuntimeError("OpenAI API não configurada. Defina OPENAI_API_KEY.")
    client = OpenAI()
    msg = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role":"system", "content": LLM_SYSTEM},
            {"role":"user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    return msg.choices[0].message.content

SAFE_PATTERN = re.compile(
    r"(import\s+(os|sys|subprocess|shutil|pathlib|socket|requests|urllib|ctypes|pickle|dill|base64|popen))|"
    r"(__import__|exec|eval|compile|open\(|rm -rf|pip install|!pip|%run|system\()",
    re.IGNORECASE,
)

def safe_exec_user_code(code: str, df: pd.DataFrame) -> Dict[str, Any]:
    if SAFE_PATTERN.search(code or ""):
        raise RuntimeError("Código gerado considerado inseguro e foi bloqueado.")
    local_vars = {"df": df, "np": np, "pd": pd, "plt": plt, "px": px}
    global_vars = {}
    exec(code, global_vars, local_vars)  # nosec - streamlit isolated
    if "result" not in local_vars:
        raise RuntimeError("O código não definiu a variável 'result'.")
    res = local_vars["result"]
    # sanity
    if not isinstance(res, dict):
        raise RuntimeError("A variável 'result' deve ser um dict.")
    return res

# -------------- UI --------------
st.set_page_config(page_title=APP_TITLE, page_icon="📊", layout="wide")
st.title(APP_TITLE)
st.caption("Faça upload de um CSV, explore com EDA automática e faça perguntas em linguagem natural.")

with st.sidebar:
    st.header("1) Upload do CSV")
    csv_file = st.file_uploader("Selecione um arquivo .csv", type=["csv"], accept_multiple_files=False)
    st.divider()
    st.header("Configurações")
    use_llm = st.toggle("Usar LLM (GPT-4.1-mini) para queries em linguagem natural", value=True)
    if use_llm:
        st.info("Defina a variável de ambiente OPENAI_API_KEY no Colab/terminal antes de iniciar.", icon="🔑")
    st.markdown("**Modelo**")
    st.code(MODEL_NAME, language="text")
    st.divider()
    st.header("Memória do Agente")
    show_memory = st.checkbox("Exibir memória (notas e Q&A anteriores)", value=False)

if csv_file is None:
    st.info("📎 Envie um CSV para começar.")
    st.stop()

# Load CSV
raw_bytes = csv_file.getvalue()
mem_key = dataset_id_from_bytes(raw_bytes)
memory = load_memory(mem_key)

@st.cache_data(show_spinner=False)
def load_df(_bytes: bytes) -> pd.DataFrame:
    # Try to detect encoding
    try:
        df = pd.read_csv(io.BytesIO(_bytes))
    except UnicodeDecodeError:
        df = pd.read_csv(io.BytesIO(_bytes), encoding="latin1")
    except Exception as e:
        raise e
    return df

df = load_df(raw_bytes)
st.success(f"CSV carregado: **{csv_file.name}** — {df.shape[0]} linhas x {df.shape[1]} colunas")

if show_memory:
    with st.expander("🧠 Memória deste dataset"):
        st.subheader("Notas")
        if memory["notes"]:
            for n in memory["notes"]:
                st.markdown(f"- {n.get('text')} _(em {n.get('ts')})_")
        else:
            st.write("Sem notas ainda.")
        st.subheader("Perguntas & Respostas anteriores")
        if memory["qa"]:
            for q in memory["qa"][-10:]:
                st.markdown(f"**Q:** {q.get('q')}  \n**A:** {q.get('a')}  \n_(em {q.get('ts')})_")
        else:
            st.write("Sem histórico.")

# ----- Tabs -----
tab_overview, tab_eda, tab_qna, tab_insights = st.tabs(
    ["📄 Visão Geral", "🔎 EDA Guiado", "❓ Pergunte ao Agente"]
    #["📄 Visão Geral", "🔎 EDA Guiado", "❓ Pergunte ao Agente", "📝 Conclusões"]
)

with tab_overview:
    st.subheader("Perfil dos Dados")
    prof = basic_profile(df)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Linhas", prof["rows"])
    c2.metric("Colunas", prof["cols"])
    c3.metric("Numéricas", len(prof["numeric_cols"]))
    c4.metric("Categ./Bool", len(prof["categorical_cols"]))

    st.markdown("**Colunas Numéricas:** " + (", ".join(prof["numeric_cols"]) or "—"))
    st.markdown("**Colunas Categóricas/Bool:** " + (", ".join(prof["categororical_cols"]) if 'categororical_cols' in prof else ", ".join(prof['categorical_cols']) or "—"))
    st.markdown("**Possíveis Datas:** " + (", ".join(prof["datetime_cols"]) or "—"))

    st.subheader("Resumo Estatístico (numéricos)")
    desc = describe_numeric(df)
    if desc.empty:
        st.info("Não há colunas numéricas para descrever.")
    else:
        st.dataframe(desc, use_container_width=True)

    st.subheader("Distribuições (histogramas)")
    num_cols = prof["numeric_cols"]
    if num_cols:
        col = st.selectbox("Escolha a coluna numérica", num_cols)
        bins = st.slider("Número de bins", 5, 100, 30)
        fig, ax = plt.subplots()
        ax.hist(df[col].dropna(), bins=bins)
        ax.set_title(f"Histograma: {col}")
        st.pyplot(fig)
    else:
        st.info("Sem colunas numéricas.")

with tab_eda:
    st.subheader("Frequências (Top 10) por coluna categórica")
    freqs = freq_tables(df)
    if freqs:
        for col, table in freqs.items():
            st.markdown(f"**{col}**")
            st.dataframe(table, use_container_width=True)
    else:
        st.info("Sem colunas categóricas/bool.")

    st.subheader("Correlação entre variáveis numéricas")
    corr = correlation_matrix(df)
    if corr.empty:
        st.info("Correlação não disponível.")
    else:
        st.dataframe(corr, use_container_width=True)
        st.plotly_chart(px.imshow(corr, text_auto=True, aspect="auto", title="Matriz de Correlação"))

    st.subheader("Outliers (IQR)")
    if prof["numeric_cols"]:
        col = st.selectbox("Coluna para detectar outliers", prof["numeric_cols"], key="outliers_col")
        mask = detect_outliers_iqr(df[col].dropna())
        outliers_count = int(mask.sum())
        st.write(f"Outliers detectados (IQR): **{outliers_count}**")
        fig, ax = plt.subplots()
        ax.boxplot(df[col].dropna(), vert=True, showmeans=True)
        ax.set_title(f"Boxplot: {col}")
        st.pyplot(fig)
    else:
        st.info("Sem colunas numéricas para outliers.")

    st.subheader("Clusters (KMeans rápido)")
    if len(prof["numeric_cols"]) >= 2:
        k = st.slider("Número de clusters", 2, 8, 3)
        clustered, used_cols = try_kmeans(df, n_clusters=k)
        if not clustered.empty:
            xcol = st.selectbox("Eixo X", used_cols, index=0)
            ycol = st.selectbox("Eixo Y", used_cols, index=1 if len(used_cols)>1 else 0)
            st.plotly_chart(px.scatter(clustered, x=xcol, y=ycol, color="cluster", title="Clusters (KMeans)"))
        else:
            st.info("Dados insuficientes para clusterização.")
    else:
        st.info("É necessário ao menos 2 colunas numéricas para clusterização.")

with tab_qna:
    st.subheader("Faça perguntas em linguagem natural sobre o CSV")
    st.caption("Exemplos: 'Qual a média da coluna X?', 'Existe correlação entre A e B?', 'Plote a série temporal de Y por mês.'")
    user_q = st.text_input("Pergunta", key="user_q")
    run_btn = st.button("Executar", type="primary")

    if run_btn and user_q.strip():
        answer_text = ""
        answer_table = None
        answer_fig = None
        code_text = ""

        # baseline prompt with schema
        schema = {
            "columns": list(df.columns),
            "dtypes": {c: str(df[c].dtype) for c in df.columns},
            "rows": int(df.shape[0])
        }
        prompt = f"""
        Dados do CSV carregado:
        - Colunas: {schema['columns']}
        - Tipos: {schema['dtypes']}
        - Linhas: {schema['rows']}

        Pergunta do usuário: {user_q}

        Gere um snippet de código conforme as regras.
        """

        if use_llm:
            try:
                code_text = call_llm(prompt)
            except Exception as e:
                st.error(f"Falha ao chamar LLM: {e}")
                code_text = ""

        # fallback: simples heurísticas sem LLM
        if not code_text:
            # exemplo simples: descrição rápida
            code_text = textwrap.dedent("""
            # Descrição simples sem LLM
            tbl = df.describe(include='all')
            result = {
                'answer': 'Gerei uma descrição geral do dataset (fallback sem LLM).',
                'table': tbl,
                'fig': None,
            }
            """)

        with st.expander("🔧 Código gerado", expanded=False):
            st.code(code_text, language="python")

        try:
            res = safe_exec_user_code(code_text, df)
            answer_text = res.get("answer", "")
            answer_table = res.get("table", None)
            answer_fig = res.get("fig", None)
        except Exception as e:
            st.error(f"Erro ao executar o código gerado: {e}")
            answer_text = "Não foi possível executar o código gerado."
            answer_table = None
            answer_fig = None

        if answer_text:
            st.success(answer_text)
        if answer_table is not None:
            if isinstance(answer_table, pd.DataFrame):
                st.dataframe(answer_table, use_container_width=True)
            else:
                st.write(answer_table)
        if answer_fig is not None:
            # Support matplotlib figure or plotly fig
            try:
                st.pyplot(answer_fig)
            except Exception:
                try:
                    st.plotly_chart(answer_fig, use_container_width=True)
                except Exception:
                    st.info("Figura retornada mas não foi possível renderizar.")

        # Store QA in memory
        append_memory(mem_key, {
            "type": "qa",
            "q": user_q,
            "a": answer_text,
            "ts": datetime.utcnow().isoformat()
        })

st.caption("⚠️ Memória é salva localmente (arquivo .jsonl) e se perde ao encerrar o ambiente do Colab.")

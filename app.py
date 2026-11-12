import streamlit as st
import requests
from requests.auth import HTTPBasicAuth
import pandas as pd
import plotly.express as px
import os
import subprocess
from datetime import datetime, timedelta
import google.generativeai as genai
import json
import unicodedata


# --- PADRÃO TOTVS ---
TOTVS_DARK_BLUE = "#002D57"
TOTVS_CYAN = "#00A3E0"
TOTVS_GRAY = "#666666"
TOTVS_LIGHT_GRAY = "#E9E9E9"
TOTVS_PALETTE = [TOTVS_DARK_BLUE, TOTVS_CYAN, TOTVS_GRAY, "#B4D4EE", "#D1D3D4"]
# Global config for long text columns
GLOBAL_TEXT_COLUMN_CONFIG = {
    "Motivo (Jira)": st.column_config.TextColumn("Motivo (Jira)", width="large"),
    "Motivo da Pausa": st.column_config.TextColumn("Motivo da Pausa", width="large"),
    "Submotivo": st.column_config.TextColumn("Submotivo", width="large"),
    "Motivo do Atraso": st.column_config.TextColumn("Motivo do Atraso", width="large"),
}
def merge_column_config(base_config=None):
    """Merge global text config with a given column config."""
    if base_config is None:
        base_config = {}
    merged = {**GLOBAL_TEXT_COLUMN_CONFIG, **base_config}
    return merged

def apply_totvs_style(fig):
    fig.update_layout(
        font_family="Verdana",
        title_font_family="Verdana",
        legend_font_family="Verdana",
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(size=13, color="#333333"), # Fonte maior e mais nítida para slides
        title_font=dict(size=18, color=TOTVS_DARK_BLUE, family="Verdana"), # Título maior em destaque
        margin=dict(t=60, b=50, l=50, r=50), # Margens limpas e organizadas
        hovermode="x unified",
        legend=dict(
            font=dict(size=12),
            bgcolor="rgba(255, 255, 255, 0.8)",
            bordercolor="#D1D3D4",
            borderwidth=1
        )
    )
    fig.update_xaxes(
        tickfont=dict(family="Verdana", size=11),
        title_font=dict(family="Verdana", size=13),
        gridcolor="#F5F5F5",
        linecolor="#D1D3D4",
        linewidth=1.5
    )
    fig.update_yaxes(
        tickfont=dict(family="Verdana", size=11),
        title_font=dict(family="Verdana", size=13),
        gridcolor="#F5F5F5",
        linecolor="#D1D3D4",
        linewidth=1.5
    )
    
    # Remove emojis dos títulos dos gráficos de forma automática e robusta
    if fig.layout.title and fig.layout.title.text:
        title_clean = fig.layout.title.text
        for emoji in ["🚀", "🌡️", "📊", "📈", "📅", "📋", "🔎", "⚙️", "✅", "❌", "⚠️", "⚠️", "🔔", "🛠️", "🩹", "⚡", "⏱️", "🚧"]:
            title_clean = title_clean.replace(emoji, "")
        fig.update_layout(title_text=title_clean.strip())
        
    return fig

# --- ORDENAÇÃO DE FASES ---
FASE_ORDER = [
    'Novos Projetos',
    'ADERÊNCIA',
    'Implantando ERP',
    'INICIADO',
    'GO LIVE',
    'Expansão',
    'ACOMPANHAMENTO'
]

def norm_fase(txt):
    if not txt:
        return ""
    return "".join(c for c in unicodedata.normalize('NFD', str(txt)) if unicodedata.category(c) != 'Mn').lower().strip()

FASE_ORDER_MAP = {norm_fase(fase): i for i, fase in enumerate(FASE_ORDER)}

def add_trendline(fig, df, x_col, y_col, group_col=None, name="Tendência", visible=True):
    if not visible or df is None or df.empty:
        return fig
    import numpy as np
    import plotly.graph_objects as go
    try:
        # Se tiver agregação/grupos (como no Go-lives vs Conclusões)
        if group_col and group_col in df.columns:
            groups = df[group_col].unique()
            for g in groups:
                sub_df = df[df[group_col] == g].copy()
                # Ordenar para garantir sequência correta
                sub_df = sub_df.sort_values(x_col)
                x_vals = sub_df[x_col].tolist()
                y_vals = pd.to_numeric(sub_df[y_col], errors='coerce').fillna(0).values
                if len(y_vals) >= 2:
                    x_indices = np.arange(len(y_vals))
                    z = np.polyfit(x_indices, y_vals, 1)
                    p = np.poly1d(z)
                    trend_y = p(x_indices)
                    
                    color = TOTVS_CYAN if g == 'Go-live' else TOTVS_DARK_BLUE
                    fig.add_trace(go.Scatter(
                        x=x_vals, y=trend_y,
                        mode='lines',
                        name=f"Tendência {g}",
                        line=dict(dash='dash', color=color, width=2.5),
                        visible=True
                    ))
        else:
            sub_df = df.copy()
            sub_df = sub_df.sort_values(x_col)
            x_vals = sub_df[x_col].tolist()
            y_vals = pd.to_numeric(sub_df[y_col], errors='coerce').fillna(0).values
            if len(y_vals) >= 2:
                x_indices = np.arange(len(y_vals))
                z = np.polyfit(x_indices, y_vals, 1)
                p = np.poly1d(z)
                trend_y = p(x_indices)
                
                fig.add_trace(go.Scatter(
                    x=x_vals, y=trend_y,
                    mode='lines',
                    name=name,
                    line=dict(dash='dash', color="#FF6F61", width=2.5),
                    visible=True
                ))
    except Exception as e:
        pass
    return fig

def filter_columns(df, key, default_cols=None):
    """Permite selecionar colunas para exibir usando um popover com multiselect"""
    if default_cols is None:
        default_cols = list(df.columns)
    
    with st.popover("⚙️ Filtrar Colunas", use_container_width=False):
        selected_cols = st.multiselect(
            "Colunas a exibir:",
            options=list(df.columns),
            default=default_cols,
            key=f"cols_sel_{key}"
        )
        
    if not selected_cols:
        return df[default_cols]
    return df[selected_cols]

# --- CONFIGURAÇÕES DO JIRA ---
from jira_config import JIRA_URL, EMAIL, TOKEN
CATEGORIA_FIELD = "customfield_10272"
MOTIVO_PAUSA_FIELD = "customfield_10883"
TEMPERATURA_FIELD = "customfield_10916"
MOTIVO_ATRASO_FIELD = "customfield_10445"  # Tipo de Atraso (Categorizado)
MOTIVO_DETALHADO_FIELD = "customfield_10412" # Motivo atraso (Texto longo)
DATA_GOLIVE_FIELD = "customfield_10155" # Data Go Live
MOTIVO_CANCELAMENTO_FIELD = "customfield_10095" # Motivo do Cancelamento (Texto)
MOTIVO_CRITICIDADE_FIELD = "customfield_10949" # Motivo criticidade (Dropdown)
MRR_FIELD = "customfield_10182" # MRR
HISTORY_FILE = "projects_history.csv"
PSA_FILE = "psa_projects.json"
JIRA_FILE = "jira_projects.json"
CANCELAMENTOS_FILE = "cancelamentos_data.json"
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    try:
        if hasattr(st, "secrets") and "GEMINI_API_KEY" in st.secrets:
            GEMINI_API_KEY = str(st.secrets["GEMINI_API_KEY"])
    except Exception:
        pass

# --- CONFIGURAÇÕES DO FRESHDESK ---
FRESHDESK_DOMAIN = "lexossuporte"
FRESHDESK_API_KEY = "2tVHHehjWgj2YgGkkpzG"
FRESHDESK_URL = f"https://{FRESHDESK_DOMAIN}.freshdesk.com"

# --- MAPEAMENTO DE ESFORÇO (Baseado na imagem) ---
EFFORT_DATA = [
    {"status": "Novos projetos", "horas": 0.33},
    {"status": "Aderência", "horas": 6.50},
    {"status": "Implantação ERP", "horas": 6.00},
    {"status": "Conectar contas", "horas": 1.00},
    {"status": "Importar produtos", "horas": 6.50},
    {"status": "Análise de Erros", "horas": 6.50},
    {"status": "Validação do cliente", "horas": 1.50},
    {"status": "Enviar Treinamento", "horas": 4.00},
    {"status": "Reunião tira dúvidas", "horas": 2.00},
    {"status": "Teste de pedido", "horas": 5.00},
    {"status": "Termo pre Golive", "horas": 5.00},
    {"status": "Acompanhamento", "horas": 19.00},
    {"status": "Concluído", "horas": 0.00},
]
TOTAL_HOURS = sum(d["horas"] for d in EFFORT_DATA)

def get_remaining_effort(status_name):
    """Calcula o esforço restante em horas e % baseado no status atual"""
    def normalize_text(txt):
        if not txt: return ""
        return "".join(c for c in unicodedata.normalize('NFD', str(txt))
                      if unicodedata.category(c) != 'Mn').lower().strip()

    status_norm = normalize_text(status_name)
    
    # Mapeamento de sinonimos ou status de bloco
    mapping_fix = {
        "iniciado": "conectar contas",
        "concluido": "concluido",
        "done": "finalizado",
        "finalizado": "finalizado",
        "expansao": "acompanhamento",
        "golive": "acompanhamento",
        "go-live": "acompanhamento",
        "go live": "acompanhamento",
    }
    status_target = mapping_fix.get(status_norm, status_norm)
    
    found_idx = -1
    for i, d in enumerate(EFFORT_DATA):
        if normalize_text(d["status"]) == status_target:
            found_idx = i
            break
            
    if found_idx == -1:
        # Se não encontrou, se for concluído/finalizado retorna 0, senão assume total
        if any(term in status_norm for term in ["conclu", "finalized", "done"]):
            return 0.0, 0.0
        return TOTAL_HOURS, 100.0
        
    # Esforço restante = soma das horas do status atual e todos os seguintes
    remaining_h = sum(d["horas"] for d in EFFORT_DATA[found_idx:])
    remaining_p = (remaining_h / TOTAL_HOURS) * 100
    return round(remaining_h, 2), round(remaining_p, 1)

def load_project_lists():
    """Carrega as listas de projetos PSA e Jira"""
    psa = set()
    jira = set()
    if os.path.exists(PSA_FILE):
        try:
            with open(PSA_FILE, 'r') as f:
                psa = set(json.load(f))
        except: pass
    if os.path.exists(JIRA_FILE):
        try:
            with open(JIRA_FILE, 'r') as f:
                jira = set(json.load(f))
        except: pass
    return psa, jira

def save_project_lists(psa_keys, jira_keys):
    """Salva as listas de projetos PSA e Jira"""
    with open(PSA_FILE, 'w') as f:
        json.dump(list(psa_keys), f)
    with open(JIRA_FILE, 'w') as f:
        json.dump(list(jira_keys), f)

def load_cancelamentos_data():
    if os.path.exists(CANCELAMENTOS_FILE):
        try:
            with open(CANCELAMENTOS_FILE, 'r') as f:
                return json.load(f)
        except: return {}
    return {}

def save_cancelamentos_data(data):
    with open(CANCELAMENTOS_FILE, 'w') as f:
        json.dump(data, f)


st.set_page_config(page_title="Jira Dashboard Generator", layout="wide")

# Força a fonte Verdana globalmente no Streamlit
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Verdana&display=swap');
    html, body, [class*="css"], .stMarkdown, .stText, .stButton, .stSelectbox, .stMultiSelect, .stMetric {
        font-family: 'Verdana', sans-serif !important;
    }
    .main {
        background-color: #F8F9FA;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #002D57 !important;
        font-family: 'Verdana', sans-serif !important;
    }
    </style>
""", unsafe_allow_html=True)

st.title("Jira Dashboard Generator")
st.sidebar.header("Configurações")

# --- FUNÇÕES DE IA ---

def get_issue_comments(issue_key, max_comments=10):
    """Busca os comentários mais recentes de uma issue do Jira"""
    url = f"{JIRA_URL}/rest/api/3/issue/{issue_key}/comment"
    auth = HTTPBasicAuth(EMAIL, TOKEN)
    headers = {"Accept": "application/json"}
    try:
        response = requests.get(
            url, headers=headers, auth=auth,
            params={"maxResults": max_comments, "orderBy": "-created"}
        )
        if response.status_code != 200:
            return []
        data = response.json()
        comments = []
        for c in data.get('comments', []):
            # Extrai texto limpo do formato ADF (Atlassian Document Format)
            body = c.get('body', {})
            text_parts = []
            if isinstance(body, dict):
                for block in body.get('content', []):
                    for inline in block.get('content', []):
                        if inline.get('type') == 'text':
                            text_parts.append(inline.get('text', ''))
            elif isinstance(body, str):
                text_parts = [body]
            author = c.get('author', {}).get('displayName', 'Desconhecido')
            date = c.get('created', '')[:10]
            text = ' '.join(text_parts).strip()
            if text:
                comments.append(f"[{date} - {author}]: {text}")
        return comments
    except Exception:
        return []


def get_gemini_model():
    """Descobre e armazena na sessão o modelo do Gemini válido e suportado pela API local"""
    if 'available_gemini_model' not in st.session_state:
        try:
            all_models = genai.list_models()
            models = [m.name for m in all_models if 'generateContent' in m.supported_generation_methods]
            
            # Prioridade 1: 1.5 Flash (Super Rápido)
            chosen = next((m for m in models if 'gemini-1.5-flash' in m), None)
            
            # Prioridade 2: 1.0 Pro Original (Estável e Leve)
            if not chosen:
                chosen = next((m for m in models if 'gemini-1.0-pro' in m and 'latest' not in m), None)
                
            # Prioridade 3: Gemini Pro Básico
            if not chosen:
                chosen = next((m for m in models if m.endswith('gemini-pro')), None)
                
            # Prioridade 4: O primeiro que não gere conflitos óbvios
            if not chosen:
                gemini_models = [m for m in models if 'gemini' in m.lower()]
                chosen = gemini_models[0] if gemini_models else 'models/gemini-pro'
                
            st.session_state['available_gemini_model'] = chosen
        except Exception as e:
            # Fallback seguro absoluto em caso de API Key offline para listagem
            st.session_state['available_gemini_model'] = 'models/gemini-pro'
            
    return st.session_state['available_gemini_model']

def get_ai_understanding(issue_key, summary, motivo, submotivo, comments):
    """Usa o Gemini para gerar um entendimento do motivo da pausa com base nos comentários"""
    if not GEMINI_API_KEY:
        return "⚠️ Configure a variável de ambiente GEMINI_API_KEY para usar esta função."
    if not comments:
        return "Sem comentários disponíveis para análise."
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        model_name = get_gemini_model()
        model = genai.GenerativeModel(model_name)
        
        comments_text = "\n".join(comments[-10:])  # Últimos 10 comentários
        prompt = f"""Você é um analista de projetos de TI. Analise os comentários abaixo de um projeto Jira pausado e forneça um entendimento claro e conciso (máximo 2 frases em português) do MOTIVO REAL da pausa, baseando-se no contexto dos comentários.

Projeto: {summary}
Motivo registrado: {motivo}
Submotivo registrado: {submotivo}

Comentários mais recentes:
{comments_text}

Resposta (apenas o entendimento, sem introdução ou rótulos):"""
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Erro na análise IA: {str(e)[:100]}"


def analyze_paused_projects(df_pausados):
    """Executa análise de IA para todos os projetos pausados e retorna o df atualizado"""
    resultados = {}
    progress = st.progress(0, text="Analisando comentários com IA...")
    total = len(df_pausados)
    
    for i, (_, row) in enumerate(df_pausados.iterrows()):
        issue_key = row['Key']
        comments = get_issue_comments(issue_key)
        understanding = get_ai_understanding(
            issue_key,
            row['Summary'],
            row['Motivo da Pausa'],
            row['Submotivo'],
            comments
        )
        resultados[issue_key] = understanding
        progress.progress((i + 1) / total, text=f"Analisando {issue_key}... ({i+1}/{total})")
    
    progress.empty()
    return resultados


def get_ai_understanding_atraso(issue_key, summary, motivo, dias_atraso, comments):
    """Usa o Gemini para gerar um entendimento do motivo do atraso com base nos comentários"""
    if not GEMINI_API_KEY:
        return "⚠️ Configure a variável de ambiente GEMINI_API_KEY para usar esta função."
    if not comments:
        return "Sem comentários disponíveis para análise."
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        model_name = get_gemini_model()
        model = genai.GenerativeModel(model_name)
        
        comments_text = "\n".join(comments[-10:])  # Últimos 10 comentários
        prompt = f"""Você é um analista de projetos de TI. Analise os comentários abaixo de um projeto Jira atrasado ({dias_atraso} dias de atraso) e forneça um entendimento claro e conciso (máximo 2 frases em português) do MOTIVO REAL do atraso, baseando-se no contexto dos comentários.

Projeto: {summary}
Motivo de atraso registrado: {motivo}
Dias de atraso: {dias_atraso}

Comentários mais recentes:
{comments_text}

Resposta (apenas o entendimento, sem introdução ou rótulos):"""
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Erro na análise IA: {str(e)[:100]}"


def analyze_delayed_projects(df_atrasados):
    """Executa análise de IA para todos os projetos atrasados e retorna um dict com os resultados"""
    resultados = {}
    progress = st.progress(0, text="Analisando comentários com IA...")
    total = len(df_atrasados)
    
    for i, (_, row) in enumerate(df_atrasados.iterrows()):
        issue_key = row['Key']
        comments = get_issue_comments(issue_key)
        motivo = row.get('Motivo Atraso', row.get('Motivo do Atraso', ''))
        dias = row.get('dias_atraso_calc', row.get('Dias_Atraso', row.get('Atraso (Dias)', '')))
        summary = row.get('Summary', row.get('Projeto', ''))
        
        understanding = get_ai_understanding_atraso(
            issue_key,
            summary,
            motivo,
            dias,
            comments
        )
        resultados[issue_key] = understanding
        progress.progress((i + 1) / total, text=f"Analisando {issue_key}... ({i+1}/{total})")
    
    progress.empty()
    return resultados



def get_ai_understanding_temp(issue_key, summary, temperatura, comments):
    """Usa o Gemini para gerar um entendimento do motivo da temperatura de um projeto com base nos comentários"""
    if not GEMINI_API_KEY:
        return "⚠️ Configure a variável de ambiente GEMINI_API_KEY para usar esta função."
    if not comments:
        return "Sem comentários disponíveis para análise."
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        
        model_name = get_gemini_model()
        model = genai.GenerativeModel(model_name)
        
        comments_text = "\n".join(comments[-10:])  # Últimos 10 comentários
        prompt = f"""Você é um analista de projetos de TI. Analise os comentários abaixo deste projeto do Jira e forneça um entendimento claro e conciso (máximo 2 frases em português) da situação atual explicando o motivo do projeto estar com a temperatura '{temperatura}', baseando-se no contexto.

Projeto: {summary}
Temperatura atual: {temperatura}

Comentários mais recentes:
{comments_text}

Resumo da situação (apenas o entendimento, sem introdução):"""
        
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Erro na análise IA: {str(e)[:100]}"

def analyze_temp_projects(df_temp):
    """Executa análise de IA para projetos filtrados por temperatura e retorna um dict com os resultados"""
    resultados = {}
    progress = st.progress(0, text="Analisando comentários com IA...")
    total = len(df_temp)
    
    for i, (_, row) in enumerate(df_temp.iterrows()):
        issue_key = row['Key']
        comments = get_issue_comments(issue_key)
        understanding = get_ai_understanding_temp(
            issue_key,
            row['Summary'],
            row['Temperatura'],
            comments
        )
        resultados[issue_key] = understanding
        progress.progress((i + 1) / total, text=f"Analisando {issue_key}... ({i+1}/{total})")
    
    progress.empty()
    return resultados

def find_freshdesk_id(jira_nome):
    all_companies = get_freshdesk_companies()
    if not all_companies: return None
    
    if os.path.exists("freshdesk_mapping.json"):
        try:
            with open("freshdesk_mapping.json", "r", encoding="utf-8") as f:
                manual_map = json.load(f)
                if jira_nome in manual_map:
                    return manual_map[jira_nome]
        except: pass


        
    def norm(txt):
        t = "".join(c for c in unicodedata.normalize('NFD', str(txt))
                      if unicodedata.category(c) != 'Mn').lower().strip()
        for suffix in ['- totvs', '- winthor', '- protheus', '(api)', '(craw)', '(full e hub)', 'totvs moda', 'protheus']:
            t = t.replace(suffix, '').strip()
        return t

    n = norm(jira_nome)
    for c in all_companies:
        if norm(c['name']) == n:
            return c['id']
    return None

def analyze_single_client_criticality(issue_key, summary, temperatura, fase):
    if not GEMINI_API_KEY:
        return "⚠️ Configure a variável de ambiente GEMINI_API_KEY."
        
    # Carga mínima extrema (apenas 2 últimos) para garantir nenhum timeout
    jira_comments = get_issue_comments(issue_key, max_comments=3)
    
    fd_id = find_freshdesk_id(summary)
    fd_tickets_text = []
    
    if fd_id:
        tickets = get_freshdesk_tickets(company_id=fd_id)
        # Filtra os 3 mais recentes apenas
        tickets = sorted(tickets, key=lambda x: x.get('created_at', ''), reverse=True)[:3]
        for t in tickets:
            status_map = {2: 'Aberto', 3: 'Pendente', 4: 'Resolvido', 5: 'Fechado'}
            status = status_map.get(t.get('status', 0), 'Desconhecido')
            fd_tickets_text.append(f"[{t.get('created_at', '')[:10]} - Status: {status}]: {t.get('subject', '')}")
            
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model_name = get_gemini_model()
        model = genai.GenerativeModel(model_name)
        
        jira_text = "\n".join(jira_comments) if jira_comments else "Sem comentários recentes no Jira."
        fd_text = "\n".join(fd_tickets_text) if fd_tickets_text else "Sem tickets recentes no Freshdesk."
        
        prompt = f"""Analise os dados do Jira e Freshdesk deste cliente e faça um breve resumo cronológico da criticidade atual.

**PROJETO:** {summary}
**FASE ATUAL:** {fase}
**TEMPERATURA:** {temperatura}

**Comentários recentes (Jira):**
{jira_text}

**Tickets recentes (Freshdesk):**
{fd_text}

**Instruções:**
- Faça um resumo em tópicos curtíssimos cruzando as duas fontes.
- Destaque bloqueios cruciais.
- Conclua com um rápido diagnóstico. Responda APENAS a análise."""
        
        # Deixa a IA responder sem limite de tokens, limitando apenas por temperatura para precisão
        config = genai.types.GenerationConfig(temperature=0.3)
        
        # Implementa uma tentativa de retry simples contra flake do Google Cloud 504 Timeout
        import time
        max_retries = 2
        for attempt in range(max_retries):
            try:
                # Retirado timeout duro limitador para deixar a biblioteca cuidar da recursão se precisar
                response = model.generate_content(prompt, generation_config=config)
                return response.text.strip()
            except Exception as inner_e:
                if attempt < max_retries - 1:
                    time.sleep(2)  # aguarda antes de bater na API de novo
                else:
                    raise inner_e
                
    except Exception as e:
        return f"Erro na análise IA: {str(e)[:100]}"

# Funções de API
def get_jira_projects():
    """Busca todos os projetos (espaços) do Jira"""
    url = f"{JIRA_URL}/rest/api/3/project"
    auth = HTTPBasicAuth(EMAIL, TOKEN)
    headers = {"Accept": "application/json"}
    try:
        response = requests.get(url, headers=headers, auth=auth)
        if response.status_code == 200:
            return sorted(response.json(), key=lambda x: x['name'])
        else:
            st.error(f"Erro ao buscar projetos: {response.status_code}")
            return []
    except Exception as e:
        st.error(f"Erro de conexão ao buscar projetos: {e}")
        return []

def get_issue_types_for_project(project_key):
    """Busca os tipos de issue disponíveis em um projeto"""
    url = f"{JIRA_URL}/rest/api/3/project/{project_key}"
    auth = HTTPBasicAuth(EMAIL, TOKEN)
    try:
        response = requests.get(url, auth=auth)
        if response.status_code == 200:
            issue_types = response.json().get('issueTypes', [])
            types = sorted(set(it['name'] for it in issue_types))
            return types
        return []
    except:
        return []

# --- FUNÇÕES FRESHDESK ---
@st.cache_data(ttl=3600)
def get_freshdesk_companies():
    """Busca todas as empresas do Freshdesk com paginação"""
    companies = []
    page = 1
    while True:
        url = f"{FRESHDESK_URL}/api/v2/companies?per_page=100&page={page}"
        auth = (FRESHDESK_API_KEY, "X")
        try:
            r = requests.get(url, auth=auth)
            if r.status_code != 200: break
            data = r.json()
            if not data: break
            companies.extend(data)
            page += 1
            if len(data) < 100: break
        except: break
    return companies

@st.cache_data(ttl=600)
def get_freshdesk_tickets(company_id=None):
    """Busca tickets do Freshdesk (últimos 30 dias por padrão)"""
    tickets = []
    page = 1
    while True:
        url = f"{FRESHDESK_URL}/api/v2/tickets?per_page=100&page={page}"
        if company_id:
            url += f"&company_id={company_id}"
        
        auth = (FRESHDESK_API_KEY, "X")
        try:
            r = requests.get(url, auth=auth)
            if r.status_code != 200: break
            data = r.json()
            if not data: break
            tickets.extend(data)
            page += 1
            if len(data) < 100 or page > 5: break # Limite de seguranca
        except: break
    return tickets

def get_jira_issues(project_key, issue_type=None):
    """Busca TODAS as issues do Jira com paginação completa"""
    url = f"{JIRA_URL}/rest/api/3/search/jql"
    auth = HTTPBasicAuth(EMAIL, TOKEN)
    headers = {"Accept": "application/json"}
    fields = f'summary,status,priority,assignee,issuetype,labels,created,duedate,resolutiondate,statuscategorychangedate,{CATEGORIA_FIELD},{MOTIVO_PAUSA_FIELD},{TEMPERATURA_FIELD},{MOTIVO_ATRASO_FIELD},{MOTIVO_DETALHADO_FIELD},{DATA_GOLIVE_FIELD},{MOTIVO_CANCELAMENTO_FIELD},{MOTIVO_CRITICIDADE_FIELD},{MRR_FIELD},customfield_10015'
    
    # Monta o JQL baseado no tipo selecionado
    if issue_type and issue_type != "Todos":
        jql = f'project = "{project_key}" AND issuetype = "{issue_type}" ORDER BY created ASC'
    else:
        jql = f'project = "{project_key}" ORDER BY created ASC'

    all_issues = []
    next_page_token = None

    try:
        while True:
            query = {'jql': jql, 'maxResults': 100, 'fields': fields, 'expand': 'changelog'}
            if next_page_token:
                query['nextPageToken'] = next_page_token

            response = requests.get(url, headers=headers, params=query, auth=auth)

            if response.status_code != 200:
                st.error(f"Erro ao conectar ao Jira: {response.status_code} - {response.text}")
                return None

            data = response.json()
            issues = data.get('issues', [])
            all_issues.extend(issues)

            next_page_token = data.get('nextPageToken')
            if not next_page_token or len(issues) == 0:
                break

        return all_issues

    except Exception as e:
        st.error(f"Ocorreu um erro na requisição: {e}")
        return None

def process_data(issues):
    """Transforma o JSON do Jira em um DataFrame do Pandas"""
    data = []
    for issue in issues:
        fields = issue['fields']
        
        # Extração da Categoria / ISM Responsável (campo customizado)
        categoria = fields.get(CATEGORIA_FIELD)
        if isinstance(categoria, dict): 
            categoria_name = categoria.get('displayName', categoria.get('value', 'Sem ISM'))
        elif categoria is not None:
            categoria_name = str(categoria)
        else:
            categoria_name = 'Sem ISM'

        # Extração de Labels (Etiquetas)
        labels_list = fields.get('labels', [])
        labels_str = " ".join([l.upper() for l in labels_list])
        
        # Categorias de Status por Labels
        is_pausado = "PAUSADO" in labels_str or "PAUSADOS" in labels_str
        is_critico = "CRÍTICO" in labels_str or "CRITICO" in labels_str
        
        # Data de criação, vencimento e início
        created_date = fields.get('created', '')
        created_dt = datetime.strptime(created_date[:10], "%Y-%m-%d") if created_date else None
        
        due_date = fields.get('duedate', '')
        due_dt = datetime.strptime(due_date[:10], "%Y-%m-%d") if due_date else None
        
        # Obter a Previsão Original do changelog do Jira
        original_due_str = None
        changelog = issue.get('changelog', {})
        histories = changelog.get('histories', [])
        histories_sorted = sorted(histories, key=lambda h: h.get('created', ''))
        for history in histories_sorted:
            for item in history.get('items', []):
                if item.get('field') == 'duedate':
                    val_from = item.get('from')
                    if val_from:
                        original_due_str = val_from
                        break
                    else:
                        original_due_str = item.get('to')
                        break
            if original_due_str:
                break
        if not original_due_str:
            original_due_str = due_date[:10] if due_date else None
            
        original_due_dt = datetime.strptime(original_due_str[:10], "%Y-%m-%d") if original_due_str else None
        
        # Extração da Data de Início
        start_date_raw = fields.get('customfield_10015')
        start_dt = datetime.strptime(start_date_raw[:10], "%Y-%m-%d") if start_date_raw else None
        
        # Extração da Data Go Live
        go_live_date = fields.get(DATA_GOLIVE_FIELD, '')
        go_live_dt = datetime.strptime(go_live_date[:10], "%Y-%m-%d") if go_live_date else None
        
        hoje = datetime.now()
        
        # Nova Definição de Atraso: Somente se ultrapassou a data limite E não tem Go Live realizado (comparando apenas a data)
        is_atrasado = (due_dt is not None and hoje.date() > due_dt.date() and go_live_dt is None)

        # Extração do Motivo da Pausa + Submotivo
        motivo_raw = fields.get(MOTIVO_PAUSA_FIELD)
        if isinstance(motivo_raw, dict):
            motivo_pausa = motivo_raw.get('value', 'Não informado')
            child = motivo_raw.get('child')
            submotivo_pausa = child.get('value', 'Não informado') if isinstance(child, dict) else 'Não informado'
        else:
            motivo_pausa = 'Não informado'
            submotivo_pausa = 'Não informado'
        
        # Extração da Temperatura
        temp_raw = fields.get(TEMPERATURA_FIELD)
        if isinstance(temp_raw, dict):
            temperatura = temp_raw.get('value', 'Estável')
        else:
            temperatura = 'Estável'
            
        if temperatura == 'Normal':
            temperatura = 'Estável'
        
        # Extrações de Atraso
        motivo_atraso_raw = fields.get(MOTIVO_ATRASO_FIELD)
        if isinstance(motivo_atraso_raw, dict):
            motivo_atraso = motivo_atraso_raw.get('value', 'Não informado')
        else:
            motivo_atraso = str(motivo_atraso_raw) if motivo_atraso_raw else 'Não informado'
            
        # Extração do Motivo Detalhado (Texto longo que pode vir em ADF)
        motivo_detalhado_raw = fields.get(MOTIVO_DETALHADO_FIELD, "")
        motivo_detalhado = ""
        if isinstance(motivo_detalhado_raw, dict):
            text_parts = []
            for block in motivo_detalhado_raw.get('content', []):
                for inline in block.get('content', []):
                    if inline.get('type') == 'text':
                        text_parts.append(inline.get('text', ''))
            motivo_detalhado = " ".join(text_parts).strip()
        else:
            motivo_detalhado = str(motivo_detalhado_raw) if motivo_detalhado_raw else ""
            
        # Definir a data de início de projeto de forma coerente com a previsão original
        dias_projeto_ref = start_dt if start_dt else created_dt
        if original_due_dt and dias_projeto_ref and dias_projeto_ref > original_due_dt:
            dias_projeto_ref = created_dt
            if dias_projeto_ref and dias_projeto_ref > original_due_dt:
                dias_projeto_ref = original_due_dt
                
        dias_projeto = (hoje.date() - dias_projeto_ref.date()).days if dias_projeto_ref else 0
        dias_atraso = (hoje.date() - original_due_dt.date()).days if (is_atrasado and original_due_dt) else 0



        # Data de atualização formatada
        updated_date = fields.get('updated', '')
        if updated_date:
            updated_dt = datetime.strptime(updated_date[:10], "%Y-%m-%d")
        else:
            updated_dt = None

        # Resolução
        res_date = fields.get('resolutiondate')
        res_dt = datetime.strptime(res_date[:10], "%Y-%m-%d") if res_date else None

        # Categoria de Status (new, indeterminate, done)
        st_cat = fields['status']['statusCategory']['key']

        # Data de mudança de categoria de status (oficial do Jira)
        sc_date = fields.get('statuscategorychangedate')
        sc_dt = datetime.strptime(sc_date[:10], "%Y-%m-%d") if sc_date else None

        # Extrações de Cancelamento e Criticidade
        motivo_cancel_raw = fields.get(MOTIVO_CANCELAMENTO_FIELD, "")
        if isinstance(motivo_cancel_raw, dict): # ADF check
            text_parts = []
            for block in motivo_cancel_raw.get('content', []):
                for inline in block.get('content', []):
                    if inline.get('type') == 'text':
                        text_parts.append(inline.get('text', ''))
            motivo_cancel = " ".join(text_parts).strip()
        else:
            motivo_cancel = str(motivo_cancel_raw) if motivo_cancel_raw else ""
            
        motivo_crit_raw = fields.get(MOTIVO_CRITICIDADE_FIELD)
        if isinstance(motivo_crit_raw, list):
            # Caso seja multi-select (lista de dicionários)
            motivos = [opt.get('value', '') for opt in motivo_crit_raw if isinstance(opt, dict)]
            motivo_crit = ", ".join(filter(None, motivos)) if motivos else 'Não informado'
        elif isinstance(motivo_crit_raw, dict):
            motivo_crit = motivo_crit_raw.get('value', 'Não informado')
        else:
            motivo_crit = str(motivo_crit_raw) if motivo_crit_raw else 'Não informado'

        mrr_raw = fields.get(MRR_FIELD)
        try:
            mrr_jira = float(mrr_raw) if mrr_raw is not None else 0.0
        except:
            mrr_jira = 0.0

        data.append({
            "Key": issue['key'],
            "Summary": fields['summary'],
            "Fase": fields['status']['name'],
            "Status_Category": st_cat,
            "Assignee": fields['assignee']['displayName'] if fields['assignee'] else "Unassigned",
            "Categoria": categoria_name,
            "Etiquetas": ", ".join(labels_list),
            "LabelList": labels_list,
            "Motivo da Pausa": motivo_pausa,
            "Submotivo": submotivo_pausa,
            "Pausado": is_pausado,
            "Atrasado": is_atrasado,
            "Critico": is_critico,
            "Ultima_Atualizacao": updated_dt,
            "Temperatura": temperatura,
            "Motivo Atraso": motivo_atraso,
            "Motivo Detalhado": motivo_detalhado,
            "Motivo Cancelamento": motivo_cancel,
            "Motivo Criticidade": motivo_crit,
            "Criado_Em": created_dt,
            "Data_Limite": due_dt,
            "Data_Limite_Original": original_due_dt,
            "Data_Go_Live": go_live_dt,
            "Resolução_Em": res_dt,
            "Status_Cat_Change": sc_dt,
            "Dias_Projeto": dias_projeto,
            "Dias_Atraso": dias_atraso,
            "MRR_Jira": mrr_jira
        })
    return pd.DataFrame(data)

# Carregamento Inicial de Projetos
if 'projects' not in st.session_state:
    with st.spinner('Carregando espaços do Jira...'):
        st.session_state['projects'] = get_jira_projects()

if st.session_state['projects']:
    project_options = {p['name']: p['key'] for p in st.session_state['projects']}
    selected_project_name = st.sidebar.selectbox(
        "Selecione o Espaço", 
        options=list(project_options.keys()),
        index=list(project_options.keys()).index("Implantação Lexos") if "Implantação Lexos" in project_options else 0
    )
    project_key = project_options[selected_project_name]

    # Carregar tipos de issue ao mudar de projeto
    cache_key = f'issue_types_{project_key}'
    if cache_key not in st.session_state:
        with st.spinner('Carregando tipos de issue...'):
            types = get_issue_types_for_project(project_key)
            st.session_state[cache_key] = ["Todos"] + types if types else ["Todos"]

    issue_type_options = st.session_state.get(cache_key, ["Todos"])
    selected_issue_type = st.sidebar.selectbox(
        "Tipo de Registro",
        options=issue_type_options,
        index=issue_type_options.index("História") if "História" in issue_type_options else 0
    )

    chart_type = st.sidebar.selectbox("Escolha o Filtro/Métrica", 
        ["Visão Geral — Gestão", "Gráfico por Fase", "Projetos por Consultor (ISM)", "Motivo das Pausas", "Gráfico de Temperatura", "Evolução Semanal", "Fluxo Mensal", "Go-lives por Mês", "Clientes Atrasados", "Controle de Prazos e Replanejamentos", "Previsão de Entrega", "Relatório de Cancelamento", "Previsão de Go-live", "Relatório PSA/Jira", "Tickets Freshdesk", "Impedimentos por Mês", "Tabela Completa"])

    # Botão de Busca
    if st.sidebar.button("Gerar Dashboard"):
        with st.spinner(f'Buscando dados de {selected_project_name}...'):
            issues_json = get_jira_issues(project_key, selected_issue_type)
            if issues_json is not None and len(issues_json) > 0:
                st.session_state['df_raw'] = process_data(issues_json)
                st.session_state['last_project'] = selected_project_name
                st.session_state['last_issue_type'] = selected_issue_type
                
                # Salva a foto diária automaticamente
                try:
                    subprocess.run(["python", "snapshot_diario.py"], capture_output=True)
                    st.toast("Snapshot diário registrado no repositório!")
                except:
                    pass
            elif issues_json is not None and len(issues_json) == 0:
                st.warning(f"Nenhuma issue do tipo '{selected_issue_type}' encontrada em '{selected_project_name}'. Tente selecionar outro tipo de registro.")
                st.session_state['df_raw'] = None
            else:
                st.session_state['df_raw'] = None

    # Exibição do Dash
    if 'df_raw' in st.session_state and st.session_state['df_raw'] is not None:
        df = st.session_state['df_raw']
        
        # Sidebar: Filtros Adicionais
        st.sidebar.markdown("---")
        st.sidebar.subheader("Filtros Rápidos")
        
        # Filtro de Fase
        available_phases = sorted(
            df['Fase'].unique(),
            key=lambda x: FASE_ORDER_MAP.get(norm_fase(x), 999)
        )
        selected_phases = st.sidebar.multiselect(
            "Filtrar Fase:", 
            options=available_phases, 
            default=[p for p in available_phases if p.lower() not in ['concluído', 'concluido', 'cancelado', 'done', 'canceled', 'finalizado']]
        )
        
        # Filtro de Etiquetas
        all_labels = set()
        for labels in df['LabelList']:
            all_labels.update(labels)
        available_labels = sorted(list(all_labels))
        
        selected_labels = st.sidebar.multiselect(
            "Filtrar por Etiquetas (Labels):",
            options=available_labels,
            default=[] # Por padrão mostra tudo, a menos que selecione algo
        )
        
        # Opção de Linha de Tendência para slides/apresentações
        exibir_tendencia = st.sidebar.checkbox("Exibir Linha de Tendência nos Gráficos", value=True)
        
        # Aplicar Filtros
        df_filtered = df[df['Fase'].isin(selected_phases)]
        if selected_labels:
            # Filtra se ao menos uma das etiquetas selecionadas estiver na lista de labels da issue
            df_filtered = df_filtered[df_filtered['LabelList'].apply(lambda x: any(l in x for l in selected_labels))]
        
        if not df_filtered.empty:
            if chart_type == "Controle de Prazos e Replanejamentos":
                st.subheader("Dashboard: Controle de Prazo")
            else:
                st.subheader(f"Dashboard: {st.session_state['last_project']}")
            
            # Métricas
            if chart_type not in ["Visão Geral — Gestão", "Controle de Prazos e Replanejamentos"]:
                c1, c2 = st.columns(2)
                c1.metric("Total Projetos", len(df_filtered))
                
                impedidos = df_filtered[df_filtered['Pausado']].shape[0]
                if chart_type != "Motivo das Pausas":
                    c2.metric("Impedidos", impedidos)

            # Gráficos
            if chart_type == "Visão Geral — Gestão":
                # --- VISÃO GERAL ---
                hoje = datetime.now()
                total_ativos = len(df_filtered)
                
                # Projetos ativos no final do mês anterior para o delta
                fim_mes_anterior = datetime(hoje.year, hoje.month, 1) - timedelta(days=1)
                
                def is_active_at_date(row, target_date):
                    issued_at = row['Criado_Em']
                    if pd.isnull(issued_at) or issued_at > target_date:
                        return False
                    if row['Status_Category'] == 'done':
                        resolved_at = row['Status_Cat_Change']
                        if pd.notnull(resolved_at) and resolved_at <= target_date:
                            return False
                    return True
                    
                df_prev_active = df[df.apply(is_active_at_date, axis=1, target_date=fim_mes_anterior)]
                total_anterior = len(df_prev_active)
                delta_total = total_ativos - total_anterior
                
                if delta_total < 0:
                    total_delta_str = f"↓ {abs(delta_total)} vs mês anterior"
                    total_delta_color = "#8B1E0F"  # Vermelho
                elif delta_total > 0:
                    total_delta_str = f"↑ {delta_total} vs mês anterior"
                    total_delta_color = "#2E7D32"  # Verde
                else:
                    total_delta_str = "— vs mês anterior"
                    total_delta_color = "#666666"

                # Impedidos
                impedidos_count = df_filtered[df_filtered['Pausado']].shape[0]
                perc_impedidos = (impedidos_count / total_ativos * 100) if total_ativos > 0 else 0
                
                # Em Risco / Críticos
                criticos_count = df_filtered[df_filtered['Temperatura'].str.contains('Critico|Crítico', na=False, case=False)].shape[0]
                em_risco_count = df_filtered[df_filtered['Temperatura'].str.contains('Risco', na=False, case=False)].shape[0]
                total_risco_criticos = criticos_count + em_risco_count
                
                # Atrasados
                df_atrasados_card = df_filtered[
                    (df_filtered['Atrasado']) & 
                    (~df_filtered['Fase'].astype(str).str.lower().str.contains("implantando erp", na=False)) &
                    (~df_filtered['Motivo Atraso'].astype(str).str.lower().str.contains("implantacao de erp|implantação de erp|implantando erp", na=False))
                ]
                atrasados_count = df_atrasados_card.shape[0]
                perc_atrasados = (atrasados_count / total_ativos * 100) if total_ativos > 0 else 0
                
                # Churn Mensal (Cancelados no mês atual vs anterior)
                inicio_mes_atual = datetime(hoje.year, hoje.month, 1)
                fim_mes_atual = hoje.replace(hour=23, minute=59, second=59)
                df_exits_atual = df[(df['Status_Category'] == 'done') & (df['Status_Cat_Change'] >= inicio_mes_atual) & (df['Status_Cat_Change'] <= fim_mes_atual)]
                churn_atual = df_exits_atual[df_exits_atual['Fase'].str.upper().str.contains("CANCEL", na=False)].shape[0]
                
                inicio_mes_anterior = datetime(fim_mes_anterior.year, fim_mes_anterior.month, 1)
                df_exits_anterior = df[(df['Status_Category'] == 'done') & (df['Status_Cat_Change'] >= inicio_mes_anterior) & (df['Status_Cat_Change'] <= fim_mes_anterior)]
                churn_anterior = df_exits_anterior[df_exits_anterior['Fase'].str.upper().str.contains("CANCEL", na=False)].shape[0]
                
                delta_churn = churn_atual - churn_anterior
                if delta_churn < 0:
                    churn_delta_str = f"↓ {abs(delta_churn)} vs mês anterior"
                    churn_delta_color = "#2E7D32"  # Verde (menos cancelamentos é bom!)
                elif delta_churn > 0:
                    churn_delta_str = f"↑ {delta_churn} vs mês anterior"
                    churn_delta_color = "#8B1E0F"  # Vermelho
                else:
                    churn_delta_str = "— vs mês anterior"
                    churn_delta_color = "#666666"

                # Semana de segunda-feira a domingo da semana passada
                inicio_passada = (hoje - timedelta(days=hoje.weekday() + 7)).replace(hour=0, minute=0, second=0, microsecond=0)
                fim_passada = inicio_passada + timedelta(days=6)
                semana_str = f"Semana {inicio_passada.strftime('%d/%m')} a {fim_passada.strftime('%d/%m')}"
                
                meses_pt = {
                    1: "jan", 2: "fev", 3: "mar", 4: "abr", 5: "mai", 6: "jun",
                    7: "jul", 8: "ago", 9: "set", 10: "out", 11: "nov", 12: "dez"
                }
                ref_mes_str = f"{meses_pt.get(hoje.month, 'mai')}/{hoje.year}"

                html_code = f"""<style>
.gestao-dashboard-header {{
    margin-bottom: 25px;
    font-family: 'Verdana', sans-serif;
}}
.gestao-dashboard-title {{
    font-size: 1.8rem;
    font-weight: 700;
    color: #002D57;
    margin: 0;
}}
.gestao-dashboard-subtitle {{
    font-size: 0.95rem;
    color: #666666;
    margin-top: 5px;
}}
.gestao-dashboard-container {{
    display: flex;
    gap: 16px;
    width: 100%;
    margin-bottom: 25px;
}}
.gestao-card {{
    flex: 1;
    background-color: #F8F7F4;
    border: 1px solid #E5E3DC;
    border-radius: 12px;
    padding: 22px 20px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    min-height: 160px;
    font-family: 'Verdana', sans-serif;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    transition: transform 0.2s, box-shadow 0.2s;
}}
.gestao-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 4px 8px rgba(0,0,0,0.05);
}}
.gestao-card-title {{
    font-size: 0.75rem;
    font-weight: 700;
    color: #666666;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    margin: 0 0 10px 0;
}}
.gestao-card-value {{
    font-size: 3rem;
    font-weight: 700;
    line-height: 1;
    margin: 5px 0 12px 0;
}}
.gestao-card-sub {{
    font-size: 0.8rem;
    font-weight: 500;
    color: #333333;
    line-height: 1.3;
}}
.gestao-divider {{
    border: none;
    border-top: 1px solid #E5E3DC;
    margin: 30px 0;
    width: 100%;
}}
@media (max-width: 992px) {{
    .gestao-dashboard-container {{
        flex-wrap: wrap;
        gap: 12px;
    }}
    .gestao-card {{
        min-width: calc(33.33% - 8px);
        min-height: 140px;
    }}
}}
@media (max-width: 768px) {{
    .gestao-card {{
        min-width: calc(50% - 6px);
    }}
}}
@media (max-width: 480px) {{
    .gestao-card {{
        min-width: 100%;
    }}
}}
</style>
<div class="gestao-dashboard-header">
<div class="gestao-dashboard-title">Análise de Gestão — Implantação Lexos</div>
<div class="gestao-dashboard-subtitle">Semana {semana_str} · {total_ativos} projetos ativos · Referência: {ref_mes_str}</div>
</div>
<div class="gestao-dashboard-container">
<div class="gestao-card">
<div class="gestao-card-title">TOTAL PROJETOS</div>
<div class="gestao-card-value" style="color: #222222;">{total_ativos}</div>
<div class="gestao-card-sub" style="color: {total_delta_color};">{total_delta_str}</div>
</div>
<div class="gestao-card">
<div class="gestao-card-title">IMPEDIDOS</div>
<div class="gestao-card-value" style="color: #8B1E0F;">{impedidos_count}</div>
<div class="gestao-card-sub">{perc_impedidos:.1f}% da carteira</div>
</div>
<div class="gestao-card">
<div class="gestao-card-title">EM RISCO / CRÍTICOS</div>
<div class="gestao-card-value" style="color: #B58900;">{total_risco_criticos}</div>
<div class="gestao-card-sub">{criticos_count} críticos · {em_risco_count} em risco</div>
</div>
<div class="gestao-card">
<div class="gestao-card-title">ATRASADOS</div>
<div class="gestao-card-value" style="color: #C08A3E;">{atrasados_count}</div>
<div class="gestao-card-sub">{perc_atrasados:.1f}% da carteira</div>
</div>
<div class="gestao-card">
<div class="gestao-card-title">CHURN MENSAL</div>
<div class="gestao-card-value" style="color: #2E7D32;">{churn_atual}</div>
<div class="gestao-card-sub" style="color: {churn_delta_color};">{churn_delta_str}</div>
</div>
</div>
<hr class="gestao-divider">"""
                st.markdown(html_code, unsafe_allow_html=True)

                # Gráficos de suporte abaixo
                col_chart1, col_chart2 = st.columns(2)
                
                with col_chart1:
                    fig_temp = px.pie(
                        df_filtered, 
                        names='Temperatura', 
                        title='Risco dos Clientes (Temperatura)',
                        hole=0.45,
                        color='Temperatura',
                        color_discrete_map={
                            'Crítico': TOTVS_DARK_BLUE, 
                            'Critico': TOTVS_DARK_BLUE, 
                            'Em Risco': TOTVS_CYAN,
                            'Quente': TOTVS_DARK_BLUE,
                            'Morna': TOTVS_CYAN,
                            'Estável': TOTVS_GRAY,
                            'Fria': '#B4D4EE',
                            'Pausado': TOTVS_LIGHT_GRAY
                        }
                    )
                    fig_temp.update_traces(
                        textinfo='percent+value',
                        textposition='inside',
                        marker=dict(line=dict(color='#FFFFFF', width=2))
                    )
                    fig_temp.update_layout(
                        height=400,
                        margin=dict(t=40, b=20, l=10, r=10),
                        showlegend=True,
                        legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
                    )
                    fig_temp = apply_totvs_style(fig_temp)
                    st.plotly_chart(fig_temp, use_container_width=True)
                    
                with col_chart2:
                    df_fase_counts = df_filtered['Fase'].value_counts().reset_index()
                    df_fase_counts['sort_idx'] = df_fase_counts['Fase'].apply(lambda x: FASE_ORDER_MAP.get(norm_fase(x), 999))
                    df_fase_counts = df_fase_counts.sort_values('sort_idx').drop(columns=['sort_idx'])
                    
                    sorted_phases = sorted(
                        df_fase_counts['Fase'].tolist(),
                        key=lambda x: FASE_ORDER_MAP.get(norm_fase(x), 999)
                    )
                    
                    fig_fase = px.bar(
                        df_fase_counts, 
                        x='Fase', y='count', 
                        title='Volume de Projetos por Fase',
                        labels={'count':'Projetos', 'Fase':'Fase'},
                        color_discrete_sequence=[TOTVS_CYAN],
                        category_orders={"Fase": sorted_phases},
                        text_auto=True
                    )
                    fig_fase = apply_totvs_style(fig_fase)
                    fig_fase.update_layout(
                        height=400,
                        margin=dict(t=40, b=20, l=10, r=10)
                    )
                    st.plotly_chart(fig_fase, use_container_width=True)
            
            elif chart_type == "Gráfico por Fase":
                df_fase_counts = df_filtered['Fase'].value_counts().reset_index()
                df_fase_counts['sort_idx'] = df_fase_counts['Fase'].apply(lambda x: FASE_ORDER_MAP.get(norm_fase(x), 999))
                df_fase_counts = df_fase_counts.sort_values('sort_idx').drop(columns=['sort_idx'])
                
                sorted_phases = sorted(
                    df_fase_counts['Fase'].tolist(),
                    key=lambda x: FASE_ORDER_MAP.get(norm_fase(x), 999)
                )
                
                fig = px.bar(
                    df_fase_counts, 
                    x='Fase', y='count', 
                    title=f'Quantos projetos em cada Fase?',
                    labels={'count':'Projetos', 'Fase':'Fase'},
                    color_discrete_sequence=[TOTVS_CYAN],
                    category_orders={"Fase": sorted_phases},
                    text_auto=True
                )
                fig = apply_totvs_style(fig)
            elif chart_type == "Projetos por Consultor (ISM)":
                st.write("### 📊 Projetos por Consultor (ISM)")
                
                # Container para os filtros e controles do gráfico
                col_c1, col_c2 = st.columns([1, 1])
                with col_c1:
                    field_choice = st.radio(
                        "Agrupar por:",
                        options=["ISM Responsável (Campo Customizado)", "Responsável (Assignee)"],
                        horizontal=True,
                        help="Escolha se deseja visualizar pelo campo customizado 'ISM Responsável' (campo oficial de consultor) ou pelo campo padrão 'Assignee' do Jira."
                    )
                
                group_col = "Categoria" if "ISM" in field_choice else "Assignee"
                group_label = "ISM Responsável" if "ISM" in field_choice else "Responsável (Assignee)"
                
                # Copiar dados para não alterar o dataframe original e normalizar nomes
                df_group = df_filtered.copy()
                
                # Tratar "Unassigned" para "Sem Responsável" no campo Assignee
                if group_col == "Assignee":
                    df_group[group_col] = df_group[group_col].replace({"Unassigned": "Sem Responsável"})
                
                with col_c2:
                    hide_unassigned = st.checkbox(
                        "Ocultar projetos sem consultor atribuído",
                        value=False,
                        help="Oculta os projetos marcados como 'Sem ISM' ou 'Sem Responsável'."
                    )
                
                if hide_unassigned:
                    df_group = df_group[~df_group[group_col].isin(["Sem ISM", "Sem Responsável", "None", "Unassigned"])]
                
                if df_group.empty:
                    st.warning("Nenhum dado disponível com os filtros selecionados.")
                else:
                    # Contagem de projetos por consultor
                    df_counts = df_group[group_col].value_counts().reset_index()
                    df_counts.columns = [group_label, "Qtd Projetos"]
                    # Ordena em ordem crescente de Qtd Projetos para que no gráfico horizontal os maiores fiquem no topo
                    df_counts = df_counts.sort_values(by="Qtd Projetos", ascending=True)
                    
                    # KPIs principais
                    total_p = len(df_group)
                    active_consultants = df_group[group_col].nunique()
                    avg_p = total_p / active_consultants if active_consultants > 0 else 0
                    
                    kpi_c1, kpi_c2, kpi_c3 = st.columns(3)
                    with kpi_c1:
                        st.metric("Total de Projetos", f"{total_p}")
                    with kpi_c2:
                        st.metric("Consultores Ativos", f"{active_consultants}")
                    with kpi_c3:
                        st.metric("Média de Projetos / Consultor", f"{avg_p:.1f}")
                    
                    # Gráfico Horizontal
                    fig = px.bar(
                        df_counts,
                        x="Qtd Projetos",
                        y=group_label,
                        orientation='h',
                        title=f"Número de Projetos por {group_label}",
                        labels={"Qtd Projetos": "Quantidade de Projetos", group_label: "Consultor (ISM)"},
                        color="Qtd Projetos",
                        color_continuous_scale=[TOTVS_GRAY, TOTVS_CYAN, TOTVS_DARK_BLUE],
                        text_auto=True
                    )
                    
                    fig = apply_totvs_style(fig)
                    fig.update_layout(
                        height=max(400, len(df_counts) * 35),
                        coloraxis_showscale=False,
                        yaxis=dict(categoryorder='total ascending')
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Seção de Detalhamento por Consultor (Drill-down)
                    st.markdown("---")
                    st.write("### 🔍 Detalhamento dos Projetos por Consultor")
                    
                    consultores_disponiveis = sorted(df_group[group_col].unique())
                    selected_consultor = st.selectbox(
                        "Selecione um Consultor para ver seus projetos:",
                        options=consultores_disponiveis
                    )
                    
                    if selected_consultor:
                        df_consultor_proj = df_group[df_group[group_col] == selected_consultor].copy()
                        
                        st.markdown(f"Exibindo **{len(df_consultor_proj)}** projetos de **{selected_consultor}**")
                        
                        # Resumo de fases e criticidade em colunas
                        col_det1, col_det2 = st.columns([1, 2])
                        with col_det1:
                            st.markdown("**Status / Fase dos Projetos:**")
                            fase_counts = df_consultor_proj['Fase'].value_counts()
                            for fase, count in fase_counts.items():
                                st.write(f"- **{fase}**: {count} projeto(s)")
                                
                            st.markdown("**Criticidade (Temperatura):**")
                            temp_counts = df_consultor_proj['Temperatura'].value_counts()
                            for temp, count in temp_counts.items():
                                emoji_map = {"Crítico": "🔴", "Critico": "🔴", "Em Risco": "🟡", "Estável": "🟢", "Pausado": "⚪"}
                                st.write(f"- {emoji_map.get(temp, '🔵')} **{temp}**: {count} projeto(s)")
                        
                        with col_det2:
                            st.markdown("**Lista de Projetos:**")
                            df_display = df_consultor_proj[['Key', 'Summary', 'Fase', 'Temperatura', 'Data_Limite', 'Atrasado']].copy()
                            df_display = df_display.rename(columns={
                                'Summary': 'Projeto',
                                'Data_Limite': 'Data Limite',
                            })
                            # Formatar data limite de forma robusta
                            df_display['Data Limite'] = pd.to_datetime(df_display['Data Limite'], errors='coerce').dt.strftime('%d/%m/%Y').fillna('Sem data')
                            # Formatar coluna Atrasado
                            df_display['Atrasado'] = df_display['Atrasado'].map({True: "⚠️ Sim", False: "Não"})
                            
                            st.data_editor(
                                df_display,
                                use_container_width=True,
                                hide_index=True,
                                disabled=True,
                                key="tbl_proj_consultor"
                            )
            
            elif chart_type == "Gráfico de Temperatura":
                fig = px.pie(
                    df_filtered, 
                    names='Temperatura', 
                    title='Distribuição de Temperatura dos Clientes',
                    hole=0.45,
                    color='Temperatura',
                    color_discrete_map={
                        'Crítico': TOTVS_DARK_BLUE, 
                        'Critico': TOTVS_DARK_BLUE, 
                        'Em Risco': TOTVS_CYAN,
                        'Quente': TOTVS_DARK_BLUE,
                        'Morna': TOTVS_CYAN,
                        'Estável': TOTVS_GRAY,
                        'Fria': '#B4D4EE',
                        'Pausado': TOTVS_LIGHT_GRAY
                    }
                )
                fig.update_traces(
                    textinfo='percent+value+label',
                    textposition='outside',
                    insidetextorientation='horizontal',
                    marker=dict(line=dict(color='#FFFFFF', width=2))
                )
                fig.update_layout(
                    height=650,
                    margin=dict(t=50, b=50, l=0, r=0),
                    showlegend=True,
                    legend=dict(orientation="h", yanchor="bottom", y=-0.1, xanchor="center", x=0.5),
                )
                fig = apply_totvs_style(fig)
                
                # Habilita seleção no gráfico
                temp_event = st.plotly_chart(
                    fig, use_container_width=True,
                    on_select="rerun", key="temp_pie_chart"
                )

                # --- Detectar clique no gráfico para drill-down ---
                drill_temp = None
                if temp_event and temp_event.selection and temp_event.selection.get('points'):
                    # No Plotly Pie, o label da fatia clicada vem em 'label'
                    drill_temp = temp_event.selection['points'][0].get('label', None)

                # --- Filtro manual de drill-down ---
                st.markdown("#### 🔎 Filtrar Detalhes por Temperatura")
                temps_disponiveis = ['Todos'] + sorted(df_filtered['Temperatura'].unique())
                
                selected_temp_drill = st.selectbox(
                    "Selecione a Temperatura para ver a lista:",
                    options=temps_disponiveis,
                    index=temps_disponiveis.index(drill_temp) if drill_temp and drill_temp in temps_disponiveis else 0,
                    key="temp_drill_select"
                )

                # --- Aplicar filtro e exibir tabela ---
                df_temp_drill = df_filtered.copy()
                if selected_temp_drill != 'Todos':
                    df_temp_drill = df_temp_drill[df_temp_drill['Temperatura'] == selected_temp_drill]

                st.markdown(f"#### 📋 {len(df_temp_drill)} cliente(s) com Temperatura: **{selected_temp_drill}**")
                
                df_temp_display = df_temp_drill[['Key', 'Summary', 'Fase', 'Assignee', 'Temperatura', 'Motivo Criticidade']].copy()
                df_temp_display = df_temp_display.rename(columns={'Summary': 'Projeto', 'Assignee': 'Responsável', 'Motivo Criticidade': 'Motivo da Criticidade'})
                df_temp_display = df_temp_display.sort_values('Projeto')
                df_temp_display = filter_columns(df_temp_display, "temp_drill")
                
                # Exibe a tabela simplificada
                st.data_editor(
                    df_temp_display,
                    use_container_width=True,
                    hide_index=True,
                    disabled=True
                )
                
                # --- Nova Lógica: Análise Profunda Individual ---
                st.markdown("---")
                st.markdown("#### 🤖 Análise Profunda de Criticidade com IA")
                st.write("A IA irá cruzar o histórico do projeto no **Jira** com os tickets recentes no **Freshdesk** para criar um panorama cronológico detalhado do momento atual.")
                
                # Input para selecionar um único projeto para investigar a fundo
                projetos_disponiveis = sorted(df_temp_display['Projeto'].tolist())
                selected_projeto_ai = st.selectbox("Selecione o Projeto para Análise Profunda:", options=[""] + projetos_disponiveis)
                
                if st.button("Analisar Criticidade do Cliente com IA", type="primary") and selected_projeto_ai:
                    if not GEMINI_API_KEY:
                        st.warning("Configure a variável de ambiente `GEMINI_API_KEY` para usar esta função.")
                    else:
                        with st.spinner(f"Processando IA para {selected_projeto_ai} (Extraindo dados do Jira e Freshdesk)..."):
                            # Identifica os dados cruciais da linha selecionada
                            row_proj = df_temp_drill[df_temp_drill['Summary'] == selected_projeto_ai].iloc[0]
                            issue_key = row_proj['Key']
                            temperatura = row_proj['Temperatura']
                            fase = row_proj['Fase']
                            
                            ai_result = analyze_single_client_criticality(issue_key, selected_projeto_ai, temperatura, fase)
                            
                        st.success("Análise concluída!")
                        if 'available_gemini_model' in st.session_state:
                            st.caption(f"ℹ️ Modelo utilizado: `{st.session_state['available_gemini_model']}`")
                        
                        st.markdown(ai_result)

                
            elif chart_type == "Fluxo Mensal":
                st.subheader("📊 Fluxo Mensal de Projetos (Últimos 6 Meses)")

                # ── Filtro de séries ─────────────────────────────────────────────
                todas_series = ["Total Projetos", "Entradas", "Concluídos", "Churn"]
                series_selecionadas = st.multiselect(
                    "🔍 Filtrar séries exibidas no gráfico:",
                    options=todas_series,
                    default=todas_series,
                    key="fluxo_mensal_series"
                )
                if not series_selecionadas:
                    series_selecionadas = todas_series  # fallback: mostra tudo
                
                hoje = datetime.now()
                # 6 meses (contando o mês atual)
                mes_atual = hoje.month
                ano_atual = hoje.year
                
                mes_inicio = mes_atual - 5
                ano_inicio = ano_atual
                if mes_inicio <= 0:
                    mes_inicio += 12
                    ano_inicio -= 1
                
                # Lista de meses
                meses_lista = []
                for i in range(6):
                    m = mes_inicio + i
                    a = ano_inicio
                    if m > 12:
                        m -= 12
                        a += 1
                    meses_lista.append((a, m))
                
                evolution_list = []
                import calendar
                
                for ano, mes in meses_lista:
                    # Encontra o último dia do mês para considerar o "fim" do período
                    if ano == ano_atual and mes == mes_atual:
                        fim_mes = hoje
                    else:
                        ultimo_dia = calendar.monthrange(ano, mes)[1]
                        fim_mes = datetime(ano, mes, ultimo_dia, 23, 59, 59)
                    
                    inicio_mes = datetime(ano, mes, 1)
                    
                    def is_active_at_month(row, end_month):
                        issued_at = row['Criado_Em']
                        if pd.isnull(issued_at) or issued_at > end_month:
                            return False
                        
                        if row['Status_Category'] == 'done':
                            resolved_at = row['Status_Cat_Change']
                            if pd.notnull(resolved_at) and resolved_at <= end_month:
                                return False
                        return True

                    df_at_month = df[df.apply(is_active_at_month, axis=1, end_month=fim_mes)]
                    
                    entries = df[(df['Criado_Em'] >= inicio_mes) & (df['Criado_Em'] <= fim_mes)].shape[0]
                    
                    df_exits = df[
                        (df['Status_Category'] == 'done') & 
                        (df['Status_Cat_Change'] >= inicio_mes) & 
                        (df['Status_Cat_Change'] <= fim_mes)
                    ]
                    
                    churn = df_exits[df_exits['Fase'].str.upper().str.contains("CANCEL", na=False)].shape[0]
                    concluidos = df_exits.shape[0] - churn
                    
                    evolution_list.append({
                        "Mês": f"{mes:02d}/{ano}",
                        "Total Projetos": df_at_month.shape[0],
                        "Entradas": entries,
                        "Concluídos": concluidos,
                        "Churn": churn
                    })
                
                df_evo_mensal = pd.DataFrame(evolution_list)
                
                # Tabela de Métricas Comparativas
                if len(df_evo_mensal) >= 2:
                    current = df_evo_mensal.iloc[-1]
                    previous = df_evo_mensal.iloc[-2]
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Projetos Neste Mês", current["Total Projetos"], int(current["Total Projetos"] - previous["Total Projetos"]))
                    m2.metric("Entradas Mensal", current["Entradas"], int(current["Entradas"] - previous["Entradas"]))
                    m3.metric("Concluídos Mensal", current["Concluídos"], int(current["Concluídos"] - previous["Concluídos"]))
                    m4.metric("Churn Mensal", current["Churn"], int(current["Churn"] - previous["Churn"]), delta_color="inverse")

                # Título dinâmico baseado nas séries selecionadas
                titulo_series = ", ".join(series_selecionadas)
                st.markdown(f"#### {titulo_series}")

                _color_map_mensal = {
                    "Total Projetos": TOTVS_DARK_BLUE,
                    "Entradas": TOTVS_CYAN,
                    "Concluídos": TOTVS_GRAY,
                    "Churn": TOTVS_LIGHT_GRAY
                }
                # Filtra apenas as colunas selecionadas que existem no DataFrame
                _series_validas = [s for s in series_selecionadas if s in df_evo_mensal.columns]

                if _series_validas:
                    fig_flow_mensal = px.bar(
                        df_evo_mensal, x="Mês", y=_series_validas,
                        barmode="group",
                        title=f"Fluxo Mensal — {titulo_series}",
                        color_discrete_map={k: v for k, v in _color_map_mensal.items() if k in _series_validas},
                        text_auto=True
                    )
                    fig_flow_mensal = apply_totvs_style(fig_flow_mensal)
                    if "Total Projetos" in _series_validas:
                        fig_flow_mensal = add_trendline(fig_flow_mensal, df_evo_mensal, "Mês", "Total Projetos", name="Tendência Total", visible=exibir_tendencia)
                    if "Concluídos" in _series_validas:
                        fig_flow_mensal = add_trendline(fig_flow_mensal, df_evo_mensal, "Mês", "Concluídos", name="Tendência Concluídos", visible=exibir_tendencia)
                    st.plotly_chart(fig_flow_mensal, use_container_width=True)
                else:
                    st.info("Selecione ao menos uma série para exibir o gráfico.")

                # ── Racional do Volume Total — Seleção por Mês ────────────────────
                if len(df_evo_mensal) >= 1:
                    st.markdown("#### 📐 Racional do Volume por Mês")

                    # Lista de meses disponíveis para seleção
                    meses_disponiveis = df_evo_mensal["Mês"].tolist()

                    # Seletor horizontal com radio buttons — padrão = último mês
                    mes_selecionado = st.radio(
                        "Selecione o mês para ver o cálculo:",
                        options=meses_disponiveis,
                        index=len(meses_disponiveis) - 1,   # padrão = mês mais recente
                        horizontal=True,
                        key="racional_mes_select"
                    )

                    # Índice do mês selecionado no DataFrame
                    idx_sel = df_evo_mensal[df_evo_mensal["Mês"] == mes_selecionado].index[0]
                    row_sel = df_evo_mensal.loc[idx_sel]

                    # Dados do mês selecionado
                    mes_ref_label  = row_sel["Mês"]
                    entradas_mes   = int(row_sel["Entradas"])
                    concluidos_mes = int(row_sel["Concluídos"])
                    churn_mes      = int(row_sel["Churn"])
                    total_sel      = int(row_sel["Total Projetos"])

                    # Base = mês anterior (se existir na janela)
                    idx_pos = df_evo_mensal.index.get_loc(idx_sel)
                    if idx_pos > 0:
                        row_ant        = df_evo_mensal.iloc[idx_pos - 1]
                        mes_ant_label  = row_ant["Mês"]
                        total_anterior = int(row_ant["Total Projetos"])
                        base_display   = str(total_anterior)
                        base_label     = f"Base ({mes_ant_label})"
                        variacao       = total_sel - total_anterior
                        variacao_sinal = f"+{variacao}" if variacao > 0 else str(variacao)
                        variacao_cor   = "#2E7D32" if variacao <= 0 else "#8B1E0F"
                        nota_variacao  = f'&nbsp;|&nbsp; <b>Variação vs mês anterior:</b> <span class="racional-variacao">{variacao_sinal} projetos</span>'
                        nota_formula   = f"Total {mes_ref_label} = Base ({mes_ant_label}) + Entradas − Concluídos − Churn"
                    else:
                        # Primeiro mês da janela — sem base anterior disponível
                        mes_ant_label  = "—"
                        total_anterior = None
                        base_display   = "—"
                        base_label     = "Base (ant.)"
                        variacao_cor   = "#888888"
                        nota_variacao  = ""
                        nota_formula   = f"Total {mes_ref_label} = Base anterior + Entradas − Concluídos − Churn"

                    racional_html = f"""
<style>
.racional-box {{
    background: linear-gradient(135deg, #f8f9fb 0%, #eef2f8 100%);
    border: 1px solid #dbe3ef;
    border-left: 5px solid #002D57;
    border-radius: 10px;
    padding: 20px 24px;
    margin: 6px 0 10px 0;
    font-family: 'Verdana', sans-serif;
}}
.racional-title {{
    font-size: 0.78rem;
    font-weight: 700;
    color: #002D57;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 14px;
}}
.racional-formula {{
    display: flex;
    flex-wrap: wrap;
    align-items: center;
    gap: 8px;
    font-size: 0.9rem;
    font-weight: 600;
    color: #333;
    margin-bottom: 14px;
}}
.racional-pill {{
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    background: #fff;
    border: 1.5px solid #c8d4e8;
    border-radius: 8px;
    padding: 6px 14px;
    min-width: 72px;
}}
.racional-pill-label {{
    font-size: 0.62rem;
    color: #888;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-bottom: 2px;
}}
.racional-pill-value {{
    font-size: 1.3rem;
    font-weight: 700;
    color: #002D57;
    line-height: 1;
}}
.racional-op {{
    font-size: 1.4rem;
    color: #555;
    font-weight: 700;
    padding: 0 2px;
}}
.racional-result {{
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    background: #002D57;
    border-radius: 8px;
    padding: 6px 18px;
    min-width: 80px;
}}
.racional-result-label {{
    font-size: 0.62rem;
    color: #a0b8d8;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.4px;
    margin-bottom: 2px;
}}
.racional-result-value {{
    font-size: 1.3rem;
    font-weight: 700;
    color: #fff;
    line-height: 1;
}}
.racional-note {{
    font-size: 0.75rem;
    color: #555;
    line-height: 1.6;
    border-top: 1px solid #dbe3ef;
    padding-top: 10px;
    margin-top: 4px;
}}
.racional-variacao {{
    display: inline-block;
    font-weight: 700;
    color: {variacao_cor};
}}
</style>
<div class="racional-box">
  <div class="racional-title">📐 Racional — Como chegamos a {total_sel} projetos em {mes_ref_label}</div>
  <div class="racional-formula">
    <div class="racional-pill">
      <span class="racional-pill-label">{base_label}</span>
      <span class="racional-pill-value">{base_display}</span>
    </div>
    <span class="racional-op">+</span>
    <div class="racional-pill">
      <span class="racional-pill-label">Entradas</span>
      <span class="racional-pill-value">{entradas_mes}</span>
    </div>
    <span class="racional-op">−</span>
    <div class="racional-pill">
      <span class="racional-pill-label">Concluídos</span>
      <span class="racional-pill-value">{concluidos_mes}</span>
    </div>
    <span class="racional-op">−</span>
    <div class="racional-pill">
      <span class="racional-pill-label">Churn</span>
      <span class="racional-pill-value">{churn_mes}</span>
    </div>
    <span class="racional-op">=</span>
    <div class="racional-result">
      <span class="racional-result-label">Total {mes_ref_label}</span>
      <span class="racional-result-value">{total_sel}</span>
    </div>
  </div>
  <div class="racional-note">
    <b>Fórmula:</b> {nota_formula}
    {nota_variacao}
    &nbsp;|&nbsp; <b>Entradas</b> = projetos criados no mês &nbsp;·&nbsp;
    <b>Concluídos</b> = encerrados com sucesso &nbsp;·&nbsp;
    <b>Churn</b> = cancelados no mês.
  </div>
</div>
"""
                    st.markdown(racional_html, unsafe_allow_html=True)
                # ── Fim do Racional ────────────────────────────────────────────────

                # ── Drill-down: Projetos por Categoria no Mês Selecionado ──────────
                st.markdown("#### 🔎 Detalhar Projetos do Mês")

                _col_cat, _col_mes_drill = st.columns([1, 2])

                with _col_cat:
                    _categoria_drill = st.selectbox(
                        "Categoria:",
                        options=["Entradas", "Concluídos", "Cancelados (Churn)"],
                        key="fluxo_drill_categoria"
                    )

                with _col_mes_drill:
                    _mes_drill = st.selectbox(
                        "Mês:",
                        options=df_evo_mensal["Mês"].tolist(),
                        index=len(df_evo_mensal) - 1,
                        key="fluxo_drill_mes"
                    )

                # Recupera as datas do mês selecionado para o drill
                _ano_d, _mes_d = int(_mes_drill.split("/")[1]), int(_mes_drill.split("/")[0])
                _inicio_d = datetime(_ano_d, _mes_d, 1)
                import calendar as _cal
                if _ano_d == ano_atual and _mes_d == mes_atual:
                    _fim_d = datetime.now()
                else:
                    _fim_d = datetime(_ano_d, _mes_d, _cal.monthrange(_ano_d, _mes_d)[1], 23, 59, 59)

                if _categoria_drill == "Entradas":
                    _df_drill = df[
                        (df['Criado_Em'] >= _inicio_d) & (df['Criado_Em'] <= _fim_d)
                    ].copy()
                    _label_drill = "entradas"
                elif _categoria_drill == "Concluídos":
                    _df_exits_d = df[
                        (df['Status_Category'] == 'done') &
                        (df['Status_Cat_Change'] >= _inicio_d) &
                        (df['Status_Cat_Change'] <= _fim_d)
                    ]
                    _df_drill = _df_exits_d[
                        ~_df_exits_d['Fase'].str.upper().str.contains("CANCEL", na=False)
                    ].copy()
                    _label_drill = "concluídos"
                else:  # Cancelados (Churn)
                    _df_exits_d = df[
                        (df['Status_Category'] == 'done') &
                        (df['Status_Cat_Change'] >= _inicio_d) &
                        (df['Status_Cat_Change'] <= _fim_d)
                    ]
                    _df_drill = _df_exits_d[
                        _df_exits_d['Fase'].str.upper().str.contains("CANCEL", na=False)
                    ].copy()
                    _label_drill = "cancelados"

                st.markdown(f"**{len(_df_drill)} projeto(s) {_label_drill} em {_mes_drill}**")

                if not _df_drill.empty:
                    # Seleciona colunas relevantes disponíveis
                    _cols_drill = [c for c in ['Key', 'Summary', 'Fase', 'Assignee', 'Criado_Em', 'Status_Cat_Change'] if c in _df_drill.columns]
                    _df_drill_display = _df_drill[_cols_drill].copy()
                    _df_drill_display = _df_drill_display.rename(columns={
                        'Summary': 'Projeto',
                        'Assignee': 'Responsável',
                        'Criado_Em': 'Criado Em',
                        'Status_Cat_Change': 'Dt. Conclusão/Cancel.'
                    })
                    # Formata datas
                    for _dcol in ['Criado Em', 'Dt. Conclusão/Cancel.']:
                        if _dcol in _df_drill_display.columns:
                            _df_drill_display[_dcol] = pd.to_datetime(_df_drill_display[_dcol], errors='coerce').dt.strftime('%d/%m/%Y')
                    st.dataframe(_df_drill_display.sort_values('Projeto') if 'Projeto' in _df_drill_display.columns else _df_drill_display,
                                 use_container_width=True, hide_index=True)
                else:
                    st.info(f"Nenhum projeto {_label_drill} encontrado em {_mes_drill}.")
                # ── Fim do Drill-down ──────────────────────────────────────────────

                with st.expander("📄 Ver Dados da Evolução Mensal (Tabela)"):
                    df_evo_mensal_sorted = df_evo_mensal.sort_index(ascending=False)
                    df_evo_mensal_filtered = filter_columns(df_evo_mensal_sorted, "evo_mensal")
                    st.dataframe(df_evo_mensal_filtered, use_container_width=True, hide_index=True)
                
            elif chart_type == "Evolução Semanal":
                st.markdown("### Evolução Semanal da Operação (Últimas 6 Semanas)")
                
                # Cálculo da evolução (Reconstrução histórica baseada nas datas de criação e mudança de status)
                @st.cache_data
                def calculate_evolution(df_all):
                    today = datetime.now().replace(hour=23, minute=59, second=59)
                    evolution_list = []
                    
                    # Vamos olhar as últimas 6 semanas
                    for i in range(5, -1, -1):
                        # Ponto de referência: Domingo daquela semana
                        week_end = (today - timedelta(days=today.weekday())).replace(hour=23, minute=59, second=59) - timedelta(days=(i*7) - 6)
                        week_start = week_end - timedelta(days=6, hours=23, minutes=59, seconds=59)
                        
                        # Definição de Saída: Somente se a Categoria for 'done'
                        # Se não for 'done', o projeto ainda está ativo na carteira
                        def is_active_at(row, end_week):
                            issued_at = row['Criado_Em']
                            if issued_at > end_week:
                                return False
                            
                            # Se o projeto está concluído HOJE, verificamos QUANDO ele foi concluído
                            if row['Status_Category'] == 'done':
                                resolved_at = row['Status_Cat_Change']
                                if resolved_at and resolved_at <= end_week:
                                    return False
                            
                            return True

                        df_at_week = df_all[df_all.apply(is_active_at, axis=1, end_week=week_end)]
                        
                        # Entradas desta semana
                        entries = df_all[(df_all['Criado_Em'] >= week_start) & (df_all['Criado_Em'] <= week_end)].shape[0]
                        
                        # Saídas desta semana diferenciadas
                        df_exits = df_all[
                            (df_all['Status_Category'] == 'done') & 
                            (df_all['Status_Cat_Change'] >= week_start) & 
                            (df_all['Status_Cat_Change'] <= week_end)
                        ]
                        
                        churn = df_exits[df_exits['Fase'].str.upper().str.contains("CANCEL", na=False)].shape[0]
                        concluidos = df_exits.shape[0] - churn
                        
                        # Impedidos e Saúde (Re-adicionando após remoção acidental)
                        impedidos = df_at_week[df_at_week['Pausado']].shape[0]
                        andamento = df_at_week[~df_at_week['Pausado']].shape[0]
                        
                        critico = df_at_week[df_at_week['Temperatura'].str.contains('Critico|Crítico', na=False, case=False)].shape[0]
                        risco = df_at_week[df_at_week['Temperatura'].str.contains('Risco', na=False, case=False)].shape[0]
                        estavel = df_at_week[df_at_week['Temperatura'].str.contains('Normal|Estavel|Estável', na=False, case=False)].shape[0]
                        
                        evolution_list.append({
                            "Semana": week_end.strftime("%d/%m"),
                            "Total Projetos": df_at_week.shape[0],
                            "Entradas": entries,
                            "Concluídos": concluidos,
                            "Churn": churn,
                            "Impedidos": impedidos,
                            "Em Andamento": andamento,
                            "Crítico": critico,
                            "Em Risco": risco,
                            "Estável": estavel
                        })
                    return pd.DataFrame(evolution_list)

                # Pool inteligente: Projetos ativos HOJE + projetos que foram finalizados/cancelados nas últimas 6 semanas
                # Isso permite ver as Saídas sem inflar o total com o histórico de anos do Jira
                hoje_ref = datetime.now().replace(hour=23, minute=59, second=59)
                seis_semanas_atras = hoje_ref - timedelta(days=42)
                
                # Filtramos os concluídos/cancelados recentemente usando a Categoria oficial
                df_recent_resolutions = df[
                    (df['Status_Category'] == 'done') &
                    (df['Status_Cat_Change'] >= seis_semanas_atras)
                ].copy()
                
                # Combinamos com os projetos que estão ativos agora (df_filtered)
                df_evo_source = pd.concat([df_filtered, df_recent_resolutions]).drop_duplicates(subset='Key')
                
                df_evo = calculate_evolution(df_evo_source)
                
                # Métricas da Última Semana vs Anterior
                if len(df_evo) >= 2:
                    current = df_evo.iloc[-1]
                    previous = df_evo.iloc[-2]
                    
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Projetos Hoje", current["Total Projetos"], int(current["Total Projetos"] - previous["Total Projetos"]))
                    m2.metric("Entradas (Sem.)", current["Entradas"], int(current["Entradas"] - previous["Entradas"]))
                    m3.metric("Concluídos (Sem.)", current["Concluídos"], int(current["Concluídos"] - previous["Concluídos"]))
                    m4.metric("Churn (Sem.)", current["Churn"], int(current["Churn"] - previous["Churn"]), delta_color="inverse")
                # As métricas gerais já foram exibidas acima
                
                # Gráfico 2: Andamento vs Impedidos (Barras Empilhadas)
                st.markdown("#### Saúde da Carteira - Impedimentos")
                fig_health = px.bar(
                    df_evo, x="Semana", y=["Em Andamento", "Impedidos"],
                    title="Projetos em Andamento vs Impedidos",
                    color_discrete_map={"Em Andamento": TOTVS_CYAN, "Impedidos": TOTVS_DARK_BLUE},
                    text_auto=True
                )
                fig_health = apply_totvs_style(fig_health)
                fig_health = add_trendline(fig_health, df_evo, "Semana", "Em Andamento", name="Tendência Em Andamento", visible=exibir_tendencia)
                st.plotly_chart(fig_health, use_container_width=True)
                
                # Gráfico 3: Distribuição de Temperatura (Barras)
                st.markdown("#### Evolução da Temperatura")
                fig_temp = px.bar(
                    df_evo, x="Semana", y=["Crítico", "Em Risco", "Estável"],
                    barmode="group",
                    title="Evolução do Risco dos Clientes",
                    color_discrete_map={"Crítico": TOTVS_DARK_BLUE, "Em Risco": TOTVS_CYAN, "Estável": TOTVS_GRAY},
                    text_auto=True
                )
                fig_temp = apply_totvs_style(fig_temp)
                fig_temp = add_trendline(fig_temp, df_evo, "Semana", "Crítico", name="Tendência Crítico", visible=exibir_tendencia)
                st.plotly_chart(fig_temp, use_container_width=True)
                
                # Tabela de Dados
                with st.expander("📄 Ver Dados da Evolução (Tabela)"):
                    df_evo_sorted = df_evo.sort_index(ascending=False)
                    df_evo_filtered = filter_columns(df_evo_sorted, "evo_semanal")
                    st.dataframe(df_evo_filtered, use_container_width=True, hide_index=True)

            elif chart_type == "Go-lives por Mês":
                st.subheader("Go-lives vs Conclusões por Mês (Últimos 6 Meses)")
                
                hoje = datetime.now()
                # Encontra o mês e ano iniciais (6 meses atrás contando com o atual)
                mes_atual = hoje.month
                ano_atual = hoje.year
                
                mes_inicio = mes_atual - 5
                ano_inicio = ano_atual
                if mes_inicio <= 0:
                    mes_inicio += 12
                    ano_inicio -= 1
                
                data_inicio = datetime(ano_inicio, mes_inicio, 1)
                
                termos_golive = ["GOLIVE", "GO LIVE", "GO-LIVE", "CONC", "FINALIZADO", "ENTREGUE", "IDERENCIA", "ADERENCIA", "ADERÊNCIA", "EXPANSAO", "EXPANSÃO", "ACOMPANHAMENTO"]
                
                if 'Data_Go_Live' not in df.columns:
                    df['Data_Go_Live'] = pd.NaT
                    st.warning("🔄 Alguns dados podem estar desatualizados no cache. Clique em 'Gerar Dashboard' no menu lateral para atualizar.")
                
                # Projetos que atingiram Go-live
                df_go_lives = df[
                    (pd.notnull(df['Data_Go_Live'])) &
                    (df['Data_Go_Live'] >= data_inicio) & 
                    (~df['Fase'].str.upper().str.contains("CANCEL", na=False))
                ].copy()
                
                if not df_go_lives.empty:
                    df_go_lives['Mês_Ano'] = df_go_lives['Data_Go_Live'].dt.strftime('%m/%Y')
                    df_go_lives['Tipo'] = 'Go-live'
                    df_go_lives['Data do Evento'] = df_go_lives['Data_Go_Live']
                else:
                    df_go_lives = pd.DataFrame(columns=list(df.columns) + ['Mês_Ano', 'Tipo', 'Data do Evento'])

                # Projetos Concluídos (Fechados no sistema)
                df_conclusoes = df[
                    (df['Status_Category'] == 'done') &
                    (df['Status_Cat_Change'] >= data_inicio) &
                    (~df['Fase'].str.upper().str.contains("CANCEL", na=False))
                ].copy()
                
                if not df_conclusoes.empty:
                    df_conclusoes['Mês_Ano'] = df_conclusoes['Status_Cat_Change'].dt.strftime('%m/%Y')
                    df_conclusoes['Tipo'] = 'Conclusão'
                    df_conclusoes['Data do Evento'] = df_conclusoes['Status_Cat_Change']
                else:
                    df_conclusoes = pd.DataFrame(columns=list(df.columns) + ['Mês_Ano', 'Tipo', 'Data do Evento'])

                df_combined = pd.concat([df_go_lives, df_conclusoes])
                
                if not df_combined.empty:
                    periodo_meses = []
                    for i in range(6):
                        m = mes_inicio + i
                        a = ano_inicio
                        if m > 12:
                            m -= 12
                            a += 1
                        periodo_meses.append(f"{m:02d}/{a}")
                    
                    df_agrupado = df_combined.groupby(['Mês_Ano', 'Tipo']).size().reset_index(name='Quantidade')
                    
                    all_combinations = pd.DataFrame([(m, t) for m in periodo_meses for t in ['Go-live', 'Conclusão']], columns=['Mês_Ano', 'Tipo'])
                    df_agrupado = pd.merge(all_combinations, df_agrupado, on=['Mês_Ano', 'Tipo'], how='left').fillna(0)
                    
                    fig = px.bar(
                        df_agrupado,
                        x='Mês_Ano', y='Quantidade', color='Tipo',
                        barmode='group',
                        title='Volume Histórico: Go-lives vs Conclusões',
                        labels={'Quantidade': 'Qtd de Projetos', 'Mês_Ano': 'Mês/Ano', 'Tipo': 'Evento'},
                        text_auto=True,
                        color_discrete_map={'Go-live': TOTVS_CYAN, 'Conclusão': TOTVS_DARK_BLUE}
                    )
                    fig = apply_totvs_style(fig)
                    fig = add_trendline(fig, df_agrupado, 'Mês_Ano', 'Quantidade', group_col='Tipo', visible=exibir_tendencia)
                    
                    golive_event = st.plotly_chart(
                        fig, use_container_width=True,
                        on_select="rerun", key="golive_bar_chart"
                    )

                    drill_month = None
                    drill_type = None
                    if golive_event and golive_event.selection and golive_event.selection.get('points'):
                        pt = golive_event.selection['points'][0]
                        drill_month = pt.get('x', None)
                        drill_type = pt.get('legendgroup', None)

                    st.markdown("#### Filtrar Detalhamento")
                    meses_disp = ['Todos'] + periodo_meses
                    tipos_disp = ['Todos', 'Go-live', 'Conclusão']
                    
                    col_m, col_t = st.columns(2)
                    with col_m:
                        selected_month_drill = st.selectbox(
                            "Selecione o Mês:",
                            options=meses_disp,
                            index=meses_disp.index(drill_month) if drill_month and drill_month in meses_disp else 0,
                            key="golive_drill_select"
                        )
                    with col_t:
                        selected_type_drill = st.selectbox(
                            "Tipo de Evento:",
                            options=tipos_disp,
                            index=tipos_disp.index(drill_type) if drill_type and drill_type in tipos_disp else 0,
                            key="golive_type_select"
                        )

                    df_final_drill = df_combined.copy()
                    if selected_month_drill != 'Todos':
                        df_final_drill = df_final_drill[df_final_drill['Mês_Ano'] == selected_month_drill]
                    if selected_type_drill != 'Todos':
                        df_final_drill = df_final_drill[df_final_drill['Tipo'] == selected_type_drill]
                    
                    st.subheader(f"📋 Lista de Clientes ({len(df_final_drill)} projetos)")
                    df_drill_display = df_final_drill[['Tipo', 'Key', 'Summary', 'Data do Evento', 'Fase', 'Assignee']].copy()
                    
                    # Formata a data se não for nula
                    df_drill_display['Data do Evento'] = pd.to_datetime(df_drill_display['Data do Evento']).dt.strftime('%d/%m/%Y').fillna('')
                    
                    df_drill_display = df_drill_display.rename(columns={
                        'Tipo': 'Tipo de Evento',
                        'Summary': 'Projeto', 
                        'Assignee': 'Responsável'
                    })
                    
                    df_drill_sorted = df_drill_display.sort_values(['Tipo de Evento', 'Data do Evento'], ascending=[False, False])
                    df_drill_filtered = filter_columns(df_drill_sorted, "golive_drill")
                    
                    st.data_editor(
                        df_drill_filtered,
                        use_container_width=True,
                        hide_index=True,
                        disabled=True
                    )
                else:
                    st.info("Nenhum Go-live ou Conclusão encontrado nos últimos 6 meses.")

            elif chart_type == "Clientes Atrasados":
                # Filtra apenas atrasados, desconsiderando clientes que estão implantando ERP
                df_atrasados = df_filtered[
                    (df_filtered['Atrasado']) & 
                    (~df_filtered['Fase'].astype(str).str.lower().str.contains("implantando erp", na=False)) &
                    (~df_filtered['Motivo Atraso'].astype(str).str.lower().str.contains("implantacao de erp|implantação de erp|implantando erp", na=False))
                ].copy()
                
                c1, c2 = st.columns(2)
                c1.metric("Total Projetos (Filtro)", len(df_filtered))
                c2.metric("Projetos Atrasados", len(df_atrasados), delta=None, delta_color="inverse")
                
                if not df_atrasados.empty:
                    st.subheader("Distribuição por Tipo de Atraso")
                    fig = px.bar(
                        df_atrasados['Motivo Atraso'].value_counts().reset_index(),
                        x='Motivo Atraso', y='count',
                        title='Principais Tipos de Atraso',
                        labels={'count': 'Qtd Projetos', 'Motivo Atraso': 'Tipo de Atraso'},
                        color_discrete_sequence=[TOTVS_CYAN],
                        text_auto=True
                    )
                    fig = apply_totvs_style(fig)
                    # Habilita seleção no gráfico
                    delay_event = st.plotly_chart(
                        fig, use_container_width=True,
                        on_select="rerun", key="delay_bar_chart"
                    )

                    # --- Detectar clique no gráfico para drill-down ---
                    drill_delay = None
                    if delay_event and delay_event.selection and delay_event.selection.get('points'):
                        drill_delay = delay_event.selection['points'][0].get('x', None)

                    # --- Filtro manual de drill-down ---
                    st.markdown("#### Filtrar por Tipo de Atraso")
                    motivos_atraso_disp = ['Todos'] + sorted(df_atrasados['Motivo Atraso'].unique())
                    
                    selected_delay_drill = st.selectbox(
                        "Selecione o Motivo para ver a lista:",
                        options=motivos_atraso_disp,
                        index=motivos_atraso_disp.index(drill_delay) if drill_delay and drill_delay in motivos_atraso_disp else 0,
                        key="delay_drill_select"
                    )

                    # --- Aplicar filtro na tabela final ---
                    df_final_atrasados = df_atrasados.copy()
                    if selected_delay_drill != 'Todos':
                        df_final_atrasados = df_final_atrasados[df_final_atrasados['Motivo Atraso'] == selected_delay_drill]

                    # --- Análise de IA por Comentários ---
                    ai_cache_key = 'ai_atraso_analysis'
                    col_btn1, col_btn2 = st.columns([2, 5])
                    with col_btn1:
                        if st.button("🤖 Analisar Comentários com IA", key="btn_ai_atraso_analysis_main",
                                     help="Busca os comentários de cada projeto atrasado e usa o Gemini para gerar um entendimento do motivo real do atraso."):
                            if not GEMINI_API_KEY:
                                st.warning("Configure a variável de ambiente `GEMINI_API_KEY` para usar esta função.")
                            else:
                                resultados = analyze_delayed_projects(df_atrasados)
                                st.session_state[ai_cache_key] = resultados
                                st.success("Análise concluída!")

                    with col_btn2:
                        if ai_cache_key in st.session_state:
                            st.caption(f"✅ Análise IA disponível para {len(st.session_state[ai_cache_key])} projetos. Clique novamente para atualizar.")

                    st.subheader(f"Detalhamento: {len(df_final_atrasados)} Clientes ({selected_delay_drill})")
                    df_atrasados_display = df_final_atrasados[['Key', 'Summary', 'Data_Limite_Original', 'Motivo Atraso', 'Motivo Detalhado', 'Dias_Atraso', 'Dias_Projeto', 'Assignee']].copy()
                    # Formata a Previsão Original (primeiro prazo definido no Jira via changelog)
                    df_atrasados_display['Data_Limite_Original'] = pd.to_datetime(df_atrasados_display['Data_Limite_Original']).dt.strftime('%d/%m/%Y').fillna('—')
                    df_atrasados_display = df_atrasados_display.rename(columns={
                        'Summary': 'Projeto', 
                        'Data_Limite_Original': 'Previsão Original',
                        'Motivo Atraso': 'Tipo de Atraso',
                        'Motivo Detalhado': 'Motivo do Atraso',
                        'Dias_Atraso': 'Tempo de Atraso (Dias)', 
                        'Dias_Projeto': 'Tempo de Projeto (Dias)',
                        'Assignee': 'Responsável'
                    })
                    
                    if ai_cache_key in st.session_state:
                        ai_map = st.session_state[ai_cache_key]
                        df_atrasados_display['🤖 Entendimento do Atraso (IA)'] = df_atrasados_display['Key'].map(
                            lambda k: ai_map.get(k, '—')
                        )
                        if 'available_gemini_model' in st.session_state:
                            st.caption(f"ℹ️ Modelo utilizado: `{st.session_state['available_gemini_model']}`")

                    df_atrasados_sorted = df_atrasados_display.sort_values('Tempo de Atraso (Dias)', ascending=False)
                    df_atrasados_filtered = filter_columns(df_atrasados_sorted, "atrasados")
                    col_config = {}
                    if "🤖 Entendimento do Atraso (IA)" in df_atrasados_filtered.columns:
                        col_config["🤖 Entendimento do Atraso (IA)"] = st.column_config.TextColumn(
                            "🤖 Entendimento do Atraso (IA)",
                            width="large",
                            help="Dê um clique duplo para ver o texto completo"
                        )
                    if "Motivo do Atraso" in df_atrasados_filtered.columns:
                        col_config["Motivo do Atraso"] = st.column_config.TextColumn(width="large")
                    if "Projeto" in df_atrasados_filtered.columns:
                        col_config["Projeto"] = st.column_config.TextColumn(width="medium")
                    
                    st.data_editor(
                        df_atrasados_filtered,
                        use_container_width=True,
                        hide_index=True,
                        column_config=col_config,
                        disabled=True
                    )

                    # Detalhes do Projeto (para garantir integridade da leitura)
                    if ai_cache_key in st.session_state:
                        st.markdown("---")
                        st.subheader("📝 Detalhes da Análise IA")
                        proj_list = df_atrasados_display['Projeto'].tolist()
                        if proj_list:
                            sel_proj = st.selectbox("Selecione um projeto para ver a análise completa:", proj_list, key="sel_atrasados_main_ai_proj")
                            sel_rows = df_atrasados_display[df_atrasados_display['Projeto'] == sel_proj]
                            if not sel_rows.empty and '🤖 Entendimento do Atraso (IA)' in sel_rows.columns:
                                full_text = sel_rows['🤖 Entendimento do Atraso (IA)'].values[0]
                                st.info(f"**{sel_proj}**: \n\n {full_text}")
                else:
                    st.success("Tudo em dia! Nenhum projeto atrasado encontrado nos filtros atuais.")

            elif chart_type == "Controle de Prazos e Replanejamentos":
                hoje = datetime.now()
                
                # Separar projetos de Implantação de ERP
                is_erp = (df_filtered['Fase'].astype(str).str.lower().str.contains("implantando erp", na=False)) | \
                         (df_filtered['Motivo Atraso'].astype(str).str.lower().str.contains("implantacao de erp|implantação de erp|implantando erp", na=False))
                
                df_erp = df_filtered[is_erp].copy()
                df_ativos = df_filtered[~is_erp].copy()
                
                # Garantir formato de data
                df_ativos['Data_Limite'] = pd.to_datetime(df_ativos['Data_Limite'])
                df_ativos['Data_Limite_Original'] = pd.to_datetime(df_ativos['Data_Limite_Original'])
                df_erp['Data_Limite'] = pd.to_datetime(df_erp['Data_Limite'])
                
                total_ativos = len(df_ativos)
                
                # 1. nº projetos dentro do prazo (Atrasado == False)
                df_dentro = df_ativos[~df_ativos['Atrasado']]
                df_fora = df_ativos[df_ativos['Atrasado']]
                
                no_prazo_count = len(df_dentro)
                fora_prazo_count = len(df_fora)
                
                # Replanejados e Atualmente dentro do prazo
                df_repl_dentro = df_dentro[
                    (df_dentro['Data_Limite'].notnull()) & 
                    (df_dentro['Data_Limite_Original'].notnull()) & 
                    (df_dentro['Data_Limite'] != df_dentro['Data_Limite_Original'])
                ]
                
                # Replanejados e Atualmente no prazo, com prazo original já vencido (expirado)
                df_repl_dentro_orig_passada = df_repl_dentro[
                    df_repl_dentro['Data_Limite_Original'].dt.date < hoje.date()
                ]
                
                repl_dentro_count = len(df_repl_dentro)
                repl_dentro_exp_count = len(df_repl_dentro_orig_passada)
                
                # --- KPI Cards ---
                st.markdown("""
                <style>
                .prazos-cards { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 28px; }
                .prazos-card {
                    flex: 1; min-width: 220px; background: #FFFFFF; border: 1px solid #E5E3DC;
                    border-top: 4px solid #002D57; border-radius: 10px; padding: 16px 14px 12px 14px;
                    font-family: 'Verdana', sans-serif; box-shadow: 0 2px 6px rgba(0,0,0,0.04);
                    transition: transform 0.18s, box-shadow 0.18s;
                }
                .prazos-card:hover { transform: translateY(-3px); box-shadow: 0 6px 14px rgba(0,0,0,0.09); }
                .prazos-card-title { font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: #888; margin: 0 0 6px 0; }
                .prazos-card-value { font-size: 2.2rem; font-weight: 700; line-height: 1; margin: 0 0 4px 0; }
                .prazos-card-sub { font-size: 0.75rem; color: #555; margin: 0; }
                .prazos-card-ok { border-top-color: #2E7D32; }
                .prazos-card-nok { border-top-color: #8B1E0F; }
                .prazos-card-repl { border-top-color: #00A3E0; }
                </style>
                """, unsafe_allow_html=True)
                
                perc_dentro = (no_prazo_count / total_ativos * 100) if total_ativos > 0 else 0
                perc_fora = (fora_prazo_count / total_ativos * 100) if total_ativos > 0 else 0
                
                st.markdown(f"""
                <div class="prazos-cards">
                  <div class="prazos-card">
                    <div class="prazos-card-title">Total de Projetos Ativos</div>
                    <div class="prazos-card-value" style="color:#002D57;">{total_ativos}</div>
                    <div class="prazos-card-sub">Portfólio sob filtros (exclui ERP)</div>
                  </div>
                  <div class="prazos-card prazos-card-ok">
                    <div class="prazos-card-title">Projetos Dentro do Prazo</div>
                    <div class="prazos-card-value" style="color:#2E7D32;">{no_prazo_count}</div>
                    <div class="prazos-card-sub">{perc_dentro:.1f}% dos projetos ativos</div>
                  </div>
                  <div class="prazos-card prazos-card-nok">
                    <div class="prazos-card-title">Projetos Fora do Prazo</div>
                    <div class="prazos-card-value" style="color:#8B1E0F;">{fora_prazo_count}</div>
                    <div class="prazos-card-sub">{perc_fora:.1f}% dos projetos ativos</div>
                  </div>
                  <div class="prazos-card prazos-card-repl">
                    <div class="prazos-card-title">Replanejados & Atualmente no Prazo</div>
                    <div class="prazos-card-value" style="color:#00A3E0;">{repl_dentro_count}</div>
                    <div class="prazos-card-sub">{repl_dentro_exp_count} tinham prazo original já vencido</div>
                  </div>
                </div>
                """, unsafe_allow_html=True)
                
                # --- Seleção da Referência de Cálculo de Atraso ---
                st.markdown("### 📊 Análise por Faixas de Atraso")
                opcao_ref = st.radio(
                    "Calcular tempo de atraso com base em:",
                    ["Data Limite Atual (Prazo repactuado)", "Data Limite Original (Primeiro prazo planejado)"],
                    horizontal=True,
                    key="prazos_ref_radio"
                )
                
                use_original = "Original" in opcao_ref
                
                # Filtrar apenas projetos atrasados que possuem a data de referência preenchida
                df_atraso_ref = df_fora.copy()
                if use_original:
                    df_atraso_ref = df_atraso_ref[df_atraso_ref['Data_Limite_Original'].notnull()]
                    # Calcular atraso em dias em relação à data original
                    df_atraso_ref['dias_atraso_calc'] = df_atraso_ref['Data_Limite_Original'].apply(
                        lambda x: (hoje.date() - x.date()).days
                    )
                else:
                    df_atraso_ref = df_atraso_ref[df_atraso_ref['Data_Limite'].notnull()]
                    # Calcular atraso em dias em relação à data atual
                    df_atraso_ref['dias_atraso_calc'] = df_atraso_ref['Data_Limite'].apply(
                        lambda x: (hoje.date() - x.date()).days
                    )
                
                # Definir faixas de atraso
                def classificar_faixa(dias):
                    if dias <= 30:
                        return "1. Até 30 dias"
                    elif dias <= 90:
                        return "2. 31 a 90 dias"
                    elif dias <= 180:
                        return "3. 91 dias a 6 meses"
                    else:
                        return "4. Mais de 6 meses"
                
                if not df_atraso_ref.empty:
                    df_atraso_ref['Faixa_Atraso'] = df_atraso_ref['dias_atraso_calc'].apply(classificar_faixa)
                else:
                    df_atraso_ref['Faixa_Atraso'] = pd.Series(dtype=str)
                
                # Contagens por faixa
                faixas_ordem = ["1. Até 30 dias", "2. 31 a 90 dias", "3. 91 dias a 6 meses", "4. Mais de 6 meses"]
                faixas_counts = {}
                for f in faixas_ordem:
                    if not df_atraso_ref.empty:
                        faixas_counts[f] = df_atraso_ref[df_atraso_ref['Faixa_Atraso'] == f].shape[0]
                    else:
                        faixas_counts[f] = 0
                
                df_faixas = pd.DataFrame({
                    "Faixa de Atraso": list(faixas_counts.keys()),
                    "Qtd Projetos": list(faixas_counts.values())
                })
                
                col1, col2 = st.columns([3, 2])
                
                with col1:
                    # Gráfico de Barras das faixas de atraso
                    fig = px.bar(
                        df_faixas,
                        x='Faixa de Atraso', y='Qtd Projetos',
                        title=f'Quantidade de Projetos por Faixa de Atraso ({opcao_ref})',
                        labels={'Qtd Projetos': 'Nº de Projetos', 'Faixa de Atraso': 'Faixa de Atraso'},
                        color_discrete_sequence=[TOTVS_DARK_BLUE],
                        text_auto=True
                    )
                    fig = apply_totvs_style(fig)
                    # Habilita clique para filtro (drill-down)
                    faixa_selection = st.plotly_chart(
                        fig, use_container_width=True,
                        on_select="rerun", key="faixas_bar_chart"
                    )
                
                with col2:
                    # Tabela resumida das faixas
                    st.markdown("<div style='margin-top: 40px;'></div>", unsafe_allow_html=True)
                    st.dataframe(
                        df_faixas,
                        use_container_width=True,
                        hide_index=True
                    )
                
                # --- Drill-down ou exibição geral de atrasados ---
                drill_faixa = None
                if faixa_selection and faixa_selection.selection and faixa_selection.selection.get('points'):
                    drill_faixa = faixa_selection.selection['points'][0].get('x', None)
                
                st.markdown("#### Detalhamento de Projetos Atrasados")
                
                # Seletor para filtrar a tabela manualmente ou ver a selecionada no gráfico
                opcoes_tabela = ["Todos os Atrasados"] + faixas_ordem
                default_idx = opcoes_tabela.index(drill_faixa) if drill_faixa in opcoes_tabela else 0
                
                faixa_selecionada = st.selectbox(
                    "Selecione a faixa para ver os detalhes:",
                    options=opcoes_tabela,
                    index=default_idx,
                    key="prazos_faixa_select"
                )
                
                df_tabela_atraso = df_atraso_ref.copy()
                if faixa_selecionada != "Todos os Atrasados":
                    df_tabela_atraso = df_tabela_atraso[df_tabela_atraso['Faixa_Atraso'] == faixa_selecionada]
                
                # --- Análise de IA por Comentários ---
                ai_cache_key = 'ai_atraso_analysis'
                col_btn1, col_btn2 = st.columns([2, 5])
                with col_btn1:
                    if st.button("🤖 Analisar Comentários com IA", key="btn_ai_atraso_analysis",
                                 help="Busca os comentários de cada projeto atrasado e usa o Gemini para gerar um entendimento do motivo real do atraso."):
                        if not GEMINI_API_KEY:
                            st.warning("Configure a variável de ambiente `GEMINI_API_KEY` para usar esta função.")
                        else:
                            resultados = analyze_delayed_projects(df_atraso_ref)
                            st.session_state[ai_cache_key] = resultados
                            st.success("Análise concluída!")

                with col_btn2:
                    if ai_cache_key in st.session_state:
                        st.caption(f"✅ Análise IA disponível para {len(st.session_state[ai_cache_key])} projetos. Clique novamente para atualizar.")

                if not df_tabela_atraso.empty:
                    # Preparar colunas para exibição
                    df_tabela_atraso['Data_Limite'] = df_tabela_atraso['Data_Limite'].dt.strftime('%d/%m/%Y').fillna('—')
                    df_tabela_atraso['Data_Limite_Original'] = df_tabela_atraso['Data_Limite_Original'].dt.strftime('%d/%m/%Y').fillna('—')
                    
                    df_display = df_tabela_atraso[[
                        'Key', 'Summary', 'Data_Limite_Original', 'Data_Limite', 
                        'dias_atraso_calc', 'Fase', 'Assignee', 'Motivo Atraso'
                    ]].copy()
                    
                    df_display = df_display.rename(columns={
                        'Summary': 'Projeto',
                        'Data_Limite_Original': 'Prazo Original',
                        'Data_Limite': 'Prazo Atual',
                        'dias_atraso_calc': 'Atraso (Dias)',
                        'Assignee': 'Responsável',
                        'Motivo Atraso': 'Motivo do Atraso'
                    })
                    
                    if ai_cache_key in st.session_state:
                        ai_map = st.session_state[ai_cache_key]
                        df_display['🤖 Entendimento do Atraso (IA)'] = df_display['Key'].map(
                            lambda k: ai_map.get(k, '—')
                        )
                        if 'available_gemini_model' in st.session_state:
                            st.caption(f"ℹ️ Modelo utilizado: `{st.session_state['available_gemini_model']}`")
                    
                    df_display_sorted = df_display.sort_values('Atraso (Dias)', ascending=False)
                    df_display_filtered = filter_columns(df_display_sorted, "prazos_detalhe")
                    
                    col_config_dict = {
                        "Motivo do Atraso": st.column_config.TextColumn(width="medium")
                    }
                    if "🤖 Entendimento do Atraso (IA)" in df_display_filtered.columns:
                        col_config_dict["🤖 Entendimento do Atraso (IA)"] = st.column_config.TextColumn(
                            "🤖 Entendimento do Atraso (IA)",
                            width="large",
                            help="Dê um clique duplo para ver o texto completo"
                        )
                    if "Projeto" in df_display_filtered.columns:
                        col_config_dict["Projeto"] = st.column_config.TextColumn(width="medium")

                    col_config = merge_column_config(col_config_dict)
                    
                    st.data_editor(
                        df_display_filtered,
                        use_container_width=True,
                        hide_index=True,
                        column_config=col_config,
                        disabled=True
                    )

                    # Detalhes do Projeto (para garantir integridade da leitura)
                    if ai_cache_key in st.session_state:
                        st.markdown("---")
                        st.subheader("📝 Detalhes da Análise IA")
                        proj_list = df_display['Projeto'].tolist()
                        if proj_list:
                            sel_proj = st.selectbox("Selecione um projeto para ver a análise completa:", proj_list, key="sel_atraso_ai_proj")
                            sel_rows = df_display[df_display['Projeto'] == sel_proj]
                            if not sel_rows.empty and '🤖 Entendimento do Atraso (IA)' in sel_rows.columns:
                                full_text = sel_rows['🤖 Entendimento do Atraso (IA)'].values[0]
                                st.info(f"**{sel_proj}**: \n\n {full_text}")
                else:
                    st.info(f"Nenhum projeto encontrado para a seleção: {faixa_selecionada}")
                
                # --- Seção de Replanejados e no Prazo ---
                st.markdown("---")
                st.markdown("### 🔄 Detalhamento de Projetos Replanejados e atualmente no Prazo")
                st.markdown(
                    f"Atualmente, **{repl_dentro_count} projetos** foram replanejados (tiveram alteração de data limite) "
                    f"e estão dentro do novo cronograma pactuado. Desses, **{repl_dentro_exp_count} projetos** estariam "
                    f"atrasados caso a data original fosse mantida."
                )
                
                if not df_repl_dentro.empty:
                    df_repl_display = df_repl_dentro.copy()
                    
                    # Identificar se a data original já expirou
                    df_repl_display['Original Expirado'] = df_repl_display['Data_Limite_Original'].dt.date.apply(
                        lambda d: "Sim ⚠️" if pd.notnull(d) and d < hoje.date() else "Não"
                    )
                    
                    df_repl_display['Data_Limite'] = df_repl_display['Data_Limite'].dt.strftime('%d/%m/%Y').fillna('—')
                    df_repl_display['Data_Limite_Original'] = df_repl_display['Data_Limite_Original'].dt.strftime('%d/%m/%Y').fillna('—')
                    
                    df_repl_table = df_repl_display[[
                        'Key', 'Summary', 'Data_Limite_Original', 'Data_Limite', 
                        'Original Expirado', 'Fase', 'Assignee'
                    ]].copy()
                    
                    df_repl_table = df_repl_table.rename(columns={
                        'Summary': 'Projeto',
                        'Data_Limite_Original': 'Prazo Original',
                        'Data_Limite': 'Prazo Repactuado',
                        'Assignee': 'Responsável'
                    })
                    
                    df_repl_table_sorted = df_repl_table.sort_values('Original Expirado', ascending=False)
                    df_repl_table_filtered = filter_columns(df_repl_table_sorted, "repl_detalhe")
                    
                    st.data_editor(
                        df_repl_table_filtered,
                        use_container_width=True,
                        hide_index=True,
                        disabled=True
                    )
                else:
                    st.info("Nenhum projeto replanejado no prazo encontrado.")

                # --- Seção de Implantação de ERP ---
                st.markdown("---")
                st.markdown("### ⚙️ Implantação de ERP")
                st.markdown(
                    "Estes são os projetos que estão na fase de Implantação de ERP ou cujo motivo de atraso "
                    "está associado à Implantação de ERP. Eles são acompanhados de forma independente."
                )
                
                with st.container(border=True):
                    st.markdown("#### 📊 Sub-dashboard: Implantação de ERP")
                    
                    total_erp = len(df_erp)
                    atrasados_erp = df_erp[df_erp['Atrasado']].shape[0]
                    no_prazo_erp = total_erp - atrasados_erp
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Total Projetos ERP", total_erp)
                    c2.metric("Atrasados ERP", atrasados_erp, delta=None, delta_color="inverse")
                    c3.metric("No Prazo ERP", no_prazo_erp)
                    
                    if total_erp > 0:
                        erp_status_df = pd.DataFrame({
                            "Status": ["Atrasado", "No Prazo"],
                            "Qtd": [atrasados_erp, no_prazo_erp]
                        })
                        fig_erp = px.pie(
                            erp_status_df,
                            values='Qtd',
                            names='Status',
                            title='Status das Implantações de ERP',
                            color='Status',
                            color_discrete_map={'Atrasado': '#8B1E0F', 'No Prazo': '#2E7D32'},
                            hole=0.4
                        )
                        fig_erp = apply_totvs_style(fig_erp)
                        st.plotly_chart(fig_erp, use_container_width=True, key="erp_donut_chart")
                    
                    st.markdown("##### Detalhamento dos Projetos ERP")
                    if not df_erp.empty:
                        df_erp_display = df_erp.copy()
                        df_erp_display['Data_Limite'] = pd.to_datetime(df_erp_display['Data_Limite']).dt.strftime('%d/%m/%Y').fillna('—')
                        df_erp_display['Status do Prazo'] = df_erp_display['Atrasado'].apply(lambda x: "Atrasado ⚠️" if x else "No Prazo ✅")
                        
                        df_erp_table = df_erp_display[[
                            'Key', 'Summary', 'Fase', 'Data_Limite', 'Status do Prazo', 'Motivo Atraso', 'Assignee'
                        ]].copy()
                        
                        df_erp_table = df_erp_table.rename(columns={
                            'Summary': 'Projeto',
                            'Data_Limite': 'Data Limite',
                            'Motivo Atraso': 'Motivo do Atraso',
                            'Assignee': 'Responsável'
                        })
                        
                        df_erp_table_sorted = df_erp_table.sort_values('Status do Prazo', ascending=True)
                        df_erp_table_filtered = filter_columns(df_erp_table_sorted, "erp_detalhe")
                        
                        col_config_erp = merge_column_config({
                            "Motivo do Atraso": st.column_config.TextColumn(width="medium")
                        })
                        
                        st.data_editor(
                            df_erp_table_filtered,
                            use_container_width=True,
                            hide_index=True,
                            column_config=col_config_erp,
                            disabled=True
                        )
                    else:
                        st.info("Nenhum projeto em implantação de ERP encontrado nos filtros atuais.")

            elif chart_type == "Previsão de Entrega":
                # ---------------------------------------------------------------
                # PREVISÃO DE ENTREGA — DESVIO E SAÚDE
                # Regras:
                #   - Main: No prazo (<=0 ou <=14), Em risco (15-45), Crítico (>45)
                #   - Implantando ERP: Apenas no total e card específico (sem gráfico)
                #   - Novos Projetos / Aderência: Incluídos no Main
                #   - Acompanhamento: Gráfico específico de tempo total, saudável até 14d após GL.
                #   - Expansão: Apenas número no KPI, sem considerar limite.
                # ---------------------------------------------------------------
                import plotly.graph_objects as go

                RISCO_DIAS  = 14   # limiar inferior: > 14 dias → em risco
                CRITICO_DIAS = 45  # limiar superior: > 45 dias → crítico

                hoje = datetime.now()

                FASE_ERP = "implantando erp"
                FASES_ACOMP = ["acompanhamento"]
                FASES_EXP = ["expansao", "expansão"]

                def _norm_fase(f):
                    return unicodedata.normalize('NFD', str(f)).encode('ascii', 'ignore').decode('utf-8').lower().strip()

                df_prev_full = df_filtered.copy()

                # --- Separação dos Datasets ---
                fases_ignorar_main = FASES_ACOMP + FASES_EXP
                df_main = df_prev_full[~df_prev_full['Fase'].apply(lambda f: _norm_fase(f) in fases_ignorar_main)].copy()
                df_acomp = df_prev_full[df_prev_full['Fase'].apply(lambda f: _norm_fase(f) in FASES_ACOMP)].copy()
                df_exp = df_prev_full[df_prev_full['Fase'].apply(lambda f: _norm_fase(f) in FASES_EXP)].copy()

                # --- Lógica Main ---
                def calc_desvio(row):
                    due = row['Data_Limite']
                    go_live = row['Data_Go_Live']
                    if pd.isnull(due):
                        return 0
                    if pd.notnull(go_live):
                        return 0
                    return (hoje - due).days

                df_main['Desvio_Dias'] = df_main.apply(calc_desvio, axis=1)
                df_main['_eh_erp'] = df_main['Fase'].apply(lambda f: _norm_fase(f) == FASE_ERP)

                def classifica(row):
                    if row['_eh_erp']:
                        return 'Implantando ERP'
                    d = row['Desvio_Dias']
                    if d <= 0:
                        return 'No prazo'
                    elif d <= RISCO_DIAS:
                        return 'No prazo'
                    elif d <= CRITICO_DIAS:
                        return 'Em risco'
                    else:
                        return 'Crítico'

                df_main['Status_Tempo'] = df_main.apply(classifica, axis=1)

                # --- KPIs ---
                total_projetos   = len(df_prev_full)
                df_contagem      = df_main[~df_main['_eh_erp']]
                no_prazo_count   = (df_contagem['Status_Tempo'] == 'No prazo').sum()
                em_risco_count   = (df_contagem['Status_Tempo'] == 'Em risco').sum()
                critico_count    = (df_contagem['Status_Tempo'] == 'Crítico').sum()
                perc_no_prazo    = (no_prazo_count / len(df_contagem) * 100) if len(df_contagem) > 0 else 0
                perc_em_risco    = (em_risco_count / len(df_contagem) * 100) if len(df_contagem) > 0 else 0
                perc_critico     = (critico_count  / len(df_contagem) * 100) if len(df_contagem) > 0 else 0
                num_expansao     = len(df_exp)
                num_acomp        = len(df_acomp)

                # --- Lógica Acompanhamento ---
                def calc_saude_acomp(row):
                    gl = row['Data_Go_Live']
                    if pd.isnull(gl):
                        return 'Excedido / Sem Go-live'
                    dias_pos = (hoje - gl).days
                    if dias_pos <= 14:
                        return 'Saudável (≤ 14d)'
                    else:
                        return 'Excedido (> 14d)'

                df_acomp['Status_Acomp'] = df_acomp.apply(calc_saude_acomp, axis=1)
                df_acomp['Dias_Pos_GL'] = df_acomp['Data_Go_Live'].apply(lambda gl: (hoje - gl).days if pd.notnull(gl) else 0)

                # ---- CSS dos cards ----
                st.markdown("""
<style>
.prev-header { margin-bottom: 24px; font-family: 'Verdana', sans-serif; }
.prev-title { font-size: 1.6rem; font-weight: 700; color: #002D57; margin: 0 0 4px 0; }
.prev-subtitle { font-size: 0.85rem; color: #888; margin: 0; }
.prev-cards { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 28px; }
.prev-card {
    flex: 1; min-width: 150px; background: #FFFFFF; border: 1px solid #E5E3DC;
    border-top: 4px solid #002D57; border-radius: 10px; padding: 16px 14px 12px 14px;
    font-family: 'Verdana', sans-serif; box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    transition: transform 0.18s, box-shadow 0.18s;
}
.prev-card:hover { transform: translateY(-3px); box-shadow: 0 6px 14px rgba(0,0,0,0.09); }
.prev-card-title { font-size: 0.65rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.6px; color: #888; margin: 0 0 6px 0; }
.prev-card-value { font-size: 2.2rem; font-weight: 700; line-height: 1; margin: 0 0 4px 0; }
.prev-card-sub { font-size: 0.7rem; color: #555; margin: 0; }
.prev-card-ok  { border-top-color: #2E7D32; }
.prev-card-rsk { border-top-color: #C08A3E; }
.prev-card-crt { border-top-color: #8B1E0F; }
.prev-card-erp { border-top-color: #002D57; }
.prev-card-exp { border-top-color: #666666; }
.prev-card-acp { border-top-color: #00A3E0; }
.prev-section-title { font-size: 0.8rem; font-weight: 700; letter-spacing: 1px; text-transform: uppercase; color: #444; margin: 0 0 16px 0; padding-bottom: 6px; border-bottom: 2px solid #E5E3DC; }
</style>""", unsafe_allow_html=True)

                ref_data = hoje.strftime('%d/%m/%Y')

                st.markdown(f"""
<div class="prev-header">
  <div class="prev-title">Painel de Entregas e Acompanhamento</div>
  <div class="prev-subtitle">Referência: {ref_data} &nbsp;·&nbsp; {total_projetos} projetos no portfólio</div>
</div>
<div class="prev-cards">
  <div class="prev-card">
    <div class="prev-card-title">Total de projetos</div>
    <div class="prev-card-value" style="color:#222;">{total_projetos}</div>
    <div class="prev-card-sub">Geral em execução</div>
  </div>
  <div class="prev-card prev-card-ok">
    <div class="prev-card-title">No prazo</div>
    <div class="prev-card-value" style="color:#2E7D32;">{no_prazo_count}</div>
    <div class="prev-card-sub">{perc_no_prazo:.0f}% da implantação</div>
  </div>
  <div class="prev-card prev-card-rsk">
    <div class="prev-card-title">Em risco</div>
    <div class="prev-card-value" style="color:#C08A3E;">{em_risco_count}</div>
    <div class="prev-card-sub">Desvio 15–45 dias</div>
  </div>
  <div class="prev-card prev-card-crt">
    <div class="prev-card-title">Críticos</div>
    <div class="prev-card-value" style="color:#8B1E0F;">{critico_count}</div>
    <div class="prev-card-sub">Desvio &gt; 45 dias</div>
  </div>
  <div class="prev-card prev-card-erp">
    <div class="prev-card-title">Implantando ERP</div>
    <div class="prev-card-value" style="color:#002D57;">{df_main['_eh_erp'].sum()}</div>
    <div class="prev-card-sub">Fase em andamento</div>
  </div>
  <div class="prev-card prev-card-acp">
    <div class="prev-card-title">Acompanhamento</div>
    <div class="prev-card-value" style="color:#00A3E0;">{num_acomp}</div>
    <div class="prev-card-sub">Pós Go-live</div>
  </div>
  <div class="prev-card prev-card-exp">
    <div class="prev-card-title">Em Expansão</div>
    <div class="prev-card-value" style="color:#666666;">{num_expansao}</div>
    <div class="prev-card-sub">Retornos pós-conclusão</div>
  </div>
</div>""", unsafe_allow_html=True)

                # =====================================================================
                # GRÁFICO 1: DESVIO POR PROJETO (Implantação Regular)
                # =====================================================================
                st.markdown('<div class="prev-section-title">Previsão de Entrega — Desvio por Projeto</div>', unsafe_allow_html=True)

                COR_PRAZO  = '#2E7D32'   # verde escuro
                COR_RISCO  = '#C08A3E'   # âmbar
                COR_CRITICO = '#8B1E0F'  # vermelho
                COR_ERP    = '#6B8DB5'   # azul acinzentado

                def cor_status(row):
                    s = row['Status_Tempo']
                    if s == 'No prazo':        return COR_PRAZO
                    elif s == 'Em risco':      return COR_RISCO
                    elif s == 'Crítico':       return COR_CRITICO
                    else:                      return COR_ERP

                ordem_status = {'Crítico': 0, 'Em risco': 1, 'No prazo': 2, 'Implantando ERP': 3}
                df_chart_prev = df_main.copy()
                df_chart_prev['_ordem'] = df_chart_prev['Status_Tempo'].map(ordem_status)
                df_chart_prev = df_chart_prev.sort_values(['_ordem', 'Desvio_Dias'], ascending=[True, False])

                df_chart_prev['Cor'] = df_chart_prev.apply(cor_status, axis=1)
                df_chart_prev['Desvio_Plot'] = df_chart_prev['Desvio_Dias'].apply(lambda d: max(d, 1) if d > 0 else d)

                df_chart_prev['Label_Projeto'] = df_chart_prev['Summary'].apply(
                    lambda s: (s[:38] + '…') if len(s) > 38 else s
                )

                fig_prev = go.Figure()

                for status, cor in [('Crítico', COR_CRITICO), ('Em risco', COR_RISCO), ('No prazo', COR_PRAZO)]:
                    subset = df_chart_prev[df_chart_prev['Status_Tempo'] == status]
                    if subset.empty:
                        continue
                    fig_prev.add_trace(go.Bar(
                        name=status,
                        y=subset['Label_Projeto'],
                        x=subset['Desvio_Plot'],
                        orientation='h',
                        marker_color=cor,
                        opacity=0.9,
                        hovertemplate=(
                            '<b>%{y}</b><br>'
                            'Desvio: %{customdata[0]:+d} dias<br>'
                            'Data limite: %{customdata[1]}<br>'
                            'Fase: %{customdata[2]}<extra></extra>'
                        ),
                        customdata=list(zip(
                            subset['Desvio_Dias'],
                            subset['Data_Limite'].apply(
                                lambda d: d.strftime('%d/%m/%Y') if pd.notnull(d) else '—'
                            ),
                            subset['Fase']
                        ))
                    ))

                fig_prev.add_vline(x=0, line_dash='solid', line_color='#333333', line_width=1.5)
                fig_prev.add_vline(
                    x=RISCO_DIAS, line_dash='dot', line_color=COR_RISCO, line_width=1,
                    annotation_text=f'+{RISCO_DIAS}d (risco)',
                    annotation_position='top right',
                    annotation_font=dict(size=10, color=COR_RISCO)
                )
                fig_prev.add_vline(
                    x=CRITICO_DIAS, line_dash='dot', line_color=COR_CRITICO, line_width=1,
                    annotation_text=f'+{CRITICO_DIAS}d (crítico)',
                    annotation_position='top right',
                    annotation_font=dict(size=10, color=COR_CRITICO)
                )

                altura_grafico = max(350, len(df_chart_prev[df_chart_prev['Status_Tempo'] != 'Implantando ERP']) * 28 + 100)

                fig_prev.update_layout(
                    barmode='overlay',
                    height=altura_grafico,
                    font_family='Verdana',
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(size=12, color='#333'),
                    margin=dict(t=20, b=60, l=20, r=40),
                    legend=dict(
                        orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0,
                        font=dict(size=11), itemsizing='constant', bgcolor='rgba(255,255,255,0.85)',
                        bordercolor='#E5E3DC', borderwidth=1, title_text=''
                    ),
                    xaxis=dict(
                        title='Desvio em dias (positivo = atrasado)',
                        gridcolor='#F0F0F0', zeroline=True, zerolinecolor='#333', zerolinewidth=1.5,
                        ticksuffix='d', tickfont=dict(size=10)
                    ),
                    yaxis=dict(showgrid=False, tickfont=dict(size=11), autorange='reversed'),
                    hovermode='y unified'
                )

                st.plotly_chart(fig_prev, use_container_width=True)

                # =====================================================================
                # GRÁFICO 2: ACOMPANHAMENTO PÓS GO-LIVE
                # =====================================================================
                if not df_acomp.empty:
                    st.markdown('<div class="prev-section-title" style="margin-top: 30px;">Tempo Total em Acompanhamento (Pós Go-live)</div>', unsafe_allow_html=True)
                    
                    COR_SAUDAVEL = '#00A3E0' # Azul TOTVS
                    COR_EXCEDIDO = '#C08A3E' # Âmbar
                    
                    df_chart_acomp = df_acomp.copy()
                    df_chart_acomp = df_chart_acomp.sort_values('Dias_Projeto', ascending=True)
                    
                    df_chart_acomp['Label_Projeto'] = df_chart_acomp['Summary'].apply(
                        lambda s: (s[:38] + '…') if len(s) > 38 else s
                    )

                    fig_acomp = go.Figure()
                    
                    for status, cor in [('Saudável (≤ 14d)', COR_SAUDAVEL), ('Excedido (> 14d)', COR_EXCEDIDO), ('Excedido / Sem Go-live', COR_EXCEDIDO)]:
                        subset = df_chart_acomp[df_chart_acomp['Status_Acomp'] == status]
                        if subset.empty:
                            continue
                        fig_acomp.add_trace(go.Bar(
                            name=status,
                            y=subset['Label_Projeto'],
                            x=subset['Dias_Projeto'],
                            orientation='h',
                            marker_color=cor,
                            opacity=0.9,
                            hovertemplate=(
                                '<b>%{y}</b><br>'
                                'Tempo Total: %{x} dias<br>'
                                'Dias pós Go-live: %{customdata[0]} dias<br>'
                                'Data Go-live: %{customdata[1]}<extra></extra>'
                            ),
                            customdata=list(zip(
                                subset['Dias_Pos_GL'],
                                subset['Data_Go_Live'].apply(
                                    lambda d: d.strftime('%d/%m/%Y') if pd.notnull(d) else '—'
                                )
                            ))
                        ))
                    
                    altura_acomp = max(250, len(df_chart_acomp) * 28 + 100)
                    fig_acomp.update_layout(
                        barmode='stack',
                        height=altura_acomp,
                        font_family='Verdana',
                        plot_bgcolor='white',
                        paper_bgcolor='white',
                        font=dict(size=12, color='#333'),
                        margin=dict(t=20, b=60, l=20, r=40),
                        legend=dict(
                            orientation='h', yanchor='bottom', y=1.01, xanchor='left', x=0,
                            font=dict(size=11), bgcolor='rgba(255,255,255,0.85)',
                            bordercolor='#E5E3DC', borderwidth=1, title_text=''
                        ),
                        xaxis=dict(
                            title='Tempo Total de Projeto (dias)',
                            gridcolor='#F0F0F0', zeroline=True, zerolinecolor='#333', zerolinewidth=1.5,
                            ticksuffix='d'
                        ),
                        yaxis=dict(showgrid=False, tickfont=dict(size=11), autorange='reversed'),
                        hovermode='y unified'
                    )
                    st.plotly_chart(fig_acomp, use_container_width=True)

                # ---- Tabela detalhada ----
                with st.expander('📋 Ver tabela detalhada de projetos'):
                    df_main_tbl = df_main[['Summary', 'Fase', 'Assignee', 'Data_Limite', 'Data_Go_Live', 'Status_Tempo', 'Desvio_Dias']].copy()
                    df_main_tbl['Status_Acomp'] = None

                    df_acomp_tbl = df_acomp[['Summary', 'Fase', 'Assignee', 'Data_Limite', 'Data_Go_Live', 'Status_Acomp']].copy()
                    df_acomp_tbl['Status_Tempo'] = None
                    df_acomp_tbl['Desvio_Dias'] = 0

                    df_exp_tbl = df_exp[['Summary', 'Fase', 'Assignee', 'Data_Limite', 'Data_Go_Live']].copy()
                    df_exp_tbl['Status_Tempo'] = None
                    df_exp_tbl['Status_Acomp'] = None
                    df_exp_tbl['Desvio_Dias'] = 0

                    df_tabela_prev = pd.concat([df_main_tbl, df_acomp_tbl, df_exp_tbl], ignore_index=True)
                    
                    # Consolidar o Status para a tabela
                    def status_tabela(row):
                        f = _norm_fase(row['Fase'])
                        if f in FASES_ACOMP: return f"Acompanhamento: {row['Status_Acomp']}"
                        if f in FASES_EXP:   return "Expansão"
                        if f == FASE_ERP:    return "Implantando ERP"
                        return row['Status_Tempo']
                        
                    df_tabela_prev['Status'] = df_tabela_prev.apply(status_tabela, axis=1)
                    
                    df_tabela_prev = df_tabela_prev.rename(columns={
                        'Summary':      'Projeto',
                        'Fase':         'Fase Atual',
                        'Assignee':     'Responsável',
                        'Data_Limite':  'Data Limite',
                        'Data_Go_Live': 'Data Go-live',
                        'Desvio_Dias':  'Desvio (dias)'
                    })
                    df_tabela_prev['Data Limite']  = pd.to_datetime(df_tabela_prev['Data Limite']).dt.strftime('%d/%m/%Y').fillna('—')
                    df_tabela_prev['Data Go-live'] = pd.to_datetime(df_tabela_prev['Data Go-live']).dt.strftime('%d/%m/%Y').fillna('—')
                    
                    df_tabela_prev = df_tabela_prev.drop(columns=['Status_Tempo', 'Status_Acomp'])
                    df_tabela_prev = df_tabela_prev.sort_values(['Status', 'Desvio (dias)'], ascending=[True, False])

                    def highlight_status(row):
                        s = str(row['Status'])
                        if 'Crítico' in s or 'Excedido' in s: return ['background-color:#FDE8E8'] * len(row)
                        elif 'Em risco' in s:                 return ['background-color:#FEF3E2'] * len(row)
                        elif 'No prazo' in s or 'Saudável' in s: return ['background-color:#EBF5EB'] * len(row)
                        else:                                 return ['background-color:#EEF2F7'] * len(row)

                    st.dataframe(
                        df_tabela_prev.style.apply(highlight_status, axis=1),
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            'Desvio (dias)': st.column_config.NumberColumn(
                                'Desvio (dias)', format='%+d d'
                            )
                        }
                    )

            elif chart_type == "Relatório de Cancelamento":
                st.subheader("Relatório de Cancelamentos (Últimos 12 Meses)")

                hoje = datetime.now()
                mes_atual = hoje.month
                ano_atual = hoje.year

                mes_inicio = mes_atual - 11
                ano_inicio = ano_atual
                if mes_inicio <= 0:
                    mes_inicio += 12
                    ano_inicio -= 1

                data_limite_12_meses = datetime(ano_inicio, mes_inicio, 1)

                # Filtra cancelados no período — usa df completo (não df_filtered) para ignorar o filtro de fase
                df_cancelados = df[
                    (df['Fase'].str.upper().str.contains("CANCEL", na=False)) &
                    (df['Status_Category'] == 'done') &
                    (df['Status_Cat_Change'] >= data_limite_12_meses)
                ].copy()

                # Métricas de topo
                total_cancelados = len(df_cancelados)
                c1, c2 = st.columns(2)
                c1.metric("Total Projetos (Carteira Atual)", len(df_filtered))
                c2.metric("Cancelamentos (Últimos 12 Meses)", total_cancelados)

                if not df_cancelados.empty:
                    df_cancelados['Mês_Ano'] = df_cancelados['Status_Cat_Change'].dt.strftime('%m/%Y')

                    # Ordem cronológica dos meses
                    periodo_meses = []
                    for i in range(12):
                        m = mes_inicio + i
                        a = ano_inicio
                        if m > 12:
                            m -= 12
                            a += 1
                        periodo_meses.append(f"{m:02d}/{a}")

                    df_agrupado = df_cancelados.groupby('Mês_Ano').size().reindex(
                        periodo_meses, fill_value=0
                    ).reset_index(name='Cancelamentos')
                    df_agrupado.columns = ['Mês/Ano', 'Cancelamentos']

                    fig = px.bar(
                        df_agrupado, x='Mês/Ano', y='Cancelamentos',
                        title='Volume de Cancelamentos por Mês',
                        labels={'Cancelamentos': 'Qtd Cancelamentos', 'Mês/Ano': 'Mês/Ano'},
                        text_auto=True,
                        color_discrete_sequence=[TOTVS_DARK_BLUE]
                    )
                    fig = apply_totvs_style(fig)
                    fig = add_trendline(fig, df_agrupado, 'Mês/Ano', 'Cancelamentos', name="Tendência Cancelamentos", visible=exibir_tendencia)
                    cancel_event = st.plotly_chart(
                        fig, use_container_width=True,
                        on_select="rerun", key="cancel_bar_chart"
                    )

                    # --- Drill-down: detecta clique no gráfico ---
                    drill_cancel_mes = None
                    if cancel_event and cancel_event.selection and cancel_event.selection.get('points'):
                        drill_cancel_mes = cancel_event.selection['points'][0].get('x', None)

                    st.markdown("---")
                    st.subheader("Detalhamento dos Cancelamentos e Complemento de Dados")

                    # --- Filtro de mês ---
                    st.markdown("#### Filtrar por Mês")
                    meses_cancel_disp = ['Todos'] + periodo_meses
                    selected_cancel_mes = st.selectbox(
                        "Selecione o mês para ver os cancelamentos:",
                        options=meses_cancel_disp,
                        index=meses_cancel_disp.index(drill_cancel_mes) if drill_cancel_mes and drill_cancel_mes in meses_cancel_disp else 0,
                        key="cancel_mes_select"
                    )

                    # Aplica filtro
                    df_cancel_filtrado = df_cancelados.copy()
                    if selected_cancel_mes != 'Todos':
                        df_cancel_filtrado = df_cancel_filtrado[df_cancel_filtrado['Mês_Ano'] == selected_cancel_mes]

                    st.markdown(f"**{len(df_cancel_filtrado)} cancelamento(s) — {selected_cancel_mes}**")

                    df_display_cancel = df_cancel_filtrado[
                        ['Key', 'Summary', 'Motivo Cancelamento', 'Status_Cat_Change', 'Assignee', 'MRR_Jira']
                    ].copy()
                    
                    # Carrega dados salvos
                    cancel_data = load_cancelamentos_data()
                    
                    # Prepara colunas editáveis
                    df_display_cancel['Origem'] = df_display_cancel['Key'].apply(lambda k: cancel_data.get(k, {}).get('origem', None))
                    df_display_cancel['Motivo (Manual)'] = df_display_cancel['Key'].apply(lambda k: cancel_data.get(k, {}).get('motivo_manual', ''))
                    
                    def resolve_mrr(row):
                        k = row['Key']
                        # Prioriza o MRR que já existe no Jira se ele for maior que zero
                        if row['MRR_Jira'] > 0:
                            return row['MRR_Jira']
                        # Caso contrário, usa o salvo manualmente se existir
                        if k in cancel_data and 'mrr' in cancel_data[k]:
                            return cancel_data[k]['mrr']
                        return row['MRR_Jira']
                        
                    df_display_cancel['MRR (R$)'] = df_display_cancel.apply(resolve_mrr, axis=1)
                    
                    df_display_cancel['Status_Cat_Change'] = df_display_cancel['Status_Cat_Change'].dt.strftime('%d/%m/%Y')
                    
                    df_display_cancel = df_display_cancel.rename(columns={
                        'Summary': 'Cliente',
                        'Motivo Cancelamento': 'Motivo (Jira)',
                        'Status_Cat_Change': 'Data do Cancelamento',
                        'Assignee': 'Responsável'
                    })
                    
                    # Organiza as colunas de forma lógica
                    col_order = ['Key', 'Cliente', 'Origem', 'MRR (R$)', 'Motivo (Jira)', 'Motivo (Manual)', 'Data do Cancelamento', 'Responsável']
                    df_display_cancel = df_display_cancel[[c for c in col_order if c in df_display_cancel.columns]]
                    
                    st.info("Preencha a 'Origem', 'MRR' e o 'Motivo (Manual)' diretamente na tabela abaixo. Os dados são salvos automaticamente.")
                    
                    edited_cancel = st.data_editor(
                        df_display_cancel,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Key": st.column_config.TextColumn("Key", disabled=True),
                            "Cliente": st.column_config.TextColumn("Cliente", disabled=True),
                            "Origem": st.column_config.SelectboxColumn("Origem", options=["AR", "Inbound"], required=False),
                            "MRR (R$)": st.column_config.NumberColumn("MRR (R$)", format="%.2f", min_value=0.0),
                            "Motivo (Jira)": st.column_config.TextColumn("Motivo (Jira)", width="large", disabled=True),
                            "Motivo (Manual)": st.column_config.TextColumn("Motivo (Manual)", width="large"),
                            "Data do Cancelamento": st.column_config.TextColumn("Data do Cancelamento", disabled=True),
                            "Responsável": st.column_config.TextColumn("Responsável", disabled=True)
                        },
                        disabled=["Key", "Cliente", "Motivo (Jira)", "Data do Cancelamento", "Responsável"],
                        key="cancel_editor"
                    )
                    
                    # Salva alterações
                    has_changes = False
                    for _, row in edited_cancel.iterrows():
                        key = row['Key']
                        origem = row.get('Origem')
                        mrr = row.get('MRR (R$)')
                        motivo = row.get('Motivo (Manual)')
                        
                        if key not in cancel_data:
                            cancel_data[key] = {}
                        
                        # Atualiza se mudou
                        if cancel_data[key].get('origem') != origem or \
                           cancel_data[key].get('mrr') != mrr or \
                           cancel_data[key].get('motivo_manual') != motivo:
                            cancel_data[key]['origem'] = origem
                            cancel_data[key]['mrr'] = mrr
                            cancel_data[key]['motivo_manual'] = motivo
                            has_changes = True
                            
                    if has_changes:
                        save_cancelamentos_data(cancel_data)
                else:
                    st.info("Nenhum cancelamento encontrado nos últimos 12 meses.")

            elif chart_type == "Previsão de Go-live":
                st.subheader("Previsão de Go-live (Próximas 6 Semanas)")
                
                hoje = datetime.now()
                # Define o início da semana atual (segunda-feira)
                inicio_semana = hoje - timedelta(days=hoje.weekday())
                fim_periodo = inicio_semana + timedelta(weeks=6)
                
                # Filtra projetos com Data_Limite no intervalo das próximas 6 semanas
                df_previsao = df_filtered[
                    (df_filtered['Data_Limite'] >= inicio_semana) & 
                    (df_filtered['Data_Limite'] <= fim_periodo)
                ].copy()
                
                if not df_previsao.empty:
                    # Cria rótulo de semana: "Semana de DD/MM" (baseado na segunda-feira da semana da data limite)
                    df_previsao['Segunda_Semana'] = df_previsao['Data_Limite'].apply(
                        lambda x: (x - timedelta(days=x.weekday())).strftime('%d/%m')
                    )
                    df_previsao['Rótulo_Semana'] = "Semana de " + df_previsao['Segunda_Semana']
                    
                    # Ordem das semanas para o gráfico
                    periodo_semanas = []
                    for i in range(6):
                        s = (inicio_semana + timedelta(weeks=i)).strftime('%d/%m')
                        periodo_semanas.append(f"Semana de {s}")

                    # Contagem por semana
                    df_semanas = df_previsao['Rótulo_Semana'].value_counts().reindex(periodo_semanas, fill_value=0).reset_index()
                    df_semanas.columns = ['Semana', 'Qtd Go-lives']

                    fig = px.bar(
                        df_semanas,
                        x='Semana', y='Qtd Go-lives',
                        title='Cronograma de Conclusão Estimado',
                        labels={'Qtd Go-lives': 'Projetos', 'Semana': 'Período'},
                        text_auto=True,
                        color_discrete_sequence=[TOTVS_CYAN]
                    )
                    fig = apply_totvs_style(fig)
                    fig = add_trendline(fig, df_semanas, 'Semana', 'Qtd Go-lives', name="Tendência Cronograma", visible=exibir_tendencia)
                    st.plotly_chart(fig, use_container_width=True)

                    # --- NOVA SEÇÃO: Resultados dos Go-lives da Semana Anterior ---
                    st.markdown("---")
                    
                    # Normalização para garantir comparação correta de datas
                    hoje_zero = hoje.replace(hour=0, minute=0, second=0, microsecond=0)
                    inicio_passada = (hoje_zero - timedelta(days=hoje_zero.weekday() + 7))
                    fim_passada = inicio_passada + timedelta(days=6, hours=23, minutes=59, seconds=59)
                    label_sem_passada = f"{inicio_passada.strftime('%d/%m')} a {fim_passada.strftime('%d/%m/%Y')}"

                    # Filtra os projetos que tinham previsão para a semana passada,
                    # excluindo os que já haviam ido a go-live ou sido concluídos ANTES do início da semana passada.
                    df_prev_passada = df[
                        (df['Data_Limite'] >= inicio_passada) &
                        (df['Data_Limite'] <= fim_passada) &
                        ~((df['Data_Go_Live'] < inicio_passada) | 
                          ((df['Status_Category'] == 'done') & (df['Status_Cat_Change'] < inicio_passada)))
                    ].copy()

                    # Realizados (Projetos cuja Data Go Live efetiva ocorreu DENTRO da semana passada)
                    df_realizados = df[
                        (df['Data_Go_Live'] >= inicio_passada) & 
                        (df['Data_Go_Live'] <= fim_passada)
                    ].copy()

                    # Não Concluídos (Pendentes/Atrasados)
                    df_pendentes = df_prev_passada[
                        ~(((df_prev_passada['Data_Go_Live'] >= inicio_passada) & (df_prev_passada['Data_Go_Live'] <= fim_passada)) |
                          ((df_prev_passada['Status_Category'] == 'done') & (df_prev_passada['Status_Cat_Change'] >= inicio_passada) & (df_prev_passada['Status_Cat_Change'] <= fim_passada)))
                    ].copy()

                    st.subheader(f"Desempenho Go-lives: Semana Anterior ({label_sem_passada})")

                    m1, m2, m3 = st.columns(3)
                    m1.metric("Previstos", len(df_prev_passada))
                    m2.metric("Realizados", len(df_realizados), 
                              delta=f"{len(df_realizados)/len(df_prev_passada)*100:.0f}%" if len(df_prev_passada) > 0 else None)
                    m3.metric("Não Concluídos", len(df_pendentes), 
                              delta=f"-{len(df_pendentes)}" if len(df_pendentes) > 0 else None, 
                              delta_color="inverse")

                    if not df_pendentes.empty:
                        # Gráfico por Tipo de Atraso (dos não concluídos)
                        motivos_pendentes = df_pendentes['Motivo Atraso'].value_counts().reset_index()
                        motivos_pendentes.columns = ['Tipo de Atraso', 'Qtd']

                        fig_pend = px.bar(
                            motivos_pendentes,
                            x='Tipo de Atraso', y='Qtd',
                            title='Motivos dos Go-lives NÃO Concluídos (Semana Anterior)',
                            labels={'Qtd': 'Qtd Projetos', 'Tipo de Atraso': 'Tipo de Atraso'},
                            text_auto=True,
                            color_discrete_sequence=[TOTVS_DARK_BLUE]
                        )
                        fig_pend = apply_totvs_style(fig_pend)
                        st.plotly_chart(fig_pend, use_container_width=True)

                        # Tabela detalhada dos pendentes
                        st.markdown("#### Detalhes dos Projetos Não Concluídos")
                        df_pend_display = df_pendentes[
                            ['Key', 'Summary', 'Data_Limite', 'Fase', 'Motivo Atraso', 'Motivo Detalhado', 'Assignee']
                        ].copy()
                        df_pend_display['Data_Limite'] = df_pend_display['Data_Limite'].dt.strftime('%d/%m/%Y')
                        df_pend_display = df_pend_display.rename(columns={
                            'Summary': 'Cliente',
                            'Data_Limite': 'Previsão Original',
                            'Motivo Atraso': 'Tipo de Atraso',
                            'Motivo Detalhado': 'Motivo do Atraso',
                            'Assignee': 'Responsável'
                        })
                        df_pend_sorted = df_pend_display.sort_values('Previsão Original')
                        df_pend_filtered = filter_columns(df_pend_sorted, "prev_pendentes")
                        col_config = {}
                        if "Motivo do Atraso" in df_pend_filtered.columns:
                            col_config["Motivo do Atraso"] = st.column_config.TextColumn(width="large")
                        st.data_editor(
                            df_pend_filtered,
                            use_container_width=True,
                            hide_index=True,
                            column_config=col_config,
                            disabled=True,
                            key="table_pend_passada"
                        )
                    
                    if not df_realizados.empty:
                        with st.expander("Ver Projetos Concretizados (Realizados)"):
                            df_real_display = df_realizados[['Key', 'Summary', 'Data_Go_Live', 'Fase', 'Assignee']].copy()
                            df_real_display['Data_Go_Live'] = df_real_display['Data_Go_Live'].dt.strftime('%d/%m/%Y')
                            df_real_filtered = filter_columns(df_real_display, "prev_realizados")
                            st.dataframe(df_real_filtered, use_container_width=True, hide_index=True)

                    if df_prev_passada.empty:
                        st.info("Nenhum Go-live estava previsto para a semana anterior.")
                    else:
                        st.success("Todos os clientes com Go-live previsto para esta semana já foram concluídos!")

                    # --- Lista por semana (existente) ---
                    st.markdown("---")
                    st.markdown("#### Lista de Projetos por Semana")
                    df_sem_cols_dummy = pd.DataFrame(columns=['Key', 'Projeto', 'Previsão Go-live', 'Fase', 'Responsável'])
                    df_sem_cols_filtered = filter_columns(df_sem_cols_dummy, "prev_semanal_list")
                    selected_sem_cols = list(df_sem_cols_filtered.columns)

                    for sem in periodo_semanas:
                        df_sem = df_previsao[df_previsao['Rótulo_Semana'] == sem]
                        if not df_sem.empty:
                            with st.expander(f"{sem} ({len(df_sem)} projetos)"):
                                df_sem_display = df_sem[['Key', 'Summary', 'Data_Limite', 'Fase', 'Assignee']].copy()
                                df_sem_display['Data_Limite'] = df_sem_display['Data_Limite'].dt.strftime('%d/%m/%Y')
                                df_sem_display = df_sem_display.rename(columns={
                                    'Summary': 'Projeto', 
                                    'Data_Limite': 'Previsão Go-live',
                                    'Assignee': 'Responsável'
                                })
                                st.dataframe(df_sem_display[selected_sem_cols], use_container_width=True, hide_index=True)
                else:
                    st.info("Nenhuma previsão de Go-live encontrada para as próximas 6 semanas com os filtros atuais.")


            elif chart_type == "Motivo das Pausas":

                df_chart = df_filtered.copy()
                
                def adjust_motivo(row):
                    m = row['Motivo da Pausa']
                    sm = row['Submotivo']
                    is_pausado = any('PAUSADO' in str(l).upper() for l in row.get('LabelList', []))
                    if not is_pausado:
                        return pd.Series(['Em andamento', 'Sem Impedimento'])
                    
                    if m == 'Não informado' or not m or str(m).strip() == '':
                        m = 'Não iniciado'
                    if sm == 'Não informado' or not sm or str(sm).strip() == '':
                        sm = '(Sem submotivo)'
                    return pd.Series([m, sm])
                
                if not df_chart.empty:
                    df_chart[['Motivo da Pausa', 'Submotivo']] = df_chart.apply(adjust_motivo, axis=1)

                df_pausados = df_chart[df_chart['Submotivo'] != 'Sem Impedimento'].copy()
                
                total_projetos = len(df_chart)
                com_impedimentos = len(df_pausados)
                sem_impedimentos = total_projetos - com_impedimentos

                st.markdown(f'''
                <style>
                .kpi-container-pausas {{ display: flex; justify-content: space-around; margin-bottom: 30px; margin-top: 10px; gap: 10px; }}
                .kpi-box-pausas {{ flex: 1; padding: 15px; border: 2px solid #69d2e7; border-radius: 8px; text-align: center; background-color: #FFFFFF; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
                .kpi-title-pausas {{ font-size: 0.85rem; font-weight: 800; color: #69d2e7; text-transform: uppercase; margin-bottom: 5px; }}
                .kpi-value-pausas {{ font-size: 2.2rem; font-weight: normal; color: #333333; }}
                </style>
                <div class="kpi-container-pausas">
                    <div class="kpi-box-pausas">
                        <div class="kpi-title-pausas">TOTAL PROJETOS</div>
                        <div class="kpi-value-pausas">{total_projetos}</div>
                    </div>
                    <div class="kpi-box-pausas">
                        <div class="kpi-title-pausas">SEM IMPEDIMENTOS</div>
                        <div class="kpi-value-pausas">{sem_impedimentos}</div>
                    </div>
                    <div class="kpi-box-pausas">
                        <div class="kpi-title-pausas">COM IMPEDIMENTOS</div>
                        <div class="kpi-value-pausas">{com_impedimentos}</div>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                
                if not df_chart.empty:
                    col_chart, col_highlights = st.columns([6, 4])
                    
                    with col_chart:
                        df_grouped = df_pausados.groupby(['Motivo da Pausa', 'Submotivo']).size().reset_index(name='Quantidade')
                        all_submotivos = sorted(df_grouped['Submotivo'].unique())
                        
                        
                        # Ordenação para a X-axis e Stacks
                        # 1. Obter o total por Motivo da Pausa para ordenar a barra
                        order_motivos = df_grouped.groupby('Motivo da Pausa')['Quantidade'].sum().sort_values(ascending=False).index.tolist()
                        
                        # Garantir que "Em andamento" e "Não iniciado" tenham uma ordem fixa se quisermos, mas descending é melhor
                        # 2. Ordenar o dataframe para que a legenda e os stacks fiquem do maior pro menor
                        df_grouped = df_grouped.sort_values(['Motivo da Pausa', 'Quantidade'], ascending=[True, False])
                        
                        all_submotivos = df_grouped['Submotivo'].unique()
                        
                        custom_colors = {
                            'Sem Impedimento': '#4285F4',
                            'Necessidade de': '#F4A460',
                            'Não Iniciado': '#AB47BC',
                            'Ação de PRJ': '#9CCC65',
                            'Problema de Servi': '#26C6DA',
                            'Comercial': '#FFCA28',
                            '(Sem submotivo)': '#E0E0E0'
                        }
                        
                        fallback_colors = [
                            '#EF5350', '#EC407A', '#7E57C2', '#5C6BC0', '#29B6F6', '#26A69A', 
                            '#66BB6A', '#D4E157', '#FFEE58', '#FFA726', '#FF7043', '#8D6E63', 
                            '#BDBDBD', '#78909C'
                        ]
                        
                        color_map = {}
                        color_idx = 0
                        for sub in all_submotivos:
                            matched = False
                            for k, v in custom_colors.items():
                                if str(sub).startswith(k): 
                                    color_map[sub] = v
                                    matched = True
                                    break
                            if not matched:
                                color_map[sub] = fallback_colors[color_idx % len(fallback_colors)]
                                color_idx += 1

                        fig_stacked = px.bar(
                            df_grouped, x='Motivo da Pausa', y='Quantidade', color='Submotivo',
                            color_discrete_map=color_map,
                            labels={'Quantidade': 'Nº de Projetos', 'Motivo da Pausa': ''},
                            text='Quantidade', barmode='stack',
                        )
                        
                        fig_stacked.update_layout(
                            height=550,
                            margin=dict(t=30, b=80, l=10, r=10),
                            legend=dict(
                                orientation='h', 
                                yanchor='top', 
                                y=-0.15, 
                                xanchor='center', 
                                x=0.5,
                                title='',
                                font=dict(size=16)
                            ),
                            xaxis=dict(showgrid=False, title='', tickfont=dict(weight='bold'), categoryorder='array', categoryarray=order_motivos),
                            yaxis=dict(showgrid=False, showticklabels=False, title=''),
                            plot_bgcolor='white',
                            paper_bgcolor='white'
                        )
                        
                        fig_stacked.update_traces(
                            textposition='inside', 
                            textfont=dict(color='white', weight='bold', size=18),
                            textangle=0,
                            insidetextanchor='middle'
                        )

                        stacked_event = st.plotly_chart(fig_stacked, use_container_width=True, on_select="rerun", key="stacked_chart_full")

                    with col_highlights:
                        # --- Cálculo de Highlights e Lowlights (Semana Anterior: Segunda a Domingo) ---
                        hoje = datetime.now()
                        inicio_semana = (hoje - timedelta(days=hoje.weekday() + 7)).replace(hour=0, minute=0, second=0, microsecond=0)
                        fim_semana = (inicio_semana + timedelta(days=6)).replace(hour=23, minute=59, second=59, microsecond=59)
                        label_semana = f"{inicio_semana.strftime('%d/%m')} A {fim_semana.strftime('%d/%m')}"
                        
                        # Highlights: Go-lives com base na Data Real do Golive (Data_Go_Live)
                        df_high = df[
                            (df['Data_Go_Live'] >= inicio_semana) & 
                            (df['Data_Go_Live'] <= fim_semana) &
                            (~df['Fase'].str.upper().str.contains("CANCEL", na=False))
                        ].copy() if 'Data_Go_Live' in df.columns else pd.DataFrame()
                        
                        # Lowlights: Cancelamentos com base na Categoria do status atualizada
                        df_low = df[
                            (df['Status_Cat_Change'] >= inicio_semana) & 
                            (df['Status_Cat_Change'] <= fim_semana) &
                            (df['Fase'].str.upper().str.contains("CANCEL", na=False))
                        ].copy() if 'Status_Cat_Change' in df.columns else pd.DataFrame()

                        if not df_high.empty:
                            df_high['View_Date'] = df_high['Data_Go_Live']
                        if not df_low.empty:
                            df_low['View_Date'] = df_low['Status_Cat_Change']

                        # Estilização CSS para os boxes modernos
                        st.markdown(f"""
                        <style>
                        .metric-box {{ border: 1px solid {TOTVS_LIGHT_GRAY}; border-radius: 4px; padding: 15px; margin-bottom: 20px; background-color: #FFFFFF; font-family: 'Verdana', sans-serif; }}
                        .box-header {{ font-weight: 700; font-size: 1.1rem; margin-bottom: 5px; color: {TOTVS_DARK_BLUE}; }}
                        .box-subtitle {{ font-size: 0.7rem; color: {TOTVS_GRAY}; text-transform: uppercase; font-weight: 600; margin-bottom: 12px; }}
                        .item-row {{ display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem; padding: 5px 0; border-bottom: 1px solid {TOTVS_LIGHT_GRAY}; }}
                        .badge-high {{ background-color: {TOTVS_CYAN}; color: white; padding: 2px 8px; border-radius: 2px; font-size: 0.7rem; font-weight: 700; }}
                        .badge-low {{ background-color: {TOTVS_DARK_BLUE}; color: white; padding: 2px 8px; border-radius: 2px; font-size: 0.7rem; font-weight: 700; }}
                        </style>
                        """, unsafe_allow_html=True)
                        
                        # Box Highlights
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="box-header">◆ Highlights</div>
                            <div class="box-subtitle">GO-LIVES ({len(df_high)}) — SEMANA {label_semana}</div>
                        """, unsafe_allow_html=True)
                        if not df_high.empty:
                            for _, row in df_high.iterrows():
                                dt_str = row['View_Date'].strftime('%d/%m') if pd.notnull(row.get('View_Date')) else '-'
                                st.markdown(f"""
                                <div class="item-row">
                                    <span>• {row['Summary']}</span>
                                    <span class="badge-high">{dt_str}</span>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='color:#999; font-style:italic; font-size:0.85rem;'>Nenhum go-live esta semana.</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)
                        
                        # Box Lowlights
                        st.markdown(f"""
                        <div class="metric-box">
                            <div class="box-header">▼ Lowlights</div>
                            <div class="box-subtitle">CANCELAMENTOS ({len(df_low)}) — SEMANA {label_semana}</div>
                        """, unsafe_allow_html=True)
                        if not df_low.empty:
                            for _, row in df_low.iterrows():
                                dt_str = row['View_Date'].strftime('%d/%m') if pd.notnull(row.get('View_Date')) else '-'
                                st.markdown(f"""
                                <div class="item-row">
                                    <span>• {row['Summary']}</span>
                                    <span class="badge-low">{dt_str}</span>
                                </div>
                                """, unsafe_allow_html=True)
                        else:
                            st.markdown("<div style='color:#999; font-style:italic; font-size:0.85rem;'>Nenhum cancelamento esta semana.</div>", unsafe_allow_html=True)
                        st.markdown("</div>", unsafe_allow_html=True)

                    drill_motivo = None
                    drill_submotivo = None

                    if stacked_event and stacked_event.selection and stacked_event.selection.get('points'):
                        pt = stacked_event.selection['points'][0]
                        drill_motivo = pt.get('x', None)
                        drill_submotivo = pt.get('legendgroup', None)

                    st.markdown("#### 🔎 Filtros de Drill-Down")
                    motivos_disponiveis = ['Todos'] + sorted(df_chart['Motivo da Pausa'].unique())
                    
                    fc1, fc2 = st.columns(2)
                    with fc1:
                        motivo_sel = st.selectbox(
                            "Filtrar por Motivo:",
                            options=motivos_disponiveis,
                            index=motivos_disponiveis.index(drill_motivo) if drill_motivo and drill_motivo in motivos_disponiveis else 0,
                            key="drill_motivo_select"
                        )
                    with fc2:
                        if motivo_sel != 'Todos':
                            subs = ['Todos'] + sorted(df_chart[df_chart['Motivo da Pausa'] == motivo_sel]['Submotivo'].unique())
                        else:
                            subs = ['Todos'] + sorted(df_chart['Submotivo'].unique())
                        submotivo_sel = st.selectbox(
                            "Filtrar por Submotivo:",
                            options=subs,
                            index=subs.index(drill_submotivo) if drill_submotivo and drill_submotivo in subs else 0,
                            key="drill_sub_select"
                        )

                    df_drill = df_chart.copy()
                    if motivo_sel != 'Todos':
                        df_drill = df_drill[df_drill['Motivo da Pausa'] == motivo_sel]
                    if submotivo_sel != 'Todos':
                        df_drill = df_drill[df_drill['Submotivo'] == submotivo_sel]

                    titulo_tabela = f"📋 {len(df_drill)} projeto(s)"
                    if motivo_sel != 'Todos':
                        titulo_tabela += f" — Motivo: **{motivo_sel}**"
                    if submotivo_sel != 'Todos':
                        titulo_tabela += f" / Submotivo: **{submotivo_sel}**"

                    st.markdown(f"#### {titulo_tabela}")

                    # --- Análise de IA por Comentários ---
                    ai_cache_key = 'ai_pause_analysis'
                    col_btn1, col_btn2 = st.columns([2, 5])
                    with col_btn1:
                        if st.button("🤖 Analisar Comentários com IA", key="btn_ai_analysis",
                                     help="Busca os comentários de cada projeto impedido e usa o Gemini para gerar um entendimento do motivo real da pausa."):
                            if not GEMINI_API_KEY:
                                st.warning("Configure a variável de ambiente `GEMINI_API_KEY` para usar esta função.")
                            else:
                                resultados = analyze_paused_projects(df_pausados)
                                st.session_state[ai_cache_key] = resultados
                                st.success("Análise concluída!")

                    with col_btn2:
                        if ai_cache_key in st.session_state:
                            st.caption(f"✅ Análise IA disponível para {len(st.session_state[ai_cache_key])} projetos. Clique novamente para atualizar.")

                    # Monta tabela com coluna de IA se disponível
                    df_display = df_drill[['Key', 'Summary', 'Fase', 'Motivo da Pausa', 'Submotivo', 'Assignee']].copy()
                    df_display = df_display.rename(columns={'Summary': 'Projeto', 'Assignee': 'Responsável'})

                    if ai_cache_key in st.session_state:
                        ai_map = st.session_state[ai_cache_key]
                        df_display['🤖 Entendimento do Impedimento (IA)'] = df_display['Key'].map(
                            lambda k: ai_map.get(k, '—')
                        )
                        # Mostra qual modelo foi usado para debugar o 404
                        if 'available_gemini_model' in st.session_state:
                            st.caption(f"ℹ️ Modelo utilizado: `{st.session_state['available_gemini_model']}`")

                    df_display_sorted = df_display.sort_values(['Motivo da Pausa', 'Submotivo'])
                    df_display_filtered = filter_columns(df_display_sorted, "pausas_detalhes")
                    col_config = {}
                    if "🤖 Entendimento do Impedimento (IA)" in df_display_filtered.columns:
                        col_config["🤖 Entendimento do Impedimento (IA)"] = st.column_config.TextColumn(
                            "🤖 Entendimento do Impedimento (IA)",
                            width="large",
                            help="Dê um clique duplo para ver o texto completo"
                        )
                    if "Projeto" in df_display_filtered.columns:
                        col_config["Projeto"] = st.column_config.TextColumn(width="medium")
                    if "Motivo da Pausa" in df_display_filtered.columns:
                        col_config["Motivo da Pausa"] = st.column_config.TextColumn(width="large")
                    if "Submotivo" in df_display_filtered.columns:
                        col_config["Submotivo"] = st.column_config.TextColumn(width="large")
                        
                    # Usando data_editor para permitir redimensionar colunas mais facilmente
                    selected_rows = st.data_editor(
                        df_display_filtered,
                        use_container_width=True,
                        hide_index=True,
                        column_config=col_config,
                        disabled=True # Só visualização, mas permite redimensionar
                    )
                    
                    # Detalhes do Projeto (para garantir integridade da leitura)
                    if ai_cache_key in st.session_state:
                        st.markdown("---")
                        st.subheader("📝 Detalhes da Análise")
                        proj_list = df_display['Projeto'].tolist()
                        sel_proj = st.selectbox("Selecione um projeto para ver a análise completa:", proj_list)
                        full_text = df_display[df_display['Projeto'] == sel_proj]['🤖 Entendimento do Impedimento (IA)'].values[0]
                        st.info(f"**{sel_proj}**: \n\n {full_text}")
                        
                        # Botão para salvar no histórico com IA
                        if st.button("💾 Salvar esta Foto com IA no Histórico"):
                            today = datetime.now().strftime("%Y-%m-%d")
                            df_save = df_display.copy()
                            df_save['Data_Snapshot'] = today
                            if os.path.exists(HISTORY_FILE):
                                df_hist = pd.read_csv(HISTORY_FILE)
                                # Se já tem dados do dia, remove antes de concatenar (evita duplicidade manual)
                                df_hist = df_hist[df_hist['Data_Snapshot'] != today]
                                df_save = pd.concat([df_hist, df_save], ignore_index=True)
                            df_save.to_csv(HISTORY_FILE, index=False, encoding='utf-8-sig')
                            st.success("Snapshot atualizado com as análises da IA!")

                else:
                    st.info("Nenhum projeto impedido encontrado com os filtros atuais.")


            elif chart_type == "Relatório PSA/Jira":
                st.subheader("📊 Relatório de Esforço PSA vs Jira")
                st.info("Novos projetos são categorizados como **PSA** por padrão. Marque os que devem ser **Jira**.")

                psa_keys, jira_keys = load_project_lists()
                
                # Carrega keys históricas para distinguir o que é "novo"
                history_keys = set()
                if os.path.exists(HISTORY_FILE):
                    try:
                        df_hist = pd.read_csv(HISTORY_FILE)
                        history_keys = set(df_hist['Key'].unique())
                    except: pass



                # Prepara o DF para edição
                df_relatorio = df_filtered.copy()
                
                def identify_type(key):
                    if key in psa_keys: return True
                    if key in jira_keys: return False
                    # Se está no histórico mas não na lista PSA, é um projeto antigo (Jira)
                    if key in history_keys: return False
                    # Se não está em lugar nenhum, é novo -> PSA
                    return True
                
                df_relatorio['PSA?'] = df_relatorio['Key'].apply(identify_type)
                
                # Calcula esforço para cada linha
                esforcos = df_relatorio['Fase'].apply(get_remaining_effort)
                df_relatorio['Horas Restantes'] = esforcos.apply(lambda x: x[0])
                df_relatorio['% Restante'] = esforcos.apply(lambda x: x[1])
                
                # Reordena colunas para facilitar visualização
                cols = ['PSA?', 'Key', 'Summary', 'Fase', 'Horas Restantes', '% Restante', 'Assignee']
                df_relatorio = df_relatorio[cols]
                
                # Interface de marcação
                edited_df = st.data_editor(
                    df_relatorio,
                    column_config={
                        "PSA?": st.column_config.CheckboxColumn("PSA?", help="Marque se este projeto é PSA"),
                        "Horas Restantes": st.column_config.NumberColumn(format="%.2f h"),
                        "% Restante": st.column_config.NumberColumn(format="%.1f %%"),
                        "Summary": "Projeto"
                    },
                    use_container_width=True,
                    hide_index=True,
                    key="psa_editor"
                )
                
                if st.button("💾 Salvar Classificações", type="primary"):
                    # Atualiza as listas com base no que está na tela
                    current_psa = set(edited_df[edited_df['PSA?'] == True]['Key'])
                    current_jira = set(edited_df[edited_df['PSA?'] == False]['Key'])
                    
                    # Para garantir que não percamos dados de outros espaços/filtros:
                    # Adicionamos os novos e removemos o que foi alterado especificamente no DF atual
                    visible_keys = set(df_relatorio['Key'])
                    
                    # Remove do set global o que está visível para re-inserir atualizado
                    psa_keys = psa_keys - visible_keys
                    jira_keys = jira_keys - visible_keys
                    
                    # Adiciona as novas classificações
                    psa_keys.update(current_psa)
                    jira_keys.update(current_jira)
                    
                    save_project_lists(psa_keys, jira_keys)
                    st.success("Classificações de projetos atualizadas!")
                    st.rerun()

                st.markdown("---")
                
                # Separação dos Blocos
                df_psa_final = edited_df[edited_df['PSA?'] == True]
                df_jira_final = edited_df[edited_df['PSA?'] == False]
                
                col1, col2 = st.columns(2)
                
                # Filtro de colunas para os blocos PSA/Jira
                df_block_cols_dummy = pd.DataFrame(columns=['Key', 'Summary', 'Fase', 'Horas Restantes', '% Restante'])
                df_block_cols_filtered = filter_columns(df_block_cols_dummy, "psa_jira_blocks")
                selected_block_cols = list(df_block_cols_filtered.columns)

                with col1:
                    st.markdown("### Bloco PSA")
                    total_h_psa = df_psa_final['Horas Restantes'].sum()
                    max_h_psa = len(df_psa_final) * TOTAL_HOURS
                    perc_remanescente_psa = (total_h_psa / max_h_psa * 100) if max_h_psa > 0 else 0
                    
                    c1a, c1b, c1c = st.columns(3)
                    c1a.metric("Projetos PSA", len(df_psa_final))
                    c1b.metric("Esforço Total", f"{total_h_psa:.2f} h")
                    c1c.metric("% Restante", f"{perc_remanescente_psa:.1f}%")
                    
                    st.dataframe(
                        df_psa_final[selected_block_cols], 
                        use_container_width=True, hide_index=True
                    )

                with col2:
                    st.markdown("### Bloco Jira")
                    total_h_jira = df_jira_final['Horas Restantes'].sum()
                    max_h_jira = len(df_jira_final) * TOTAL_HOURS
                    perc_remanescente_jira = (total_h_jira / max_h_jira * 100) if max_h_jira > 0 else 0
                    
                    c2a, c2b, c2c = st.columns(3)
                    c2a.metric("Projetos Jira", len(df_jira_final))
                    c2b.metric("Esforço Total", f"{total_h_jira:.2f} h")
                    c2c.metric("% Restante", f"{perc_remanescente_jira:.1f}%")
                    
                    st.dataframe(
                        df_jira_final[selected_block_cols], 
                        use_container_width=True, hide_index=True
                    )
                
                # Resumo Geral
                # Resumo Geral
                st.markdown("### Resumo de Esforço Restante por Tipo")
                resumo_data = pd.DataFrame({
                    "Tipo": ["PSA", "Jira"],
                    "Esforço (Horas)": [total_h_psa, total_h_jira],
                    "Qtd Projetos": [len(df_psa_final), len(df_jira_final)]
                })
                fig_resumo = px.bar(
                    resumo_data, x="Tipo", y="Esforço (Horas)", 
                    color="Tipo", text_auto=True,
                    color_discrete_map={"PSA": TOTVS_DARK_BLUE, "Jira": TOTVS_CYAN}
                )
                st.plotly_chart(fig_resumo, use_container_width=True)

                # --- Nova Seção: Capacidade e Previsão ---
                st.markdown("### Capacidade e Previsão de Entrega")
                # Parâmetros solicitados
                CONSULTORES_PSA = 2
                CONSULTORES_JIRA = 4
                HORAS_NOMINAIS = 160
                HORAS_UTEIS = 140 # Consideradas para o cálculo

                st.markdown(f"Considerando **{CONSULTORES_JIRA}** analistas para Jira e **{CONSULTORES_PSA}** para PSA.")
                
                cap_psa = CONSULTORES_PSA * HORAS_UTEIS
                cap_jira = CONSULTORES_JIRA * HORAS_UTEIS
                
                # Cálculo de meses (esforço total / capacidade mensal)
                meses_psa = total_h_psa / cap_psa if cap_psa > 0 else 0
                meses_jira = total_h_jira / cap_jira if cap_jira > 0 else 0
                
                col_cap1, col_cap2 = st.columns(2)
                
                with col_cap1:
                    st.metric("Meses para zerar PSA", f"{meses_psa:.1f} meses", 
                              help=f"Cálculo: {total_h_psa:.0f}h / ({CONSULTORES_PSA} cons. x {HORAS_UTEIS}h)")
                    st.caption(f"Capacidade: **{cap_psa}h / mês** ({CONSULTORES_PSA} analistas)")
                    
                with col_cap2:
                    st.metric("Meses para zerar Jira", f"{meses_jira:.1f} meses",
                              help=f"Cálculo: {total_h_jira:.0f}h / ({CONSULTORES_JIRA} cons. x {HORAS_UTEIS}h)")
                    st.caption(f"Capacidade: **{cap_jira}h / mês** ({CONSULTORES_JIRA} analistas)")
                
                # Gráfico Comparativo
                df_cap = pd.DataFrame([
                    {"Tipo": "PSA", "Métrica": "Backlog Total (h)", "Valor": total_h_psa},
                    {"Tipo": "PSA", "Métrica": "Capacidade Mensal (h)", "Valor": cap_psa},
                    {"Tipo": "Jira", "Métrica": "Backlog Total (h)", "Valor": total_h_jira},
                    {"Tipo": "Jira", "Métrica": "Capacidade Mensal (h)", "Valor": cap_jira}
                ])
                
                fig_cap = px.bar(
                    df_cap, x="Tipo", y="Valor", color="Métrica",
                    barmode="group", text_auto=".0f",
                    title="Backlog Acumulado vs. Capacidade Mensal de Entrega",
                    color_discrete_map={"Backlog Total (h)": TOTVS_DARK_BLUE, "Capacidade Mensal (h)": TOTVS_CYAN}
                )
                st.plotly_chart(apply_totvs_style(fig_cap), use_container_width=True)
                
                st.info(f"💡 **Premissa de Cálculo:** Embora a jornada nominal seja de {HORAS_NOMINAIS}h/mês, estamos considerando apenas **{HORAS_UTEIS}h úteis** por analista para a resolução deste backlog.")

                # --- Nova Seção: Evolução Mensal PSA vs Jira (Desde Mar/2026) ---
                st.markdown("---")
                st.markdown("### 📈 Evolução Mensal PSA vs Jira (Desde Mar/2026)")
                
                start_date_evo = datetime(2026, 3, 1)
                hoje_evo = datetime.now()
                
                meses_evo_lista = []
                temp_date_evo = start_date_evo
                while temp_date_evo <= hoje_evo:
                    meses_evo_lista.append((temp_date_evo.year, temp_date_evo.month))
                    if temp_date_evo.month == 12:
                        temp_date_evo = datetime(temp_date_evo.year + 1, 1, 1)
                    else:
                        temp_date_evo = datetime(temp_date_evo.year, temp_date_evo.month + 1, 1)
                
                evolution_psa_jira = []
                import calendar
                
                for ano, mes in meses_evo_lista:
                    if ano == hoje_evo.year and mes == hoje_evo.month:
                        fim_mes = hoje_evo
                    else:
                        ultimo_dia = calendar.monthrange(ano, mes)[1]
                        fim_mes = datetime(ano, mes, ultimo_dia, 23, 59, 59)
                        
                    def is_active_at_month_evo(row, end_month):
                        issued_at = row['Criado_Em']
                        if pd.isnull(issued_at) or issued_at > end_month:
                            return False
                        
                        if row['Status_Category'] == 'done':
                            resolved_at = row['Status_Cat_Change']
                            if pd.notnull(resolved_at) and resolved_at <= end_month:
                                return False
                        return True
                        
                    df_active = df[df.apply(is_active_at_month_evo, axis=1, end_month=fim_mes)].copy()
                    df_active['PSA?'] = df_active['Key'].apply(identify_type)
                    
                    psa_count = df_active[df_active['PSA?'] == True].shape[0]
                    jira_count = df_active[df_active['PSA?'] == False].shape[0]
                    
                    evolution_psa_jira.append({
                        "Mês": f"{mes:02d}/{ano}",
                        "PSA": psa_count,
                        "Jira": jira_count,
                        "Total": psa_count + jira_count
                    })
                
                df_evo_psa_jira = pd.DataFrame(evolution_psa_jira)
                
                if not df_evo_psa_jira.empty and len(df_evo_psa_jira) >= 1:
                    jira_start = df_evo_psa_jira.iloc[0]["Jira"]
                    jira_current = df_evo_psa_jira.iloc[-1]["Jira"]
                    
                    if jira_start > 0:
                        reducao_jira = ((jira_start - jira_current) / jira_start) * 100
                    else:
                        reducao_jira = 0.0
                        
                    col_met1, col_met2, col_met3 = st.columns(3)
                    with col_met1:
                        st.metric(
                            label="Redução de Projetos Jira",
                            value=f"{reducao_jira:.1f}%",
                            delta=f"-{jira_start - jira_current} projetos (de {jira_start} para {jira_current})",
                            delta_color="normal"
                        )
                    with col_met2:
                        st.metric(
                            label="Projetos PSA Atuais",
                            value=f"{df_evo_psa_jira.iloc[-1]['PSA']}",
                            delta=f"{df_evo_psa_jira.iloc[-1]['PSA'] - df_evo_psa_jira.iloc[0]['PSA']} vs Mar/2026"
                        )
                    with col_met3:
                        st.metric(
                            label="Total de Projetos Ativos",
                            value=f"{df_evo_psa_jira.iloc[-1]['Total']}",
                            delta=f"{df_evo_psa_jira.iloc[-1]['Total'] - df_evo_psa_jira.iloc[0]['Total']} vs Mar/2026"
                        )
                        
                    df_melted = df_evo_psa_jira.melt(
                        id_vars=["Mês"],
                        value_vars=["PSA", "Jira"],
                        var_name="Tipo",
                        value_name="Quantidade"
                    )
                    
                    fig_evo = px.line(
                        df_melted,
                        x="Mês",
                        y="Quantidade",
                        color="Tipo",
                        markers=True,
                        title="Evolução Mensal do Número de Projetos Ativos",
                        color_discrete_map={"PSA": TOTVS_DARK_BLUE, "Jira": TOTVS_CYAN}
                    )
                    
                    fig_evo = apply_totvs_style(fig_evo)
                    fig_evo.update_traces(line=dict(width=3), marker=dict(size=8))
                    st.plotly_chart(fig_evo, use_container_width=True)
                    
                    with st.expander("📄 Ver tabela detalhada da evolução mensal"):
                        st.dataframe(
                            df_evo_psa_jira.rename(columns={
                                "Mês": "Mês de Referência",
                                "PSA": "Qtd Projetos PSA",
                                "Jira": "Qtd Projetos Jira",
                                "Total": "Total Projetos Ativos"
                            }),
                            use_container_width=True,
                            hide_index=True
                        )

            elif chart_type == "Tickets Freshdesk":
                st.subheader("Freshdesk: Suporte em Implantação")
                
                # Funções de normalização e verificação de fase final
                def _is_fase_final(fase):
                    if not fase: return False
                    fase_norm = unicodedata.normalize('NFD', str(fase)).encode('ascii', 'ignore').decode('utf-8').lower().strip()
                    return any(term in fase_norm for term in ['conclu', 'cancel', 'done', 'finali'])

                def norm_client(txt):
                    if not txt: return ""
                    t = "".join(c for c in unicodedata.normalize('NFD', str(txt))
                                  if unicodedata.category(c) != 'Mn').lower().strip()
                    # Remove tudo após o primeiro '-' ou '(' para agrupar projetos do mesmo cliente
                    t = t.split('-')[0].split('(')[0].strip()
                    # Limpeza adicional de termos comuns se sobrarem
                    for term in ['totvs', 'winthor', 'protheus', 'lexos', 'moda']:
                        t = t.replace(term, '').strip()
                    return t

                pos_phases = ["Expansão", "Acompanhamento"]
                pos_phases_norm = [unicodedata.normalize('NFD', p).encode('ascii', 'ignore').decode('utf-8').lower().strip() for p in pos_phases]

                # Identificar Clientes Finalizados (que possuem pelo menos um projeto Concluído/Cancelado no Jira)
                # Usamos o DF original (df) para garantir que pegamos o histórico completo, não só o filtrado
                finalized_clients = set()
                for _, row in df.iterrows():
                    if _is_fase_final(row['Fase']):
                        finalized_clients.add(norm_client(row['Summary']))

                # 1. Obter clientes do Jira (conforme filtros atuais da sidebar)
                # Lógica: 
                # - Excluir projetos que já estão em fase final.
                # - Se o cliente já tem ALGUM projeto finalizado, ele só deve aparecer se o projeto atual for Pós-Golive (Expansão/Acompanhamento).
                #   Isso remove projetos "esquecidos" em fases de implantação para clientes que já deram Go Live.
                
                def _should_include_in_report(row):
                    fase = row['Fase']
                    if _is_fase_final(fase):
                        return False
                    
                    client_norm = norm_client(row['Summary'])
                    fase_norm = unicodedata.normalize('NFD', str(fase)).encode('ascii', 'ignore').decode('utf-8').lower().strip()
                    
                    if client_norm in finalized_clients:
                        # Se o cliente já é "Finalizado", só incluímos se for fase de Pós-Golive
                        return fase_norm in pos_phases_norm
                    
                    return True

                jira_clients_df = df_filtered[df_filtered.apply(_should_include_in_report, axis=1)][['Summary', 'Fase']].copy()
                jira_clients_df.columns = ['Cliente', 'Fase']
                
                with st.spinner("Sincronizando empresas do Freshdesk..."):
                    all_companies = get_freshdesk_companies()
                
                if all_companies:
                    # Carrega arquivo de mapeamento manual (De/Para)
                    MANUAL_MAP_FILE = "freshdesk_mapping.json"
                    manual_map = {}
                    if os.path.exists(MANUAL_MAP_FILE):
                        try:
                            with open(MANUAL_MAP_FILE, 'r', encoding='utf-8') as f:
                                manual_map = json.load(f)
                        except:
                            manual_map = {}

                    # 2. Associação Jira -> Freshdesk
                    def norm(txt):
                        return norm_client(txt)
                    
                    fd_map = {norm(c['name']): c for c in all_companies}
                    
                    matched = []
                    unmatched_jira = []

                    for _, row in jira_clients_df.iterrows():
                        jira_nome = str(row['Cliente'])
                        n = norm(jira_nome)

                        if jira_nome in manual_map:
                            # Tenta pelo mapeamento manual primeiro
                            fd_id = manual_map[jira_nome]
                            # Acha o nome na lista do FD
                            fd_name = next((c['name'] for c in all_companies if str(c['id']) == str(fd_id)), "Desconhecido")
                            matched.append({
                                'JiraName': jira_nome,
                                'Fase': row['Fase'],
                                'FDId': fd_id,
                                'FDName': fd_name
                            })
                        elif n in fd_map:
                            # Acha pelo nome limpo
                            matched.append({
                                'JiraName': jira_nome,
                                'Fase': row['Fase'],
                                'FDId': fd_map[n]['id'],
                                'FDName': fd_map[n]['name']
                            })
                        else:
                            unmatched_jira.append(jira_nome)
                    
                    df_mapping = pd.DataFrame(matched)
                    
                    # Interface de Configuração - De/Para
                    if unmatched_jira:
                        with st.expander("Configurar Mapeamento (De/Para) de Clientes"):
                            st.write("Estes projetos do Jira não foram encontrados automaticamente no Freshdesk. Associe-os abaixo:")
                            
                            fd_options = [{"label": c['name'], "value": c['id']} for c in all_companies]
                            fd_options_sorted = sorted(fd_options, key=lambda x: x["label"])
                            
                            c1, c2 = st.columns([1, 1])
                            with c1:
                                proj_to_map = st.selectbox("Selecione o Projeto no Jira", [""] + sorted(list(set(unmatched_jira))))
                            with c2:
                                emp_options = [""] + [f"{x['label']} (ID: {x['value']})" for x in fd_options_sorted]
                                fd_to_map = st.selectbox("Selecione a Empresa no Freshdesk", emp_options)
                            
                            if st.button("Salvar Associação"):
                                if proj_to_map and fd_to_map:
                                    fd_id_str = fd_to_map.split("(ID: ")[-1].replace(")", "")
                                    manual_map[proj_to_map] = int(fd_id_str)
                                    with open(MANUAL_MAP_FILE, 'w', encoding='utf-8') as f:
                                        json.dump(manual_map, f, ensure_ascii=False, indent=4)
                                    st.success(f"Associação salva! {proj_to_map} -> {fd_to_map.split(' (ID')[0]}")
                                    st.rerun()
                                else:
                                    st.warning("Selecione ambos os campos para salvar.")

                    st.write("---")

                    if df_mapping.empty:
                        st.warning("Nenhum cliente do filtro atual foi encontrado no Freshdesk.")
                    else:
                        st.success(f"{len(df_mapping)} clientes vinculados com sucesso (Total no filtro atual: {len(jira_clients_df)}).")
                        
                        # Filtro de Seleção
                        options = ["Todos os vinculados"] + sorted(df_mapping['JiraName'].tolist())
                        sel_client = st.selectbox("Selecione o Cliente para detalhamento:", options)
                        
                        if st.button("Carregar Tickets do Freshdesk"):
                            tickets_list = []
                            
                            with st.spinner("Buscando tickets no Freshdesk..."):
                                if sel_client == "Todos os vinculados":
                                    prog = st.progress(0)
                                    for i, row in df_mapping.iterrows():
                                        tks = get_freshdesk_tickets(company_id=row['FDId'])
                                        for t in tks:
                                            t['jira_fase'] = row['Fase']
                                            t['jira_client'] = row['JiraName']
                                        tickets_list.extend(tks)
                                        prog.progress((i+1)/len(df_mapping))
                                    prog.empty()
                                else:
                                    row = df_mapping[df_mapping['JiraName'] == sel_client].iloc[0]
                                    tks = get_freshdesk_tickets(company_id=row['FDId'])
                                    for t in tks:
                                        t['jira_fase'] = row['Fase']
                                        t['jira_client'] = row['JiraName']
                                    tickets_list.extend(tks)
                            
                            if tickets_list:
                                df_tks = pd.DataFrame([{
                                    "ID": x['id'],
                                    "Assunto": x['subject'],
                                    "Cliente": x['jira_client'],
                                    "Fase": x['jira_fase'],
                                    "Data": x['created_at'][:10],
                                    "Status": x.get('status', 0)
                                } for x in tickets_list])
                                
                                # Filtrar apenas os tickets com status ABERTO (Freshdesk Status 2 = Open)
                                df_tks = df_tks[df_tks['Status'] == 2]

                                
                                # Exportar contagem para o expander lá embaixo (usando strip para evitar problemas)
                                st.session_state['last_fd_counts'] = df_tks.groupby('Cliente').size().to_dict()
                                st.session_state['last_fd_tickets_view'] = df_tks # Guarda o DF para renderizar mesmo sem o botão clicado

                            else:
                                st.info("Nenhum ticket encontrado nos últimos 30 dias para a seleção.")
                                st.session_state['last_fd_tickets_view'] = None
                                st.session_state['last_fd_counts'] = {}

                        # Renderizar dados salvos se existirem (para persistir após interação com a tabela)
                        if not st.session_state.get('last_fd_tickets_view') is None and not st.button("Carregar Tickets do Freshdesk", key="hidden_btn_check"):
                            # Repetir a lógica de renderização se já houver dados
                            df_tks = st.session_state['last_fd_tickets_view']
                            
                            # Normalização de fase
                            def normalize_phase(p):
                                return unicodedata.normalize('NFD', str(p)).encode('ascii', 'ignore').decode('utf-8').lower().strip()

                            def render_dashboard(df_tickets, title, valid_phases):
                                # Reutiliza a normalização global para consistência
                                valid_norm = [normalize_phase(p) for p in valid_phases]

                                # Validação cruzada: apenas clientes que AINDA estão ativos no Jira
                                # (jira_clients_df já exclui projetos Concluídos/Cancelados)
                                # Isso garante que dados em cache (session_state) não exibam projetos finalizados
                                clientes_ativos = set(jira_clients_df['Cliente'].str.strip().str.lower())
                                df_tickets = df_tickets[df_tickets['Cliente'].str.strip().str.lower().isin(clientes_ativos)]

                                df_sec = df_tickets[df_tickets['Fase'].apply(lambda x: normalize_phase(x) in valid_norm)]
                                
                                st.markdown(f"### {title}")
                                
                                if df_sec.empty:
                                    st.info(f"Nenhum ticket encontrado para as fases de {title}.")
                                    return
                                    
                                total_tickets = len(df_sec)
                                
                                # Obter todos os projetos no Jira que pertencem a essa fase para cálculo exato
                                total_projetos_fase = len(jira_clients_df[jira_clients_df['Fase'].apply(lambda x: normalize_phase(x) in valid_norm)])
                                
                                # Clientes que abriram tickets (foco do relatório)
                                clientes_com_ticket = df_sec['Cliente'].nunique()
                                
                                # Calcular projetos sem ticket
                                projetos_sem_ticket = max(0, total_projetos_fase - clientes_com_ticket)
                                
                                media = total_tickets / clientes_com_ticket if clientes_com_ticket > 0 else 0
                                
                                # Layout escuro estilo dashboard da imagem enviada
                                c1, c2, c3, c4 = st.columns(4)
                                # Usando HTML para cards (Padrão TOTVS)
                                c1.markdown(f"<div style='background-color:#FFFFFF; padding:20px; border-radius:4px; border: 1px solid {TOTVS_LIGHT_GRAY}; border-left: 5px solid {TOTVS_DARK_BLUE};'><h6 style='color:{TOTVS_GRAY}; margin:0; font-family:Verdana; font-size:0.7rem;'>TOTAL DE TICKETS</h6><h2 style='color:{TOTVS_DARK_BLUE}; margin:0; font-family:Verdana;'>{total_tickets}</h2></div>", unsafe_allow_html=True)
                                c2.markdown(f"<div style='background-color:#FFFFFF; padding:20px; border-radius:4px; border: 1px solid {TOTVS_LIGHT_GRAY}; border-left: 5px solid {TOTVS_CYAN};'><h6 style='color:{TOTVS_GRAY}; margin:0; font-family:Verdana; font-size:0.7rem;'>CLIENTES COM TICKET</h6><h2 style='color:{TOTVS_DARK_BLUE}; margin:0; font-family:Verdana;'>{clientes_com_ticket}</h2></div>", unsafe_allow_html=True)
                                c3.markdown(f"<div style='background-color:#FFFFFF; padding:20px; border-radius:4px; border: 1px solid {TOTVS_LIGHT_GRAY}; border-left: 5px solid {TOTVS_GRAY};'><h6 style='color:{TOTVS_GRAY}; margin:0; font-family:Verdana; font-size:0.7rem;'>PROJETOS S/ TICKET</h6><h2 style='color:{TOTVS_DARK_BLUE}; margin:0; font-family:Verdana;'>{projetos_sem_ticket}</h2></div>", unsafe_allow_html=True)
                                c4.markdown(f"<div style='background-color:#FFFFFF; padding:20px; border-radius:4px; border: 1px solid {TOTVS_LIGHT_GRAY}; border-left: 5px solid #B4D4EE;'><h6 style='color:{TOTVS_GRAY}; margin:0; font-family:Verdana; font-size:0.7rem;'>MÉDIA POR CLIENTE</h6><h2 style='color:{TOTVS_DARK_BLUE}; margin:0; font-family:Verdana;'>{media:.1f}</h2></div>", unsafe_allow_html=True)
                                st.write("")
                                
                                # Top 10 clients chart
                                top10 = df_sec.groupby('Cliente').size().reset_index(name='Tickets').sort_values('Tickets', ascending=False).head(10)
                                
                                fig = px.bar(
                                    top10, x='Cliente', y='Tickets',
                                    title="Top 10 Clientes por Tickets",
                                    text_auto=True,
                                    color_discrete_sequence=[TOTVS_CYAN]
                                )
                                fig = apply_totvs_style(fig)
                                fig.update_layout(
                                    xaxis=dict(showgrid=False, title=""),
                                )
                                st.plotly_chart(fig, use_container_width=True)
                                
                                # Expansível com o detalhe
                                with st.expander(f"Ver lista de tickets ({title})"):
                                    df_sec_display = df_sec[['ID', 'Assunto', 'Cliente', 'Fase', 'Data']].copy()
                                    df_sec_filtered = filter_columns(df_sec_display, f"fd_tks_{title}")
                                    st.dataframe(df_sec_filtered, use_container_width=True, hide_index=True)

                            # Fases definidas no início do bloco Freshdesk
                            # pre_phases = implantação ativa
                            pre_phases = ["Iniciado", "Implantando ERP", "Aderência", "Novos projetos", "Golive", "Go live", "Go-live"]
                            # pos_phases já definida acima

                            # Define um fundo escuro só para a visualização dos dashboards se desejado
                            st.markdown("""
                            <style>
                            div[data-testid="stVerticalBlock"] > div { margin-bottom: 0.5rem; }
                            </style>
                            """, unsafe_allow_html=True)
                            
                            # Renderizar painéis pre e pos Golive
                            render_dashboard(df_tks, "Serviços - Pré Golive", pre_phases)
                            st.markdown("---")
                            render_dashboard(df_tks, "Serviços - Pós Golive", pos_phases)
                            
                            # Renderizar "Outras Fases" pra não perder projetos da carteira que caírem fora da esteira regular
                            todas_fases_mapeadas = [normalize_phase(x) for x in pre_phases + pos_phases]
                            outros_projetos = jira_clients_df[~jira_clients_df['Fase'].apply(lambda x: normalize_phase(x) in todas_fases_mapeadas)]
                            df_tks_outros = df_tks[~df_tks['Fase'].apply(lambda x: normalize_phase(x) in todas_fases_mapeadas)]
                            
                            if not outros_projetos.empty:
                                st.markdown("---")
                                render_dashboard(df_tks, "Serviços - Outras Fases", list(outros_projetos['Fase'].unique()))
                else:
                    st.error("Erro ao conectar com Freshdesk.")

            elif chart_type == "Impedimentos por Mês":
                import calendar as _cal

                # ── Paleta de cores por motivo de pausa ───────────────────────
                CORES_MOTIVO = {
                    "Sem Impedimento":      "#27AE60",
                    "Implantação ERP":      "#2980B9",
                    "Sem Retorno Cliente":  "#E67E22",
                    "Validação Cliente":    "#E74C3C",
                    "Despriorizado Cliente":"#8E44AD",
                    "Outros":               "#95A5A6",
                }

                # ── KPI Cards ─────────────────────────────────────────────────
                hoje_imp = datetime.now()

                total_geral     = len(df)
                cancelados_tot  = df[df['Fase'].str.upper().str.contains("CANCEL", na=False)].shape[0]
                concluidos_tot  = df[
                    (df['Status_Category'] == 'done') &
                    (~df['Fase'].str.upper().str.contains("CANCEL", na=False))
                ].shape[0]
                ativos_tot      = df[df['Status_Category'] != 'done'].shape[0]
                impedidos_tot   = df[
                    (df['Status_Category'] != 'done') & (df['Pausado'] == True)
                ].shape[0]
                em_andamento_tot = df[
                    (df['Status_Category'] != 'done') & (df['Pausado'] == False)
                ].shape[0]
                implantacao_tot = df[
                    (df['Status_Category'] != 'done') &
                    (df['Fase'].str.lower().str.contains("implanta", na=False)) &
                    (~df['Fase'].str.upper().str.contains("CANCEL", na=False))
                ].shape[0]

                kpi_html = f"""
<style>
.imp-kpi-row {{
    display: flex;
    gap: 14px;
    flex-wrap: wrap;
    margin-bottom: 28px;
    font-family: 'Segoe UI', Verdana, sans-serif;
}}
.imp-kpi-card {{
    flex: 1;
    min-width: 120px;
    background: #fff;
    border: 1px solid #e0e4ec;
    border-radius: 10px;
    padding: 18px 16px 14px 16px;
    display: flex;
    flex-direction: column;
    gap: 4px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    transition: box-shadow 0.2s;
}}
.imp-kpi-card:hover {{
    box-shadow: 0 4px 12px rgba(0,0,0,0.10);
}}
.imp-kpi-label {{
    font-size: 0.72rem;
    font-weight: 700;
    color: #444;
    margin: 0;
}}
.imp-kpi-value {{
    font-size: 2.6rem;
    font-weight: 700;
    color: #111;
    line-height: 1.1;
    margin: 4px 0 2px 0;
}}
.imp-kpi-sub {{
    font-size: 0.70rem;
    color: #888;
    margin: 0;
}}
</style>
<div class="imp-kpi-row">
  <div class="imp-kpi-card">
    <span class="imp-kpi-label">Implantação</span>
    <span class="imp-kpi-value">{implantacao_tot}</span>
    <span class="imp-kpi-sub">Implantações</span>
  </div>
  <div class="imp-kpi-card">
    <span class="imp-kpi-label">Em Andamento</span>
    <span class="imp-kpi-value">{em_andamento_tot}</span>
    <span class="imp-kpi-sub">Implantações</span>
  </div>
  <div class="imp-kpi-card">
    <span class="imp-kpi-label">Impedidos</span>
    <span class="imp-kpi-value">{impedidos_tot}</span>
    <span class="imp-kpi-sub">Implantações</span>
  </div>
  <div class="imp-kpi-card">
    <span class="imp-kpi-label">Concluído</span>
    <span class="imp-kpi-value">{concluidos_tot}</span>
    <span class="imp-kpi-sub">Implantações</span>
  </div>
  <div class="imp-kpi-card">
    <span class="imp-kpi-label">Total</span>
    <span class="imp-kpi-value">{total_geral}</span>
    <span class="imp-kpi-sub">Implantações</span>
  </div>
  <div class="imp-kpi-card">
    <span class="imp-kpi-label">Cancelados</span>
    <span class="imp-kpi-value">{cancelados_tot}</span>
    <span class="imp-kpi-sub">Implantações</span>
  </div>
</div>
"""
                st.markdown(kpi_html, unsafe_allow_html=True)

                # ── Controles do gráfico ──────────────────────────────────────
                col_ctrl1, col_ctrl2 = st.columns([2, 2])
                with col_ctrl1:
                    janela_meses = st.select_slider(
                        "Janela histórica (meses):",
                        options=[6, 12, 18, 24],
                        value=24,
                        key="imp_janela"
                    )
                with col_ctrl2:
                    modo_grafico = st.radio(
                        "Modo do gráfico:",
                        options=["Empilhado", "Agrupado"],
                        horizontal=True,
                        key="imp_modo"
                    )
                barmode_val = "stack" if modo_grafico == "Empilhado" else "group"

                # ── Pool de dados: ativos hoje + finalizados nos últimos N meses ─
                ref_inicio_pool = hoje_imp - timedelta(days=janela_meses * 31)
                df_done_recentes = df[
                    (df['Status_Category'] == 'done') &
                    (df['Status_Cat_Change'] >= ref_inicio_pool)
                ].copy()
                df_pool = pd.concat([df_filtered, df_done_recentes]).drop_duplicates(subset='Key')

                # ── Série histórica mês a mês ──────────────────────────────────
                mes_atual_i  = hoje_imp.month
                ano_atual_i  = hoje_imp.year
                meses_imp    = []
                for i in range(janela_meses - 1, -1, -1):
                    m = mes_atual_i - i
                    a = ano_atual_i
                    while m <= 0:
                        m += 12
                        a -= 1
                    meses_imp.append((a, m))

                imp_records = []
                meses_pt_imp = {
                    1:"jan",2:"fev",3:"mar",4:"abr",5:"mai",6:"jun",
                    7:"jul",8:"ago",9:"set",10:"out",11:"nov",12:"dez"
                }

                for ano_i, mes_i in meses_imp:
                    if ano_i == ano_atual_i and mes_i == mes_atual_i:
                        fim_mes_i = hoje_imp
                    else:
                        ultimo_dia_i = _cal.monthrange(ano_i, mes_i)[1]
                        fim_mes_i = datetime(ano_i, mes_i, ultimo_dia_i, 23, 59, 59)

                    label_mes = f"{meses_pt_imp[mes_i]}/{str(ano_i)[2:]}"

                    def _ativo_mes(row, fim):
                        criado = row['Criado_Em']
                        if pd.isnull(criado) or criado > fim:
                            return False
                        if row['Status_Category'] == 'done':
                            sc = row['Status_Cat_Change']
                            if pd.notnull(sc) and sc <= fim:
                                return False
                        return True

                    df_ativos_mes = df_pool[df_pool.apply(_ativo_mes, axis=1, fim=fim_mes_i)]

                    for _, row in df_ativos_mes.iterrows():
                        if row['Pausado']:
                            raw_motivo = str(row.get('Motivo da Pausa', '') or '').strip()
                            if raw_motivo in ('', 'Não informado', 'nan'):
                                raw_motivo = 'Outros'
                            if raw_motivo not in CORES_MOTIVO:
                                raw_motivo = 'Outros'
                            motivo_cat = raw_motivo
                        else:
                            motivo_cat = 'Sem Impedimento'

                        imp_records.append({
                            "Mês": label_mes,
                            "Motivo": motivo_cat,
                            "Key": row['Key'],
                            "Summary": row['Summary'],
                            "Fase": row['Fase'],
                            "Assignee": row.get('Assignee', ''),
                            "Submotivo": row.get('Submotivo', ''),
                        })

                df_imp_long = pd.DataFrame(imp_records)

                if df_imp_long.empty:
                    st.info("Sem dados de impedimentos no período selecionado.")
                else:
                    # Ordenação das categorias (Sem Impedimento primeiro)
                    ordem_motivos = ["Sem Impedimento"] + [
                        m for m in CORES_MOTIVO.keys() if m != "Sem Impedimento"
                    ]

                    # Agrupamento para o gráfico
                    df_imp_chart = (
                        df_imp_long
                        .groupby(["Mês", "Motivo"])
                        .size()
                        .reset_index(name="Tarefas")
                    )

                    # Garante ordem cronológica dos meses no eixo X
                    ordem_x = [f"{meses_pt_imp[m]}/{str(a)[2:]}" for a, m in meses_imp]
                    df_imp_chart["Mês"] = pd.Categorical(df_imp_chart["Mês"], categories=ordem_x, ordered=True)
                    df_imp_chart = df_imp_chart.sort_values("Mês")

                    fig_imp = px.bar(
                        df_imp_chart,
                        x="Mês",
                        y="Tarefas",
                        color="Motivo",
                        barmode=barmode_val,
                        title="Em Andamento / Impedidos",
                        labels={"Tarefas": "Tarefas", "Mês": "", "Motivo": ""},
                        color_discrete_map=CORES_MOTIVO,
                        category_orders={"Motivo": ordem_motivos},
                        text_auto=True,
                    )
                    fig_imp.update_traces(
                        textfont_size=10,
                        textposition="outside" if barmode_val == "group" else "inside"
                    )
                    fig_imp.update_layout(
                        height=480,
                        margin=dict(t=50, b=40, l=10, r=10),
                        legend=dict(
                            orientation="h",
                            yanchor="top",
                            y=-0.18,
                            xanchor="center",
                            x=0.5
                        ),
                        xaxis=dict(tickangle=-45),
                        plot_bgcolor="white",
                        paper_bgcolor="white",
                    )
                    fig_imp = apply_totvs_style(fig_imp)
                    st.plotly_chart(fig_imp, use_container_width=True)

                    # ── Drill-down: selecionar mês e ver projetos ──────────────
                    st.markdown("#### 🔎 Detalhar Projetos do Mês")
                    col_d1, col_d2 = st.columns([2, 2])
                    with col_d1:
                        meses_drill = ["Todos"] + ordem_x
                        mes_drill_sel = st.selectbox(
                            "Selecione o mês:",
                            options=meses_drill,
                            index=len(meses_drill) - 1,
                            key="imp_drill_mes"
                        )
                    with col_d2:
                        motivos_drill = ["Todos"] + ordem_motivos
                        motivo_drill_sel = st.selectbox(
                            "Selecione o motivo:",
                            options=motivos_drill,
                            key="imp_drill_motivo"
                        )

                    df_drill = df_imp_long.copy()
                    if mes_drill_sel != "Todos":
                        df_drill = df_drill[df_drill["Mês"] == mes_drill_sel]
                    if motivo_drill_sel != "Todos":
                        df_drill = df_drill[df_drill["Motivo"] == motivo_drill_sel]

                    df_drill_display = df_drill[
                        ["Mês", "Motivo", "Key", "Summary", "Fase", "Submotivo", "Assignee"]
                    ].rename(columns={
                        "Summary": "Projeto",
                        "Motivo": "Motivo da Pausa",
                        "Assignee": "Responsável",
                        "Submotivo": "Submotivo"
                    }).sort_values(["Mês", "Motivo da Pausa", "Projeto"])

                    st.markdown(f"**{len(df_drill_display)} projeto(s) encontrado(s)**")
                    st.data_editor(
                        df_drill_display,
                        use_container_width=True,
                        hide_index=True,
                        disabled=True
                    )

            elif chart_type == "Tabela Completa":
                st.write("### Detalhamento dos Projetos")
                df_completa = df_filtered[['Key', 'Summary', 'Fase', 'Categoria', 'Etiquetas', 'Assignee']].copy()
                df_completa_filtered = filter_columns(df_completa, "tabela_completa")
                st.dataframe(df_completa_filtered, use_container_width=True)

            with st.expander("Explorar Todos os Dados"):
                df_explorer = df_filtered.copy()
                if chart_type == "Tickets Freshdesk" and 'last_fd_counts' in st.session_state:
                    counts = st.session_state['last_fd_counts']
                    # Limpa as strings para garantir a associação independente de espaços extras ou maiúsculas/minúsculas
                    counts_clean = {str(k).strip().lower(): v for k, v in counts.items()}
                    df_explorer['Tickets Freshdesk'] = df_explorer['Summary'].map(lambda x: counts_clean.get(str(x).strip().lower(), 0))
                df_explorer_filtered = filter_columns(df_explorer, "explorar_todos")
                st.dataframe(df_explorer_filtered, use_container_width=True)
        else:
            st.warning("Nenhum dado com os filtros atuais.")
else:
    st.warning("Não foi possível carregar os projetos. Verifique sua conexão e TOKEN.")

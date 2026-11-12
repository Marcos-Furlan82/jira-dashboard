import os

def load_env():
    # 1. Tentar ler do Streamlit Cloud Secrets (se estiver rodando no Streamlit Cloud)
    try:
        import streamlit as st
        if hasattr(st, "secrets"):
            for key in ["JIRA_URL", "JIRA_EMAIL", "JIRA_TOKEN", "GEMINI_API_KEY"]:
                if key in st.secrets and key not in os.environ:
                    os.environ[key] = str(st.secrets[key])
    except Exception:
        pass

    # 2. Tentar ler do arquivo local .env
    root_dir = os.path.dirname(os.path.abspath(__file__))
    env_path = os.path.join(root_dir, ".env")
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    val = v.strip()
                    # Remove surrounding quotes if present
                    if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
                        val = val[1:-1]
                    if k.strip() not in os.environ:
                        os.environ[k.strip()] = val

# Load env variables on import
load_env()

JIRA_URL = os.environ.get("JIRA_URL", "https://lexos-totvs.atlassian.net/").rstrip("/") + "/"
EMAIL = os.environ.get("JIRA_EMAIL", "lexos.jira@gmail.com")
TOKEN = os.environ.get("JIRA_TOKEN", "")


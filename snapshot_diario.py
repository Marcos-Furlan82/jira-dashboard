import os
import json
import requests
import pandas as pd
from datetime import datetime
from requests.auth import HTTPBasicAuth

from jira_config import JIRA_URL, EMAIL, TOKEN
CATEGORIA_FIELD = "customfield_10272"
MOTIVO_PAUSA_FIELD = "customfield_10883"
HISTORY_FILE = "projects_history.csv"

def get_all_projects_data(project_key="IL"):
    """Busca dados de todos os projetos (Histórias) do espaço"""
    url = f"{JIRA_URL}/rest/api/3/search/jql"
    auth = HTTPBasicAuth(EMAIL, TOKEN)
    headers = {"Accept": "application/json"}
    fields = f'summary,status,assignee,issuetype,labels,updated,{CATEGORIA_FIELD},{MOTIVO_PAUSA_FIELD}'
    
    all_issues = []
    next_page_token = None
    
    while True:
        query = {
            'jql': f'project = "{project_key}" AND issuetype = Historia ORDER BY updated DESC',
            'maxResults': 100,
            'fields': fields
        }
        if next_page_token:
            query['nextPageToken'] = next_page_token

        response = requests.get(url, headers=headers, params=query, auth=auth)
        if response.status_code != 200:
            print(f"Erro: {response.status_code}")
            break

        data = response.json()
        issues = data.get('issues', [])
        all_issues.extend(issues)
        
        next_page_token = data.get('nextPageToken')
        if not next_page_token or len(issues) == 0:
            break
            
    return all_issues

def save_snapshot():
    """Salva a 'foto' diária dos projetos no CSV historico"""
    issues = get_all_projects_data()
    if not issues:
        print("Nenhum dado encontrado para salvar.")
        return

    today = datetime.now().strftime("%Y-%m-%d")
    rows = []

    for issue in issues:
        fields = issue['fields']
        
        # Categorias baseadas em Labels e Status
        labels = [l.upper() for l in fields.get('labels', [])]
        is_pausado = "PAUSADO" in " ".join(labels) or "PAUSADOS" in " ".join(labels)
        is_atrasado = "ATRASADO" in " ".join(labels) or "ATRASADOS" in " ".join(labels)
        is_critico = "CRÍTICO" in " ".join(labels) or "CRITICO" in " ".join(labels)
        
        # Motivo e Submotivo
        motivo_raw = fields.get(MOTIVO_PAUSA_FIELD)
        motivo = "Não informado"
        submotivo = "Não informado"
        if isinstance(motivo_raw, dict):
            motivo = motivo_raw.get('value', 'Não informado')
            child = motivo_raw.get('child')
            if isinstance(child, dict):
                submotivo = child.get('value', 'Não informado')

        rows.append({
            "Data_Snapshot": today,
            "Key": issue['key'],
            "Projeto": fields['summary'],
            "ISM_Responsavel": fields['assignee']['displayName'] if fields['assignee'] else "Não atribuído",
            "Pausado": "Sim" if is_pausado else "Não",
            "Atrasado": "Sim" if is_atrasado else "Não",
            "Critico": "Sim" if is_critico else "Não",
            "Motivo_Pausa": motivo if is_pausado else "-",
            "Submotivo_Pausa": submotivo if is_pausado else "-",
            "Ultima_Atualizacao_Jira": fields['updated']
        })

    df_new = pd.DataFrame(rows)
    
    if os.path.exists(HISTORY_FILE):
        df_old = pd.read_csv(HISTORY_FILE)
        # Remove dados do mesmo dia se já existirem (para permitir re-execução)
        df_old = df_old[df_old['Data_Snapshot'] != today]
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new

    df_final.to_csv(HISTORY_FILE, index=False, encoding='utf-8-sig')
    print(f"Snapshot de {today} salvo com sucesso ({len(rows)} projetos).")

if __name__ == "__main__":
    save_snapshot()

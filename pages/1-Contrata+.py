import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
from scipy.stats import norm

# ==========================================
# CONFIGURAÇÃO DA PÁGINA
# ==========================================
st.set_page_config(page_title="Análise Contrata+", page_icon="🎯", layout="wide")

# ==========================================
# SISTEMA DE LOGIN (NOVO)
# ==========================================
def check_password():
    """Retorna True se o usuário inseriu a senha correta configurada no secrets."""
    if "password_correct" not in st.session_state:
        # Inicializa o estado da senha como falso
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    # Interface de bloqueio/login
    st.title("🔐 Acesso Restrito - Contrata+")
    st.markdown("Por favor, insira a senha do projeto para acessar o laboratório de dados.")
    
    senha_input = st.text_input("Digite a senha:", type="password")
    
    if st.button("Entrar"):
        # Puxa a senha do arquivo secrets.toml
        # Usamos st.secrets.get() por segurança, caso a chave não exista
        senha_correta = st.secrets.get("password", "") 
        
        if senha_input == senha_correta:
            st.session_state["password_correct"] = True
            st.rerun() # Reinicia a página para carregar o dashboard
        else:
            st.error("🚫 Senha incorreta.")
    
    return False

# BLOQUEIO DE EXECUÇÃO: O código para aqui se a senha não estiver correta
if not check_password():
    st.stop()
    
# ==========================================
# FUNÇÕES ESTATÍSTICAS E DE PARSING
# ==========================================
def format_p_valor(p):
    """Formata o p-valor: decimal para valores comuns, científica para muito baixos."""
    if pd.isna(p): return "-"
    if p < 0.001:
        return f"{p:.4e}" # Ex: 1.2345e-05
    return f"{p:.4f}"

def calcular_teste_z(x1, n1, x2, n2):
    """Calcula o Teste Z para Duas Proporções e retorna o p-valor."""
    if n1 == 0 or n2 == 0:
        return np.nan, np.nan
    p1, p2 = x1 / n1, x2 / n2
    p_pool = (x1 + x2) / (n1 + n2)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    p_valor = 2 * (1 - norm.cdf(abs(z)))
    return z, p_valor

def parse_data_ptbr(data_str):
    """Lê o formato 'ter., fev. 11, 2025' e converte para Datetime do Pandas."""
    if pd.isna(data_str): return pd.NaT
    data_str = str(data_str).lower().strip()
    meses = {'jan': '01', 'fev': '02', 'mar': '03', 'abr': '04', 'mai': '05', 'jun': '06',
             'jul': '07', 'ago': '08', 'set': '09', 'out': '10', 'nov': '11', 'dez': '12'}
    
    match = re.search(r'([a-zç]+)\.?\s+(\d{1,2}),\s+(\d{4})', data_str)
    if match:
        mes_str, dia, ano = match.groups()
        mes = meses.get(mes_str[:3], '01')
        return pd.to_datetime(f"{ano}-{mes}-{dia.zfill(2)}")
    
    # Fallback para caso alguma data já esteja no padrão ISO
    try:
        return pd.to_datetime(data_str)
    except:
        return pd.NaT

# ==========================================
# 1. FUNÇÃO DE CARREGAMENTO (VIA SECRETS)
# ==========================================
@st.cache_data(ttl=600)
def carregar_dados_contrata():
    try:
        url_mens = st.secrets["contrata_mais"]["url_mensagens"]
        url_cad = st.secrets["contrata_mais"]["url_cadastros"]
        
        df_mens = pd.read_csv(url_mens, dtype={'CNPJ': str}) 
        df_cad = pd.read_csv(url_cad, dtype={'CNPJ': str})
        
        return df_mens, df_cad
    except KeyError:
        st.error("❌ A seção [contrata_mais] ou as URLs não foram encontradas. Verifique o seu arquivo secrets.toml.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Erro ao baixar os dados: {e}")
        st.stop()

# ==========================================
# 2. PROCESSAMENTO E CRUZAMENTO DE DADOS
# ==========================================
df_mensagens, df_cadastros = carregar_dados_contrata()

try:
    # Limpeza de CNPJ
    df_mensagens['CNPJ_clean'] = df_mensagens['CNPJ'].astype(str).str.replace(r'\D', '', regex=True)
    df_cadastros['CNPJ_clean'] = df_cadastros['CNPJ'].astype(str).str.replace(r'\D', '', regex=True)

    # Conversão de Datas Segura
    if 'data_envio' in df_mensagens.columns:
        df_mensagens['data_envio_dt'] = pd.to_datetime(df_mensagens['data_envio'], errors='coerce')
    
    df_cadastros['data_cadastro_dt'] = df_cadastros['Data de cadastro'].apply(parse_data_ptbr)

    # Remoção de Duplicatas e Cruzamento
    df_cadastros_unicos = df_cadastros.drop_duplicates(subset=['CNPJ_clean'])
    df_analise = pd.merge(
        df_mensagens, 
        df_cadastros_unicos[['CNPJ_clean', 'Data de cadastro', 'data_cadastro_dt', 'UF', 'Ativo']], 
        on='CNPJ_clean', 
        how='left'
    )

    df_analise['converteu'] = df_analise['Data de cadastro'].notna()
    df_analise['mensagem_lida'] = pd.to_numeric(df_analise['mensagem_lida'], errors='coerce').fillna(0)

except Exception as e:
    st.error(f"Erro no processamento: {e}")
    st.stop()

# ==========================================
# 3. CONSTRUÇÃO DA BARRA LATERAL (SIDEBAR)
# ==========================================
st.sidebar.success("✅ Dados carregados via Google Drive")

if st.sidebar.button("🔄 Forçar Atualização dos Dados"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.divider()

total_enviado = len(df_analise)
total_lido = df_analise['mensagem_lida'].sum()
total_convertido = df_analise['converteu'].sum()
notificacoes_enviadas = df_analise['notificacao_push_enviada'].notna().sum() if 'notificacao_push_enviada' in df_analise.columns else 0

st.sidebar.markdown("### 🚩 Status Geral")
st.sidebar.write(f"- **Enviadas:** {total_enviado}")
st.sidebar.write(f"- **Lidas:** {int(total_lido)}")
st.sidebar.write(f"- **Cadastros:** {int(total_convertido)}")

with st.sidebar.expander("🛠️ Ver Colunas Processadas"):
    st.write(df_analise.columns.tolist())

st.sidebar.divider()
csv_export = df_analise.to_csv(index=False).encode('utf-8')
st.sidebar.download_button(label="📥 Baixar Base Cruzada", data=csv_export, file_name="base_contrata_mais.csv", mime="text/csv", use_container_width=True)

# ==========================================
# 4. INTERFACE PRINCIPAL (DASHBOARD)
# ==========================================
st.title("🎯 Experimento Comportamental: Contrata+")
st.markdown("Avaliação de gatilhos comportamentais para o cadastro de MEIs em compras públicas.")

# SISTEMA DE ABAS (TABS)
tab_geral, tab_estatistica = st.tabs(["📊 Visão Geral do Funil", "🔬 Testes Estatísticos e Baseline"])

# ---------------------------------------------------------
# ABA 1: VISÃO GERAL (O Dashboard que já tínhamos)
# ---------------------------------------------------------
with tab_geral:
    st.subheader("Estatísticas Totais do Disparo")
    taxa_leitura_geral = (total_lido / total_enviado) * 100 if total_enviado > 0 else 0
    taxa_conversao_geral = (total_convertido / total_enviado) * 100 if total_enviado > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total de MEIs Abordados", f"{total_enviado}")
    c2.metric("Taxa de Abertura (Lidas)", f"{int(total_lido)}", f"{taxa_leitura_geral:.4f}%")
    c3.metric("Cadastros Efetivados", f"{int(total_convertido)}")
    c4.metric("Taxa de Conversão Geral", f"{taxa_conversao_geral:.4f}%")
    st.divider()

    coluna_grupo = 'Grupo'
    if coluna_grupo in df_analise.columns:
        df_grupos = df_analise.groupby(coluna_grupo).agg({'CNPJ_clean': 'count', 'mensagem_lida': 'sum', 'converteu': 'sum'}).reset_index()
        df_grupos.columns = ['Grupo', 'Enviados', 'Lidas', 'Convertidos']
        df_grupos['Taxa de Leitura (%)'] = (df_grupos['Lidas'] / df_grupos['Enviados'] * 100).round(4)
        df_grupos['Taxa de Conversão (%)'] = (df_grupos['Convertidos'] / df_grupos['Enviados'] * 100).round(4)
        df_grupos = df_grupos.sort_values('Taxa de Conversão (%)', ascending=False)

        col_tabela, col_grafico = st.columns([1, 2])
        with col_tabela:
            st.write("**Resumo por Mensagem:**")
            st.dataframe(df_grupos, use_container_width=True, hide_index=True,
                         column_config={"Taxa de Leitura (%)": st.column_config.NumberColumn(format="%.4f %%"),
                                        "Taxa de Conversão (%)": st.column_config.NumberColumn(format="%.4f %%")})
        with col_grafico:
            metrica_alvo = st.radio("Métrica para o gráfico:", ["Taxa de Conversão (%)", "Taxa de Leitura (%)"], horizontal=True)
            fig_conv = px.bar(df_grupos.sort_values(metrica_alvo, ascending=False), x='Grupo', y=metrica_alvo,
                              text=metrica_alvo, title=f"Comparativo de {metrica_alvo.replace(' (%)', '')} por Grupo",
                              color='Grupo', color_discrete_sequence=px.colors.qualitative.Prism)
            fig_conv.update_traces(texttemplate='%{text:.4f}%', textposition='outside')
            fig_conv.update_layout(yaxis=dict(range=[0, df_grupos[metrica_alvo].max() * 1.2]))
            st.plotly_chart(fig_conv, use_container_width=True)

    st.subheader("🌪️ Funil de Engajamento")
    fig_funil = go.Figure(go.Funnel(
        y=['Mensagens Enviadas', 'Mensagens Lidas', 'Cadastros Realizados'],
        x=[total_enviado, int(total_lido), int(total_convertido)],
        textinfo="value+percent initial", marker=dict(color=["#B3CDE3", "#8C96C6", "#8856A7"])
    ))
    st.plotly_chart(fig_funil, use_container_width=True)

# ---------------------------------------------------------
# ABA 2: LABORATÓRIO ESTATÍSTICO
# ---------------------------------------------------------
with tab_estatistica:
    st.markdown("### 📅 1. Cálculo da Linha de Base (Baseline Orgânico)")
    st.info("""
    Para comparar o período de intervenção (26/03/2026 a 05/04/2026) com períodos históricos sem campanha, 
    informe o tamanho da base de MEIs elegíveis para o denominador das taxas históricas.
    """)
    
    # Denominador para o cálculo da taxa histórica
    base_mei_estado = st.number_input("Tamanho da Base Geral de MEIs elegíveis (Denominador Histórico):", min_value=100, value=total_enviado, step=1000)

    # As DUAS janelas exatas que você solicitou (11 dias cada)
    janelas_baseline = [
        {"nome": "Imediatamente Anterior", "inicio": "2026-01-30", "fim": "2026-02-09"},
        {"nome": "Ano Anterior", "inicio": "2025-03-26", "fim": "2025-04-05"}
    ]
    
    cadastros_base_historica = []
    
    col_hist1, col_hist2 = st.columns(2)
    
    # Lógica iterativa para calcular volume de cada janela
    for idx, j in enumerate(janelas_baseline):
        inicio, fim = pd.to_datetime(j["inicio"]), pd.to_datetime(j["fim"])
        # Conta na tabela de cadastros ÚNICOS (já limpa de duplicatas no início do código)
        qtd = df_cadastros_unicos[
            (df_cadastros_unicos['data_cadastro_dt'] >= inicio) & 
            (df_cadastros_unicos['data_cadastro_dt'] <= fim)
        ].shape[0]
        cadastros_base_historica.append(qtd)
        
        # Exibe os volumes encontrados
        if idx == 0: col_hist1.metric(f"Cadastros: {j['nome']}", qtd, f"{j['inicio']} a {j['fim']}")
        else: col_hist2.metric(f"Cadastros: {j['nome']}", qtd, f"{j['inicio']} a {j['fim']}")

    # Média e Taxa da Linha de Base
    media_cadastros_baseline = np.mean(cadastros_base_historica)
    taxa_baseline_media = (media_cadastros_baseline / base_mei_estado)

    st.markdown(f"**Média do Baseline (11 dias):** {media_cadastros_baseline:.1f} cadastros ➔ **Taxa Orgânica:** {(taxa_baseline_media * 100):.4f}%")
    st.divider()
    
    st.markdown("### 📊 2. Efeito Incremental Consolidado e Por Mensagem")
    
    # Filtro do período da intervenção
    if 'data_envio_dt' in df_analise.columns:
        df_campanha = df_analise[
            (df_analise['data_envio_dt'] >= pd.to_datetime("2026-03-26")) & 
            (df_analise['data_envio_dt'] <= pd.to_datetime("2026-04-05"))
        ]
        if df_campanha.empty: df_campanha = df_analise
    else:
        df_campanha = df_analise

    # 2.1 EFEITO CONSOLIDADO (Toda a intervenção vs Baseline)
    cadastros_consolidado = df_campanha['converteu'].sum()
    expostos_consolidado = len(df_campanha)
    taxa_consolidada = cadastros_consolidado / expostos_consolidado if expostos_consolidado > 0 else 0
    
    z_cons, p_val_cons = calcular_teste_z(cadastros_consolidado, expostos_consolidado, media_cadastros_baseline, base_mei_estado)
    inc_cons = (taxa_consolidada - taxa_baseline_media) * 100
    
    st.success(f"**EFEITO INCREMENTAL TOTAL:** A campanha gerou uma taxa de **{(taxa_consolidada*100):.4f}%** vs **{(taxa_baseline_media*100):.4f}%** do orgânico. Incremento de **{inc_cons:+.4f} p.p.** (P-Valor: {format_p_valor(p_val_cons)} ➔ {'✅ Significativo' if p_val_cons < 0.05 else '❌ Não Significativo'})")
    
    # 2.2 TABELA 1: Cada mensagem vs Baseline
    if coluna_grupo in df_campanha.columns:
        grupos_stats = df_campanha.groupby(coluna_grupo).agg(
            cadastros=('converteu', 'sum'),
            expostos=('CNPJ_clean', 'count')
        ).reset_index()

        resultados_tab1 = []
        for _, row in grupos_stats.iterrows():
            g_nome = row[coluna_grupo]
            x_camp, n_camp = row['cadastros'], row['expostos']
            taxa_camp = x_camp / n_camp if n_camp > 0 else 0
            
            z, p_val = calcular_teste_z(x_camp, n_camp, media_cadastros_baseline, base_mei_estado)
            incremento_pp = (taxa_camp - taxa_baseline_media) * 100
            
            resultados_tab1.append({
                "Gatilho (Mensagem)": g_nome,
                "Expostos": n_camp,
                "Cadastros": x_camp,
                "Taxa Mensagem (%)": f"{(taxa_camp * 100):.4f}%",
                "Efeito vs Baseline": f"{incremento_pp:+.4f} p.p.",
                "P-Valor": format_p_valor(p_val), 
                "Significância (p<0.05)": "✅" if p_val < 0.05 else "❌"
            })
        
        st.write("Tabela 1: Impacto das Mensagens vs. Linha de Base Orgânica")
        df_tab1 = pd.DataFrame(resultados_tab1)
        st.dataframe(df_tab1, use_container_width=True, hide_index=True)

        st.divider()

        # 2.3 TABELA 2: Comparação par a par entre os gatilhos (A vs B, B vs C, etc)
        st.markdown("### ⚔️ 3. Teste A/B/C (Comparação Direta Entre as Mensagens)")
        
        resultados_tab2 = []
        nomes_grupos = grupos_stats[coluna_grupo].tolist()
        
        for i in range(len(nomes_grupos)):
            for j in range(i + 1, len(nomes_grupos)):
                gA, gB = nomes_grupos[i], nomes_grupos[j]
                
                xA = grupos_stats.loc[grupos_stats[coluna_grupo] == gA, 'cadastros'].values[0]
                nA = grupos_stats.loc[grupos_stats[coluna_grupo] == gA, 'expostos'].values[0]
                xB = grupos_stats.loc[grupos_stats[coluna_grupo] == gB, 'cadastros'].values[0]
                nB = grupos_stats.loc[grupos_stats[coluna_grupo] == gB, 'expostos'].values[0]
                
                taxaA, taxaB = xA / nA, xB / nB
                diff = (taxaA - taxaB) * 100
                
                z, p_val = calcular_teste_z(xA, nA, xB, nB)
                vencedor = gA if diff > 0 and p_val < 0.05 else (gB if diff < 0 and p_val < 0.05 else "Empate Técnico")

                resultados_tab2.append({
                    "Comparação": f"{gA} vs {gB}",
                    "Taxa A (%)": f"{(taxaA * 100):.4f}%",
                    "Taxa B (%)": f"{(taxaB * 100):.4f}%",
                    "Diferença (p.p.)": f"{diff:+.4f}",
                    "P-Valor": format_p_valor(p_val),
                    "Vencedor Estatístico": vencedor
                })
        
        st.write("Tabela 2: Diferença de Conversão e Significância entre Grupos Experimentais")
        df_tab2 = pd.DataFrame(resultados_tab2)
        st.dataframe(df_tab2, use_container_width=True, hide_index=True)
        
        st.info("💡 **Nota Metodológica:** O Teste Z bicaudal foi aplicado. P-Valores menores que 0.05 (sinalizados com ✅ ou com um Vencedor declarado) garantem com 95% de confiança que os efeitos observados se devem ao gatilho comportamental testado, e não ao acaso.")
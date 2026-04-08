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
    if 'data_leitura' in df_mensagens.columns:
        df_mensagens['data_leitura_dt'] = pd.to_datetime(df_mensagens['data_leitura'], errors='coerce')
    
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
    st.markdown("### A - Teste Estatístico dos Tipos de Mensagens (A/B/C)")
    st.info("Comparação global e pareada (1-1) entre os grupos de disparos efetuados (ex: Emoção e Ego, Aversão a Perdas e Normas Sociais).")
    
    # Prepara base (filtramos quem tem Grupo preenchido)
    df_testes = df_analise.dropna(subset=['Grupo'])
    grupos_unicos = df_testes['Grupo'].unique().tolist()
    
    if len(grupos_unicos) < 2:
        st.warning("Não há grupos o suficiente para realizar testes estatísticos comparativos.")
        st.stop()

    import scipy.stats as stats
    import io
    import itertools

    # Função de Teste Qui-Quadrado Global (K-grupos para proporções)
    def test_chi2_global(df, grupo_col, metrica_col):
        # Tabela de contingência cruzada: Linhas (Grupos) vs Colunas (Converteu 0/1)
        tab = pd.crosstab(df[grupo_col], df[metrica_col])
        if tab.shape[1] < 2:
            return np.nan, np.nan, np.nan
        chi2, p_val, dof, ex = stats.chi2_contingency(tab)
        
        # Cramér's V (Tamanho de Efeito)
        n = tab.sum().sum()
        min_dim = min(tab.shape) - 1
        v_cramer = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 and n > 0 else 0
        return chi2, p_val, v_cramer

    # Função para Teste Z Pareado entre 2 proporções específicas
    def test_pairwise_prop(df, gA, gB, grupo_col, metrica_col):
        nA = len(df[df[grupo_col] == gA])
        nB = len(df[df[grupo_col] == gB])
        xA = df[df[grupo_col] == gA][metrica_col].sum()
        xB = df[df[grupo_col] == gB][metrica_col].sum()
        
        pA = xA / nA if nA > 0 else 0
        pB = xB / nB if nB > 0 else 0
        
        diff = (pA - pB) * 100
        z, p_val = calcular_teste_z(xA, nA, xB, nB)
        
        # Cohen's h (Tamanho Erfeito p/ proporções)
        h_cohen = 2 * (np.arcsin(np.sqrt(pA)) - np.arcsin(np.sqrt(pB)))
        
        return pA, pB, diff, p_val, h_cohen

    metricas_alvo = {
        "Taxa de Leitura": "mensagem_lida",
        "Taxa de Conversão": "converteu"
    }
    
    tabelas_finais = {} # Dicionário mestre para Excel
    
    for nome_metrica, col_metrica in metricas_alvo.items():
        st.markdown(f"#### 🎯 Métrica Principal: {nome_metrica}")
        
        # 1. Teste Global (Chi-Quadrado de Pearson)
        chi2, p_val_g, v_cramer = test_chi2_global(df_testes, 'Grupo', col_metrica)
        
        res_global = pd.DataFrame([{
            "Teste Global": "Qui-Quadrado (Independência)",
            "Estatística (X²)": f"{chi2:.3f}" if pd.notna(chi2) else "-",
            "P-Valor": format_p_valor(p_val_g),
            "Significância": "✅ Há diferença genuína (p < 0.05)" if pd.notna(p_val_g) and p_val_g < 0.05 else "❌ Grupos Não Diferem Estruturalmente",
            "Efeito da Intervenção (Cramér's V)": f"{v_cramer:.3f}" if pd.notna(v_cramer) else "-"
        }])
        
        st.write(f"**Comparação Global: Existe alguma diferença estatística na {nome_metrica} entre os Tipos de Mensagens?**")
        st.dataframe(res_global, use_container_width=True, hide_index=True)
        tabelas_finais[f"{nome_metrica} - Global"] = res_global
        
        # 2. Testes Pareados Post-Hoc (Gatilho A vs Gatilho B)
        st.write(f"**Combates Diretos (Confrontos de Pares): Qual gatilho performa melhor na {nome_metrica}?**")
        
        combinacoes = list(itertools.combinations(grupos_unicos, 2))
        resultados_pareados = []
        
        for gA, gB in combinacoes:
            pA, pB, diff, p_val_par, h_cohen = test_pairwise_prop(df_testes, gA, gB, 'Grupo', col_metrica)
            
            # Correção de Bonferroni (Tornar o teste mais robusto para múltiplas comparações simultâneas e não cometer Falso Positivo)
            alfa_bonf = 0.05 / len(combinacoes) if len(combinacoes) > 0 else 0.05
            
            sig = f"✅ (p < {alfa_bonf:.4f})" if pd.notna(p_val_par) and p_val_par < alfa_bonf else "❌"
            vencedor = gA if diff > 0 and '✅' in sig else (gB if diff < 0 and '✅' in sig else "Empate Técnico")
            
            resultados_pareados.append({
                "Confronto Pareado": f"{gA} vs {gB}",
                "Desempenho A (%)": f"{(pA * 100):.4f}%",
                "Desempenho B (%)": f"{(pB * 100):.4f}%",
                "Diferença (p.p.)": f"{diff:+.4f}",
                "P-Valor Simples": format_p_valor(p_val_par),
                "Significante? (Bonferroni)": sig,
                "Intensidade (Cohen's h)": f"{h_cohen:+.3f}",
                "Gatilho Vencedor": vencedor
            })
            
        df_pareados = pd.DataFrame(resultados_pareados)
        st.dataframe(df_pareados, use_container_width=True, hide_index=True)
        tabelas_finais[f"{nome_metrica} - Pareada"] = df_pareados
        
        st.info("💡 **Nota Metodológica:** O Teste Global Qui-Quadrado aponta se no emaranhado geral há impacto comportamental. Embaixo, a Matriz Pareada cruza gatilho-a-gatilho usando o Teste Z aplicado com Correção de Bonferroni (mais robusto contra falsos positivos). Os tamanhos de efeito (Cramér's V e Cohen's h) demonstram se além de dar p-valor vencedor, a intensidade real também foi substancial.")
        
        st.divider()

    st.markdown("### B - Comparação Orgânica de Cadastros Diários")
    st.info("Projeção de performance da Janela da Intervenção versus janelas temporais de tamanho equivalente (11 dias) num contexto completamente orgânico (sem campanhas).")

    # Definição das janelas especificadas
    janelas_analise = [
        {"Nome": "🎯 Período do Disparo (Intervenção)", "Início": "2026-03-26", "Fim": "2026-04-05"},
        {"Nome": "Imediatamente Pré-Disparo", "Início": "2026-01-30", "Fim": "2026-02-09"},
        {"Nome": "Exato Período no Ano Anterior", "Início": "2025-03-26", "Fim": "2025-04-05"},
        {"Nome": "Histórico Jan/1", "Início": "2026-01-20", "Fim": "2026-01-30"},
        {"Nome": "Histórico Jan/2", "Início": "2026-01-25", "Fim": "2026-02-04"},
        {"Nome": "Histórico Mar/Ant", "Início": "2025-03-19", "Fim": "2025-03-29"},
        {"Nome": "Histórico Abr/Post", "Início": "2025-04-16", "Fim": "2025-04-26"},
    ]

    resultados_janelas = []
    dict_series_dias = {}
    
    for janela in janelas_analise:
        dt_inicio = pd.to_datetime(janela["Início"])
        dt_fim = pd.to_datetime(janela["Fim"])
        
        # Filtragem precisa no unicos de cadastro
        df_j = df_cadastros_unicos[
            (df_cadastros_unicos['data_cadastro_dt'] >= dt_inicio) & 
            (df_cadastros_unicos['data_cadastro_dt'] <= dt_fim)
        ]
        
        # Garantindo array contínuo de 11 dias onde dias de 0 vendas recebem 0 (para médias / desvios reais e corretos)
        dias_range = pd.date_range(start=dt_inicio, end=dt_fim)
        cadastros_por_dia = df_j.groupby('data_cadastro_dt').size().reindex(dias_range, fill_value=0)
        
        dict_series_dias[janela["Nome"]] = cadastros_por_dia
        
        total_cads = cadastros_por_dia.sum()
        media_cads = cadastros_por_dia.mean()
        mediana_cads = cadastros_por_dia.median()
        std_cads = cadastros_por_dia.std(ddof=1) if len(cadastros_por_dia) > 1 else 0
        
        resultados_janelas.append({
            "Cenário (Curva Orgânica vs Intervenção)": janela["Nome"],
            "Período Real": f"{dt_inicio.strftime('%d/%m/%Y')} a {dt_fim.strftime('%d/%m/%Y')}",
            "Nº de Cadastros (11 dias)": int(total_cads),
            "Média Diária": round(media_cads, 2),
            "Mediana Diária": round(mediana_cads, 2),
            "Desvio Padrão": round(std_cads, 2)
        })

    df_res_b = pd.DataFrame(resultados_janelas)
    
    # Aplica formatação visual para destacar a Intervenção a olho nu
    def highlight_intervencao(row):
        if "Intervenção" in row['Cenário (Curva Orgânica vs Intervenção)']:
            return ['background-color: rgba(136, 86, 167, 0.2); font-weight: bold'] * len(row)
        return [''] * len(row)
    
    st.dataframe(df_res_b.style.apply(highlight_intervencao, axis=1), use_container_width=True, hide_index=True)
    tabelas_finais["Cadastros Históricos (Geral)"] = df_res_b
    
    st.markdown("#### 🔬 Teste de Médias Estatísticas (Disparo vs Históricos)")
    st.write("Este teste avalia, dia contra dia, se o aumento/redução do seu volume de conversão tem validação científica comparado ao cenário normal (Welch's t-test para amostras contínuas não lineares) ou se foi ao acaso.")
    
    # Referência da Intervenção para confrontamento
    nome_int = "🎯 Período do Disparo (Intervenção)"
    array_int = dict_series_dias.get(nome_int)
    
    if array_int is not None:
        n_int = len(array_int)
        resultados_testes_b = []
        
        for nome_base, array_base in dict_series_dias.items():
            if nome_base == nome_int: 
                continue
                
            n_base = len(array_base)
            mean_int, mean_base = array_int.mean(), array_base.mean()
            std_int, std_base = array_int.std(ddof=1), array_base.std(ddof=1)
            
            diff_medias = mean_int - mean_base
            
            # Teste T de Welch (amostras independentes, variâncias heterogêneas permitidas)
            t_stat, p_val = stats.ttest_ind(array_int, array_base, equal_var=False)
            
            # Cohen's d para independent t-test
            d_cohen = 0
            if (n_int + n_base - 2) > 0:
                var_int, var_base = std_int**2, std_base**2
                s_pool = np.sqrt(((n_int - 1) * var_int + (n_base - 1) * var_base) / (n_int + n_base - 2))
                if s_pool != 0:
                    d_cohen = diff_medias / s_pool
            
            resultados_testes_b.append({
                "Cenário Alvo": f"vs {nome_base}",
                "Desempenho da Intervenção": f"≈ {mean_int:.2f} cads/dia",
                "Desempenho do Orgânico": f"≈ {mean_base:.2f} cads/dia",
                "Efeito Bruto (Δ)": f"{diff_medias:+.2f} cads",
                "P-Valor": format_p_valor(p_val),
                "Parecer Teste-t": "✅ Diferença Científica" if (pd.notna(p_val) and p_val < 0.05) else "❌ Estatisticamente Igual",
                "Impacto do Efeito (Cohen's d)": f"{d_cohen:+.3f}"
            })
            
        df_testes_b = pd.DataFrame(resultados_testes_b)
        st.dataframe(df_testes_b, use_container_width=True, hide_index=True)
        tabelas_finais["Teste de Hipóteses (B)"] = df_testes_b

    st.divider()

    st.markdown("### 📊 C - Representações Visuais Diárias")
    st.markdown("#### 📦 Dispersão Diária por Janela (Acumulado Diário de Cadastros)")
    
    lista_df_box = []
    for nome_janela, serie_diaria in dict_series_dias.items():
        # Criação de um DataFrame desconstruído para o Boxplot
        df_temp = pd.DataFrame({
            "Cenário Alvo": nome_janela,
            "Cadastros no Dia": serie_diaria.values
        })
        lista_df_box.append(df_temp)
    
    df_box_total = pd.concat(lista_df_box)
    fig_b_box = px.box(
        df_box_total, x="Cenário Alvo", y="Cadastros no Dia", color="Cenário Alvo", points="all",
        title="O Nível de Constância: Dispersão de Cadastros (Boxplot)", color_discrete_sequence=px.colors.qualitative.Prism
    )
    fig_b_box.update_layout(xaxis_title="", showlegend=False, xaxis=dict(tickangle=-20))
    st.plotly_chart(fig_b_box, use_container_width=True)

    st.divider()

    st.markdown("#### 🗓️ Densidade Histórica (Mapas de Calor das Datas Reais)")
    st.info("O calendário ilustra em quais dias exatos do mês houve pico e esfriamento num panorama geral do tempo.")
    
    def plot_calendar_heatmap(df_dados, col_data, title, colorscale):
        # Filtra campos não preenchidos
        serie_valida = df_dados[col_data].dropna()
        if serie_valida.empty:
            return go.Figure().update_layout(title=f"{title} (Sem dados suficientes)")
            
        counts = serie_valida.groupby(serie_valida.dt.date).size().reset_index(name='count')
        counts.columns = ['date', 'count']
        counts['date'] = pd.to_datetime(counts['date'])
        counts['dia'] = counts['date'].dt.day
        counts['mes_ano'] = counts['date'].dt.strftime('%m/%Y')
        
        # Garante a ordem correta para o Eixo Y
        counts['ano'] = counts['date'].dt.year
        counts['mes_num'] = counts['date'].dt.month
        counts = counts.sort_values(['ano', 'mes_num'])
        
        df_pivot = counts.pivot(index='mes_ano', columns='dia', values='count').fillna(0)
        
        # Matriz preenchida (Dias de 1 a 31 forçadamente para visual de mês fechado)
        df_pivot = df_pivot.reindex(columns=range(1, 32), fill_value=np.nan)
        
        # Ordem descrescente (Mais recente em cima) no Y
        Y_ord = counts[['ano', 'mes_num', 'mes_ano']].drop_duplicates().sort_values(['ano', 'mes_num'], ascending=False)['mes_ano'].tolist()
        df_pivot = df_pivot.reindex(index=Y_ord)
        
        fig = go.Figure(data=go.Heatmap(
            z=df_pivot.values,
            x=df_pivot.columns,
            y=df_pivot.index,
            colorscale=colorscale,
            xgap=2, ygap=2,
            hovertemplate="<b>%{y} / Dia %{x}</b><br>Quantidade: %{z}<extra></extra>"
        ))
        
        fig.update_layout(
            title=title,
            xaxis=dict(tickmode='linear', dtick=1, title="Dia Mês", side='top'),
            yaxis=dict(title="Ciclos"),
            height=max(250, len(Y_ord)*40),
            margin=dict(t=80, b=10)
        )
        return fig

    def plot_weekday_heatmap(df_dados, col_data, title, colorscale):
        serie_valida = df_dados[col_data].dropna()
        if serie_valida.empty:
            return go.Figure().update_layout(title=f"{title} (Sem dados)")
            
        df_temp = pd.DataFrame({'date': serie_valida})
        df_temp['date'] = pd.to_datetime(df_temp['date'])
        
        map_dias = {0: '1-Seg', 1: '2-Ter', 2: '3-Qua', 3: '4-Qui', 4: '5-Sex', 5: '6-Sáb', 6: '7-Dom'}
        df_temp['dia_semana'] = df_temp['date'].dt.dayofweek.map(map_dias)
        df_temp['mes_ano'] = df_temp['date'].dt.strftime('%m/%Y')
        df_temp['ano'] = df_temp['date'].dt.year
        df_temp['mes_num'] = df_temp['date'].dt.month
        
        counts = df_temp.groupby(['ano', 'mes_num', 'mes_ano', 'dia_semana']).size().reset_index(name='count')
        counts = counts.sort_values(['ano', 'mes_num'])
        df_pivot = counts.pivot(index='mes_ano', columns='dia_semana', values='count').fillna(0)
        
        dias_ord = ['1-Seg', '2-Ter', '3-Qua', '4-Qui', '5-Sex', '6-Sáb', '7-Dom']
        df_pivot = df_pivot.reindex(columns=dias_ord, fill_value=np.nan)
        dias_labels = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']
        
        Y_ord = counts[['ano', 'mes_num', 'mes_ano']].drop_duplicates().sort_values(['ano', 'mes_num'], ascending=False)['mes_ano'].tolist()
        df_pivot = df_pivot.reindex(index=Y_ord)
        
        fig = go.Figure(data=go.Heatmap(
            z=df_pivot.values,
            x=dias_labels,
            y=df_pivot.index,
            colorscale=colorscale,
            xgap=2, ygap=2,
            hovertemplate="<b>%{y} / %{x}</b><br>Quantidade Agrupada: %{z}<extra></extra>"
        ))
        fig.update_layout(
            title=title, xaxis=dict(side='top'), yaxis=dict(title="Ciclos"),
            height=max(250, len(Y_ord)*40), margin=dict(t=80, b=10)
        )
        return fig

    # Plota a matriz de Dias do Mês Lado a Lado
    col_hm1, col_hm2 = st.columns(2)
    with col_hm1:
        fig_hm_cad = plot_calendar_heatmap(df_cadastros_unicos, 'data_cadastro_dt', 'Volume Geral Diário (Cadastros)', 'Purples')
        st.plotly_chart(fig_hm_cad, use_container_width=True)
    with col_hm2:
        if 'data_leitura_dt' in df_mensagens.columns:
            fig_hm_lei = plot_calendar_heatmap(df_mensagens, 'data_leitura_dt', 'Tracking de Interação (Mensagens Lidas)', 'Blues')
            st.plotly_chart(fig_hm_lei, use_container_width=True)
        else:
            st.warning("Data de leitura de mensagens não encontrada. O gráfico não pôde ser gerado.")

    st.write("")
    st.markdown("#### 📅 Frequência de Engajamento e Sazonalidade")
    st.info("Navegue pelas abas abaixo para investigar especificamente quais dias da semana ou meses atraem organicamente as maiores avalanches de cadastros.")
    
    tab_hm, tab_box, tab_bar = st.tabs(["🔥 Heatmap Diário (Matriz)", "📦 Boxplot da Sazonalidade", "📊 Acumulado Total"])
    
    with tab_hm:
        fig_hm_cad_sem = plot_weekday_heatmap(df_cadastros_unicos, 'data_cadastro_dt', 'Picos de Cadastro (Seg-Dom)', 'Greens')
        st.plotly_chart(fig_hm_cad_sem, use_container_width=True)
        
    with tab_box:
        # Prepara a base agregada por dia exato para Boxplot real
        df_todas_datas = df_cadastros_unicos['data_cadastro_dt'].dropna().dt.date.value_counts().reset_index()
        df_todas_datas.columns = ['data', 'cadastros']
        df_todas_datas['data'] = pd.to_datetime(df_todas_datas['data'])
        
        map_dias_s = {0: '1-Seg', 1: '2-Ter', 2: '3-Qua', 3: '4-Qui', 4: '5-Sex', 5: '6-Sáb', 6: '7-Dom'}
        df_todas_datas['dia_semana'] = df_todas_datas['data'].dt.dayofweek.map(map_dias_s)
        
        map_meses = {1: '01-Jan', 2: '02-Fev', 3: '03-Mar', 4: '04-Abr', 5: '05-Mai', 6: '06-Jun', 7: '07-Jul', 8: '08-Ago', 9: '09-Set', 10: '10-Out', 11: '11-Nov', 12: '12-Dez'}
        df_todas_datas['mes_nome'] = df_todas_datas['data'].dt.month.map(map_meses)
        
        tipo_box = st.radio("Selecione o Eixo Base para o Boxplot:", ["Dias da Semana (Seg-Dom)", "Meses do Ano (Jan-Dez)"], horizontal=True, key="rad_box")
        
        if "Dias" in tipo_box:
            df_bx_plot = df_todas_datas.sort_values('dia_semana')
            fig_bx = px.box(df_bx_plot, x='dia_semana', y='cadastros', color='dia_semana', points="all", title="Boxplot: Extensão de Cadastros por Dia da Semana")
            fig_bx.update_layout(xaxis_title="Dias da Semana", yaxis_title="Cadastros Realizados no Dia")
        else:
            df_bx_plot = df_todas_datas.sort_values('mes_nome')
            fig_bx = px.box(df_bx_plot, x='mes_nome', y='cadastros', color='mes_nome', points="all", title="Boxplot: Extensão de Cadastros por Mês")
            fig_bx.update_layout(xaxis_title="Meses", yaxis_title="Cadastros Realizados no Dia")
            
        fig_bx.update_layout(showlegend=False)
        st.plotly_chart(fig_bx, use_container_width=True)

    with tab_bar:
        tipo_bar = st.radio("Selecione o Eixo Base para o Acumulado Total:", ["Dias da Semana (Seg-Dom)", "Meses do Ano (Jan-Dez)"], horizontal=True, key="rad_bar")
        
        if "Dias" in tipo_bar:
            df_bar_plot = df_todas_datas.groupby('dia_semana')['cadastros'].sum().reset_index().sort_values('dia_semana')
            fig_bar = px.bar(df_bar_plot, x='dia_semana', y='cadastros', text='cadastros', color='dia_semana', title="Soma Bruta Histórica (Por Dia da Semana)")
            fig_bar.update_layout(xaxis_title="Dias da Semana", yaxis_title="Cadastros Acumulados")
        else:
            df_bar_plot = df_todas_datas.groupby('mes_nome')['cadastros'].sum().reset_index().sort_values('mes_nome')
            fig_bar = px.bar(df_bar_plot, x='mes_nome', y='cadastros', text='cadastros', color='mes_nome', title="Soma Bruta Histórica (Por Mês)")
            fig_bar.update_layout(xaxis_title="Meses do Ano", yaxis_title="Cadastros Acumulados")
            
        fig_bar.update_traces(textposition='outside')
        fig_bar.update_layout(showlegend=False, yaxis=dict(range=[0, df_bar_plot['cadastros'].max() * 1.2]))
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()

    # 3. Exportação para Excel
    @st.cache_data
    def gerar_excel_estatistica_abc(tabelas_dict):
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for title, df_tbl in tabelas_dict.items():
                # Regras Excel: 31 caracteres max e sem barras "/"
                sheet_name = title.replace("/", "-")[:31]
                df_tbl.to_excel(writer, sheet_name=sheet_name, index=False)
        return output.getvalue()

    col_vazia, col_btn = st.columns([3, 1])
    with col_btn:
        st.download_button(
            label="📥 Exportar Matriz Analítica AB/C (Excel)",
            data=gerar_excel_estatistica_abc(tabelas_finais),
            file_name="matriz_estatistica_mensagens.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
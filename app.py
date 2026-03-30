import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import pingouin as pg

st.set_page_config(page_title="Análise Comportamental - Impact Hub", layout="wide")

st.title("📊 Dashboard de Ciências Comportamentais")
st.markdown("Análise de inclusão econômica e fortalecimento de MEIs.")

# --- FASE 1: UPLOAD E LIMPEZA ---
uploaded_file = st.sidebar.file_uploader("Envie a Tabela-mãe (CSV ou XLSX)", type=['csv', 'xlsx'])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        df_raw = pd.read_csv(uploaded_file)
    else:
        df_raw = pd.read_excel(uploaded_file)

    # Limpeza de nomes (Equivalente ao janitor::clean_names)
    df = df_raw.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('[^\w\s]', '', regex=True)

    # --- FASE 2: TRANSFORMAÇÃO (Lógica do seu R) ---
    with st.expander("Configurações de Processamento"):
        # Ajuste da Taxa de Adesão
        col_adesao = 'taxa_de_semanas_ativas_interacao_ao_menos_uma_vez_na_semana'
        if col_adesao in df.columns:
            df['taxa_adesao_num'] = df[col_adesao].astype(str).str.replace('%', '').str.replace(',', '.')
            df['taxa_adesao_num'] = pd.to_numeric(df['taxa_adesao_num'], errors='coerce')
            df['taxa_adesao_num'] = np.where(df['taxa_adesao_num'] > 1, df['taxa_adesao_num'] / 100, df['taxa_adesao_num'])

        # Presença Mentoria
        df['presenca_mentoria'] = df['presenca_em_quantas_mentorias'].replace(['Nenhuma', np.nan], '0')
        df['presenca_mentoria'] = pd.to_numeric(df['presenca_mentoria'], errors='coerce').fillna(0)

        # Criação das Flags de Grupos (G0 a G5)
        df['G0'] = np.where(df['grupo'] == "Grupo Controle", 1, 0)
        df['G1'] = np.where(df['grupo'] == "Grupo informativo/formal", 1, 0)
        df['G2'] = np.where(df['grupo'] == "Grupo Padrão/acolhedor", 1, 0)
        df['G3'] = np.where((df['G2'] == 1) & (df['presenca_mentoria'] > 0), 1, 0)
        
        # Subamostras (usando regex como no seu R)
        df['G4'] = np.where((df['G2'] == 1) & (df.get('subamostra_nao_convidadas_mentoria_rj_sp', '').astype(str).str.contains("Sim")), 1, 0)
        df['G5'] = np.where((df['G2'] == 1) & (df.get('subamostra_convidadas_mentoria_rj_sp', '').astype(str).str.contains("Sim")), 1, 0)

    # Mostrar contagem
    st.sidebar.subheader("Contagem por Grupo")
    contagem = {
        "G0 (Controle)": df['G0'].sum(),
        "G1 (Formal)": df['G1'].sum(),
        "G2 (Acolhedor)": df['G2'].sum(),
        "G3 (Mentoria)": df['G3'].sum()
    }
    st.sidebar.write(contagem)

    # --- FASE 3: ANÁLISE ESTATÍSTICA ---
    tab1, tab2, tab3 = st.tabs(["Estatísticas Descritivas", "Modelos de Regressão", "Visualizações"])

    with tab1:
        st.header("Descritivas por Grupo")
        # Criando o dataframe empilhado para comparação (como no seu R)
        conditions = [
            (df['G0'] == 1), (df['G1'] == 1), (df['G2'] == 1), 
            (df['G3'] == 1), (df['G4'] == 1), (df['G5'] == 1)
        ]
        choices = ["G0 (Controle)", "G1 (Formal)", "G2 (Acolhedor Total)", 
                   "G3 (Mentoria)", "G4 (Não Convidadas)", "G5 (Convidadas)"]
        
        # Nota: Como uma pessoa pode estar em mais de um grupo (ex: G2 e G3), 
        # criamos uma base longa para o plot
        df_plot_list = []
        for cond, label in zip(conditions, choices):
            temp = df[cond].copy()
            temp['grupo_comparacao'] = label
            df_plot_list.append(temp)
        df_plot = pd.concat(df_plot_list)

        resumo = df_plot.groupby('grupo_comparacao')['taxa_adesao_num'].agg(['count', 'mean', 'std', 'median']).reset_index()
        st.dataframe(resumo.style.format({'mean': '{:.2%}', 'std': '{:.2%}', 'median': '{:.2%}'}))

    with tab2:
        st.header("Análise de Hipóteses (Regressão Linear)")
        target_h = st.selectbox("Selecione a Hipótese para Testar", 
                                ["H1: G3 vs G0", "H2: G1 vs G0", "H3: G2 vs G1"])
        
        if target_h == "H1: G3 vs G0":
            df_h = df_plot[df_plot['grupo_comparacao'].isin(["G0 (Controle)", "G3 (Mentoria)"])]
        elif target_h == "H2: G1 vs G0":
            df_h = df_plot[df_plot['grupo_comparacao'].isin(["G0 (Controle)", "G1 (Formal)"])]
        else:
            df_h = df_plot[df_plot['grupo_comparacao'].isin(["G1 (Formal)", "G2 (Acolhedor Total)"])]

        # Regressão OLS
        model = smf.ols('taxa_adesao_num ~ grupo_comparacao', data=df_h).fit()
        st.text(model.summary().as_text())

        # Tamanho do Efeito (Cohen's d) usando Pingouin
        st.subheader("Tamanho do Efeito (Cohen's d)")
        g_names = df_h['grupo_comparacao'].unique()
        if len(g_names) == 2:
            d_val = pg.compute_effsize(df_h[df_h['grupo_comparacao'] == g_names[0]]['taxa_adesao_num'],
                                       df_h[df_h['grupo_comparacao'] == g_names[1]]['taxa_adesao_num'],
                                       eftype='cohen')
            st.metric("Cohen's d", round(d_val, 3))

    with tab3:
        st.header("Visualização de Impacto")
        fig = px.box(df_plot, x="grupo_comparacao", y="taxa_adesao_num", 
                     color="grupo_comparacao", points="all",
                     title="Impacto das Intervenções na Taxa de Adesão")
        fig.update_layout(yaxis_tickformat='.0%')
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Aguardando upload da tabela-mãe para iniciar as análises.")
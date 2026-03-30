import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import pingouin as pg
from scipy.stats import fisher_exact

import io
from fpdf import FPDF

# Configuração da Página
st.set_page_config(page_title="Relatório Ciências Comportamentais", layout="wide")
st.title("📊 Análise Comportamental - Inclusão Econômica")
st.markdown("Projeto Sebrae / CINCO / Impact Hub - Avaliação de Intervenções e Hábito Financeiro")

def gerar_excel_completo(df_plot, hipoteses, metricas):
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        # 1. Aba de Resumo Descritivo
        df_descritivo_full = []
        for nome_label, col_base in metricas.items():
            resumo = df_plot.groupby('grupo_comparacao')[col_base].agg(['count', 'mean', 'std', 'median']).reset_index()
            resumo['Métrica'] = nome_label
            df_descritivo_full.append(resumo)
        pd.concat(df_descritivo_full).to_excel(writer, sheet_name='Estatísticas_Descritivas', index=False)
        
        # 2. Aba de Regressões (Roda todas as H de uma vez para o arquivo)
        lista_regressoes = []
        for h_nome, (ref, teste) in hipoteses.items():
            tab, d, insight = testar_hipotese(df_plot, 'taxa_adesao_num', ref, teste)
            tab['Hipótese_Analizada'] = h_nome
            tab['Cohen_d'] = d
            lista_regressoes.append(tab)
        pd.concat(lista_regressoes).to_excel(writer, sheet_name='Resultados_Hipoteses', index=False)
        
        # 3. Aba de Dados Brutos
        df_plot.to_excel(writer, sheet_name='Dados_Base', index=False)
    return output.getvalue()

def gerar_pdf_relatorio(df_plot, hipoteses):
    pdf = FPDF()
    pdf.add_page()
    # Usar 'latin-1' ou 'UTF-8' dependendo da versão, fpdf2 suporta UTF-8 por padrão
    pdf.set_font("helvetica", "B", 16)
    pdf.cell(0, 10, "Sumario Executivo - Ciencias Comportamentais", ln=True, align='C')
    pdf.ln(10)

    for h_nome, (ref, teste) in hipoteses.items():
        tab, d, insight = testar_hipotese(df_plot, 'taxa_adesao_num', ref, teste)
        pdf.set_font("helvetica", "B", 12)
        pdf.cell(0, 10, h_nome.encode('latin-1', 'replace').decode('latin-1'), ln=True)
        pdf.set_font("helvetica", "", 10)
        
        # Limpar emojis e markdown para não quebrar o PDF
        txt_limpo = insight.replace("**", "").replace("✅", "").replace("⚠️", "").replace("⚖️", "")
        pdf.multi_cell(0, 5, txt_limpo.encode('latin-1', 'replace').decode('latin-1'))
        pdf.cell(0, 10, f"Cohen's d: {d:.3f}", ln=True)
        pdf.ln(5)
    
    return pdf.output()

def gerar_pdf_relatorio(df_plot, hipoteses):
    """Gera um PDF executivo com os principais insights."""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 10, "Relatorio Executivo - Ciencias Comportamentais", ln=True, align='C')
    pdf.set_font("Arial", "", 10)
    pdf.cell(0, 10, "Parceria: Sebrae / CINCO / Impact Hub", ln=True, align='C')
    pdf.ln(10)

    for h_nome, (ref, teste) in hipoteses.items():
        tab, d, insight = testar_hipotese(df_plot, 'taxa_adesao_num', ref, teste)
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, f"Analise: {h_nome}", ln=True)
        pdf.set_font("Arial", "", 10)
        
        # Limpa markdown do insight para o PDF
        txt_limpo = insight.replace("**", "").replace("✅", "").replace("⚠️", "").replace("⚖️", "")
        pdf.multi_cell(0, 5, txt_limpo)
        pdf.cell(0, 5, f"Forca do Efeito (Cohen's d): {d:.3f}", ln=True)
        pdf.ln(5)
        
    return pdf.output()

def check_password():
    """Retorna True se o usuário inseriu a senha correta."""
    if "password_correct" not in st.session_state:
        # Inicializa o estado da senha
        st.session_state["password_correct"] = False

    if st.session_state["password_correct"]:
        return True

    # Interface de login
    st.title("🔐 Acesso Restrito")
    password = st.text_input("Digite a senha do projeto:", type="password")
    
    if st.button("Entrar"):
        # Substitua 'suasenha123' pela senha que desejar
        if password == "hub_sebrae_2026": 
            st.session_state["password_correct"] = True
            st.rerun() # Reinicia para mostrar o app
        else:
            st.error("🚫 Senha incorreta.")
    
    return False

# SÓ EXECUTA O RESTANTE DO APP SE A SENHA FOR CORRETA
if not check_password():
    st.stop()  # Interrompe a execução aqui

# --- Daqui para baixo entra o seu código original de análise ---
st.success("Acesso autorizado!")

# --- FUNÇÕES AUXILIARES ---
def limpar_nomes_colunas(df):
    """Imita o janitor::clean_names do R, incluindo remoção de acentos e desduplicação."""
    # 1. Limpeza de texto
    df.columns = (
        df.columns.str.strip().str.lower()
        .str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')
        .str.replace(r'[^\w\s]', '', regex=True)
        .str.replace(r'\s+', '_', regex=True)
    )
    
    # 2. Desduplicação de nomes de colunas
    cols = pd.Series(df.columns)
    for dup in cols[cols.duplicated()].unique():
        # Adiciona _1, _2, etc., nas colunas repetidas
        cols[cols[cols == dup].index.values.tolist()] = [
            f"{dup}_{i+1}" if i != 0 else dup for i in range(sum(cols == dup))
        ]
    df.columns = cols
    
    return df

def testar_hipotese(df, var_dependente, grupo_ref, grupo_teste):
    """Roda Regressão OLS com tratamento de p-valor científico e casos marginais."""
    df_h = df[df['grupo_comparacao'].isin([grupo_ref, grupo_teste])].copy()
    
    # 1. Regressão e Limpeza
    formula = f"{var_dependente} ~ C(grupo_comparacao, Treatment(reference='{grupo_ref}'))"
    try:
        model = smf.ols(formula, data=df_h).fit()
        tabela_resultados = model.summary2().tables[1].reset_index()
        
        # Limpeza de nomes para exibição
        tabela_resultados['index'] = tabela_resultados['index'].str.replace(
            r"C\(grupo_comparacao, Treatment\(reference='.*'\)\)\[T\.", "Efeito: ", regex=True
        ).str.replace(r"\]", "", regex=True)
        tabela_resultados['index'] = tabela_resultados['index'].replace({"Intercept": f"Base ({grupo_ref})"})
        tabela_resultados.rename(columns={'index': 'Variável/Grupo'}, inplace=True)
        
        linha_teste = tabela_resultados[tabela_resultados['Variável/Grupo'] == f"Efeito: {grupo_teste}"]
        coef_val = linha_teste['Coef.'].values[0] if not linha_teste.empty else 0
        pval = linha_teste['P>|t|'].values[0] if not linha_teste.empty else 1
    except Exception as e:
        return pd.DataFrame({'Erro': [str(e)]}), np.nan, "Erro no processamento estatístico."

    # 2. Cohen's d
    g_teste_vals = df_h[df_h['grupo_comparacao'] == grupo_teste][var_dependente].dropna()
    g_ref_vals = df_h[df_h['grupo_comparacao'] == grupo_ref][var_dependente].dropna()
    d_val = pg.compute_effsize(g_teste_vals, g_ref_vals, eftype='cohen') if len(g_teste_vals) > 0 else np.nan

    # 3. Definição das Variáveis de Texto (Onde estava o erro)
    direcao = "aumentou" if coef_val > 0 else "reduziu"
    abs_coef = abs(coef_val) # <--- A definição que faltava
    
    # Formatação do p-valor (Científica se < 0.001)
    pval_formatado = f"{pval:.2e}" if pval < 0.001 else f"{pval:.3f}"
    
    if pval < 0.05:
        emoji, status = "✅", "é **estatisticamente significativa**"
    elif 0.05 <= pval < 0.10:
        emoji, status = "⚠️", "é **marginalmente significativa**"
    else:
        emoji, status = "⚖️", "**NÃO** é estatisticamente significativa"

    # 4. Montagem do Texto de Insight
    texto = f"{emoji} **O que os dados dizem:** A diferença entre os grupos {status} (p-valor = {pval_formatado}).\n\n"
    texto += f"A intervenção '{grupo_teste}' **{direcao}** o resultado em **{abs_coef:.1%}** (pontos percentuais) em relação ao grupo de referência."
    
    if pval < 0.10:
        # Magnitude do efeito
        if abs(d_val) < 0.2: mag = "muito pequena"
        elif abs(d_val) < 0.5: mag = "pequena"
        elif abs(d_val) < 0.8: mag = "média"
        else: mag = "grande"
        
        texto += f" O impacto prático deste efeito é considerado **{mag}** (Cohen's d = {d_val:.2f})."
        
        if 0.05 <= pval < 0.10:
            texto += "\n\n*Nota: O resultado é marginal (tendência). Pode indicar que o efeito existe, mas o tamanho da amostra (N) limita o poder estatístico.*"
    else:
        texto += " Como o p-valor é superior a 0.10, esta variação é tratada estatisticamente como ruído."

    return tabela_resultados, d_val, texto

# --- CORES PADRÃO ---
CORES_GRUPOS = {
    "G0 (Controle)": "#E0E0E0", "G1 (Formal)": "#B3CDE3", 
    "G2 (Acolhedor Total)": "#8C96C6", "G4 (Não Convidadas)": "#8856A7", 
    "G5 (Convidadas)": "#810F7C", "G3 (Mentoria/Quem Foi)": "#000000",
    "G2 (Acolhedora)": "#009E73" # Usado no gráfico de tempo
}

# ==============================================================================
# FASE 1: UPLOAD E PREPARAÇÃO DA BASE
# ==============================================================================
uploaded_file = st.sidebar.file_uploader("📂 Envie a Tabela-mãe (CSV ou XLSX)", type=['csv', 'xlsx'])

if uploaded_file:
    with st.spinner('Lendo e limpando os dados...'):
        if uploaded_file.name.endswith('.csv'):
            df_bruto = pd.read_csv(uploaded_file)
        else:
            df_bruto = pd.read_excel(uploaded_file)

        df = limpar_nomes_colunas(df_bruto)

        # Transformações Numéricas
        colunas_taxa = {
            'taxa_adesao_num': 'taxa_de_semanas_ativas_interacao_ao_menos_uma_vez_na_semana',
            'taxa_transacoes_num': 'taxa_semana_ativas_registro_transacoes',
            'taxa_metas_num': 'taxa_semana_ativas_metas',
            'taxa_conjunta_num': 'taxa_semana_ativas_transacoes_e_metas'
        }

        for nova_col, col_antiga in colunas_taxa.items():
            if col_antiga in df.columns:
                df[nova_col] = df[col_antiga].astype(str).str.replace('%', '').str.replace(',', '.')
                df[nova_col] = pd.to_numeric(df[nova_col], errors='coerce')
                df[nova_col] = np.where(df[nova_col] > 1, df[nova_col] / 100, df[nova_col])

        # Presença Mentoria
        df['presenca_mentoria'] = df.get('presenca_em_quantas_mentorias', '0').replace(['Nenhuma', np.nan], '0')
        df['presenca_mentoria'] = pd.to_numeric(df['presenca_mentoria'], errors='coerce').fillna(0)

        # ==============================================================================
        # FASE 2: CRIAÇÃO DOS GRUPOS E EMPILHAMENTO (BIND_ROWS)
        # ==============================================================================
        df['G0'] = np.where(df.get('grupo', '') == "Grupo Controle", 1, 0)
        df['G1'] = np.where(df.get('grupo', '') == "Grupo informativo/formal", 1, 0)
        df['G2'] = np.where(df.get('grupo', '') == "Grupo Padrão/acolhedor", 1, 0)
        df['G3'] = np.where((df['G2'] == 1) & (df['presenca_mentoria'] > 0), 1, 0)
        
        # Subamostras com os nomes exatos processados
        serie_g4 = df.get('subamostra_nao_convidadas_mentoria_rjsp', pd.Series('', index=df.index)).astype(str)
        serie_g5 = df.get('subamostra_convidadas_mentoria_rjsp', pd.Series('', index=df.index)).astype(str)

        df['G4'] = np.where((df['G2'] == 1) & (serie_g4.str.contains("Sim", case=False, na=False)), 1, 0)
        df['G5'] = np.where((df['G2'] == 1) & (serie_g5.str.contains("Sim", case=False, na=False)), 1, 0)
        # Criar df_plot empilhado (permitindo sobreposição de grupos como no R)
        dfs_para_empilhar = []
        mapa_grupos = [
            ('G0', "G0 (Controle)"), ('G1', "G1 (Formal)"), ('G2', "G2 (Acolhedor Total)"),
            ('G4', "G4 (Não Convidadas)"), ('G5', "G5 (Convidadas)"), ('G3', "G3 (Mentoria/Quem Foi)")
        ]
        
        for flag, label in mapa_grupos:
            df_temp = df[df[flag] == 1].copy()
            df_temp['grupo_comparacao'] = label
            dfs_para_empilhar.append(df_temp)
            
        df_plot = pd.concat(dfs_para_empilhar)
        df_plot['grupo_comparacao'] = pd.Categorical(df_plot['grupo_comparacao'], categories=[m[1] for m in mapa_grupos], ordered=True)

        hipoteses = {
                "H1: Plano Completo (G3 vs G0)": ("G0 (Controle)", "G3 (Mentoria/Quem Foi)"),
                "H2a: Informativo vs Controle (G1 vs G0)": ("G0 (Controle)", "G1 (Formal)"),
                "H2b: Acolhedor vs Controle (G2 vs G0)": ("G0 (Controle)", "G2 (Acolhedor Total)"),
                "H3: Linguagem (G2 vs G1)": ("G1 (Formal)", "G2 (Acolhedor Total)"),
                "H4: Poder da Mentoria (G3 vs G4)": ("G4 (Não Convidadas)", "G3 (Mentoria/Quem Foi)"),
                "H5: Poder do Convite (G5 vs G4)": ("G4 (Não Convidadas)", "G5 (Convidadas)")
            }

        st.sidebar.success("✅ Base processada com sucesso!")
        st.sidebar.subheader("Contagem de Flags")
        st.sidebar.write({m[1]: df[m[0]].sum() for m in mapa_grupos})

        # --- SEÇÃO DE EXPORTAÇÃO NA BARRA LATERAL ---
        st.sidebar.divider()
        st.sidebar.header("📤 Exportar Resultados")
        
        # Preparar dados para exportação
        # (Garante que as métricas e hipóteses estejam acessíveis)
        metricas_exp = {
            "Taxa Adesão": "taxa_adesao_num",
            "Taxa Transações": "taxa_transacoes_num",
            "Taxa Metas": "taxa_metas_num",
            "Taxa Conjunta": "taxa_conjunta_num"
        }
        
        # Botão Excel
        try:
            dados_excel = gerar_excel_completo(df_plot, hipoteses, metricas_exp)
            st.sidebar.download_button(
                label="📊 Baixar Relatório Excel (.xlsx)",
                data=dados_excel,
                file_name="Relatorio_Comportamental_Completo.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
        except Exception as e:
            st.sidebar.error(f"Erro ao gerar Excel: {e}") # Mostra o erro real aqui

        # Botão PDF
        try:
            # Importante: para PDF, o output de bytes pode variar conforme a versão da fpdf2
            dados_pdf = gerar_pdf_relatorio(df_plot, hipoteses)
            st.sidebar.download_button(
                label="📄 Baixar Resumo PDF (.pdf)",
                data=bytes(dados_pdf) if isinstance(dados_pdf, bytearray) else dados_pdf,
                file_name="Sumario_Executivo_Comportamental.pdf",
                mime="application/pdf"
            )
        except Exception as e:
            st.sidebar.error(f"Erro ao gerar PDF: {e}") # Mostra o erro real aqui

        # Mostrar colunas para debug, caso necessário
        with st.sidebar.expander("🛠️ Ver Colunas Processadas"):
            st.write(df.columns.tolist())

        # --- NAVEGAÇÃO POR ABAS ATUALIZADA ---
        tab0, tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "0. Descritivas dos Grupos",
            "1. Checagem de Balanceamento", 
            "2. Hipóteses (Adesão Geral)", 
            "3. Hipóteses (Transações vs Metas)", 
            "4. Comportamento Avançado", 
            "5. Deep Dive: Tom de Voz (G1 vs G2)"
        ])

        # --- ABA 0: ESTATÍSTICA DESCRITIVA ---
        with tab0:
            st.header("📊 Estatística Descritiva por Grupo")
            st.markdown("""
            Esta visão geral apresenta o comportamento médio de cada grupo experimental. 
            Utilize esta aba para validar o tamanho das amostras e comparar as tendências centrais.
            """)

            # 1. Definir as métricas principais
            metricas_analise = {
                "Taxa de Adesão (Semanas Ativas)": "taxa_adesao_num",
                "Taxa Ativas: Transações": "taxa_transacoes_num",
                "Taxa Ativas: Metas": "taxa_metas_num",
                "Taxa Ativas: Transações + Metas": "taxa_conjunta_num", # <- Novo atributo
                "Qtd. de Transações": "quantidade_de_transacoes",
                "Qtd. de Metas": "quantidade_de_metas_registradas",
                "Visualizações do Painel": "quantidade_de_visualizacoes_painel"
            }

            # 2. Seleção da métrica para detalhamento (opcional, para facilitar a leitura)
            metrica_selecionada = st.selectbox("Selecione a métrica para ver o resumo estatístico:", list(metricas_analise.keys()))
            coluna_alvo = metricas_analise[metrica_selecionada]

            # 3. Cálculo do Resumo
            resumo_estatistico = df_plot.groupby('grupo_comparacao')[coluna_alvo].agg(
                N='count',
                Média='mean',
                Mediana='median',
                Desvio_Padrão='std'
            ).reset_index()

            # 4. Formatação condicional (Porcentagem vs Absoluto)
            is_taxa = "Taxa" in metrica_selecionada
            
            format_dict = {
                'Média': '{:.2%}' if is_taxa else '{:.2f}',
                'Mediana': '{:.2%}' if is_taxa else '{:.2f}',
                'Desvio_Padrão': '{:.2%}' if is_taxa else '{:.2f}',
                'N': '{:,.0f}'
            }

            st.subheader(f"Tabela Resumo: {metrica_selecionada}")
            st.dataframe(resumo_estatistico.style.format(format_dict), use_container_width=True)

            # 5. Explicação dos resultados da Aba 0
            st.info(f"""
            **Como ler esta tabela:**
            * **N**: Indica quantas empreendedoras compõem este grupo
            * **Média**: É o valor central do grupo. {"Valores mais próximos de 100% indicam maior engajamento semanal." if is_taxa else "Valores mais altos indicam maior volume de uso da funcionalidade."}
            * **Desvio Padrão**: Mede a dispersão. Se for muito alto em relação à média, significa que o grupo é heterogêneo (algumas usam muito, outras quase nada).
            """)

            # Bônus: Um gráfico de barras comparativo rápido
            fig_bar_desc = px.bar(
                resumo_estatistico, 
                x='grupo_comparacao', 
                y='Média', 
                color='grupo_comparacao',
                color_discrete_map=CORES_GRUPOS,
                text_auto='.2%' if is_taxa else '.2s',
                title=f"Comparativo de Médias: {metrica_selecionada}"
            )
            fig_bar_desc.update_layout(showlegend=False)
            if is_taxa:
                fig_bar_desc.update_layout(yaxis_tickformat='.0%')
            st.plotly_chart(fig_bar_desc, use_container_width=True)

        # --- ABA 1: BALANCEAMENTO ---
        # --- ABA 1: BALANCEAMENTO ---
        with tab1:
            st.header("Checagem de Balanceamento (Qui-Quadrado)")
            
            vars_balanceamento = {
                "Nível de Confiança": "nivel_de_confianca",
                "Frequência de Registro": "frequencia_de_registro",
                "Tempo de Negócio": "e_o_seu_negocio_existe_ha_quanto_tempo_" # <- Adicionado o '_' no final
            }
            
            col1, col2 = st.columns(2)
            for i, (titulo, var) in enumerate(vars_balanceamento.items()):
                if var in df.columns:
                    target_col = col1 if i % 2 == 0 else col2
                    with target_col:
                        st.subheader(titulo)
                        tabela = pd.crosstab(df['grupo'], df[var])
                        st.dataframe(tabela)
                        
                        try:
                            # Teste Qui-Quadrado usando Pingouin
                            expected, observed, stats = pg.chi2_independence(df, x='grupo', y=var)
                            pval = stats[stats['test'] == 'pearson']['pval'].values[0]
                            st.info(f"**P-valor:** {pval:.4f} (Se > 0.05, grupos estão balanceados)")
                        except Exception as e:
                            st.warning(f"Não foi possível calcular o qui-quadrado: {e}")

        # --- ABA 2: HIPÓTESES (ADESÃO GERAL) ---
        with tab2:
            st.header("Avaliação Consolidada: Taxa de Adesão Geral")
            
            fig_adesao = px.box(df_plot, x="grupo_comparacao", y="taxa_adesao_num", color="grupo_comparacao",
                                color_discrete_map=CORES_GRUPOS, points="all",
                                title="Impacto das Intervenções na Criação de Hábito Financeiro")
            fig_adesao.update_layout(yaxis_tickformat='.0%', showlegend=False)
            st.plotly_chart(fig_adesao, use_container_width=True)

            st.subheader("Testes Estatísticos")
            
            h_sel = st.selectbox("Selecione a Hipótese:", list(hipoteses.keys()), key='h_adesao')
            ref, teste = hipoteses[h_sel]
            
            tabela_res, d_cohen, texto_insight = testar_hipotese(df_plot, 'taxa_adesao_num', ref, teste)
            
            # Exibe o insight (a cor da caixa muda se for marginal na função)
            if "marginalmente" in texto_insight:
                st.warning(texto_insight)
            else:
                st.info(texto_insight)
            
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.dataframe(tabela_res.style.format({
                    'Coef.': '{:.4f}', 'Std.Err.': '{:.4f}', 
                    't': '{:.2f}', 
                    'P>|t|': lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}",
                    '[0.025': '{:.4f}', '0.975]': '{:.4f}'
                }), use_container_width=True)
            with col_b:
                st.metric("Tamanho do Efeito (Cohen's d)", f"{d_cohen:.3f}" if pd.notna(d_cohen) else "N/A")

        # --- ABA 3: TRANSAÇÕES VS METAS ---
        with tab3:
            st.header("Avaliação de Hipóteses Separadas")
            
            tipo_analise = st.radio(
                "Selecione o Foco da Análise:", 
                ["Trabalho Operacional (Apenas Transações)", "Retenção Estratégica (Apenas Metas)", "Ação Conjunta (Transações + Metas)"], 
                key="radio_foco_analise"
            )
            
            if "Apenas Transações" in tipo_analise:
                var_alvo = 'taxa_transacoes_num'
            elif "Apenas Metas" in tipo_analise:
                var_alvo = 'taxa_metas_num'
            else:
                var_alvo = 'taxa_conjunta_num'
            
            fig_sep = px.box(df_plot, x="grupo_comparacao", y=var_alvo, color="grupo_comparacao",
                             color_discrete_map=CORES_GRUPOS, points="all",
                             title=f"Impacto na {tipo_analise}")
            fig_sep.update_layout(yaxis_tickformat='.0%', showlegend=False)
            st.plotly_chart(fig_sep, use_container_width=True)

            h_sel_sep = st.selectbox("Selecione a Hipótese para analisar:", list(hipoteses.keys()), key='sel_hipotese_aba3')
            ref_sep, teste_sep = hipoteses[h_sel_sep]
            
            res_tab_sep, d_cohen_sep, texto_insight_sep = testar_hipotese(df_plot, var_alvo, ref_sep, teste_sep)
            
            if "marginalmente" in texto_insight_sep:
                st.warning(texto_insight_sep)
            else:
                st.info(texto_insight_sep)
            
            col_c, col_d = st.columns([2, 1])
            with col_c:
                st.dataframe(res_tab_sep.style.format({
                    'Coef.': '{:.4f}', 'Std.Err.': '{:.4f}', 't': '{:.2f}', 
                    'P>|t|': lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}",
                    '[0.025': '{:.4f}', '0.975]': '{:.4f}'
                }), use_container_width=True)   
                
            with col_d:
                st.metric("Tamanho do Efeito (Cohen's d)", f"{d_cohen_sep:.3f}" if pd.notna(d_cohen_sep) else "N/A")

        # --- ABA 4: COMPORTAMENTO AVANÇADO ---
        with tab4:
            st.header("Análises Comportamentais Avançadas")
            st.markdown("""
            Esta aba permite explorar o comportamento das empreendedoras além das médias simples, 
            identificando clusters de uso e profundidade de adoção.
            """)

            # --- 1. EXPLORADOR 3D (FOCO EM DISTINÇÃO VISUAL) ---
            st.subheader("🛠️ Explorador Comportamental 3D (Alto Contraste)")
            st.info("""
            **Dica de Leitura:** Procure por agrupamentos de cores isoladas. A distinção de cores foi aumentada 
            para facilitar a identificação de clusters comportamentais específicos por grupo de intervenção.
            """)

            # 1. CRIANDO HIERARQUIA DE GRUPOS EXCLUSIVOS (Mantido para unicidade de pontos)
            condicoes_unicas = [
                (df['G3'] == 1),
                (df['G5'] == 1),
                (df['G4'] == 1),
                (df['G1'] == 1),
                (df['G0'] == 1)
            ]
            labels_unicos = [
                "G3 (Mentoria)", 
                "G5 (Convidadas)", 
                "G4 (Não Convidadas)", 
                "G1 (Formal)", 
                "G0 (Controle)"
            ]
            df['grupo_3d_exclusivo'] = np.select(condicoes_unicas, labels_unicos, default='G2 (Outras fora do G3, G4 e G5)')

            
            PALETA_DISTINTA_3D = {
                "G3 (Mentoria)": "#E41A1C",       # Vermelho (Alto impacto)
                "G5 (Convidadas)": "#377EB8",    # Azul (Forte)
                "G4 (Não Convidadas)": "#FF7F00",# Laranja (Vibrante)
                "G1 (Formal)": "#4DAF4A",        # Verde (Comparação)
                "G0 (Controle)": "#984EA3",       # Roxo (Controle)
                "G2 (Outras fora do G3, G4 e G5)": "#AAAAAA"              # Cinza (Baseline)
            }

            # 2. SELETORES DE EIXO (Mantidos com chaves únicas)
            opcoes_3d = {
                "Taxa Adesão (Semanas Ativas)": "taxa_adesao_num",
                "Taxa Ativas Transações": "taxa_transacoes_num",
                "Taxa Ativas Metas": "taxa_metas_num",
                "Taxa Ativas Transações + Metas": "taxa_conjunta_num",
                "Qtd. Total Transações": "quantidade_de_transacoes",
                "Qtd. Total Metas Registradas": "quantidade_de_metas_registradas",
                "Qtd. Total Visualizações Painel": "quantidade_de_visualizacoes_painel"
            }

            c1, c2, c3 = st.columns(3)
            with c1: x_lab = st.selectbox("Eixo X:", list(opcoes_3d.keys()), index=0, key="x_3d_distinto")
            with c2: y_lab = st.selectbox("Eixo Y:", list(opcoes_3d.keys()), index=4, key="y_3d_distinto")
            with c3: z_lab = st.selectbox("Eixo Z:", list(opcoes_3d.keys()), index=6, key="z_3d_distinto")

            # 3. CONSTRUÇÃO DO PLOT (AGORA COM A NOVA PALETA DISTINTA)
            fig_3d = px.scatter_3d(
                df, # Usa o DF original para unicidade de pontos
                x=opcoes_3d[x_lab], 
                y=opcoes_3d[y_lab], 
                z=opcoes_3d[z_lab],
                color='grupo_3d_exclusivo', 
                color_discrete_map=PALETA_DISTINTA_3D, # <-- Usando a nova paleta aqui
                opacity=0.75, # Levemente mais opaco para melhor distinção
                hover_data=['id', 'grupo'],
                title=f"Espaço Comportamental Real (N={len(df)} | Alto Contraste)"
            )

            # Melhoria visual dos pontos: adiciona bordas para separação
            fig_3d.update_traces(marker=dict(size=6, line=dict(width=1, color='DarkSlateGrey')))
            fig_3d.update_layout(margin=dict(l=0, r=0, b=0, t=30))
            
            # Formatação condicional de taxas
            if "Taxa" in x_lab: fig_3d.update_layout(scene=dict(xaxis=dict(tickformat=".0%")))
            if "Taxa" in y_lab: fig_3d.update_layout(scene=dict(yaxis=dict(tickformat=".0%")))
            if "Taxa" in z_lab: fig_3d.update_layout(scene=dict(zaxis=dict(tickformat=".0%")))

            st.plotly_chart(fig_3d, use_container_width=True)
            
            st.success(f"📌 Total de empreendedoras únicas plotadas: {len(df)}")

            st.divider()

            # --- 2. MATRIZ DE CORRELAÇÃO (HEATMAP INTERATIVO) ---
            st.subheader("🌡️ Matriz de Correlação Comportamental")
            
            st.info("""
            **Como ler o Heatmap:** A cor indica a força da relação entre duas métricas. 
            Valores próximos de **1.0 (Amarelo/Quente)** indicam que quando uma métrica sobe, a outra também sobe. 
            Valores próximos de **0 (Roxo/Frio)** indicam que não há relação clara entre elas.
            """)

            # 1. Lista de métricas numéricas para correlação
            opcoes_corr = {
                "Taxa Adesão": "taxa_adesao_num",
                "Taxa Transações": "taxa_transacoes_num",
                "Taxa Metas": "taxa_metas_num",
                "Taxa Conjunta": "taxa_conjunta_num",
                "Qtd. Transações": "quantidade_de_transacoes",
                "Qtd. Metas": "quantidade_de_metas_registradas",
                "Uso do Painel": "quantidade_de_visualizacoes_painel",
                "Presença Mentoria": "presenca_mentoria"
            }

            # 2. Multiselect para o usuário escolher o que correlacionar
            metricas_selecionadas = st.multiselect(
                "Selecione as métricas para a matriz de correlação:",
                options=list(opcoes_corr.keys()),
                default=list(opcoes_corr.keys())[:5], # Começa com as 5 primeiras
                key="corr_multiselect"
            )

            if len(metricas_selecionadas) > 1:
                # Mapeia os nomes amigáveis para os nomes das colunas
                cols_corr = [opcoes_corr[m] for m in metricas_selecionadas]
                
                # Calcula a matriz de correlação de Pearson
                # Usamos o df original para evitar duplicatas de usuárias
                matriz_corr = df[cols_corr].corr(method='pearson')

                # 3. Criação do Heatmap com Plotly
                fig_heat = px.imshow(
                    matriz_corr,
                    text_auto=".2f", # Mostra os valores dentro das células
                    aspect="auto",
                    color_continuous_scale='Viridis', # Escala de cores profissional
                    labels=dict(color="Correlação"),
                    x=metricas_selecionadas,
                    y=metricas_selecionadas,
                    title="Correlação de Pearson entre Métricas Selecionadas"
                )

                fig_heat.update_layout(margin=dict(l=0, r=0, b=0, t=40))
                st.plotly_chart(fig_heat, use_container_width=True)

                with st.expander("🔍 Interpretando as Correlações"):
                    st.markdown("""
                    * **Correlação Alta (> 0.7)**: Indica um vínculo forte. Se o 'Uso do Painel' tem alta correlação com 'Taxa de Metas', as intervenções que levam ao painel são cruciais para o hábito estratégico.
                    * **Correlação Baixa (< 0.3)**: Indica que os comportamentos são independentes.
                    * **Diagonal Principal**: Sempre será 1.0, pois representa a métrica correlacionada com ela mesma.
                    """)
            else:
                st.warning("Selecione pelo menos duas métricas para gerar a matriz de correlação.")

            st.divider()



            # --- 2. CURVA DE SOBREVIVÊNCIA DO HÁBITO (ATUALIZADA COM TODOS OS GRUPOS) ---
            st.subheader("📈 Curva de Sobrevivência do Hábito")
            
            # 1. CRIANDO A HIERARQUIA PARA O GRÁFICO DE LINHAS
            # Note a lógica para o G2 (Outras): estar no G2 mas NÃO estar nos subgrupos G3, G4 ou G5
            condicoes_surv = [
                (df['G3'] == 1),
                (df['G5'] == 1),
                (df['G4'] == 1),
                ((df['G2'] == 1) & (df['G3'] == 0) & (df['G4'] == 0) & (df['G5'] == 0)),
                (df['G1'] == 1),
                (df['G0'] == 1)
            ]
            
            labels_surv = [
                "G3 (Mentoria)", 
                "G5 (Convidadas)", 
                "G4 (Não Convidadas)", 
                "G2 (Outras/Acolhedor)", 
                "G1 (Formal)", 
                "G0 (Controle)"
            ]
            
            df['grupo_survival'] = np.select(condicoes_surv, labels_surv, default='IGNORAR')

            # 2. FILTRAGEM E PREPARAÇÃO (TIDY DATA)
            df_deep = df[df['grupo_survival'] != 'IGNORAR'].copy()
            cols_sem = [c for c in df_deep.columns if 'interacoes_semana_' in c]
            
            if cols_sem:
                # Transformar para formato longo (long format)
                df_t = df_deep.melt(
                    id_vars=['grupo_survival'], 
                    value_vars=cols_sem, 
                    var_name='semana', 
                    value_name='interacoes'
                )
                
                # Extrair o número da semana
                df_t['sem_num'] = df_t['semana'].str.extract(r'(\d+)').astype(int)
                
                # Calcular a taxa de retenção (% de ativas com interação > 0)
                df_t_agg = df_t.groupby(['grupo_survival', 'sem_num'])['interacoes'].apply(
                    lambda x: (x > 0).mean()
                ).reset_index()

                # 3. DEFINIÇÃO DE CORES (Mantendo a paleta distinta do 3D para consistência)
                PALETA_SURVIVAL = {
                    "G3 (Mentoria)": "#E41A1C",       # Vermelho
                    "G5 (Convidadas)": "#377EB8",    # Azul
                    "G4 (Não Convidadas)": "#FF7F00",# Laranja
                    "G2 (Outras/Acolhedor)": "#FFFF33", # Amarelo (ou outra cor distinta)
                    "G1 (Formal)": "#4DAF4A",        # Verde
                    "G0 (Controle)": "#984EA3"        # Roxo
                }

                # 4. PLOTAGEM DA CURVA
                fig_t = px.line(
                    df_t_agg, 
                    x="sem_num", 
                    y="interacoes", 
                    color="grupo_survival", 
                    markers=True,
                    color_discrete_map=PALETA_SURVIVAL,
                    title="Sobrevivência do Hábito: Proporção de Usuárias Ativas por Semana"
                )
                
                fig_t.update_layout(
                    yaxis_tickformat='.0%', 
                    yaxis_title="% de Empreendedoras Ativas", 
                    xaxis_title="Semana de Participação",
                    legend_title="Grupos:",
                    hovermode="x unified"
                )
                
                # Ajuste para mostrar todas as semanas no eixo X
                fig_t.update_xaxes(dtick=1)
                
                st.plotly_chart(fig_t, use_container_width=True)
                

            else:
                st.warning("Atenção: Não foram encontradas colunas de interações semanais na planilha.")

            st.divider()

            # --- 3. MATURIDADE E CONVERSÃO ---
            col_eco1, col_eco2 = st.columns(2)
            
            with col_eco1:
                st.subheader("💎 Maturidade de Gestão")
                resumo_eco = df_plot.groupby('grupo_comparacao')[['quantidade_de_transacoes', 'quantidade_de_visualizacoes_painel']].mean().reset_index()
                resumo_eco['painel_aj_5x'] = resumo_eco['quantidade_de_visualizacoes_painel'] * 5
                
                fig_eco = go.Figure()
                fig_eco.add_trace(go.Bar(x=resumo_eco['grupo_comparacao'], y=resumo_eco['quantidade_de_transacoes'], name="Transações", marker_color="#B3CDE3"))
                fig_eco.add_trace(go.Bar(x=resumo_eco['grupo_comparacao'], y=resumo_eco['painel_aj_5x'], name="Uso Painel (Ajustado 5x)", marker_color="#8856A7"))
                fig_eco.update_layout(barmode='group', title="Média de Ações por Usuária", margin=dict(t=40, b=0))
                st.plotly_chart(fig_eco, use_container_width=True)

            with col_eco2:
                st.subheader("🎯 Taxa de Conversão Real")
                df_plot['fez_algo'] = (df_plot['quantidade_de_transacoes'] > 0).astype(int)
                df_plot['viu_algo'] = (df_plot['quantidade_de_visualizacoes_painel'] > 0).astype(int)
                resumo_conv = df_plot.groupby('grupo_comparacao')[['fez_algo', 'viu_algo']].mean().reset_index()
                
                fig_conv = go.Figure()
                fig_conv.add_trace(go.Bar(x=resumo_conv['grupo_comparacao'], y=resumo_conv['fez_algo'], name="Transações", marker_color="#B3CDE3"))
                fig_conv.add_trace(go.Bar(x=resumo_conv['grupo_comparacao'], y=resumo_conv['viu_algo'], name="Painel", marker_color="#8856A7"))
                fig_conv.update_layout(barmode='group', yaxis_tickformat='.0%', title="% da Base que Adotou", margin=dict(t=40, b=0))
                st.plotly_chart(fig_conv, use_container_width=True)

        # --- ABA 5: INVESTIGAÇÃO TOM DE VOZ ---
        with tab5:
            st.header("Investigação: O Efeito do Tom de Voz (G1 vs G2)")
            
            df_qualidade = df[(df['G1'] == 1) | (df['G2'] == 1)].copy()
            df_qualidade['grupo_comparacao'] = np.where(df_qualidade['G1'] == 1, "G1 (Apenas IA Formal)", "G2 (Apenas IA Acolhedora)")
            
            st.subheader("Testes de Quantidade de Interações (Regressão OLS)")
            alvo_voz = st.radio("Métrica:", ["quantidade_de_transacoes", "quantidade_de_visualizacoes_painel"])
            
            tabela_voz, d_voz, texto_voz = testar_hipotese(df_qualidade, alvo_voz, "G1 (Apenas IA Formal)", "G2 (Apenas IA Acolhedora)")
            
            if "marginalmente" in texto_voz:
                st.warning(texto_voz)
            else:
                st.info(texto_voz)
            
            col_v1, col_v2 = st.columns([2, 1])
            with col_v1:
                st.dataframe(tabela_voz.style.format({
                    'Coef.': '{:.4f}', 'Std.Err.': '{:.4f}', 't': '{:.2f}',
                    'P>|t|': lambda x: f"{x:.2e}" if x < 0.001 else f"{x:.4f}"
                }), use_container_width=True)
            with col_v2:
                st.metric("Cohen's d", f"{d_voz:.3f}" if pd.notna(d_voz) else "N/A")
                
            st.divider()
            
            st.subheader("Testes de Proporção / Conversão (Qui-Quadrado)")
            df_qualidade['fez_transacao'] = (df_qualidade['quantidade_de_transacoes'] > 0).astype(int)
            df_qualidade['viu_painel'] = (df_qualidade['quantidade_de_visualizacoes_painel'] > 0).astype(int)
            
            flag_voz = st.radio("Analisar Conversão em:", ["fez_transacao", "viu_painel"])
            
            try:
                exp, obs, stats_voz = pg.chi2_independence(df_qualidade, x='grupo_comparacao', y=flag_voz)
                st.write("**Tabela de Contingência (Observado):**")
                st.dataframe(obs)
                
                pval_voz = stats_voz[stats_voz['test'] == 'pearson']['pval'].values[0]
                cramer_voz = stats_voz[stats_voz['test'] == 'pearson']['cramer'].values[0]
                
                tabela_2x2 = pd.crosstab(df_qualidade['grupo_comparacao'], df_qualidade[flag_voz])
                odds, p_fisher = fisher_exact(tabela_2x2)
                
                # Formatação científica para o P-Valor do Qui-Quadrado se for muito baixo
                pval_voz_fmt = f"{pval_voz:.2e}" if pval_voz < 0.001 else f"{pval_voz:.4f}"
                
                col_s1, col_s2, col_s3 = st.columns(3)
                col_s1.metric("P-Valor (Qui-Quadrado)", pval_voz_fmt)
                col_s2.metric("Cramer's V", f"{cramer_voz:.3f}")
                col_s3.metric("Odds Ratio", f"{odds:.3f}")
                
            except Exception as e:
                st.error(f"Erro ao calcular estatísticas de conversão: {e}")
else:
    st.info("👈 Por favor, faça o upload do arquivo Tabela-mãe (.csv) no menu lateral para iniciar.")
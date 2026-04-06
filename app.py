import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import pingouin as pg
from scipy.stats import fisher_exact
import secrets  
import ssl
import io
import urllib.request
from fpdf import FPDF
from plotly.subplots import make_subplots
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import re
import random
from collections import Counter
from streamlit_echarts import st_echarts

@st.cache_data(show_spinner=False)
def gerar_nuvem_echarts_pt(texto, fonte="Arial", paleta=None):
    """Gera o dicionário nativo da nuvem de palavras para o ECharts."""
    if not texto.strip():
        return None

    texto = texto.lower()
    
    # Stopwords robustas em PT-BR baseadas na nossa limpeza anterior
    stopwords_pt = set([
        "o", "a", "os", "as", "um", "uma", "uns", "umas", "de", "do", "da", "dos", "das",
        "em", "no", "na", "nos", "nas", "para", "por", "com", "sem", "que", "se", "como",
        "mas", "ou", "e", "é", "não", "nao", "sim", "eu", "me", "meu", "minha", "nós", "nosso",
        "nossa", "ele", "ela", "eles", "elas", "foi", "ser", "ter", "tem", "tinha", "tudo",
        "nada", "muito", "pouco", "mais", "menos", "isso", "aquilo", "este", "esta", "esse",
        "essa", "já", "só", "também", "quando", "onde", "porque", "qual", "quais", "quem", "nan",
        "pra", "aos", "pelo", "pela", "pois", "sobre", "forma", "ainda", "estou", "tive", "fazer"
    ])

    # Regex que captura palavras incluindo acentuação (à-ÿ), mínimo de 3 letras
    palavras_limpas = re.findall(r'\b[a-zà-ÿ]{3,}\b', texto)
    palavras_filtradas = [w for w in palavras_limpas if w not in stopwords_pt]
    
    # Pega as 100 palavras mais comuns
    contagem = Counter(palavras_filtradas).most_common(100)

    # Se não houver paleta, usa os tons de roxo do projeto
    if not paleta:
        paleta = ["#8856A7", "#8C96C6", "#810F7C", "#8C6BB1", "#4D004B"]

    dados_palavras = []
    for palavra, freq in contagem:
        dados_palavras.append({
            "name": palavra.capitalize(),
            "value": freq,
            "textStyle": {
                "color": random.choice(paleta)
            }
        })

    # Dicionário que o ECharts entende instantaneamente
    opcoes_echarts = {
        "tooltip": {"show": True},
        "toolbox": {
            "feature": {
                "saveAsImage": {"show": True, "title": "Baixar Nuvem", "type": "png"}
            }
        },
        "series": [{
            "type": "wordCloud",
            "shape": "circle",
            "sizeRange": [15, 80],
            "rotationRange": [-45, 90],
            "rotationStep": 45,
            "gridSize": 8,
            "textStyle": {
                "fontFamily": fonte,
                "fontWeight": "bold"
            },
            "data": dados_palavras
        }]
    }

    return opcoes_echarts

# Configuração da Página
st.set_page_config(page_title="Relatório Ciências Comportamentais", layout="wide")
st.title("📊 Análise Comportamental - Liberdade Financeira")
st.markdown("Projeto Sebrae / CINCO / Impact Hub - Avaliação de Intervenções e Hábito Financeiro")

@st.cache_data(ttl=3600)
def load_data_from_drive():
    if "DRIVE_URL" not in st.secrets:
        st.error("❌ Chave 'DRIVE_URL' não encontrada no secrets.toml.")
        return None
        
    try:
        url = st.secrets["DRIVE_URL"]
        
        # 1. Cria o contexto que ignora a verificação de certificado
        context = ssl._create_unverified_context()
        
        # 2. Abre a URL usando esse contexto
        # O 'with' garante que a conexão seja fechada após a leitura
        with urllib.request.urlopen(url, context=context) as response:
            # 3. O pandas lê diretamente o objeto de resposta
            df_drive = pd.read_csv(response)
            
        if df_drive.empty:
            st.error("⚠️ A aba selecionada parece estar vazia. Verifique o GID.")
            return None
            
        return df_drive

    except Exception as e:
        st.error(f"⚠️ Erro ao acessar a planilha: {e}")
        return None

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
df_bruto = load_data_from_drive()

if df_bruto is not None:
        # 1. Limpeza inicial
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

        with st.sidebar:
            st.success("✅ Dados carregados via Google Drive")
            if st.button("🔄 Forçar Atualização dos Dados"):
                st.cache_data.clear()
                st.rerun()

        # --- NAVEGAÇÃO POR ABAS ATUALIZADA ---
        tab0, tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "0. Descritivas dos Grupos",
            "1. Checagem de Balanceamento", 
            "2. Hipóteses (Adesão)", 
            "3. Hipóteses (Transações/Metas)", 
            "4. Comportamento Avançado", 
            "5. Deep Dive (Tom de Voz)",
            "6. Questionários (Pré/Pós)" # <-- Nova Aba
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
            st.subheader("🛠️ Explorador Comportamental 3D")
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
                title=f"Espaço Comportamental Real (N={len(df)})"
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


            # --- 3. LABORATÓRIO DE SEGMENTAÇÃO (HETEROGENEIDADE) ---
            st.divider()
            st.subheader("🔍 Laboratório de Segmentação Qualitativa")
            
            st.info("""
            **Objetivo de Doutorado:** Esta ferramenta permite analisar a heterogeneidade do impacto. 
            Verifique se a intervenção (ex: Mentoria) teve efeitos diferentes dependendo do 
            perfil da empreendedora (ex: tempo de negócio ou nível de estresse).
            """)

            # 1. Mapeamento de Atributos Qualitativos (Eixo X ou Segmentos)
            atributos_qualitativos = {
                "Tempo de Negócio": "e_o_seu_negocio_existe_ha_quanto_tempo_",
                "Atividade Principal": "pra_gente_entender_melhor_o_seu_ramo_qual_e_a_atividade_principal_do_seu_negocio_",
                "Tamanho da Equipe": "voce_toca_o_negocio_sozinha_ou_tem_mais_gente_nesse_corre_com_voce",
                "Estresse Financeiro": "estresse_financeiro",
                "Escolaridade": "nivel_de_escolaridade",
                "Controle de Gastos (Auto-declarado)": "controle_de_gastos",
                "Separa PF da PJ": "separa_dinheiro_pessoal_do_negocio"
            }

            # 2. Mapeamento de Métricas de Engajamento (Eixo Y)
            metricas_engajamento = {
                "Taxa Adesão Geral": "taxa_adesao_num",
                "Taxa Transações": "taxa_transacoes_num",
                "Taxa Metas": "taxa_metas_num",
                "Taxa Conjunta": "taxa_conjunta_num",
                "Qtd. de Transações": "quantidade_de_transacoes",
                "Qtd. de Metas": "quantidade_de_metas_registradas",
                "Visualizações Painel": "quantidade_de_visualizacoes_painel"
            }

            col_lab1, col_lab2 = st.columns(2)
            with col_lab1:
                segmento_sel = st.selectbox("Selecione o Perfil Qualitativo (X):", list(atributos_qualitativos.keys()), key="seg_x")
            with col_lab2:
                metrica_sel = st.selectbox("Selecione a Métrica de Engajamento (Y):", list(metricas_engajamento.keys()), key="seg_y")

            # 3. Construção do Gráfico de Boxplot Facetado
            # Usamos df_plot para ter os nomes de grupos bonitos e as métricas convertidas
            
            var_x = atributos_qualitativos[segmento_sel]
            var_y = metricas_engajamento[metrica_sel]

            # Filtramos NaNs para não poluir o gráfico
            df_seg = df_plot.dropna(subset=[var_x, var_y])

            fig_seg = px.box(
                df_seg,
                x=var_x,
                y=var_y,
                color="grupo_comparacao",
                color_discrete_map=CORES_GRUPOS,
                facet_col="grupo_comparacao", # Cria um mini-gráfico para cada grupo
                facet_col_wrap=3, # Quebra a linha a cada 3 grupos
                points="outliers", # Mostra apenas os pontos fora da curva
                title=f"Distribuição de {metrica_sel} por {segmento_sel}",
                labels={var_x: segmento_sel, var_y: metrica_sel}
            )

            # Ajustes estéticos
            fig_seg.update_layout(showlegend=False, height=600)
            if "Taxa" in metrica_sel:
                fig_seg.update_layout(yaxis_tickformat='.0%')
            
            st.plotly_chart(fig_seg, use_container_width=True)

            with st.expander("💡 Como interpretar esta análise?"):
                st.markdown(f"""
                * **Consistência**: Se em todos os grupos as empreendedoras com 'Escolaridade Alta' performam melhor, a escolaridade é um preditor de sucesso independente da intervenção.
                * **Efeito Moderador**: Se no grupo 'Controle' o '{segmento_sel}' não faz diferença, mas no grupo 'Mentoria' um perfil específico decola, você descobriu que a mentoria é especialmente eficaz para aquele tipo de empreendedora.
                * **Dispersão**: Boxplots altos (caixas compridas) indicam que o comportamento do grupo é muito variado, sugerindo que outros fatores não mapeados estão influenciando o hábito.
                """)


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

        # --- ABA 6: ANÁLISE DE QUESTIONÁRIOS (PRÉ/PÓS) ---
        with tab6:
            st.header("Análise de Questionários: Pré vs Pós-Intervenção")
            st.markdown("""
            Esta seção avalia a evolução da percepção, conhecimento e confiança declarada pelas 
            empreendedoras, cruzando as respostas iniciais e finais.
            """)

            # ==============================================================================
            # DICIONÁRIO DE ENUNCIADOS
            # ==============================================================================
            dicionario_questoes = {
                1: "Nos últimos 12 meses, qual frase melhor descreve a comparação entre a renda total e os gastos na sua casa?",
                2: "O quanto as frases seguintes descrevem você ou sua situação:  [Preocupações com as despesas e compromissos financeiros são motivo de estresse na minha casa]",
                3: "O quanto as frases seguintes descrevem você ou sua situação:  [Por causa dos compromissos financeiros assumidos, o padrão de vida da minha casa foi bastante reduzido]",
                4: "O quanto as frases seguintes descrevem você ou sua situação:  [Estou apertada financeiramente]",
                5: "O quanto as frases seguintes descrevem você ou sua situação:  [Eu sei como me controlar para não gastar muito]",
                6: "O quanto as frases seguintes descrevem você ou sua situação:  [Eu sei como me obrigar a poupar]",
                7: "O quanto as frases seguintes descrevem você ou sua situação:  [Eu sei como me obrigar a cumprir minhas metas financeiras]",
                8: "Durante ou após o programa, você passou a anotar receitas e despesas com mais frequência?",
                9: "Hoje, com que frequência você registra suas finanças?",
                10: "Passou a estabelecer metas ou planos financeiros depois do programa?",
                11: "Como é o seu acompanhamento de metas (resposta aberta)",
                12: "Passou a separar o dinheiro do negócio do dinheiro pessoal (por exemplo, contas ou PIX diferentes)?",
                13: "Nível de confiança atual na sua gestão financeira"
            }

           
            cols_iniciais = ['nota_inicial_questionario'] + [f"q{i}_inicial" for i in range(1, 14)]
            cols_finais = ['nota_final_questionario'] + [f"q{i}_final" for i in range(1, 14)]
            todas_cols_num = cols_iniciais + cols_finais
            
            for col in todas_cols_num:
                if col in df_plot.columns:
                    if df_plot[col].dtype == 'object':
                        df_plot[col] = df_plot[col].astype(str).str.replace(',', '.', regex=False)
                    df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce')


            # --- 0. ESTATÍSTICA DESCRITIVA GERAL ---
            st.subheader("📋 0. Estatísticas Descritivas (Todas as Questões)")
            
            # Filtra apenas as colunas que existem no dataframe
            cols_validas = [c for c in todas_cols_num if c in df_plot.columns]
            
            if cols_validas:
                # Calcula Média, Mediana e Desvio Padrão
                df_desc = df_plot[cols_validas].agg(['mean', 'median', 'std']).T
                df_desc.columns = ['Média', 'Mediana', 'Desvio Padrão']
                df_desc = df_desc.round(2)
                
                with st.expander("Ver Tabela Completa de Estatísticas"):
                    st.dataframe(df_desc, use_container_width=True)
            else:
                st.info("Colunas numéricas não encontradas para gerar estatísticas.")

            st.divider()

            # --- 1. LABORATÓRIO DE EVOLUÇÃO (UNIFICADO) ---
            st.subheader("📊 1. Laboratório de Evolução (Pré vs Pós)")
            
            # --- CONTROLES DO LABORATÓRIO ---
            with st.expander("🛠️ Configurações do Gráfico", expanded=True):
                c1, c2, c3 = st.columns(3)
                with c1:
                    eixo_x_tipo = st.radio("Eixo X:", ["Por Grupos", "Por Questões"], key="lab_eixo")
                    agg_func = st.radio("Métrica Estatística:", ["Média", "Mediana"], key="lab_agg")
                with c2:
                    tipo_grafico = st.selectbox("Tipo de Gráfico:", ["Barras Agrupadas", "Linhas com Marcadores"], key="lab_tipo")
                    
                    if eixo_x_tipo == "Por Grupos":
                        opcoes_metrica = {"Nota Geral": "nota"}
                        for i in range(1, 14): opcoes_metrica[f"Questão {i}"] = f"q{i}"
                        metrica_id = st.selectbox("Métrica para Analisar:", list(opcoes_metrica.keys()))
                        metrica_prefixo = opcoes_metrica[metrica_id]
                    else:
                        grupos_lista = ["Todos os Grupos"] + list(df_plot['grupo_comparacao'].dropna().unique())
                        grupo_f_lab = st.selectbox("Filtrar por Grupo:", grupos_lista)
                with c3:
                    st.write("**Dica:**")
                    st.caption("A visão 'Por Grupos' foca em comparar as intervenções. A visão 'Por Questões' foca em entender a jornada de aprendizado em cada tema.")

            # --- PROCESSAMENTO DOS DADOS PARA O GRÁFICO ---
            if eixo_x_tipo == "Por Grupos":
                # CORREÇÃO: Tratamento específico para a coluna de Nota vs Questões
                if metrica_id == "Nota Geral":
                    col_ini = "nota_inicial_questionario"
                    col_fin = "nota_final_questionario"
                else:
                    col_ini = f"{metrica_prefixo}_inicial"
                    col_fin = f"{metrica_prefixo}_final"
                
                if agg_func == "Média":
                    df_res_lab = df_plot.groupby('grupo_comparacao')[[col_ini, col_fin]].mean().reset_index()
                else:
                    df_res_lab = df_plot.groupby('grupo_comparacao')[[col_ini, col_fin]].median().reset_index()
                
                x_data = df_res_lab['grupo_comparacao']
                y_ini = df_res_lab[col_ini]
                y_fin = df_res_lab[col_fin]
                titulo_lab = f"{tipo_grafico} ({agg_func}): {metrica_id} por Grupo"
            
            else: # Por Questões
                df_f_lab = df_plot if grupo_f_lab == "Todos os Grupos" else df_plot[df_plot['grupo_comparacao'] == grupo_f_lab]
                
                lista_qs = [f"q{i}" for i in range(1, 14)] + ["nota"]
                labels_qs = [f"Q{i}" for i in range(1, 14)] + ["Nota Geral"]
                
                y_ini, y_fin = [], []
                for q in lista_qs:
                    # CORREÇÃO: Tratamento específico também no loop
                    if q == "nota":
                        c_i, c_f = "nota_inicial_questionario", "nota_final_questionario"
                    else:
                        c_i, c_f = f"{q}_inicial", f"{q}_final"
                        
                    if agg_func == "Média":
                        y_ini.append(df_f_lab[c_i].mean())
                        y_fin.append(df_f_lab[c_f].mean())
                    else:
                        y_ini.append(df_f_lab[c_i].median())
                        y_fin.append(df_f_lab[c_f].median())
                
                x_data = labels_qs
                titulo_lab = f"{tipo_grafico} ({agg_func}): Jornada de Todas as Questões ({grupo_f_lab})"

            # --- CONSTRUÇÃO DO GRÁFICO ---
            fig_lab = go.Figure()

            use_secondary = True if (eixo_x_tipo == "Por Questões") else False
            if use_secondary:
                fig_lab = make_subplots(specs=[[{"secondary_y": True}]])

            if "Barras" in tipo_grafico:
                def add_bar_trace(y_vals, name, color):
                    if use_secondary:
                        fig_lab.add_trace(go.Bar(x=x_data[:-1], y=y_vals[:-1], name=f"{name} (Questões)", marker_color=color, legendgroup=name), secondary_y=False)
                        fig_lab.add_trace(go.Bar(x=[x_data[-1]], y=[y_vals[-1]], name=f"{name} (Nota)", marker_color=color, legendgroup=name, showlegend=False), secondary_y=True)
                    else:
                        fig_lab.add_trace(go.Bar(x=x_data, y=y_vals, name=name, marker_color=color))
                
                add_bar_trace(y_ini, "Pré-Intervenção", "#B3CDE3")
                add_bar_trace(y_fin, "Pós-Intervenção", "#8856A7")
                fig_lab.update_layout(barmode='group')

            else: # Linhas
                def add_line_trace(y_vals, name, color):
                    if use_secondary:
                        fig_lab.add_trace(go.Scatter(x=x_data[:-1], y=y_vals[:-1], name=f"{name} (Questões)", mode='lines+markers', line=dict(color=color, width=3), legendgroup=name), secondary_y=False)
                        fig_lab.add_trace(go.Scatter(x=[x_data[-1]], y=[y_vals[-1]], name=f"{name} (Nota)", mode='markers', marker=dict(color=color, size=12), legendgroup=name, showlegend=False), secondary_y=True)
                    else:
                        fig_lab.add_trace(go.Scatter(x=x_data, y=y_vals, name=name, mode='lines+markers', line=dict(color=color, width=3)))
                
                add_line_trace(y_ini, "Pré-Intervenção", "#B3CDE3")
                add_line_trace(y_fin, "Pós-Intervenção", "#8856A7")

            fig_lab.update_layout(title=titulo_lab, hovermode="x unified", height=500)
            if use_secondary:
                fig_lab.update_yaxes(title_text="Escala Questões (1-5)", secondary_y=False)
                fig_lab.update_yaxes(title_text="Escala Nota Geral", secondary_y=True, showgrid=False)
            else:
                fig_lab.update_yaxes(title_text=f"Valor ({agg_func})")

            st.plotly_chart(fig_lab, use_container_width=True)
            
            if eixo_x_tipo == "Por Grupos" and "Questão" in metrica_id:
                q_idx = int(metrica_id.split(" ")[1])
                st.caption(f"**Enunciado Selecionado:** {dicionario_questoes[q_idx]}")

            st.divider()

            # --- 2. BOXPLOT DINÂMICO (TODOS OS PONTOS) ---
            st.subheader("📦 2. Dispersão das Respostas (Boxplot)")
            
            df_melt_all = df_plot.melt(
                id_vars=['grupo_comparacao'], 
                value_vars=cols_validas,
                var_name='coluna_original',
                value_name='valor'
            ).dropna(subset=['valor'])
            
            df_melt_all['momento'] = np.where(df_melt_all['coluna_original'].str.contains('inicial'), 'Pré-Intervenção', 'Pós-Intervenção')
            
            def limpar_nome_metrica(nome):
                nome = nome.replace('_inicial', '').replace('_final', '')
                if 'nota' in nome: return 'Nota Geral'
                return nome.upper()
                
            df_melt_all['metrica'] = df_melt_all['coluna_original'].apply(limpar_nome_metrica)

            tipo_eixo_x = st.radio("Como você deseja agrupar o Eixo X?", 
                                   ["Por Grupos Experimentais", "Por Questões do Questionário"], horizontal=True)

            if tipo_eixo_x == "Por Grupos Experimentais":
                metricas_disponiveis = list(df_melt_all['metrica'].unique())
                metrica_sel = st.selectbox("Selecione a Métrica para analisar:", metricas_disponiveis)
                
                df_box = df_melt_all[df_melt_all['metrica'] == metrica_sel]
                
                fig_box = px.box(
                    df_box, 
                    x="grupo_comparacao", 
                    y="valor", 
                    color="momento",
                    color_discrete_map={"Pré-Intervenção": "#B3CDE3", "Pós-Intervenção": "#8856A7"},
                    points="all", 
                    title=f"Dispersão: {metrica_sel} (Por Grupo)",
                    category_orders={"momento": ["Pré-Intervenção", "Pós-Intervenção"]}
                )
                fig_box.update_layout(yaxis_title="Valor / Nota", xaxis_title="")
                fig_box.update_traces(marker=dict(size=4, opacity=0.6, line=dict(width=0)))
                
            else: # Por Questões (Eixo Y Duplo)
                grupos_disp = ["Todos os Grupos"] + list(df_melt_all['grupo_comparacao'].dropna().unique())
                grupo_sel = st.selectbox("Selecione o Grupo para analisar:", grupos_disp)
                
                df_box = df_melt_all if grupo_sel == "Todos os Grupos" else df_melt_all[df_melt_all['grupo_comparacao'] == grupo_sel]
                
                # Inicia a figura com eixo Y secundário habilitado
                fig_box = make_subplots(specs=[[{"secondary_y": True}]])
                
                df_nota = df_box[df_box['metrica'] == 'Nota Geral']
                df_q = df_box[df_box['metrica'] != 'Nota Geral']
                
                cores = {"Pré-Intervenção": "#B3CDE3", "Pós-Intervenção": "#8856A7"}

                # Traços 1: Questões normais (Q1 a Q13) -> Eixo Esquerdo
                for momento in ["Pré-Intervenção", "Pós-Intervenção"]:
                    df_q_m = df_q[df_q['momento'] == momento]
                    if not df_q_m.empty:
                        fig_box.add_trace(
                            go.Box(
                                x=df_q_m['metrica'], y=df_q_m['valor'], name=momento,
                                marker_color=cores[momento], boxpoints='all', jitter=0.4, pointpos=-1.8,
                                marker=dict(size=4, opacity=0.6, line=dict(width=0)), legendgroup=momento
                            ),
                            secondary_y=False,
                        )

                # Traços 2: Nota Geral -> Eixo Direito
                for momento in ["Pré-Intervenção", "Pós-Intervenção"]:
                    df_n_m = df_nota[df_nota['momento'] == momento]
                    if not df_n_m.empty:
                        fig_box.add_trace(
                            go.Box(
                                x=df_n_m['metrica'], y=df_n_m['valor'], name=momento,
                                marker_color=cores[momento], boxpoints='all', jitter=0.4, pointpos=-1.8,
                                marker=dict(size=4, opacity=0.6, line=dict(width=0)), legendgroup=momento, 
                                showlegend=False # Esconde para não duplicar a legenda
                            ),
                            secondary_y=True,
                        )

                # Ajustes de layout para suportar os eixos e ordenar o Eixo X
                ordem_x = ['Nota Geral'] + [f'Q{i}' for i in range(1, 14)]
                fig_box.update_layout(
                    boxmode='group', 
                    title=f"Dispersão de Todas as Questões ({grupo_sel})",
                    xaxis=dict(categoryorder='array', categoryarray=ordem_x)
                )
                
                # Configura os títulos dos eixos
                fig_box.update_yaxes(title_text="Valores (Q1 a Q13)", secondary_y=False)
                fig_box.update_yaxes(title_text="Nota Geral", secondary_y=True, showgrid=False)

            st.plotly_chart(fig_box, use_container_width=True)
            st.divider()

            # --- 3. FLUXO DE MUDANÇA (SANKEY DIAGRAM) ---
            st.subheader("🌊 3. Jornada da Resposta (Diagrama Sankey)")
            
            col_q, col_g = st.columns([2, 1])
            with col_q:
                questoes_opcoes = {f"Questão {i}": i for i in range(1, 14)}
                q_selecionada = st.selectbox("Selecione a Questão Qualitativa:", list(questoes_opcoes.keys()))
                q_num = questoes_opcoes[q_selecionada]
            
            with col_g:
                grupos_disponiveis = ["Todos os Grupos"] + list(df_plot['grupo_comparacao'].dropna().unique())
                g_sankey = st.selectbox("Filtrar Fluxo por Grupo:", grupos_disponiveis)

            st.markdown(f"**📝 Enunciado:** *{dicionario_questoes[q_num]}*")

            col_txt_ini = f"q{q_num}_texto_inicial"
            col_txt_fin = f"q{q_num}_texto_final"

            df_sankey = df_plot.dropna(subset=[col_txt_ini, col_txt_fin]).copy()
            df_sankey = df_sankey[
                (~df_sankey[col_txt_ini].astype(str).str.strip().isin(['-', '', 'nan'])) & 
                (~df_sankey[col_txt_fin].astype(str).str.strip().isin(['-', '', 'nan']))
            ]

            if g_sankey != "Todos os Grupos":
                df_sankey = df_sankey[df_sankey['grupo_comparacao'] == g_sankey]

            if not df_sankey.empty:
                fluxo = df_sankey.groupby([col_txt_ini, col_txt_fin]).size().reset_index(name='contagem')

                # 1. Define os nós originais
                nos_iniciais = [str(x) + " (Pré)" for x in fluxo[col_txt_ini].unique()]
                nos_finais = [str(x) + " (Pós)" for x in fluxo[col_txt_fin].unique()]
                todos_os_nos = nos_iniciais + nos_finais

                # 2. Aplica o estilo CSS (texto branco com contorno preto e negrito)
                labels_formatados = [
                    f"<span style='color:white; text-shadow: -1px -1px 0 #000, 1px -1px 0 #000, -1px 1px 0 #000, 1px 1px 0 #000; font-weight: bold;'>{label}</span>"
                    for label in todos_os_nos
                ]

                # 3. O mapeamento usa os nomes originais para não quebrar a lógica de ligações numéricas
                mapa_nos = {nome: i for i, nome in enumerate(todos_os_nos)}

                fontes = [mapa_nos[str(x) + " (Pré)"] for x in fluxo[col_txt_ini]]
                alvos = [mapa_nos[str(x) + " (Pós)"] for x in fluxo[col_txt_fin]]
                valores = fluxo['contagem'].tolist()

                # 4. Constrói a figura passando os labels formatados diretamente
                fig_sankey = go.Figure(data=[go.Sankey(
                    node = dict(
                        pad = 20, 
                        thickness = 20, 
                        line = dict(color = "black", width = 0.5), 
                        label = labels_formatados, # Rótulos bonitos aplicados aqui!
                        color = "#377EB8"
                    ),
                    link = dict(
                        source = fontes, 
                        target = alvos, 
                        value = valores, 
                        color = "rgba(169, 169, 169, 0.4)" # Mantém a transparência para não brigar com o texto
                    )
                )])

                # 5. Ajusta o tamanho da fonte global do gráfico e o layout
                fig_sankey.update_traces(
                    textfont=dict(size=14, family='Arial, Helvetica, sans-serif'),
                    selector=dict(type='sankey')
                )

                fig_sankey.update_layout(
                    title_text=f"Fluxo de Respostas: {q_selecionada} (N={len(df_sankey)})", 
                    font_size=12, 
                    height=500
                )
                
                st.plotly_chart(fig_sankey, use_container_width=True)
            else:
                st.warning("Não há respostas válidas pareadas para esta questão no grupo selecionado.")

            st.divider()

            # --- 4. AVALIAÇÃO FINAL DO PROGRAMA (QUALITATIVA E QUANTITATIVA) ---
            st.subheader("🗣️ 5. Avaliação Final do Programa (Feedback)")
            st.markdown("""
            Esta seção compila o feedback direto das empreendedoras no final da jornada, 
            mesclando respostas de múltipla escolha com a análise de sentimentos e sugestões dos textos abertos.
            """)

            perguntas_fechadas = {
                "Participação no Programa": "voce_participou_das_atividades_do_programa",
                "O que ajudou a continuar?": "o_que_mais_ajudou_voce_a_continuar_participando_do_programa",
                "O que dificultou a participação?": "se_voce_deixou_de_participar_ou_interagir_pouco_o_que_dificultou"
            }

            perguntas_abertas = {
                "Influência nos Hábitos": "de_que_forma_o_programa_influenciou_seus_habitos_financeiros",
                "O que mais gostou": "o_que_voce_mais_gostou_no_programa",
                "O que poderia ser melhor": "o_que_voce_acha_que_poderia_ter_sido_melhor_no_programa",
                "O que não funcionou": "o_que_na_sua_opiniao_nao_funcionou_muito_bem",
                "Sugestões para o Futuro": "que_tipo_de_apoio_conteudo_ou_formato_ajudaria_mais_a_melhorar_essa_solucao_para_outras_mulheres_empreendedoras"
            }

            tab_fechadas, tab_abertas = st.tabs(["📊 Perguntas Fechadas (Múltipla Escolha)", "☁️ Perguntas Abertas (Nuvem de Palavras)"])

            # ==========================================
            # SUB-ABA A: MÚLTIPLA ESCOLHA
            # ==========================================
            with tab_fechadas:
                col_f1, col_f2, col_f3 = st.columns([2, 1, 1])
                with col_f1:
                    pergunta_f_sel = st.selectbox("Selecione a Pergunta:", list(perguntas_fechadas.keys()))
                with col_f2:
                    grupos_f = ["Todos os Grupos"] + list(df_plot['grupo_comparacao'].dropna().unique())
                    grupo_f_sel = st.selectbox("Filtrar por Grupo (Gráfico):", grupos_f)
                with col_f3:
                    tipo_grafico_f = st.radio("Formato do Gráfico:", ["Barras", "Pizza"])

                coluna_alvo_f = perguntas_fechadas[pergunta_f_sel]

                # Filtro de dados e remoção de hífens/vazios
                df_f = df_plot.dropna(subset=[coluna_alvo_f]).copy()
                df_f = df_f[~df_f[coluna_alvo_f].astype(str).str.strip().isin(['-', '', 'nan', 'NaN'])]

                if grupo_f_sel != "Todos os Grupos":
                    df_f = df_f[df_f['grupo_comparacao'] == grupo_f_sel]

                if not df_f.empty:
                    df_counts = df_f[coluna_alvo_f].value_counts().reset_index()
                    df_counts.columns = ['Resposta', 'Quantidade']
                    df_counts = df_counts.sort_values(by='Quantidade', ascending=True)

                    if tipo_grafico_f == "Barras":
                        fig_f = px.bar(
                            df_counts, x='Quantidade', y='Resposta', orientation='h',
                            title=f"{pergunta_f_sel} (N={len(df_f)})",
                            color_discrete_sequence=['#8856A7'], text='Quantidade'
                        )
                        fig_f.update_layout(yaxis_title="", xaxis_title="Número de Empreendedoras")
                        fig_f.update_traces(textposition='outside')
                    else:
                        fig_f = px.pie(
                            df_counts, values='Quantidade', names='Resposta', 
                            title=f"{pergunta_f_sel} (N={len(df_f)})",
                            color_discrete_sequence=px.colors.sequential.Purp
                        )
                        fig_f.update_traces(textposition='inside', textinfo='percent+label')

                    st.plotly_chart(fig_f, use_container_width=True)
                else:
                    st.warning("Não há respostas válidas registradas para esta pergunta no grupo selecionado.")

            # ==========================================
            # SUB-ABA B: NUVEM DE PALAVRAS INTERATIVA (ECHARTS)
            # ==========================================
            with tab_abertas:
                col_a1, col_a2 = st.columns([2, 1])
                with col_a1:
                    pergunta_a_sel = st.selectbox("Selecione a Pergunta Aberta:", list(perguntas_abertas.keys()))
                with col_a2:
                    grupos_a = ["Todos os Grupos"] + list(df_plot['grupo_comparacao'].dropna().unique())
                    grupo_a_sel = st.selectbox("Filtrar por Grupo (Nuvem):", grupos_a)
                
                # --- NOVOS SELETORES DE ESTILO ECHART ---
                st.write("")
                col_opt1, col_opt2 = st.columns(2)
                with col_opt1:
                    estilo_fonte = st.selectbox(
                        "Estilo da Fonte:", 
                        ["Arial", "Verdana", "Courier New", "Comic Sans MS", "Impact", "Poppins"], 
                        key="font_wc"
                    )
                with col_opt2:
                    tema_cor = st.selectbox(
                        "Paleta de Cores:", 
                        ["Impacto (Roxos)", "Oceano", "Fogo", "Floresta", "Cyberpunk", "Acadêmico"], 
                        key="color_wc"
                    )
                    
                paletas = {
                    "Impacto (Roxos)": ["#8856A7", "#8C96C6", "#810F7C", "#8C6BB1", "#4D004B"],
                    "Oceano": ["#0077b6", "#00b4d8", "#90e0ef", "#03045e", "#023e8a"],
                    "Fogo": ["#ff4d00", "#ff8c00", "#ff0000", "#fad02c", "#e85d04"],
                    "Floresta": ["#2d6a4f", "#40916c", "#1b4332", "#74c69d", "#95d5b2"],
                    "Cyberpunk": ["#f72585", "#7209b7", "#3a0ca3", "#4361ee", "#4cc9f0"],
                    "Acadêmico": ["#264653", "#2a9d8f", "#e9c46a", "#f4a261", "#e76f51"]
                }
                paleta_escolhida = paletas[tema_cor]

                st.divider()

                # Extração e Filtro de Dados
                coluna_alvo_a = perguntas_abertas[pergunta_a_sel]
                df_a = df_plot.dropna(subset=[coluna_alvo_a]).copy()
                df_a = df_a[~df_a[coluna_alvo_a].astype(str).str.strip().isin(['-', '', 'nan', 'NaN'])]

                if grupo_a_sel != "Todos os Grupos":
                    df_a = df_a[df_a['grupo_comparacao'] == grupo_a_sel]

                if not df_a.empty:
                    texto_completo = " ".join(df_a[coluna_alvo_a].astype(str).tolist())
                    
                    st.markdown(f"**Tema:** {pergunta_a_sel} | **Grupo:** {grupo_a_sel} | **Respostas válidas:** {len(df_a)}")
                    
                    with st.spinner("Desenhando a nuvem..."):
                        # Chama a função que criamos no topo do arquivo
                        wc_opcoes = gerar_nuvem_echarts_pt(
                            texto_completo, 
                            fonte=estilo_fonte, 
                            paleta=paleta_escolhida
                        )
                        
                        if wc_opcoes:
                            # Renderiza o componente interativo
                            st_echarts(
                                options=wc_opcoes, 
                                height="550px", 
                                key=f"wc_final_{pergunta_a_sel}_{grupo_a_sel}_{estilo_fonte}_{tema_cor}"
                            )
                        else:
                            st.warning("Não há palavras relevantes suficientes para gerar a nuvem após a filtragem.")

                    with st.expander("📝 Ver amostra de respostas originais na íntegra"):
                        amostra = df_a[coluna_alvo_a].sample(min(5, len(df_a))).tolist()
                        for idx, resp in enumerate(amostra):
                            st.write(f"*{idx+1}. \"{resp}\"*")
                else:
                    st.warning("Sem dados textuais válidos para gerar a nuvem neste grupo.")
else:
    st.warning("Aguardando conexão com a base de dados do Google Drive...")
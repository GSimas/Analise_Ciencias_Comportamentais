import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import statsmodels.formula.api as smf
import pingouin as pg
from scipy.stats import fisher_exact

# Configuração da Página
st.set_page_config(page_title="Relatório Ciências Comportamentais", layout="wide")
st.title("📊 Análise Comportamental - Inclusão Econômica")
st.markdown("Projeto Sebrae / CINCO / Impact Hub - Avaliação de Intervenções e Hábito Financeiro")


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
    """Roda Regressão OLS, limpa os nomes das variáveis e gera interpretação automática."""
    df_h = df[df['grupo_comparacao'].isin([grupo_ref, grupo_teste])].copy()
    
    # 1. Regressão e Limpeza da Tabela
    formula = f"{var_dependente} ~ C(grupo_comparacao, Treatment(reference='{grupo_ref}'))"
    try:
        model = smf.ols(formula, data=df_h).fit()
        
        # Extrai a tabela de coeficientes como DataFrame do Pandas
        tabela_resultados = model.summary2().tables[1].reset_index()
        
        # Limpa os nomes assustadores do statsmodels
        tabela_resultados['index'] = tabela_resultados['index'].str.replace(
            r"C\(grupo_comparacao, Treatment\(reference='.*'\)\)\[T\.", "Efeito: ", regex=True
        ).str.replace(r"\]", "", regex=True)
        
        tabela_resultados['index'] = tabela_resultados['index'].replace({"Intercept": f"Base ({grupo_ref})"})
        tabela_resultados.rename(columns={'index': 'Variável/Grupo'}, inplace=True)
        
        # Pega os valores da linha de teste para escrever o texto
        linha_teste = tabela_resultados[tabela_resultados['Variável/Grupo'] == f"Efeito: {grupo_teste}"]
        coef_val = linha_teste['Coef.'].values[0] if not linha_teste.empty else 0
        pval = linha_teste['P>|t|'].values[0] if not linha_teste.empty else 1

    except Exception as e:
        tabela_resultados = pd.DataFrame({'Erro': [str(e)]})
        coef_val, pval = 0, 1
        
    # 2. Tamanho do Efeito (Cohen's d)
    g_teste_vals = df_h[df_h['grupo_comparacao'] == grupo_teste][var_dependente].dropna()
    g_ref_vals = df_h[df_h['grupo_comparacao'] == grupo_ref][var_dependente].dropna()
    
    if len(g_teste_vals) > 0 and len(g_ref_vals) > 0:
        d_val = pg.compute_effsize(g_teste_vals, g_ref_vals, eftype='cohen')
    else:
        d_val = np.nan

    # 3. Gerador de Texto Explicativo (Insights Automatizados)
    if pd.isna(d_val) or tabela_resultados.empty:
        texto_insight = "⚠️ Não há dados suficientes nesta amostra para calcular o efeito com segurança."
    else:
        # Define a significância
        if pval < 0.05:
            sig_texto = "é **estatisticamente significativa**"
            emoji = "✅"
        else:
            sig_texto = "**NÃO** é estatisticamente significativa"
            emoji = "⚖️"
            
        # Define a direção
        direcao = "aumentou" if coef_val > 0 else "reduziu"
        
        # Define o tamanho prático do efeito
        if abs(d_val) < 0.2:
            tamanho_efeito = "muito pequeno ou irrelevante"
        elif abs(d_val) < 0.5:
            tamanho_efeito = "pequeno"
        elif abs(d_val) < 0.8:
            tamanho_efeito = "médio"
        else:
            tamanho_efeito = "grande"

        texto_insight = f"{emoji} **O que os dados dizem:** A diferença entre os grupos {sig_texto} (p-valor = {pval:.3f}). "
        texto_insight += f"Em média, a intervenção do grupo '{grupo_teste}' **{direcao}** o resultado em **{abs(coef_val):.1%}** (pontos percentuais) em relação ao controle. "
        
        # Só fala do tamanho do efeito se for significativo
        if pval < 0.05:
            texto_insight += f"Na prática, o impacto dessa mudança no comportamento é considerado **{tamanho_efeito}** (Cohen's d = {d_val:.2f})."
        else:
            texto_insight += "Como o p-valor é alto, essa diferença pode ser apenas fruto do acaso (ruído estatístico)."

    return tabela_resultados, d_val, texto_insight
    
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

        st.sidebar.success("✅ Base processada com sucesso!")
        st.sidebar.subheader("Contagem de Flags")
        st.sidebar.write({m[1]: df[m[0]].sum() for m in mapa_grupos})

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
            
            # Gráfico Boxplot
            fig_adesao = px.box(df_plot, x="grupo_comparacao", y="taxa_adesao_num", color="grupo_comparacao",
                                color_discrete_map=CORES_GRUPOS, points="all",
                                title="Impacto das Intervenções na Criação de Hábito Financeiro")
            fig_adesao.update_layout(yaxis_tickformat='.0%', showlegend=False)
            st.plotly_chart(fig_adesao, use_container_width=True)

            # Testes de Hipótese
            st.subheader("Testes Estatísticos")
            hipoteses = {
                "H1: Plano Completo (G3 vs G0)": ("G0 (Controle)", "G3 (Mentoria/Quem Foi)"),
                "H2a: Informativo vs Controle (G1 vs G0)": ("G0 (Controle)", "G1 (Formal)"),
                "H2b: Acolhedor vs Controle (G2 vs G0)": ("G0 (Controle)", "G2 (Acolhedor Total)"),
                "H3: Linguagem (G2 vs G1)": ("G1 (Formal)", "G2 (Acolhedor Total)"),
                "H4: Poder da Mentoria (G3 vs G4)": ("G4 (Não Convidadas)", "G3 (Mentoria/Quem Foi)"),
                "H5: Poder do Convite (G5 vs G4)": ("G4 (Não Convidadas)", "G5 (Convidadas)")
            }

            h_sel = st.selectbox("Selecione a Hipótese:", list(hipoteses.keys()), key='h_adesao')
            ref, teste = hipoteses[h_sel]
            
            tabela_res, d_cohen, texto_insight = testar_hipotese(df_plot, 'taxa_adesao_num', ref, teste)
            
            # Mostra o texto explicativo em destaque
            st.info(texto_insight)
            
            col_a, col_b = st.columns([2, 1])
            with col_a:
                # Mostra a tabela limpa
                st.dataframe(tabela_res.style.format({
                    'Coef.': '{:.4f}', 'Std.Err.': '{:.4f}', 
                    't': '{:.2f}', 'P>|t|': '{:.4f}', 
                    '[0.025': '{:.4f}', '0.975]': '{:.4f}'
                }))
            with col_b:
                st.metric("Tamanho do Efeito (Cohen's d)", f"{d_cohen:.3f}" if pd.notna(d_cohen) else "N/A")

        # --- ABA 3: TRANSAÇÕES VS METAS ---
        with tab3:
            st.header("Avaliação de Hipóteses Separadas")
            
            # 1. Seleção do Foco - Alterei a KEY para evitar duplicidade
            tipo_analise = st.radio(
                "Selecione o Foco da Análise:", 
                [
                    "Trabalho Operacional (Apenas Transações)", 
                    "Retenção Estratégica (Apenas Metas)", 
                    "Ação Conjunta (Transações + Metas)"
                ], 
                key="radio_foco_analise"
            )
            
            # Mapeamento da variável
            if "Apenas Transações" in tipo_analise:
                var_alvo = 'taxa_transacoes_num'
            elif "Apenas Metas" in tipo_analise:
                var_alvo = 'taxa_metas_num'
            elif "Transações + Metas" in tipo_analise:
                var_alvo = 'taxa_conjunta_num'
            else:
                # Fallback de segurança (opcional)
                var_alvo = 'taxa_adesao_num'
            
            # 2. Gráfico Boxplot
            fig_sep = px.box(df_plot, x="grupo_comparacao", y=var_alvo, color="grupo_comparacao",
                             color_discrete_map=CORES_GRUPOS, points="all",
                             title=f"Impacto na {tipo_analise}")
            fig_sep.update_layout(yaxis_tickformat='.0%', showlegend=False)
            st.plotly_chart(fig_sep, use_container_width=True)

            # 3. Seleção da Hipótese - Use também uma KEY específica se houver erro aqui
            h_sel_sep = st.selectbox("Selecione a Hipótese para analisar:", list(hipoteses.keys()), key='sel_hipotese_aba3')
            ref_sep, teste_sep = hipoteses[h_sel_sep]
            
            # 4. Execução do Teste e Insights
            res_tab_sep, d_cohen_sep, texto_insight_sep = testar_hipotese(df_plot, var_alvo, ref_sep, teste_sep)
            
            # 5. Exibição
            st.info(texto_insight_sep)
            
            col_c, col_d = st.columns([2, 1])
            with col_c:
                st.subheader("Tabela de Regressão")
                st.dataframe(res_tab_sep.style.format({
                    'Coef.': '{:.4f}', 'Std.Err.': '{:.4f}', 
                    't': '{:.2f}', 'P>|t|': '{:.4f}', 
                    '[0.025': '{:.4f}', '0.975]': '{:.4f}'
                }), use_container_width=True)
                
            with col_d:
                st.subheader("Efeito Prático")
                st.metric("Cohen's d", f"{d_cohen_sep:.3f}" if pd.notna(d_cohen_sep) else "N/A")

        # --- ABA 4: COMPORTAMENTO AVANÇADO ---
        with tab4:
            st.header("Análises Comportamentais Avançadas")
            
            # G1: Curva de Sobrevivência
            st.subheader("1. Curva de Sobrevivência do Hábito")
            df_deep = df[(df['G0']==1) | (df['G1']==1) | (df['G2']==1) | (df['G3']==1)].copy()
            cond_plot = [df_deep['G3']==1, df_deep['G2']==1, df_deep['G1']==1, df_deep['G0']==1]
            choices_plot = ["G3 (Mentoria + IA)", "G2 (Acolhedora)", "G1 (Formal)", "G0 (Controle)"]
            df_deep['grupo_plot'] = np.select(cond_plot, choices_plot, default='Outros')
            
            cols_semana = [c for c in df_deep.columns if 'interacoes_semana_' in c]
            if cols_semana:
                df_tempo = df_deep.melt(id_vars=['grupo_plot'], value_vars=cols_semana, var_name='semana_nome', value_name='interacoes')
                df_tempo['semana_num'] = df_tempo['semana_nome'].str.extract(r'(\d+)').astype(float)
                df_tempo['ativa'] = np.where(df_tempo['interacoes'] > 0, 1, 0)
                
                df_tempo_agg = df_tempo.groupby(['grupo_plot', 'semana_num'])['ativa'].mean().reset_index()
                
                fig_tempo = px.line(df_tempo_agg, x="semana_num", y="ativa", color="grupo_plot", markers=True,
                                    title="Sobrevivência do Hábito ao Longo das Semanas")
                fig_tempo.update_layout(yaxis_tickformat='.0%', yaxis_title="% de Empreendedoras Ativas")
                st.plotly_chart(fig_tempo, use_container_width=True)
            else:
                st.warning("Colunas 'interacoes_semana_X' não encontradas.")

            # G2: Ecossistema de Uso e G4: Conversão (Combinados)
            col_eco1, col_eco2 = st.columns(2)
            
            with col_eco1:
                st.subheader("2. Maturidade de Gestão Financeira")
                # Médias
                resumo_eco = df_plot.groupby('grupo_comparacao')[['quantidade_de_transacoes', 'quantidade_de_visualizacoes_painel']].mean().reset_index()
                resumo_eco['painel_ajustado'] = resumo_eco['quantidade_de_visualizacoes_painel'] * 5
                
                fig_eco = go.Figure()
                fig_eco.add_trace(go.Bar(x=resumo_eco['grupo_comparacao'], y=resumo_eco['quantidade_de_transacoes'], name="Transações", marker_color="#B3CDE3"))
                fig_eco.add_trace(go.Bar(x=resumo_eco['grupo_comparacao'], y=resumo_eco['painel_ajustado'], name="Uso do Painel (*Ajustado)", marker_color="#8856A7"))
                fig_eco.update_layout(barmode='group')
                st.plotly_chart(fig_eco, use_container_width=True)

            with col_eco2:
                st.subheader("3. Taxa de Conversão (% Populacional)")
                # Taxas de conversão
                df_plot['fez_transacao'] = (df_plot['quantidade_de_transacoes'] > 0).astype(int)
                df_plot['viu_painel'] = (df_plot['quantidade_de_visualizacoes_painel'] > 0).astype(int)
                
                resumo_conv = df_plot.groupby('grupo_comparacao')[['fez_transacao', 'viu_painel']].mean().reset_index()
                
                fig_conv = go.Figure()
                fig_conv.add_trace(go.Bar(x=resumo_conv['grupo_comparacao'], y=resumo_conv['fez_transacao'], name="Fizeram Transações", marker_color="#B3CDE3", text=resumo_conv['fez_transacao'].apply(lambda x: f"{x:.1%}")))
                fig_conv.add_trace(go.Bar(x=resumo_conv['grupo_comparacao'], y=resumo_conv['viu_painel'], name="Viram Painel", marker_color="#8856A7", text=resumo_conv['viu_painel'].apply(lambda x: f"{x:.1%}")))
                fig_conv.update_layout(barmode='group', yaxis_tickformat='.0%')
                st.plotly_chart(fig_conv, use_container_width=True)

        # --- ABA 5: INVESTIGAÇÃO TOM DE VOZ ---
        with tab5:
            st.header("Investigação: O Efeito do Tom de Voz (G1 vs G2)")
            
            df_qualidade = df[(df['G1'] == 1) | (df['G2'] == 1)].copy()
            df_qualidade['grupo_comparacao'] = np.where(df_qualidade['G1'] == 1, "G1 (Apenas IA Formal)", "G2 (Apenas IA Acolhedora)")
            
            st.subheader("Testes de Quantidade de Interações (Regressão OLS)")
            alvo_voz = st.radio("Métrica:", ["quantidade_de_transacoes", "quantidade_de_visualizacoes_painel"])
            
            tabela_voz, d_voz, texto_voz = testar_hipotese(df_qualidade, alvo_voz, "G1 (Apenas IA Formal)", "G2 (Apenas IA Acolhedora)")
            
            st.info(texto_voz)
            
            col_v1, col_v2 = st.columns([2, 1])
            with col_v1:
                st.dataframe(tabela_voz.style.format({
                    'Coef.': '{:.4f}', 'Std.Err.': '{:.4f}', 
                    't': '{:.2f}', 'P>|t|': '{:.4f}'
                }))
            with col_v2:
                st.metric("Cohen's d", f"{d_voz:.3f}" if pd.notna(d_voz) else "N/A")
                
            st.divider()
            
            st.subheader("Testes de Proporção / Conversão (Qui-Quadrado)")
            df_qualidade['fez_transacao'] = (df_qualidade['quantidade_de_transacoes'] > 0).astype(int)
            df_qualidade['viu_painel'] = (df_qualidade['quantidade_de_visualizacoes_painel'] > 0).astype(int)
            
            flag_voz = st.radio("Analisar Conversão em:", ["fez_transacao", "viu_painel"])
            
            try:
                # Qui-Quadrado e Cramer's V com Pingouin
                exp, obs, stats_voz = pg.chi2_independence(df_qualidade, x='grupo_comparacao', y=flag_voz)
                st.dataframe(obs)
                
                pval_voz = stats_voz[stats_voz['test'] == 'pearson']['pval'].values[0]
                cramer_voz = stats_voz[stats_voz['test'] == 'pearson']['cramer'].values[0]
                
                # Odds Ratio com Scipy
                tabela_2x2 = pd.crosstab(df_qualidade['grupo_comparacao'], df_qualidade[flag_voz])
                odds, p_fisher = fisher_exact(tabela_2x2)
                
                col_s1, col_s2, col_s3 = st.columns(3)
                col_s1.metric("P-Valor (Qui-Quadrado)", f"{pval_voz:.4f}")
                col_s2.metric("Cramer's V", f"{cramer_voz:.3f}")
                col_s3.metric("Odds Ratio", f"{odds:.3f}")
                
            except Exception as e:
                st.error(f"Erro ao calcular estatísticas de conversão: {e}")

else:
    st.info("👈 Por favor, faça o upload do arquivo Tabela-mãe (.csv ou .xlsx) no menu lateral para iniciar.")
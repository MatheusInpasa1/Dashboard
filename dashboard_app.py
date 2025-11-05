# dashboard_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import os
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(page_title="Dashboard de Análise de Processos", layout="wide")

# Função para gerar IDs únicos
def generate_unique_key(*args):
    return "_".join(str(arg) for arg in args)

# Função para carregar dados
@st.cache_data
def carregar_dados(uploaded_file):
    """Carrega os dados do arquivo Excel com cache para melhor performance"""
    try:
        if uploaded_file.name.endswith('.csv'):
            dados = pd.read_csv(uploaded_file)
        else:
            dados = pd.read_excel(uploaded_file)
        return dados
    except Exception as e:
        st.error(f"Erro ao carregar arquivo: {str(e)}")
        return None

# Função para converter para data
def converter_para_data(coluna):
    """Tenta converter uma coluna para datetime"""
    try:
        return pd.to_datetime(coluna, dayfirst=True, errors='coerce')
    except:
        return coluna

# Função para detectar outliers usando IQR
def detectar_outliers(dados, coluna):
    if coluna not in dados.columns:
        return pd.DataFrame(), pd.Series()
    
    Q1 = dados[coluna].quantile(0.25)
    Q3 = dados[coluna].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_mask = (dados[coluna] < lower_bound) | (dados[coluna] > upper_bound)
    return dados[outliers_mask], outliers_mask

# Função para detectar outliers usando Z-score (implementação manual)
def detectar_outliers_zscore(dados, coluna, threshold=3):
    if coluna not in dados.columns:
        return pd.DataFrame(), pd.Series()
    
    data_clean = dados[coluna].dropna()
    if len(data_clean) < 2:
        return pd.DataFrame(), pd.Series()
    
    mean_val = data_clean.mean()
    std_val = data_clean.std()
    
    if std_val == 0:
        return pd.DataFrame(), pd.Series()
    
    z_scores = np.abs((data_clean - mean_val) / std_val)
    outliers_mask = z_scores > threshold
    return dados[outliers_mask], outliers_mask

# Função para calcular regressão linear manualmente
def calcular_regressao_linear(x, y):
    """Calcula regressão linear manualmente"""
    # Remover valores NaN
    mask = ~np.isnan(x) & ~np.isnan(y)
    x_clean = x[mask]
    y_clean = y[mask]
    
    if len(x_clean) < 2:
        return None, None, None
    
    n = len(x_clean)
    x_mean = np.mean(x_clean)
    y_mean = np.mean(y_clean)
    
    numerator = np.sum((x_clean - x_mean) * (y_clean - y_mean))
    denominator = np.sum((x_clean - x_mean) ** 2)
    
    if denominator == 0:
        return None, None, None
    
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    # Calcular R²
    y_pred = slope * x_clean + intercept
    ss_res = np.sum((y_clean - y_pred) ** 2)
    ss_tot = np.sum((y_clean - y_mean) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
    
    return slope, intercept, r_squared

# Função para criar gráfico Q-Q (implementação manual)
def criar_qq_plot_correto(data):
    """Cria gráfico Q-Q correto passando pelo meio dos pontos"""
    data_clean = data.dropna()
    if len(data_clean) < 2:
        return go.Figure()
    
    # Calcular quantis teóricos usando distribuição normal manualmente
    n = len(data_clean)
    # Gerar quantis teóricos para distribuição normal
    theoretical_quantiles = np.sort(np.random.normal(0, 1, n))
    sample_quantiles = np.sort(data_clean)
    
    # Normalizar os dados para melhor visualização
    sample_quantiles = (sample_quantiles - np.mean(sample_quantiles)) / np.std(sample_quantiles)
    
    # Calcular linha de tendência para o Q-Q plot
    z = np.polyfit(theoretical_quantiles, sample_quantiles, 1)
    p = np.poly1d(z)
    
    fig = go.Figure()
    
    # Adicionar pontos
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=sample_quantiles,
        mode='markers',
        name='Dados',
        marker=dict(color='blue', size=6)
    ))
    
    # Adicionar linha de tendência que passa pelo meio dos pontos
    fig.add_trace(go.Scatter(
        x=theoretical_quantiles,
        y=p(theoretical_quantiles),
        mode='lines',
        name='Linha de Tendência',
        line=dict(color='red', width=2)
    ))
    
    fig.update_layout(
        title="Gráfico Q-Q (Análise de Normalidade)",
        xaxis_title="Quantis Teóricos",
        yaxis_title="Quantis Amostrais",
        showlegend=True
    )
    
    return fig

# Função para análise de capacidade do processo
def analise_capacidade_processo(dados, coluna, lse, lie):
    """Analisa a capacidade do processo"""
    if coluna not in dados.columns:
        return None
    
    data_clean = dados[coluna].dropna()
    if len(data_clean) < 2:
        return None
    
    media = np.mean(data_clean)
    desvio_padrao = np.std(data_clean, ddof=1)
    
    resultados = {
        'media': media,
        'desvio_padrao': desvio_padrao,
        'n': len(data_clean)
    }
    
    if lse is not None and lie is not None and lse > lie:
        # Cp - Capacidade do processo
        cp = (lse - lie) / (6 * desvio_padrao)
        # Cpk - Capacidade real do processo
        cpk_u = (lse - media) / (3 * desvio_padrao) if desvio_padrao > 0 else 0
        cpk_l = (media - lie) / (3 * desvio_padrao) if desvio_padrao > 0 else 0
        cpk = min(cpk_u, cpk_l)
        
        resultados.update({
            'cp': cp,
            'cpk': cpk,
            'lse': lse,
            'lie': lie
        })
    
    return resultados

# Função para criar gráfico de controle
def criar_grafico_controle(dados, coluna_valor, coluna_data=None):
    """Cria gráfico de controle (X-bar)"""
    if coluna_valor not in dados.columns:
        return go.Figure(), 0, 0, 0
    
    data_clean = dados[[coluna_valor]].copy()
    if coluna_data and coluna_data in dados.columns:
        data_clean[coluna_data] = dados[coluna_data]
        data_clean = data_clean.sort_values(coluna_data)
    
    # Calcular limites de controle
    media = data_clean[coluna_valor].mean()
    std = data_clean[coluna_valor].std()
    
    lsc = media + 3 * std  # Limite Superior de Controle
    lic = media - 3 * std  # Limite Inferior de Controle
    lc = media             # Linha Central
    
    fig = go.Figure()
    
    # Adicionar pontos do processo
    if coluna_data:
        x_data = data_clean[coluna_data]
    else:
        x_data = list(range(len(data_clean)))
    
    fig.add_trace(go.Scatter(
        x=x_data,
        y=data_clean[coluna_valor],
        mode='lines+markers',
        name='Valores',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # Adicionar linhas de controle
    fig.add_hline(y=lsc, line_dash="dash", line_color="red", annotation_text="LSC")
    fig.add_hline(y=lc, line_dash="dash", line_color="green", annotation_text="LC")
    fig.add_hline(y=lic, line_dash="dash", line_color="red", annotation_text="LIC")
    
    fig.update_layout(
        title=f"Gráfico de Controle - {coluna_valor}",
        xaxis_title=coluna_data if coluna_data else "Amostras",
        yaxis_title=coluna_valor,
        showlegend=True
    )
    
    return fig, lsc, lc, lic

# Função para teste de normalidade manual (simplificado)
def teste_normalidade_manual(data):
    """Teste de normalidade simplificado usando assimetria e curtose"""
    data_clean = data.dropna()
    if len(data_clean) < 3:
        return 0.5  # Valor neutro se não há dados suficientes
    
    # Calcular assimetria manualmente
    mean_val = np.mean(data_clean)
    std_val = np.std(data_clean)
    if std_val == 0:
        return 0.5
    
    skewness = np.mean(((data_clean - mean_val) / std_val) ** 3)
    
    # Calcular curtose manualmente
    kurtosis = np.mean(((data_clean - mean_val) / std_val) ** 4) - 3
    
    # Estimativa simplificada de p-valor baseada na assimetria e curtose
    p_value = max(0, 1 - (abs(skewness) + abs(kurtosis)) / 2)
    return p_value

def main():
    st.title("🏭 Dashboard de Análise de Processos Industriais")
    
    # Inicializar estado da sessão
    session_defaults = {
        'dados_originais': None,
        'dados_processados': None,
        'filtro_data_limpo': False,
        'outliers_removidos': {},
        'lse_values': {},
        'lie_values': {}
    }
    
    for key, value in session_defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    
    # Sidebar para upload
    with st.sidebar:
        st.header("📁 Carregamento de Dados")
        
        uploaded_file = st.file_uploader(
            "Selecione o arquivo de dados:",
            type=['xlsx', 'xls', 'csv'],
            key=generate_unique_key("file_uploader", "main")
        )
        
        if uploaded_file is not None:
            st.success("✅ Arquivo selecionado!")
        else:
            st.info("📝 Aguardando upload do arquivo...")
            st.stop()

    # Carregar dados
    dados = carregar_dados(uploaded_file)
    
    if dados is None:
        st.error("❌ Falha ao carregar os dados.")
        st.stop()

    # Inicializar dados na sessão se necessário
    if st.session_state.dados_originais is None:
        st.session_state.dados_originais = dados.copy()
        st.session_state.dados_processados = dados.copy()

    # Processar dados
    dados_processados = st.session_state.dados_processados.copy()
    colunas_numericas = dados_processados.select_dtypes(include=[np.number]).columns.tolist()
    
    # Detectar colunas de data
    colunas_data = []
    for col in dados_processados.columns:
        if any(palavra in col.lower() for palavra in ['data', 'date', 'dia', 'time', 'hora', 'timestamp']):
            colunas_data.append(col)
            dados_processados[col] = converter_para_data(dados_processados[col])

    # Sidebar para filtros globais
    with st.sidebar:
        st.header("🎛️ Filtros Globais")
        
        # Botão para resetar todos os filtros
        if st.button("🔄 Resetar Todos os Filtros", use_container_width=True,
                    key=generate_unique_key("reset_filters", "main")):
            st.session_state.dados_processados = st.session_state.dados_originais.copy()
            st.session_state.filtro_data_limpo = False
            st.session_state.outliers_removidos = {}
            st.session_state.lse_values = {}
            st.session_state.lie_values = {}
            st.rerun()
        
        # Filtro de período
        if colunas_data:
            coluna_data_filtro = st.selectbox("Coluna de data para filtro:", colunas_data,
                                             key=generate_unique_key("data_filter_col", "main"))
            
            if pd.api.types.is_datetime64_any_dtype(dados_processados[coluna_data_filtro]):
                min_date = dados_processados[coluna_data_filtro].min()
                max_date = dados_processados[coluna_data_filtro].max()
                
                # Verificar se o filtro foi limpo
                if st.session_state.filtro_data_limpo:
                    date_range = (min_date, max_date)
                else:
                    date_range = st.date_input(
                        "Selecione o período:",
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date,
                        key=generate_unique_key("date_range", "main")
                    )
                
                # Botão para limpar filtro de data
                if st.button("❌ Limpar Filtro de Data", use_container_width=True,
                            key=generate_unique_key("clear_date_filter", "main")):
                    st.session_state.filtro_data_limpo = True
                    st.rerun()
                
                if len(date_range) == 2 and not st.session_state.filtro_data_limpo:
                    start_date, end_date = date_range
                    dados_processados = dados_processados[
                        (dados_processados[coluna_data_filtro] >= pd.Timestamp(start_date)) &
                        (dados_processados[coluna_data_filtro] <= pd.Timestamp(end_date))
                    ]
        
        # Filtro de outliers - CORRIGIDO
        st.subheader("🔍 Gerenciamento de Outliers")
        
        if colunas_numericas:
            coluna_outliers = st.selectbox("Selecione a coluna para análise de outliers:", colunas_numericas,
                                          key=generate_unique_key("outlier_col", "main"))
            
            if coluna_outliers:
                # Selecionar método de detecção de outliers
                metodo_outliers = st.radio("Método de detecção:", 
                                          ["IQR (Recomendado)", "Z-Score"],
                                          key=generate_unique_key("outlier_method", coluna_outliers))
                
                # Detectar outliers
                if metodo_outliers == "IQR (Recomendado)":
                    outliers_df, outliers_mask = detectar_outliers(dados_processados, coluna_outliers)
                else:
                    outliers_df, outliers_mask = detectar_outliers_zscore(dados_processados, coluna_outliers)
                
                st.info(f"📊 {len(outliers_df)} outliers detectados na coluna '{coluna_outliers}'")
                
                # Mostrar outliers
                if len(outliers_df) > 0:
                    with st.expander("📋 Visualizar Outliers Detectados"):
                        st.dataframe(outliers_df[[coluna_outliers]].style.format({
                            coluna_outliers: '{:.4f}'
                        }))
                
                # Opção para remover outliers
                col_btn1, col_btn2 = st.columns(2)
                with col_btn1:
                    if st.button(f"🗑️ Remover Outliers", use_container_width=True,
                                key=generate_unique_key("remove_outliers", coluna_outliers)):
                        dados_sem_outliers = dados_processados[~outliers_mask]
                        st.session_state.dados_processados = dados_sem_outliers
                        st.session_state.outliers_removidos[coluna_outliers] = True
                        st.success(f"✅ {len(outliers_df)} outliers removidos da coluna '{coluna_outliers}'")
                        st.rerun()
                
                with col_btn2:
                    if st.button(f"↩️ Restaurar Outliers", use_container_width=True,
                                key=generate_unique_key("restore_outliers", coluna_outliers)):
                        if coluna_outliers in st.session_state.outliers_removidos:
                            st.session_state.dados_processados = st.session_state.dados_originais.copy()
                            del st.session_state.outliers_removidos[coluna_outliers]
                            st.success(f"✅ Outliers restaurados para '{coluna_outliers}'")
                            st.rerun()

        # Configuração de limites de especificação
        st.subheader("🎯 Limites de Especificação")
        if colunas_numericas:
            coluna_limites = st.selectbox("Selecione a variável:", colunas_numericas,
                                         key=generate_unique_key("limits_col", "main"))
            
            col_lim1, col_lim2 = st.columns(2)
            with col_lim1:
                lse = st.number_input("LSE (Limite Superior):", 
                                     value=float(st.session_state.lse_values.get(coluna_limites, 0)),
                                     key=generate_unique_key("lse", coluna_limites))
                st.session_state.lse_values[coluna_limites] = lse
            
            with col_lim2:
                lie = st.number_input("LIE (Limite Inferior):", 
                                     value=float(st.session_state.lie_values.get(coluna_limites, 0)),
                                     key=generate_unique_key("lie", coluna_limites))
                st.session_state.lie_values[coluna_limites] = lie

    # Abas principais
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📈 Análise Temporal", 
        "📊 Estatística Detalhada", 
        "🔥 Correlações", 
        "🔍 Dispersão & Regressão",
        "🎯 Controle Estatístico",
        "📋 Resumo Executivo"
    ])

    with tab1:
        st.header("📈 Análise de Séries Temporais")
        
        if colunas_data and colunas_numericas:
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                coluna_data = st.selectbox("Coluna de Data:", colunas_data, 
                                          key=generate_unique_key("temp_data", "tab1"))
            with col2:
                coluna_valor = st.selectbox("Coluna para Análise:", colunas_numericas,
                                           key=generate_unique_key("temp_valor", "tab1"))
            with col3:
                tipo_grafico = st.selectbox("Tipo de Gráfico:", 
                                           ["Linha", "Área", "Barra", "Scatter", "Boxplot Temporal", "Histograma Temporal"],
                                           key=generate_unique_key("chart_type", "tab1"))
            
            if coluna_data and coluna_valor:
                dados_temp = dados_processados.sort_values(by=coluna_data)
                
                # Opção para remover outliers diretamente no gráfico
                remover_outliers_grafico = st.checkbox("📉 Remover outliers para visualização",
                                                      key=generate_unique_key("remove_temp_outliers", coluna_valor))
                
                if remover_outliers_grafico:
                    outliers_df, outliers_mask = detectar_outliers(dados_temp, coluna_valor)
                    dados_temp = dados_temp[~outliers_mask]
                    st.info(f"📊 {len(outliers_df)} outliers removidos para visualização")
                
                # Criar gráfico baseado no tipo selecionado
                if tipo_grafico == "Linha":
                    fig = px.line(dados_temp, x=coluna_data, y=coluna_valor, 
                                 title=f"Evolução Temporal de {coluna_valor}")
                elif tipo_grafico == "Área":
                    fig = px.area(dados_temp, x=coluna_data, y=coluna_valor,
                                 title=f"Evolução Temporal de {coluna_valor}")
                elif tipo_grafico == "Barra":
                    fig = px.bar(dados_temp, x=coluna_data, y=coluna_valor,
                                title=f"Evolução Temporal de {coluna_valor}")
                elif tipo_grafico == "Scatter":
                    fig = px.scatter(dados_temp, x=coluna_data, y=coluna_valor,
                                    title=f"Relação Temporal de {coluna_valor}")
                elif tipo_grafico == "Histograma Temporal":
                    # Criar histograma 2D
                    fig = px.density_heatmap(dados_temp, x=coluna_data, y=coluna_valor,
                                           title=f"Distribuição Temporal de {coluna_valor}")
                else:  # Boxplot Temporal
                    # Criar períodos mensais para boxplot
                    dados_temp['Periodo'] = dados_temp[coluna_data].dt.to_period('M').astype(str)
                    fig = px.box(dados_temp, x='Periodo', y=coluna_valor,
                                title=f"Distribuição Mensal de {coluna_valor}")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Análise de tendência com mais detalhes
                st.subheader("📈 Análise de Tendência Detalhada")
                if len(dados_temp) > 1:
                    # Calcular métricas de tendência
                    crescimento = ((dados_temp[coluna_valor].iloc[-1] - dados_temp[coluna_valor].iloc[0]) / 
                                 dados_temp[coluna_valor].iloc[0] * 100) if dados_temp[coluna_valor].iloc[0] != 0 else 0
                    
                    # Regressão linear para tendência
                    x = np.arange(len(dados_temp))
                    y = dados_temp[coluna_valor].values
                    coef = np.polyfit(x, y, 1)[0]
                    
                    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
                    with col_t1:
                        st.metric("Crescimento Total", f"{crescimento:.1f}%")
                    with col_t2:
                        tendencia = "↗️ Alta" if coef > 0 else "↘️ Baixa" if coef < 0 else "➡️ Estável"
                        st.metric("Tendência", tendencia)
                    with col_t3:
                        st.metric("Taxa de Variação", f"{coef:.4f}")
                    with col_t4:
                        # Calcular volatilidade (desvio padrão das variações)
                        variacoes = dados_temp[coluna_valor].pct_change().dropna()
                        volatilidade = variacoes.std() * 100 if len(variacoes) > 0 else 0
                        st.metric("Volatilidade", f"{volatilidade:.2f}%")

    with tab2:
        st.header("📊 Estatística Detalhada")
        
        if colunas_numericas:
            coluna_analise = st.selectbox("Selecione a coluna para análise:", colunas_numericas, 
                                         key=generate_unique_key("stats_col", "tab2"))
            
            if coluna_analise:
                # Opções de análise
                col_opt1, col_opt2 = st.columns(2)
                with col_opt1:
                    remover_outliers_grafico = st.checkbox("📉 Remover outliers para análise",
                                                          key=generate_unique_key("remove_stats_outliers", coluna_analise))
                with col_opt2:
                    usar_log = st.checkbox("📊 Aplicar transformação logarítmica",
                                          key=generate_unique_key("use_log", coluna_analise))
                
                dados_analise = dados_processados.copy()
                if remover_outliers_grafico:
                    outliers_df, outliers_mask = detectar_outliers(dados_analise, coluna_analise)
                    dados_analise = dados_analise[~outliers_mask]
                    st.info(f"📊 {len(outliers_df)} outliers removidos para análise")
                
                if usar_log:
                    # Adicionar uma constante pequena para evitar log(0)
                    dados_analise[coluna_analise] = np.log1p(dados_analise[coluna_analise] - dados_analise[coluna_analise].min())
                    st.info("Transformação logarítmica aplicada")
                
                # Estatísticas básicas
                st.subheader("📋 Estatísticas Descritivas Completas")
                stats_data = dados_analise[coluna_analise].describe()
                
                col1, col2, col3, col4 = st.columns(4)
                metrics = [
                    ("Média", stats_data['mean']),
                    ("Mediana", stats_data['50%']),
                    ("Moda", dados_analise[coluna_analise].mode().iloc[0] if not dados_analise[coluna_analise].mode().empty else np.nan),
                    ("Desvio Padrão", stats_data['std']),
                    ("Variância", stats_data['std']**2),
                    ("Coef. Variação", (stats_data['std']/stats_data['mean'])*100 if stats_data['mean'] != 0 else 0),
                    ("Mínimo", stats_data['min']),
                    ("Máximo", stats_data['max']),
                    ("Amplitude", stats_data['max'] - stats_data['min']),
                    ("Q1 (25%)", stats_data['25%']),
                    ("Q3 (75%)", stats_data['75%']),
                    ("IQR", stats_data['75%'] - stats_data['25%'])
                ]
                
                for i, (name, value) in enumerate(metrics):
                    with [col1, col2, col3, col4][i % 4]:
                        if not np.isnan(value):
                            st.metric(name, f"{value:.4f}" if isinstance(value, (int, float)) else str(value))
                
                # Análise de distribuição COMPLETA
                st.subheader("📈 Análise de Distribuição")
                
                dist_col1, dist_col2 = st.columns(2)
                
                with dist_col1:
                    # Coeficientes de forma
                    skewness = dados_analise[coluna_analise].skew()
                    kurtosis = dados_analise[coluna_analise].kurtosis()
                    
                    st.write("**📊 Medidas de Forma:**")
                    st.metric("Assimetria", f"{skewness:.3f}")
                    st.metric("Curtose", f"{kurtosis:.3f}")
                    
                    # Teste de normalidade manual
                    p_norm = teste_normalidade_manual(dados_analise[coluna_analise])
                    st.metric("p-valor (Normalidade Aprox.)", f"{p_norm:.4f}")
                    
                    # Interpretação
                    st.write("**📝 Interpretação:**")
                    if abs(skewness) < 0.5:
                        st.success("• Distribuição aproximadamente simétrica")
                    elif abs(skewness) < 1:
                        st.warning("• Distribuição moderadamente assimétrica")
                    else:
                        st.error("• Distribuição fortemente assimétrica")
                    
                    if abs(kurtosis) < 0.5:
                        st.success("• Curtose próxima da normal")
                    elif abs(kurtosis) < 1:
                        st.warning("• Curtose moderadamente diferente da normal")
                    else:
                        st.error("• Curtose muito diferente da normal")
                
                with dist_col2:
                    # Gráficos de distribuição
                    fig = px.histogram(dados_analise, x=coluna_analise, 
                                      title=f"Distribuição de {coluna_analise}",
                                      nbins=30, marginal="box",
                                      histnorm='probability density')
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                # Gráfico Q-Q
                st.subheader("📊 Gráfico Q-Q (Análise de Normalidade)")
                fig_qq = criar_qq_plot_correto(dados_analise[coluna_analise])
                st.plotly_chart(fig_qq, use_container_width=True)

    with tab3:
        st.header("🔥 Análise de Correlações")
        
        if len(colunas_numericas) > 1:
            # Selecionar variáveis para correlação
            st.subheader("🎯 Seleção de Variáveis")
            variaveis_selecionadas = st.multiselect(
                "Selecione as variáveis para análise de correlação:",
                options=colunas_numericas,
                default=colunas_numericas[:min(8, len(colunas_numericas))],
                key=generate_unique_key("corr_vars", "tab3")
            )
            
            if len(variaveis_selecionadas) > 1:
                # Opções de análise
                col_opt1, col_opt2 = st.columns(2)
                with col_opt1:
                    remover_outliers_corr = st.checkbox("📉 Remover outliers para análise",
                                                       key=generate_unique_key("remove_corr_outliers", "tab3"))
                with col_opt2:
                    metodo_corr = st.selectbox("Método de correlação:",
                                              ["Pearson", "Spearman"],
                                              key=generate_unique_key("corr_method", "tab3"))
                
                dados_corr = dados_processados.copy()
                if remover_outliers_corr:
                    for var in variaveis_selecionadas:
                        outliers_df, outliers_mask = detectar_outliers(dados_corr, var)
                        dados_corr = dados_corr[~outliers_mask]
                    st.info("Outliers removidos de todas as variáveis selecionadas")
                
                # Matriz de correlação
                if metodo_corr == "Pearson":
                    corr_matrix = dados_corr[variaveis_selecionadas].corr(method='pearson')
                else:
                    corr_matrix = dados_corr[variaveis_selecionadas].corr(method='spearman')
                
                fig = px.imshow(corr_matrix, 
                               title=f"Matriz de Correlação ({metodo_corr})",
                               color_continuous_scale="RdBu_r",
                               aspect="auto",
                               text_auto=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Análise detalhada de correlações
                st.subheader("🔍 Análise Detalhada das Correlações")
                
                correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        corr_value = corr_matrix.iloc[i, j]
                        correlations.append({
                            'Variável 1': corr_matrix.columns[i],
                            'Variável 2': corr_matrix.columns[j],
                            'Correlação': corr_value,
                            '|Correlação|': abs(corr_value)
                        })
                
                df_corr = pd.DataFrame(correlations)
                
                col_ana1, col_ana2 = st.columns(2)
                
                with col_ana1:
                    st.write("**📈 Top 10 Maiores Correlações:**")
                    top_correlations = df_corr.nlargest(10, '|Correlação|')
                    for _, row in top_correlations.iterrows():
                        corr_value = row['Correlação']
                        corr_color = "🟢" if corr_value > 0 else "🔴"
                        corr_strength = "Forte" if abs(corr_value) > 0.7 else "Moderada" if abs(corr_value) > 0.3 else "Fraca"
                        st.write(f"{corr_color} **{corr_value:.3f}** - {corr_strength}")
                        st.write(f"   {row['Variável 1']} ↔ {row['Variável 2']}")
                        st.write("---")
                
                with col_ana2:
                    st.write("**📉 Top 10 Menores Correlações:**")
                    bottom_correlations = df_corr.nsmallest(10, '|Correlação|')
                    for _, row in bottom_correlations.iterrows():
                        corr_value = row['Correlação']
                        corr_color = "🟢" if corr_value > 0 else "🔴"
                        corr_strength = "Fraca"
                        st.write(f"{corr_color} **{corr_value:.3f}** - {corr_strength}")
                        st.write(f"   {row['Variável 1']} ↔ {row['Variável 2']}")
                        st.write("---")

    with tab4:
        st.header("🔍 Gráficos de Dispersão com Regressão")
        
        if len(colunas_numericas) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                eixo_x = st.selectbox("Eixo X:", colunas_numericas, 
                                     key=generate_unique_key("scatter_x", "tab4"))
            with col2:
                eixo_y = st.selectbox("Eixo Y:", colunas_numericas,
                                     key=generate_unique_key("scatter_y", "tab4"))
            
            if eixo_x and eixo_y:
                # Opções avançadas
                col_opt1, col_opt2, col_opt3 = st.columns(3)
                with col_opt1:
                    remover_outliers_grafico = st.checkbox("📉 Remover outliers",
                                                          key=generate_unique_key("remove_scatter_outliers", f"{eixo_x}_{eixo_y}"))
                with col_opt2:
                    mostrar_regressao = st.checkbox("📈 Mostrar regressão", value=True,
                                                   key=generate_unique_key("show_regression", f"{eixo_x}_{eixo_y}"))
                with col_opt3:
                    color_by = st.selectbox("Colorir por:", [""] + colunas_numericas,
                                           key=generate_unique_key("color_by", f"{eixo_x}_{eixo_y}"))
                
                dados_scatter = dados_processados.copy()
                if remover_outliers_grafico:
                    outliers_x, outliers_mask_x = detectar_outliers(dados_scatter, eixo_x)
                    outliers_y, outliers_mask_y = detectar_outliers(dados_scatter, eixo_y)
                    outliers_mask = outliers_mask_x | outliers_mask_y
                    dados_scatter = dados_scatter[~outliers_mask]
                    st.info(f"📊 {outliers_mask.sum()} outliers removidos para visualização")
                
                # Gráfico de dispersão
                if color_by and color_by in dados_scatter.columns:
                    fig = px.scatter(dados_scatter, x=eixo_x, y=eixo_y, color=color_by,
                                    title=f"{eixo_y} vs {eixo_x} (Colorido por {color_by})")
                else:
                    fig = px.scatter(dados_scatter, x=eixo_x, y=eixo_y, 
                                    title=f"{eixo_y} vs {eixo_x}")
                
                # Calcular regressão linear manualmente
                if mostrar_regressao:
                    slope, intercept, r_squared = calcular_regressao_linear(
                        dados_scatter[eixo_x].values,
                        dados_scatter[eixo_y].values
                    )
                    
                    # Adicionar linha de regressão manualmente se possível
                    if slope is not None and intercept is not None:
                        x_range = np.linspace(dados_scatter[eixo_x].min(), dados_scatter[eixo_x].max(), 100)
                        y_pred = slope * x_range + intercept
                        
                        fig.add_trace(go.Scatter(
                            x=x_range,
                            y=y_pred,
                            mode='lines',
                            name='Linha de Regressão',
                            line=dict(color='red', width=3)
                        ))
                        
                        # Adicionar equação da reta
                        equation = f"y = {slope:.4f}x + {intercept:.4f}"
                        r2_text = f"R² = {r_squared:.4f}"
                        
                        fig.add_annotation(
                            x=0.05,
                            y=0.95,
                            xref="paper",
                            yref="paper",
                            text=f"<b>{equation}<br>{r2_text}</b>",
                            showarrow=False,
                            font=dict(size=14, color="black"),
                            bgcolor="white",
                            bordercolor="black",
                            borderwidth=2,
                            borderpad=4,
                            opacity=0.8
                        )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Estatísticas de correlação COMPLETAS
                st.subheader("📊 Estatísticas de Correlação e Regressão")
                
                correlacao_pearson = dados_scatter[eixo_x].corr(dados_scatter[eixo_y])
                correlacao_spearman = dados_scatter[eixo_x].corr(dados_scatter[eixo_y], method='spearman')
                
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                with col_stat1:
                    st.metric("Correlação (Pearson)", f"{correlacao_pearson:.4f}")
                with col_stat2:
                    st.metric("Correlação (Spearman)", f"{correlacao_spearman:.4f}")
                with col_stat3:
                    if r_squared is not None:
                        st.metric("Coeficiente R²", f"{r_squared:.4f}")
                with col_stat4:
                    if slope is not None:
                        st.metric("Inclinação", f"{slope:.4f}")

    with tab5:
        st.header("🎯 Controle Estatístico do Processo")
        
        if colunas_numericas:
            coluna_controle = st.selectbox("Selecione a variável para controle:", colunas_numericas,
                                          key=generate_unique_key("control_chart_var", "tab5"))
            
            coluna_data_controle = None
            if colunas_data:
                coluna_data_controle = st.selectbox("Selecione a coluna de data (opcional):", 
                                                   [""] + colunas_data,
                                                   key=generate_unique_key("control_chart_date", "tab5"))
            
            if coluna_controle:
                # Gráfico de controle
                if coluna_data_controle and coluna_data_controle != "":
                    fig_controle, lsc, lc, lic = criar_grafico_controle(
                        dados_processados, coluna_controle, coluna_data_controle
                    )
                else:
                    fig_controle, lsc, lc, lic = criar_grafico_controle(
                        dados_processados, coluna_controle
                    )
                
                st.plotly_chart(fig_controle, use_container_width=True)
                
                # Estatísticas de controle
                st.subheader("📊 Estatísticas de Controle")
                
                dados_controle = dados_processados[coluna_controle].dropna()
                media = dados_controle.mean()
                std = dados_controle.std()
                
                col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)
                with col_ctrl1:
                    st.metric("Linha Central (LC)", f"{lc:.4f}")
                    st.metric("Média", f"{media:.4f}")
                with col_ctrl2:
                    st.metric("LSC", f"{lsc:.4f}")
                    st.metric("+3σ", f"{media + 3*std:.4f}")
                with col_ctrl3:
                    st.metric("LIC", f"{lic:.4f}")
                    st.metric("-3σ", f"{media - 3*std:.4f}")
                with col_ctrl4:
                    # Pontos fora dos limites
                    pontos_fora = ((dados_controle > lsc) | (dados_controle < lic)).sum()
                    percentual_fora = (pontos_fora / len(dados_controle)) * 100
                    st.metric("Pontos Fora", f"{pontos_fora} ({percentual_fora:.1f}%)")
                
                # Análise de capacidade do processo
                st.subheader("📈 Análise de Capacidade do Processo")
                
                lse = st.session_state.lse_values.get(coluna_controle, 0)
                lie = st.session_state.lie_values.get(coluna_controle, 0)
                
                if lse != 0 and lie != 0 and lse > lie:
                    capacidade = analise_capacidade_processo(dados_processados, coluna_controle, lse, lie)
                    
                    if capacidade and 'cp' in capacidade:
                        col_cap1, col_cap2, col_cap3 = st.columns(3)
                        with col_cap1:
                            st.metric("Cp", f"{capacidade['cp']:.3f}")
                            st.metric("Cpk", f"{capacidade['cpk']:.3f}")
                        with col_cap2:
                            st.metric("LSE", f"{lse:.3f}")
                            st.metric("LIE", f"{lie:.3f}")
                        with col_cap3:
                            # Interpretação da capacidade
                            cpk = capacidade['cpk']
                            if cpk >= 1.33:
                                st.success("✅ Processo Capaz")
                            elif cpk >= 1.0:
                                st.warning("⚠️ Processo Marginalmente Capaz")
                            else:
                                st.error("❌ Processo Incapaz")

    with tab6:
        st.header("📋 Resumo Executivo")
        
        # Métricas gerais
        st.subheader("📊 Visão Geral do Processo")
        
        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
        with col_res1:
            st.metric("Total de Amostras", len(dados_processados))
            st.metric("Variáveis Numéricas", len(colunas_numericas))
        with col_res2:
            st.metric("Variáveis de Data", len(colunas_data))
            st.metric("Dados Faltantes", dados_processados.isnull().sum().sum())
        with col_res3:
            # Estatísticas gerais das variáveis numéricas
            if colunas_numericas:
                media_geral = dados_processados[colunas_numericas].mean().mean()
                st.metric("Média Geral", f"{media_geral:.2f}")
        with col_res4:
            if colunas_numericas:
                std_geral = dados_processados[colunas_numericas].std().mean()
                st.metric("Desvio Padrão Médio", f"{std_geral:.2f}")
        
        # Top variáveis mais variáveis
        st.subheader("🎯 Variáveis com Maior Variabilidade")
        if colunas_numericas:
            variabilidades = dados_processados[colunas_numericas].std().sort_values(ascending=False)
            top_variáveis = variabilidades.head(5)
            
            for var, std_val in top_variáveis.items():
                st.write(f"• **{var}**: {std_val:.4f}")
        
        # Alertas e insights
        st.subheader("🚨 Alertas e Insights")
        
        # Verificar outliers em todas as variáveis
        total_outliers = 0
        for coluna in colunas_numericas:
            outliers_df, _ = detectar_outliers(dados_processados, coluna)
            total_outliers += len(outliers_df)
        
        if total_outliers > 0:
            st.warning(f"⚠️ **{total_outliers} outliers** detectados no processo")
        
        # Verificar dados faltantes
        dados_faltantes = dados_processados.isnull().sum().sum()
        if dados_faltantes > 0:
            st.warning(f"⚠️ **{dados_faltantes} dados faltantes** encontrados")
        
        # Verificar estabilidade do processo
        if colunas_numericas:
            # Calcular coeficiente de variação médio
            coef_variacao_medio = (dados_processados[colunas_numericas].std() / 
                                 dados_processados[colunas_numericas].mean()).mean()
            
            if coef_variacao_medio > 0.5:
                st.warning("⚠️ **Alta variabilidade** detectada no processo")
            elif coef_variacao_medio < 0.1:
                st.success("✅ **Baixa variabilidade** - Processo estável")

    # Download dos dados processados
    st.sidebar.header("💾 Exportar Dados")
    csv = dados_processados.to_csv(index=False)
    st.sidebar.download_button(
        label="📥 Baixar dados processados (CSV)",
        data=csv,
        file_name="dados_processados.csv",
        mime="text/csv",
        key=generate_unique_key("download_csv", "main")
    )

if __name__ == "__main__":
    main()

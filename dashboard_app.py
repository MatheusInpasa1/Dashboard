# dashboard_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import os

# Configuração da página
st.set_page_config(page_title="Dashboard de Utilidades", layout="wide")

# Função para carregar dados
@st.cache_data
def carregar_dados(uploaded_file):
    """Carrega os dados do arquivo Excel com cache para melhor performance"""
    try:
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

# Função para detectar outliers
def detectar_outliers(dados, coluna):
    Q1 = dados[coluna].quantile(0.25)
    Q3 = dados[coluna].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_mask = (dados[coluna] < lower_bound) | (dados[coluna] > upper_bound)
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

# Função para criar gráfico Q-Q correto
def criar_qq_plot_correto(data):
    """Cria gráfico Q-Q correto passando pelo meio dos pontos"""
    data_clean = data.dropna()
    if len(data_clean) < 2:
        return go.Figure()
    
    # Calcular quantis teóricos normais
    n = len(data_clean)
    theoretical_quantiles = np.sort(np.random.normal(np.mean(data_clean), np.std(data_clean), n))
    sample_quantiles = np.sort(data_clean)
    
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

def main():
    st.title("📊 Dashboard de Utilidades - Análise Completa")
    
    # Inicializar estado da sessão
    if 'dados_originais' not in st.session_state:
        st.session_state.dados_originais = None
    if 'dados_processados' not in st.session_state:
        st.session_state.dados_processados = None
    if 'filtro_data_limpo' not in st.session_state:
        st.session_state.filtro_data_limpo = False
    if 'outliers_removidos' not in st.session_state:
        st.session_state.outliers_removidos = {}
    
    # Sidebar para upload
    with st.sidebar:
        st.header("📁 Carregamento de Dados")
        
        uploaded_file = st.file_uploader(
            "Selecione o arquivo Excel:",
            type=['xlsx', 'xls']
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
        if any(palavra in col.lower() for palavra in ['data', 'date', 'dia', 'time']):
            colunas_data.append(col)
            dados_processados[col] = converter_para_data(dados_processados[col])

    # Sidebar para filtros globais
    with st.sidebar:
        st.header("🎛️ Filtros Globais")
        
        # Botão para resetar todos os filtros
        if st.button("🔄 Resetar Todos os Filtros", use_container_width=True):
            st.session_state.dados_processados = st.session_state.dados_originais.copy()
            st.session_state.filtro_data_limpo = False
            st.session_state.outliers_removidos = {}
            st.rerun()
        
        # Filtro de período
        if colunas_data:
            coluna_data_filtro = st.selectbox("Coluna de data para filtro:", colunas_data)
            
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
                        max_value=max_date
                    )
                
                # Botão para limpar filtro de data
                if st.button("❌ Limpar Filtro de Data", use_container_width=True):
                    st.session_state.filtro_data_limpo = True
                    st.rerun()
                
                if len(date_range) == 2 and not st.session_state.filtro_data_limpo:
                    start_date, end_date = date_range
                    dados_processados = dados_processados[
                        (dados_processados[coluna_data_filtro] >= pd.Timestamp(start_date)) &
                        (dados_processados[coluna_data_filtro] <= pd.Timestamp(end_date))
                    ]
        
        # Filtro de outliers - AGORA FUNCIONAL
        st.subheader("🔍 Gerenciamento de Outliers")
        
        if colunas_numericas:
            coluna_outliers = st.selectbox("Selecione a coluna para análise de outliers:", colunas_numericas)
            
            if coluna_outliers:
                # Detectar outliers
                outliers_df, outliers_mask = detectar_outliers(dados_processados, coluna_outliers)
                st.info(f"📊 {len(outliers_df)} outliers detectados na coluna '{coluna_outliers}'")
                
                # Mostrar outliers
                if len(outliers_df) > 0:
                    with st.expander("📋 Visualizar Outliers Detectados"):
                        st.dataframe(outliers_df[[coluna_outliers]].style.format({
                            coluna_outliers: '{:.2f}'
                        }))
                
                # Opção para remover outliers
                if st.button(f"🗑️ Remover Outliers de '{coluna_outliers}'", use_container_width=True):
                    dados_sem_outliers = dados_processados[~outliers_mask]
                    st.session_state.dados_processados = dados_sem_outliers
                    st.session_state.outliers_removidos[coluna_outliers] = True
                    st.success(f"✅ {len(outliers_df)} outliers removidos da coluna '{coluna_outliers}'")
                    st.rerun()

    # Abas principais
    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Análise de Séries Temporais", 
        "📊 Estatística Detalhada", 
        "🔥 Análise de Correlações", 
        "🔍 Gráficos de Dispersão"
    ])

    with tab1:
        st.header("📈 Análise de Séries Temporais")
        
        if colunas_data and colunas_numericas:
            col1, col2, col3 = st.columns([2, 2, 1])
            
            with col1:
                coluna_data = st.selectbox("Coluna de Data:", colunas_data, key="temp_data")
            with col2:
                coluna_valor = st.selectbox("Coluna para Análise:", colunas_numericas, key="temp_valor")
            with col3:
                tipo_grafico = st.selectbox("Tipo de Gráfico:", 
                                           ["Linha", "Área", "Barra", "Scatter", "Boxplot Temporal"])
            
            if coluna_data and coluna_valor:
                dados_temp = dados_processados.sort_values(by=coluna_data)
                
                # Opção para remover outliers diretamente no gráfico
                remover_outliers_grafico = st.checkbox("📉 Remover outliers deste gráfico")
                
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
                else:  # Boxplot Temporal
                    # Criar períodos mensais para boxplot
                    dados_temp['Periodo'] = dados_temp[coluna_data].dt.to_period('M').astype(str)
                    fig = px.box(dados_temp, x='Periodo', y=coluna_valor,
                                title=f"Distribuição Mensal de {coluna_valor}")
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Estatísticas temporais COMPLETAS
                st.subheader("📊 Estatísticas Temporais Detalhadas")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Média", f"{dados_temp[coluna_valor].mean():.2f}")
                    st.metric("Mediana", f"{dados_temp[coluna_valor].median():.2f}")
                    st.metric("Moda", f"{dados_temp[coluna_valor].mode().iloc[0] if not dados_temp[coluna_valor].mode().empty else 'N/A'}")
                
                with col2:
                    st.metric("Desvio Padrão", f"{dados_temp[coluna_valor].std():.2f}")
                    st.metric("Variância", f"{dados_temp[coluna_valor].var():.2f}")
                    st.metric("Coef. Variação", f"{(dados_temp[coluna_valor].std()/dados_temp[coluna_valor].mean())*100:.1f}%")
                
                with col3:
                    st.metric("Mínimo", f"{dados_temp[coluna_valor].min():.2f}")
                    st.metric("Máximo", f"{dados_temp[coluna_valor].max():.2f}")
                    st.metric("Amplitude", f"{dados_temp[coluna_valor].max() - dados_temp[coluna_valor].min():.2f}")
                
                with col4:
                    Q1 = dados_temp[coluna_valor].quantile(0.25)
                    Q3 = dados_temp[coluna_valor].quantile(0.75)
                    st.metric("Q1 (25%)", f"{Q1:.2f}")
                    st.metric("Q3 (75%)", f"{Q3:.2f}")
                    st.metric("IQR", f"{Q3 - Q1:.2f}")
                
                # Análise de tendência
                st.subheader("📈 Análise de Tendência")
                if len(dados_temp) > 1:
                    crescimento = ((dados_temp[coluna_valor].iloc[-1] - dados_temp[coluna_valor].iloc[0]) / dados_temp[coluna_valor].iloc[0] * 100) if dados_temp[coluna_valor].iloc[0] != 0 else 0
                    
                    col_t1, col_t2, col_t3 = st.columns(3)
                    with col_t1:
                        st.metric("Crescimento Total", f"{crescimento:.1f}%")
                    with col_t2:
                        # Tendência linear simples
                        x = np.arange(len(dados_temp))
                        y = dados_temp[coluna_valor].values
                        coef = np.polyfit(x, y, 1)[0]
                        tendencia = "↗️ Alta" if coef > 0 else "↘️ Baixa" if coef < 0 else "➡️ Estável"
                        st.metric("Tendência", tendencia)
                    with col_t3:
                        st.metric("Taxa de Variação", f"{coef:.4f}")

    with tab2:
        st.header("📊 Estatística Detalhada")
        
        if colunas_numericas:
            coluna_analise = st.selectbox("Selecione a coluna para análise:", colunas_numericas, key="stats_col")
            
            if coluna_analise:
                # Opção para remover outliers diretamente no gráfico
                remover_outliers_grafico = st.checkbox("📉 Remover outliers para análise")
                
                dados_analise = dados_processados.copy()
                if remover_outliers_grafico:
                    outliers_df, outliers_mask = detectar_outliers(dados_analise, coluna_analise)
                    dados_analise = dados_analise[~outliers_mask]
                    st.info(f"📊 {len(outliers_df)} outliers removidos para análise")
                
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
                            st.metric(name, f"{value:.2f}" if isinstance(value, (int, float)) else str(value))
                
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
                                      nbins=30, marginal="box")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Gráfico Q-Q CORRIGIDO
                st.subheader("📊 Gráfico Q-Q (Análise de Normalidade)")
                fig_qq = criar_qq_plot_correto(dados_analise[coluna_analise])
                st.plotly_chart(fig_qq, use_container_width=True)
                
                # Análise de outliers
                st.subheader("🔍 Análise de Outliers")
                outliers_df, outliers_mask = detectar_outliers(dados_processados, coluna_analise)
                st.metric("Número de Outliers", len(outliers_df))
                
                if len(outliers_df) > 0:
                    with st.expander("📋 Detalhes dos Outliers"):
                        st.dataframe(outliers_df[[coluna_analise]].style.format({
                            coluna_analise: '{:.2f}'
                        }))

    with tab3:
        st.header("🔥 Análise de Correlações")
        
        if len(colunas_numericas) > 1:
            # Selecionar variáveis para correlação
            st.subheader("🎯 Seleção de Variáveis")
            variaveis_selecionadas = st.multiselect(
                "Selecione as variáveis para análise de correlação:",
                options=colunas_numericas,
                default=colunas_numericas[:min(8, len(colunas_numericas))],
                key="corr_vars"
            )
            
            if len(variaveis_selecionadas) > 1:
                # Opção para remover outliers das correlações
                remover_outliers_corr = st.checkbox("📉 Remover outliers para análise de correlação")
                
                dados_corr = dados_processados.copy()
                if remover_outliers_corr:
                    for var in variaveis_selecionadas:
                        outliers_df, outliers_mask = detectar_outliers(dados_corr, var)
                        dados_corr = dados_corr[~outliers_mask]
                    st.info("Outliers removidos de todas as variáveis selecionadas")
                
                # Matriz de correlação
                corr_matrix = dados_corr[variaveis_selecionadas].corr()
                
                fig = px.imshow(corr_matrix, 
                               title="Matriz de Correlação",
                               color_continuous_scale="RdBu_r",
                               aspect="auto",
                               text_auto=True)
                st.plotly_chart(fig, use_container_width=True)
                
                # Top correlações DETALHADO
                st.subheader("🔝 Top 10 Maiores e Menores Correlações")
                
                correlations = []
                for i in range(len(corr_matrix.columns)):
                    for j in range(i+1, len(corr_matrix.columns)):
                        correlations.append({
                            'Variável 1': corr_matrix.columns[i],
                            'Variável 2': corr_matrix.columns[j],
                            'Correlação': corr_matrix.iloc[i, j]
                        })
                
                df_corr = pd.DataFrame(correlations)
                df_corr['Abs_Correlation'] = df_corr['Correlação'].abs()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**📈 10 Maiores Correlações:**")
                    top_correlations = df_corr.nlargest(10, 'Abs_Correlation')
                    for _, row in top_correlations.iterrows():
                        corr_color = "🟢" if row['Correlação'] > 0 else "🔴"
                        corr_strength = "Forte" if abs(row['Correlação']) > 0.7 else "Moderada" if abs(row['Correlação']) > 0.3 else "Fraca"
                        st.write(f"{corr_color} **{row['Correlação']:.3f}** - {corr_strength}")
                        st.write(f"   {row['Variável 1']} ↔ {row['Variável 2']}")
                        st.write("---")
                
                with col2:
                    st.write("**📉 10 Menores Correlações:**")
                    bottom_correlations = df_corr.nsmallest(10, 'Abs_Correlation')
                    for _, row in bottom_correlations.iterrows():
                        corr_color = "🟢" if row['Correlação'] > 0 else "🔴"
                        corr_strength = "Forte" if abs(row['Correlação']) > 0.7 else "Moderada" if abs(row['Correlação']) > 0.3 else "Fraca"
                        st.write(f"{corr_color} **{row['Correlação']:.3f}** - {corr_strength}")
                        st.write(f"   {row['Variável 1']} ↔ {row['Variável 2']}")
                        st.write("---")

    with tab4:
        st.header("🔍 Gráficos de Dispersão com Regressão")
        
        if len(colunas_numericas) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                eixo_x = st.selectbox("Eixo X:", colunas_numericas, key="scatter_x")
            with col2:
                eixo_y = st.selectbox("Eixo Y:", colunas_numericas, key="scatter_y")
            
            if eixo_x and eixo_y:
                # Opção para remover outliers diretamente no gráfico
                remover_outliers_grafico = st.checkbox("📉 Remover outliers deste gráfico")
                
                dados_scatter = dados_processados.copy()
                if remover_outliers_grafico:
                    outliers_x, outliers_mask_x = detectar_outliers(dados_scatter, eixo_x)
                    outliers_y, outliers_mask_y = detectar_outliers(dados_scatter, eixo_y)
                    outliers_mask = outliers_mask_x | outliers_mask_y
                    dados_scatter = dados_scatter[~outliers_mask]
                    st.info(f"📊 {outliers_mask.sum()} outliers removidos para visualização")
                
                # Gráfico de dispersão
                fig = px.scatter(dados_scatter, x=eixo_x, y=eixo_y, 
                                title=f"{eixo_y} vs {eixo_x}")
                
                # Calcular regressão linear manualmente
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
                    
                    # Adicionar equação da reta BEM VISÍVEL
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
                
                correlacao = dados_scatter[eixo_x].corr(dados_scatter[eixo_y])
                
                col_stat1, col_stat2, col_stat3 = st.columns(3)
                with col_stat1:
                    st.metric("Coeficiente de Correlação", f"{correlacao:.4f}")
                with col_stat2:
                    if r_squared is not None:
                        st.metric("Coeficiente de Determinação (R²)", f"{r_squared:.4f}")
                with col_stat3:
                    if slope is not None:
                        st.metric("Inclinação da Reta", f"{slope:.4f}")
                
                # Mostrar equação da reta em destaque
                if slope is not None and intercept is not None:
                    st.subheader("🧮 Equação da Reta de Regressão")
                    st.info(f"**{equation}**")
                    st.info(f"**{r2_text}**")
                
                # Interpretação detalhada
                st.subheader("📝 Interpretação da Correlação")
                
                if abs(correlacao) > 0.7:
                    st.success("**Correlação Forte**")
                    st.write("• Relação muito significativa entre as variáveis")
                    st.write("• Pode indicar causalidade ou forte dependência")
                elif abs(correlacao) > 0.3:
                    st.info("**Correlação Moderada**")
                    st.write("• Relação moderadamente significativa")
                    st.write("• Pode indicar tendência ou influência parcial")
                else:
                    st.warning("**Correlação Fraca**")
                    st.write("• Relação fraca ou inexistente")
                    st.write("• Variáveis praticamente independentes")

    # Download dos dados processados
    st.sidebar.header("💾 Exportar Dados")
    csv = dados_processados.to_csv(index=False)
    st.sidebar.download_button(
        label="📥 Baixar dados processados",
        data=csv,
        file_name="dados_processados.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()

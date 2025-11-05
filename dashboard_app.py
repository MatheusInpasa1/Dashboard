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

# ========== FUNÇÕES PARA ANÁLISE DE CAPABILIDADE ==========

def calcular_indices_capabilidade(dados, coluna, lse, lie):
    """Calcula todos os índices de capabilidade"""
    if coluna not in dados.columns:
        return None
    
    data_clean = dados[coluna].dropna()
    if len(data_clean) < 2:
        return None
    
    media = np.mean(data_clean)
    desvio_padrao = np.std(data_clean, ddof=1)
    variancia = np.var(data_clean, ddof=1)
    
    resultados = {
        'media': media,
        'desvio_padrao': desvio_padrao,
        'variancia': variancia,
        'n': len(data_clean),
        'minimo': np.min(data_clean),
        'maximo': np.max(data_clean),
        'amplitude': np.max(data_clean) - np.min(data_clean)
    }
    
    if lse is not None and lie is not None and lse > lie and desvio_padrao > 0:
        # Cp - Capacidade potencial do processo
        cp = (lse - lie) / (6 * desvio_padrao)
        
        # Cpk - Capacidade real do processo
        cpk_superior = (lse - media) / (3 * desvio_padrao)
        cpk_inferior = (media - lie) / (3 * desvio_padrao)
        cpk = min(cpk_superior, cpk_inferior)
        
        # Cpm - Capabilidade considerando o alvo (assume alvo no centro)
        alvo = (lse + lie) / 2
        cpm = (lse - lie) / (6 * np.sqrt(desvio_padrao**2 + (media - alvo)**2))
        
        # Pp - Performance potencial do processo
        pp = (lse - lie) / (6 * desvio_padrao)
        
        # Ppk - Performance real do processo
        ppk_superior = (lse - media) / (3 * desvio_padrao)
        ppk_inferior = (media - lie) / (3 * desvio_padrao)
        ppk = min(ppk_superior, ppk_inferior)
        
        # Percentual fora das especificações
        z_superior = (lse - media) / desvio_padrao
        z_inferior = (media - lie) / desvio_padrao
        
        # Estimativa de percentual fora (usando distribuição normal)
        from scipy.stats import norm
        pct_fora_superior = (1 - norm.cdf(z_superior)) * 100
        pct_fora_inferior = norm.cdf(-z_inferior) * 100
        pct_total_fora = pct_fora_superior + pct_fora_inferior
        
        resultados.update({
            'lse': lse,
            'lie': lie,
            'alvo': alvo,
            'cp': cp,
            'cpk': cpk,
            'cpm': cpm,
            'pp': pp,
            'ppk': ppk,
            'cpk_superior': cpk_superior,
            'cpk_inferior': cpk_inferior,
            'z_superior': z_superior,
            'z_inferior': z_inferior,
            'pct_fora_superior': pct_fora_superior,
            'pct_fora_inferior': pct_fora_inferior,
            'pct_total_fora': pct_total_fora,
            'ppm_superior': pct_fora_superior * 10000,
            'ppm_inferior': pct_fora_inferior * 10000,
            'ppm_total': pct_total_fora * 10000
        })
    
    return resultados

def criar_histograma_capabilidade(dados, coluna, lse, lie, resultados):
    """Cria histograma com limites de especificação"""
    if coluna not in dados.columns:
        return go.Figure()
    
    data_clean = dados[coluna].dropna()
    
    fig = go.Figure()
    
    # Histograma
    fig.add_trace(go.Histogram(
        x=data_clean,
        nbinsx=30,
        name='Distribuição',
        opacity=0.7,
        marker_color='lightblue'
    ))
    
    # Linha de densidade
    fig.add_trace(go.Scatter(
        x=np.linspace(data_clean.min(), data_clean.max(), 100),
        y=stats.norm.pdf(np.linspace(data_clean.min(), data_clean.max(), 100), 
                        data_clean.mean(), data_clean.std()),
        mode='lines',
        name='Curva Normal',
        line=dict(color='red', width=2)
    ))
    
    # Limites de especificação
    if lse is not None:
        fig.add_vline(x=lse, line_dash="dash", line_color="red", 
                     annotation_text="LSE", annotation_position="top")
    
    if lie is not None:
        fig.add_vline(x=lie, line_dash="dash", line_color="red",
                     annotation_text="LIE", annotation_position="top")
    
    # Média do processo
    fig.add_vline(x=data_clean.mean(), line_dash="solid", line_color="green",
                 annotation_text="Média", annotation_position="bottom")
    
    # Alvo (centro das especificações)
    if lse is not None and lie is not None:
        alvo = (lse + lie) / 2
        fig.add_vline(x=alvo, line_dash="dot", line_color="orange",
                     annotation_text="Alvo", annotation_position="bottom")
    
    fig.update_layout(
        title=f"Histograma de Capabilidade - {coluna}",
        xaxis_title=coluna,
        yaxis_title="Frequência",
        showlegend=True,
        height=500
    )
    
    return fig

def criar_grafico_controle_capabilidade(dados, coluna, lse, lie, resultados):
    """Cria gráfico de controle para análise de capabilidade"""
    if coluna not in dados.columns:
        return go.Figure()
    
    data_clean = dados[coluna].dropna()
    
    fig = go.Figure()
    
    # Dados do processo
    fig.add_trace(go.Scatter(
        x=list(range(len(data_clean))),
        y=data_clean,
        mode='lines+markers',
        name='Valores',
        line=dict(color='blue', width=1),
        marker=dict(size=4)
    ))
    
    # Média do processo
    media = data_clean.mean()
    fig.add_hline(y=media, line_dash="solid", line_color="green",
                 annotation_text="Média", annotation_position="right")
    
    # Limites de especificação
    if lse is not None:
        fig.add_hline(y=lse, line_dash="dash", line_color="red",
                     annotation_text="LSE", annotation_position="right")
    
    if lie is not None:
        fig.add_hline(y=lie, line_dash="dash", line_color="red",
                     annotation_text="LIE", annotation_position="right")
    
    # Alvo
    if lse is not None and lie is not None:
        alvo = (lse + lie) / 2
        fig.add_hline(y=alvo, line_dash="dot", line_color="orange",
                     annotation_text="Alvo", annotation_position="right")
    
    fig.update_layout(
        title=f"Gráfico de Controle - {coluna}",
        xaxis_title="Amostra",
        yaxis_title=coluna,
        showlegend=True,
        height=400
    )
    
    return fig

def interpretar_capabilidade(resultados):
    """Fornece interpretação dos índices de capabilidade"""
    if not resultados or 'cpk' not in resultados:
        return "Dados insuficientes para análise"
    
    cpk = resultados['cpk']
    cp = resultados.get('cp', cpk)
    
    interpretacao = ""
    
    # Interpretação Cpk
    if cpk >= 1.67:
        interpretacao += "✅ **Excelente** - Processo altamente capaz (Cpk ≥ 1.67)\n"
    elif cpk >= 1.33:
        interpretacao += "✅ **Muito Bom** - Processo capaz (1.33 ≤ Cpk < 1.67)\n"
    elif cpk >= 1.0:
        interpretacao += "⚠️ **Aceitável** - Processo marginalmente capaz (1.0 ≤ Cpk < 1.33)\n"
    elif cpk >= 0.67:
        interpretacao += "❌ **Insatisfatório** - Processo incapaz (0.67 ≤ Cpk < 1.0)\n"
    else:
        interpretacao += "🚨 **Crítico** - Processo totalmente incapaz (Cpk < 0.67)\n"
    
    # Comparação Cp vs Cpk
    if 'cp' in resultados:
        diferenca = resultados['cp'] - resultados['cpk']
        if diferenca > 0.5:
            interpretacao += "\n📊 **Processo descentrado** - Grande diferença entre Cp e Cpk indica que o processo não está centrado\n"
        elif diferenca > 0.2:
            interpretacao += "\n📊 **Processo levemente descentrado** - Pequena diferença entre Cp e Cpk\n"
        else:
            interpretacao += "\n📊 **Processo bem centrado** - Cp e Cpk próximos indicam bom centramento\n"
    
    # Análise de capacidade
    if cpk >= 1.33:
        interpretacao += "\n🎯 **Recomendações:** Processo sob controle, mantenha monitoramento\n"
    elif cpk >= 1.0:
        interpretacao += "\n🎯 **Recomendações:** Melhorar centramento do processo\n"
    else:
        interpretacao += "\n🎯 **Recomendações:** Reduzir variabilidade e melhorar centramento\n"
    
    return interpretacao

# ========== FIM DAS FUNÇÕES DE CAPABILIDADE ==========

# ========== FUNÇÕES PARA CARTA DE CONTROLE ==========

def criar_carta_controle_xbar_s(dados, coluna_valor, coluna_grupo=None, tamanho_amostra=5):
    """Cria carta de controle X-bar e S"""
    if coluna_valor not in dados.columns:
        return None, None, None, None, None
    
    dados_clean = dados.copy()
    
    # Se não há coluna de grupo, criar grupos sequenciais
    if coluna_grupo is None or coluna_grupo not in dados.columns:
        dados_clean['Grupo'] = (np.arange(len(dados_clean)) // tamanho_amostra) + 1
        coluna_grupo = 'Grupo'
    
    # Agrupar dados
    grupos = dados_clean.groupby(coluna_grupo)[coluna_valor]
    
    # Calcular estatísticas por grupo
    xbar = grupos.mean()  # Média do grupo
    s = grupos.std(ddof=1)  # Desvio padrão do grupo
    n = grupos.count()  # Tamanho do grupo
    
    # Coeficientes para carta de controle
    A3 = {2: 2.659, 3: 1.954, 4: 1.628, 5: 1.427, 6: 1.287, 7: 1.182, 8: 1.099, 9: 1.032, 10: 0.975}
    B3 = {2: 0, 3: 0, 4: 0, 5: 0, 6: 0.030, 7: 0.118, 8: 0.185, 9: 0.239, 10: 0.284}
    B4 = {2: 3.267, 3: 2.568, 4: 2.266, 5: 2.089, 6: 1.970, 7: 1.882, 8: 1.815, 9: 1.761, 10: 1.716}
    
    # Usar coeficiente para n=5 como padrão se n variar
    n_medio = int(n.mean())
    coef_A3 = A3.get(n_medio, 1.427)
    coef_B3 = B3.get(n_medio, 0)
    coef_B4 = B4.get(n_medio, 2.089)
    
    # Linhas de controle para X-bar
    xbar_media = xbar.mean()
    s_media = s.mean()
    
    LSC_xbar = xbar_media + coef_A3 * s_media
    LIC_xbar = xbar_media - coef_A3 * s_media
    
    # Linhas de controle para S
    LSC_s = coef_B4 * s_media
    LIC_s = coef_B3 * s_media
    
    return xbar, s, n, (LSC_xbar, xbar_media, LIC_xbar), (LSC_s, s_media, LIC_s)

def criar_carta_controle_individual(dados, coluna_valor, coluna_tempo=None):
    """Cria carta de controle para dados individuais (I-MR)"""
    if coluna_valor not in dados.columns:
        return None, None, None, None, None
    
    dados_clean = dados.copy().sort_values(coluna_tempo) if coluna_tempo else dados.copy()
    
    # Dados individuais
    individuais = dados_clean[coluna_valor]
    
    # Amplitude móvel (MR)
    mr = individuais.diff().abs()
    
    # Linhas de controle para dados individuais
    media_i = individuais.mean()
    mr_media = mr.mean()
    
    LSC_i = media_i + 2.66 * mr_media
    LIC_i = media_i - 2.66 * mr_media
    
    # Linhas de controle para amplitude móvel
    LSC_mr = 3.267 * mr_media
    LIC_mr = 0
    
    return individuais, mr, (LSC_i, media_i, LIC_i), (LSC_mr, mr_media, LIC_mr)

def criar_carta_controle_p(dados, coluna_defeitos, coluna_tamanho_amostra, coluna_grupo=None):
    """Cria carta de controle P (proporção de defeituosos)"""
    if coluna_defeitos not in dados.columns or coluna_tamanho_amostra not in dados.columns:
        return None, None, None
    
    dados_clean = dados.copy()
    
    # Se não há coluna de grupo, criar grupos sequenciais
    if coluna_grupo is None or coluna_grupo not in dados.columns:
        dados_clean['Grupo'] = np.arange(len(dados_clean)) + 1
        coluna_grupo = 'Grupo'
    
    # Agrupar dados
    grupos = dados_clean.groupby(coluna_grupo)
    
    # Calcular proporção de defeituosos
    p = grupos[coluna_defeitos].sum() / grupos[coluna_tamanho_amostra].sum()
    n = grupos[coluna_tamanho_amostra].mean()
    
    # Linhas de controle
    p_media = p.mean()
    n_medio = n.mean()
    
    LSC_p = p_media + 3 * np.sqrt(p_media * (1 - p_media) / n_medio)
    LIC_p = max(0, p_media - 3 * np.sqrt(p_media * (1 - p_media) / n_medio))
    
    return p, n, (LSC_p, p_media, LIC_p)

def criar_carta_controle_c(dados, coluna_defeitos, coluna_grupo=None):
    """Cria carta de controle C (número de defeitos)"""
    if coluna_defeitos not in dados.columns:
        return None, None
    
    dados_clean = dados.copy()
    
    # Se não há coluna de grupo, criar grupos sequenciais
    if coluna_grupo is None or coluna_grupo not in dados.columns:
        dados_clean['Grupo'] = np.arange(len(dados_clean)) + 1
        coluna_grupo = 'Grupo'
    
    # Agrupar dados
    grupos = dados_clean.groupby(coluna_grupo)
    
    # Número de defeitos por grupo
    c = grupos[coluna_defeitos].sum()
    
    # Linhas de controle
    c_media = c.mean()
    
    LSC_c = c_media + 3 * np.sqrt(c_media)
    LIC_c = max(0, c_media - 3 * np.sqrt(c_media))
    
    return c, (LSC_c, c_media, LIC_c)

def plotar_carta_controle(valores, limites, titulo, tipo="individual"):
    """Plota uma carta de controle"""
    LSC, LC, LIC = limites
    
    fig = go.Figure()
    
    # Adicionar pontos
    fig.add_trace(go.Scatter(
        x=list(range(1, len(valores) + 1)),
        y=valores,
        mode='lines+markers',
        name='Valores',
        line=dict(color='blue', width=2),
        marker=dict(size=6)
    ))
    
    # Adicionar linhas de controle
    fig.add_hline(y=LSC, line_dash="dash", line_color="red", 
                  annotation_text="LSC", annotation_position="right")
    fig.add_hline(y=LC, line_dash="dash", line_color="green", 
                  annotation_text="LC", annotation_position="right")
    fig.add_hline(y=LIC, line_dash="dash", line_color="red", 
                  annotation_text="LIC", annotation_position="right")
    
    # Destacar pontos fora de controle
    pontos_fora = (valores > LSC) | (valores < LIC)
    if pontos_fora.any():
        indices_fora = np.where(pontos_fora)[0] + 1
        valores_fora = valores[pontos_fora]
        
        fig.add_trace(go.Scatter(
            x=indices_fora,
            y=valores_fora,
            mode='markers',
            name='Fora de Controle',
            marker=dict(color='red', size=10, symbol='x')
        ))
    
    fig.update_layout(
        title=titulo,
        xaxis_title="Amostra/Grupo",
        yaxis_title="Valor",
        showlegend=True,
        height=500,
        title_font_size=20,
        xaxis_title_font_size=16,
        yaxis_title_font_size=16
    )
    
    return fig, pontos_fora.sum()

# ========== FIM DAS FUNÇÕES DE CARTA DE CONTROLE ==========

# Função para calcular regressão linear manualmente
def calcular_regressao_linear(x, y):
    """Calcula regressão linear manualmente"""
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
    sample_mean = np.mean(sample_quantiles)
    sample_std = np.std(sample_quantiles)
    if sample_std > 0:
        sample_quantiles = (sample_quantiles - sample_mean) / sample_std
    
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
    
    if lse is not None and lie is not None and lse > lie and desvio_padrao > 0:
        # Cp - Capacidade do processo
        cp = (lse - lie) / (6 * desvio_padrao)
        # Cpk - Capacidade real do processo
        cpk_u = (lse - media) / (3 * desvio_padrao)
        cpk_l = (media - lie) / (3 * desvio_padrao)
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
    if coluna_data and coluna_data in data_clean.columns:
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

# Importar scipy.stats para análise de capabilidade
try:
    from scipy import stats
except ImportError:
    # Implementação alternativa se scipy não estiver disponível
    import math
    class stats:
        class norm:
            @staticmethod
            def cdf(x):
                """Aproximação da função de distribuição acumulada normal"""
                return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

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
    colunas_todas = dados_processados.columns.tolist()
    
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
        
        # Filtro de outliers
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

    # Abas principais - AGORA COM ANÁLISE DE CAPABILIDADE
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Análise Temporal", 
        "📊 Estatística Detalhada", 
        "🔥 Correlações", 
        "🔍 Dispersão & Regressão",
        "🎯 Carta de Controle",
        "📈 Controle Estatístico",
        "📊 Análise de Capabilidade",  # NOVA ABA
        "📋 Resumo Executivo"
    ])

    # ========== ABA 1-6 (MANTIDAS IGUAIS) ==========
    # ... (código das abas 1-6 permanece igual ao anterior)

    # ========== NOVA ABA 7: ANÁLISE DE CAPABILIDADE ==========
    with tab7:
        st.header("📊 Análise de Capabilidade do Processo")
        
        st.markdown("""
        **Análise de Capabilidade** avalia a capacidade de um processo em produzir dentro dos limites de especificação.
        Esta análise calcula índices como Cp, Cpk, Pp, Ppk e estima o percentual de produtos fora da especificação.
        """)
        
        if colunas_numericas:
            # Seleção da variável para análise
            coluna_capabilidade = st.selectbox(
                "Selecione a variável para análise de capabilidade:",
                colunas_numericas,
                key=generate_unique_key("capabilidade_col", "tab7")
            )
            
            # Configuração dos limites
            st.subheader("🎯 Configuração dos Limites de Especificação")
            
            col_lim1, col_lim2, col_lim3 = st.columns(3)
            with col_lim1:
                lse_cap = st.number_input(
                    "LSE (Limite Superior de Especificação):",
                    value=float(st.session_state.lse_values.get(coluna_capabilidade, 0)),
                    key=generate_unique_key("lse_cap", coluna_capabilidade)
                )
            
            with col_lim2:
                lie_cap = st.number_input(
                    "LIE (Limite Inferior de Especificação):",
                    value=float(st.session_state.lie_values.get(coluna_capabilidade, 0)),
                    key=generate_unique_key("lie_cap", coluna_capabilidade)
                )
            
            with col_lim3:
                alvo_cap = st.number_input(
                    "Alvo (Valor Ideal - Opcional):",
                    value=float((lse_cap + lie_cap) / 2 if lse_cap != 0 and lie_cap != 0 else 0),
                    key=generate_unique_key("alvo_cap", coluna_capabilidade)
                )
            
            # Botão para executar análise
            if st.button("📈 Executar Análise de Capabilidade", use_container_width=True,
                        key=generate_unique_key("executar_capabilidade", "tab7")):
                
                if lse_cap == 0 and lie_cap == 0:
                    st.error("❌ É necessário definir pelo menos um limite de especificação (LSE ou LIE)")
                else:
                    try:
                        # Calcular índices de capabilidade
                        resultados = calcular_indices_capabilidade(
                            dados_processados, coluna_capabilidade, lse_cap, lie_cap
                        )
                        
                        if resultados:
                            # Gráficos
                            st.subheader("📊 Visualizações da Capabilidade")
                            
                            col_graf1, col_graf2 = st.columns(2)
                            
                            with col_graf1:
                                # Histograma de capabilidade
                                fig_hist = criar_histograma_capabilidade(
                                    dados_processados, coluna_capabilidade, lse_cap, lie_cap, resultados
                                )
                                st.plotly_chart(fig_hist, use_container_width=True)
                            
                            with col_graf2:
                                # Gráfico de controle
                                fig_controle = criar_grafico_controle_capabilidade(
                                    dados_processados, coluna_capabilidade, lse_cap, lie_cap, resultados
                                )
                                st.plotly_chart(fig_controle, use_container_width=True)
                            
                            # Índices de Capabilidade
                            st.subheader("🎯 Índices de Capabilidade")
                            
                            col_idx1, col_idx2, col_idx3, col_idx4 = st.columns(4)
                            
                            with col_idx1:
                                st.metric("Cp (Capabilidade Potencial)", 
                                         f"{resultados.get('cp', 0):.3f}" if 'cp' in resultados else "N/A")
                                st.metric("Cpk (Capabilidade Real)", 
                                         f"{resultados.get('cpk', 0):.3f}" if 'cpk' in resultados else "N/A")
                            
                            with col_idx2:
                                st.metric("Pp (Performance Potencial)", 
                                         f"{resultados.get('pp', 0):.3f}" if 'pp' in resultados else "N/A")
                                st.metric("Ppk (Performance Real)", 
                                         f"{resultados.get('ppk', 0):.3f}" if 'ppk' in resultados else "N/A")
                            
                            with col_idx3:
                                st.metric("Cpm (Capabilidade com Alvo)", 
                                         f"{resultados.get('cpm', 0):.3f}" if 'cpm' in resultados else "N/A")
                                st.metric("K (Índice de Descentramento)", 
                                         f"{abs(resultados.get('cp', 0) - resultados.get('cpk', 0)):.3f}" 
                                         if 'cp' in resultados and 'cpk' in resultados else "N/A")
                            
                            with col_idx4:
                                st.metric("Z Superior", 
                                         f"{resultados.get('z_superior', 0):.2f}" if 'z_superior' in resultados else "N/A")
                                st.metric("Z Inferior", 
                                         f"{resultados.get('z_inferior', 0):.2f}" if 'z_inferior' in resultados else "N/A")
                            
                            # Estatísticas do Processo
                            st.subheader("📈 Estatísticas do Processo")
                            
                            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                            
                            with col_stat1:
                                st.metric("Média do Processo", f"{resultados['media']:.4f}")
                                st.metric("Desvio Padrão", f"{resultados['desvio_padrao']:.4f}")
                            
                            with col_stat2:
                                st.metric("LSE", f"{lse_cap:.4f}")
                                st.metric("LIE", f"{lie_cap:.4f}")
                            
                            with col_stat3:
                                st.metric("Amplitude", f"{resultados['amplitude']:.4f}")
                                st.metric("Número de Amostras", resultados['n'])
                            
                            with col_stat4:
                                if 'alvo' in resultados:
                                    st.metric("Alvo", f"{resultados['alvo']:.4f}")
                                st.metric("Variação", f"{resultados['variancia']:.4f}")
                            
                            # Análise de Não-Conformidades
                            st.subheader("🚨 Análise de Não-Conformidades")
                            
                            col_nc1, col_nc2, col_nc3 = st.columns(3)
                            
                            with col_nc1:
                                if 'pct_fora_superior' in resultados:
                                    st.metric("% Acima do LSE", f"{resultados['pct_fora_superior']:.4f}%")
                                    st.metric("PPM Acima do LSE", f"{resultados['ppm_superior']:.0f}")
                            
                            with col_nc2:
                                if 'pct_fora_inferior' in resultados:
                                    st.metric("% Abaixo do LIE", f"{resultados['pct_fora_inferior']:.4f}%")
                                    st.metric("PPM Abaixo do LIE", f"{resultados['ppm_inferior']:.0f}")
                            
                            with col_nc3:
                                if 'pct_total_fora' in resultados:
                                    st.metric("% Total Fora", f"{resultados['pct_total_fora']:.4f}%")
                                    st.metric("PPM Total", f"{resultados['ppm_total']:.0f}")
                            
                            # Interpretação
                            st.subheader("🔍 Interpretação da Capabilidade")
                            
                            interpretacao = interpretar_capabilidade(resultados)
                            st.info(interpretacao)
                            
                            # Tabela de Referência
                            st.subheader("📋 Tabela de Referência - Índices de Capabilidade")
                            
                            referencia = pd.DataFrame({
                                'Índice Cpk': ['≥ 1.67', '1.33 - 1.67', '1.0 - 1.33', '0.67 - 1.0', '< 0.67'],
                                'Classificação': ['Excelente', 'Adequado', 'Marginal', 'Inadequado', 'Inaceitável'],
                                'PPM Esperado': ['< 0.6', '0.6 - 63', '63 - 2700', '2700 - 45500', '> 45500'],
                                'Sigma Level': ['≥ 5σ', '4σ - 5σ', '3σ - 4σ', '2σ - 3σ', '< 2σ']
                            })
                            
                            st.dataframe(referencia, use_container_width=True)
                            
                        else:
                            st.error("❌ Não foi possível calcular os índices de capabilidade. Verifique os dados e limites.")
                    
                    except Exception as e:
                        st.error(f"❌ Erro na análise de capabilidade: {str(e)}")
                        st.info("💡 **Dica**: Verifique se os limites de especificação estão corretos e se há dados suficientes.")
        
        else:
            st.warning("📊 Não há variáveis numéricas para análise de capabilidade.")

    # ========== ABA 8: RESUMO EXECUTIVO ==========
    with tab8:
        st.header("📋 Resumo Executivo")
        
        # ... (código do resumo executivo permanece igual)

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

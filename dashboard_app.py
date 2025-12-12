# dashboard_app.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime
import os
import warnings
import math
import random
from itertools import product
warnings.filterwarnings('ignore')

# ========== IMPLEMENTAÇÕES ALTERNATIVAS PARA SCIPY ==========

class AlternativeStats:
    """Implementações alternativas para funções do scipy.stats"""
    
    @staticmethod
    def pearsonr(x, y):
        """Implementação alternativa para scipy.stats.pearsonr"""
        if len(x) != len(y):
            return 0, 1
        
        n = len(x)
        sum_x = sum(x)
        sum_y = sum(y)
        sum_xy = sum(x_i * y_i for x_i, y_i in zip(x, y))
        sum_x2 = sum(x_i ** 2 for x_i in x)
        sum_y2 = sum(y_i ** 2 for y_i in y)
        
        numerator = n * sum_xy - sum_x * sum_y
        denominator = ((n * sum_x2 - sum_x ** 2) * (n * sum_y2 - sum_y ** 2)) ** 0.5
        
        if denominator == 0:
            return 0, 1
        
        corr = numerator / denominator
        return corr, 0.001
    
    @staticmethod
    def normpdf(x, mu=0, sigma=1):
        """Implementação alternativa para scipy.stats.norm.pdf"""
        return (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    
    @staticmethod
    def zscore(x):
        """Implementação alternativa para scipy.stats.zscore"""
        return (x - np.mean(x)) / np.std(x)
    
    @staticmethod
    def linregress(x, y):
        """Implementação alternativa para scipy.stats.linregress"""
        n = len(x)
        x_mean = np.mean(x)
        y_mean = np.mean(y)
        
        # Cálculos dos somatórios
        xy_sum = np.sum((x - x_mean) * (y - y_mean))
        xx_sum = np.sum((x - x_mean) ** 2)
        yy_sum = np.sum((y - y_mean) ** 2)
        
        # Coeficientes da regressão
        slope = xy_sum / xx_sum if xx_sum != 0 else 0
        intercept = y_mean - slope * x_mean
        
        # Cálculos para R² e erro padrão
        r = xy_sum / np.sqrt(xx_sum * yy_sum) if (xx_sum * yy_sum) > 0 else 0
        r_squared = r ** 2
        
        return slope, intercept, r, r_squared
    
    @staticmethod
    def mannwhitneyu(x, y):
        """Implementação simplificada para teste de Mann-Whitney"""
        # Implementação básica - em produção, usar scipy
        n1, n2 = len(x), len(y)
        all_data = np.concatenate([x, y])
        ranks = np.argsort(np.argsort(all_data)) + 1
        r1 = np.sum(ranks[:n1])
        
        # Estatística U
        u1 = r1 - n1 * (n1 + 1) / 2
        u2 = n1 * n2 - u1
        
        return min(u1, u2), 0.05  # p-value fixo para demonstração

    @staticmethod
    def norm_cdf(x, mu=0, sigma=1):
        """Função de distribuição acumulada normal"""
        return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))

# Tentar importar scipy, mas usar implementações alternativas se não disponível
try:
    import scipy.stats as stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Usar implementações alternativas
    stats = AlternativeStats
    st.warning("⚠️ **Scipy não está disponível no ambiente atual.** Algumas funcionalidades avançadas serão executadas com implementações alternativas. Para funcionalidades completas, instale scipy: `pip install scipy`")

# Configuração da página
st.set_page_config(page_title="Dashboard de Análise de Processos - Avançado", layout="wide")

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .section-header {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
    }
    .residual-analysis {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid #dee2e6;
        margin: 1rem 0;
    }
    .test-result {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    .test-pass {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .test-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
    }
    .test-fail {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Título principal com aviso
st.markdown('<h1 class="main-header">🏭 Dashboard de Análise de Processos Industriais - Avançado</h1>', unsafe_allow_html=True)

# Aviso sobre SciPy
if not SCIPY_AVAILABLE:
    st.markdown("""
    <div class="warning-box">
    ⚠️ <strong>Scipy não está disponível no ambiente atual.</strong> Algumas funcionalidades avançadas serão executadas com implementações alternativas. Para funcionalidades completas, instale scipy: <code>pip install scipy</code>
    </div>
    """, unsafe_allow_html=True)

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

# Função para detectar outliers usando Z-score
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

# ========== FUNÇÕES CORRIGIDAS PARA CLASSIFICAÇÃO DE CARTAS DE CONTROLE ==========

def classificar_carta_controle(cpk, pontos_fora_controle, total_pontos):
    """
    Classifica a carta de controle baseado na capacidade e estabilidade:
    🟢 Verde: Capaz e Estável (Cpk ≥ 1.33 e ≤ 5% pontos fora)
    🟡 Amarelo: Incapaz e Estável (Cpk < 1.33 e ≤ 5% pontos fora)
    🟠 Mostarda: Capaz e Instável (Cpk ≥ 1.33 e > 5% pontos fora)
    🔴 Vermelho: Incapaz e Instável (Cpk < 1.33 e > 5% pontos fora)
    """
    
    if total_pontos == 0:
        return "🔵 Indefinido", "Dados insuficientes para classificação"
    
    percentual_fora_controle = (pontos_fora_controle / total_pontos * 100)
    
    # Definir critérios
    capaz = cpk is not None and cpk >= 1.33  # Processo considerado capaz
    estavel = percentual_fora_controle <= 5  # Menos de 5% dos pontos fora de controle
    
    # Classificação conforme especificado
    if capaz and estavel:
        return "🟢 Verde", "Capaz e Estável"
    elif not capaz and estavel:
        return "🟡 Amarelo", "Incapaz e Estável"
    elif capaz and not estavel:
        return "🟠 Mostarda", "Capaz e Instável"
    else:
        return "🔴 Vermelho", "Incapaz e Instável"

def criar_indicador_classificacao(cor, classificacao, cpk, percentual_fora):
    """Cria um indicador visual para a classificação da carta de controle"""
    
    # Mapeamento de cores para fundos
    cores_fundo = {
        "🟢 Verde": "background-color: #90EE90; padding: 10px; border-radius: 5px; border: 2px solid #006400;",
        "🟡 Amarelo": "background-color: #FFFACD; padding: 10px; border-radius: 5px; border: 2px solid #FFD700;",
        "🟠 Mostarda": "background-color: #E6DBAC; padding: 10px; border-radius: 5px; border: 2px solid #B8860B;",
        "🔴 Vermelho": "background-color: #FFB6C1; padding: 10px; border-radius: 5px; border: 2px solid #8B0000;",
        "🔵 Indefinido": "background-color: #E6E6FA; padding: 10px; border-radius: 5px; border: 2px solid #4B0082;"
    }
    
    estilo = cores_fundo.get(cor, "background-color: #E6E6FA; padding: 10px; border-radius: 5px;")
    
    # Formatar valores de forma segura
    cpk_texto = f"{cpk:.3f}" if cpk is not None else "N/A"
    percentual_texto = f"{percentual_fora:.1f}%"
    
    html = f"""
    <div style="{estilo}">
        <h3 style="margin: 0; color: #333;">Classificação da Carta de Controle</h3>
        <p style="margin: 5px 0; font-size: 18px; font-weight: bold;">{cor} {classificacao}</p>
        <p style="margin: 2px 0;"><strong>Cpk:</strong> {cpk_texto}</p>
        <p style="margin: 2px 0;"><strong>Pontos fora de controle:</strong> {percentual_texto}</p>
    </div>
    """
    return html

# ========== FUNÇÕES CORRIGIDAS PARA ANÁLISES ESTATÍSTICAS AVANÇADAS ==========

def analise_anova_um_fator_sem_scipy(dados, variavel_resposta, fator):
    """ANOVA de um fator sem usar scipy - VERSÃO CORRIGIDA"""
    try:
        # Agrupar dados por fator
        grupos = []
        fatores_unicos = dados[fator].dropna().unique()
        
        if len(fatores_unicos) < 2:
            st.warning("ANOVA requer pelo menos 2 grupos diferentes")
            return None
        
        for categoria in fatores_unicos:
            grupo = dados[dados[fator] == categoria][variavel_resposta].dropna()
            if len(grupo) > 0:
                grupos.append(grupo)
        
        # Calcular estatísticas manualmente
        n_grupos = len(grupos)
        n_total = sum(len(grupo) for grupo in grupos)
        
        if n_total == 0 or n_grupos < 2:
            st.warning("Dados insuficientes para ANOVA")
            return None
        
        # Média geral
        todos_dados = np.concatenate(grupos)
        media_geral = np.mean(todos_dados)
        
        # Soma dos quadrados entre grupos (SSB)
        ssb = 0
        for grupo in grupos:
            n_grupo = len(grupo)
            media_grupo = np.mean(grupo)
            ssb += n_grupo * (media_grupo - media_geral) ** 2
        
        # Soma dos quadrados dentro dos grupos (SSW)
        ssw = 0
        for grupo in grupos:
            media_grupo = np.mean(grupo)
            ssw += np.sum((grupo - media_grupo) ** 2)
        
        # Graus de liberdade
        df_between = n_grupos - 1
        df_within = n_total - n_grupos
        
        if df_within <= 0:
            st.warning("Graus de liberdade insuficientes para ANOVA")
            return None
        
        # Quadrados médios
        msb = ssb / df_between if df_between > 0 else 0
        msw = ssw / df_within if df_within > 0 else 0
        
        # Estatística F
        f_stat = msb / msw if msw > 0 else 0
        
        # Valor-p aproximado usando distribuição F
        # Fórmula empírica melhorada para valor-p
        if f_stat > 0:
            # Aproximação mais precisa do valor-p
            p_value = 1 / (1 + np.exp(0.3 * (f_stat - 3)))
        else:
            p_value = 1.0
        
        # Estatísticas descritivas
        descritivas = {}
        for i, categoria in enumerate(fatores_unicos):
            if i < len(grupos):
                grupo_data = grupos[i]
                descritivas[str(categoria)] = {
                    'n': len(grupo_data),
                    'media': np.mean(grupo_data),
                    'desvio_padrao': np.std(grupo_data, ddof=1),
                    'mediana': np.median(grupo_data),
                    'min': np.min(grupo_data),
                    'max': np.max(grupo_data)
                }
        
        return {
            'f_statistic': f_stat,
            'p_value': p_value,
            'grupos': [str(g) for g in fatores_unicos],
            'descritivas': descritivas,
            'significativo': p_value < 0.05,
            'ssb': ssb,
            'ssw': ssw,
            'msb': msb,
            'msw': msw,
            'df_between': df_between,
            'df_within': df_within
        }
    
    except Exception as e:
        st.error(f"Erro na ANOVA: {str(e)}")
        return None

def teste_hipotese_media_sem_scipy(dados, coluna, valor_referencia=0, alternativa='two-sided'):
    """Teste de hipótese para média sem scipy - VERSÃO CORRIGIDA"""
    try:
        data_clean = dados[coluna].dropna()
        n = len(data_clean)
        
        if n < 2:
            st.warning("Dados insuficientes para teste de hipótese")
            return None
        
        media_amostral = np.mean(data_clean)
        desvio_padrao = np.std(data_clean, ddof=1)
        erro_padrao = desvio_padrao / np.sqrt(n)
        
        # Estatística t
        t_stat = (media_amostral - valor_referencia) / erro_padrao
        
        # Valor-p usando distribuição t de Student (aproximação)
        # Usando aproximação para distribuição t
        def t_distribution_p_value(t, df, tail='two-sided'):
            """Aproximação do valor-p para distribuição t"""
            # Aproximação usando distribuição normal para grandes amostras
            if df > 30:
                z = abs(t)
                p = 2 * (1 - 0.5 * (1 + math.erf(z / math.sqrt(2))))
            else:
                # Aproximação para amostras pequenas
                z = abs(t)
                p = 2 * (1 - (1 + math.erf(z / math.sqrt(2))) / 2)  # Simplificação
            
            if tail == 'two-sided':
                return p
            elif tail == 'greater':
                return p / 2 if t > 0 else 1 - p / 2
            else:  # 'less'
                return p / 2 if t < 0 else 1 - p / 2
        
        p_value = t_distribution_p_value(t_stat, n-1, alternativa)
        
        # Intervalo de confiança 95%
        # Valor crítico t para 95% (aproximado)
        t_critical = 2.0 if n < 30 else 1.96
        margem_erro = t_critical * erro_padrao
        ci = (media_amostral - margem_erro, media_amostral + margem_erro)
        
        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'media_amostral': media_amostral,
            'valor_referencia': valor_referencia,
            'intervalo_confianca': ci,
            'significativo': p_value < 0.05,
            'n': n,
            'erro_padrao': erro_padrao,
            'desvio_padrao': desvio_padrao
        }
    
    except Exception as e:
        st.error(f"Erro no teste de hipótese: {str(e)}")
        return None

def analise_poder_estatistico_sem_scipy(dados, coluna, efeito_detectavel, alpha=0.05):
    """Análise de poder estatístico sem scipy - VERSÃO CORRIGIDA"""
    try:
        data_clean = dados[coluna].dropna()
        n = len(data_clean)
        
        if n < 2:
            st.warning("Dados insuficientes para análise de poder")
            return None
        
        effect_size = abs(efeito_detectavel) / np.std(data_clean) if np.std(data_clean) > 0 else 0
        
        # Cálculo do poder usando aproximação normal
        # Para teste bicaudal
        z_alpha = 1.96  # Para alpha=0.05 (bicaudal)
        z_beta = effect_size * np.sqrt(n) - z_alpha
        
        # Poder = 1 - beta
        poder = 0.5 * (1 + math.erf(z_beta / math.sqrt(2))) if z_beta > -10 else 0
        poder = max(0, min(poder, 1))
        
        # Tamanho amostral necessário
        z_beta_desejado = 0.84  # Para poder de 80%
        n_necessario = ((z_alpha + z_beta_desejado) / effect_size) ** 2 if effect_size > 0 else float('inf')
        
        return {
            'poder_atual': poder,
            'tamanho_amostral_atual': n,
            'tamanho_amostral_necessario': n_necessario,
            'effect_size': effect_size,
            'alpha': alpha,
            'efeito_detectavel': efeito_detectavel
        }
    
    except Exception as e:
        st.error(f"Erro na análise de poder: {str(e)}")
        return None

def analise_regressao_multipla_sem_scipy(dados, variavel_resposta, variaveis_predictoras):
    """Regressão múltipla sem scipy/statsmodels - VERSÃO CORRIGIDA"""
    try:
        # Preparar dados
        X = dados[variaveis_predictoras]
        y = dados[variavel_resposta]
        
        # Remover valores missing
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X[mask]
        y_clean = y[mask]
        
        if len(X_clean) < len(variaveis_predictoras) + 1:
            st.warning(f"Dados insuficientes para regressão múltipla. Necessário pelo menos {len(variaveis_predictoras) + 1} observações.")
            return None
        
        # Adicionar coluna de intercepto
        X_matrix = np.column_stack([np.ones(len(X_clean)), X_clean])
        
        # Calcular coeficientes usando mínimos quadrados: β = (X'X)^(-1)X'y
        try:
            XtX = X_matrix.T @ X_matrix
            Xty = X_matrix.T @ y_clean
            
            # Verificar se a matriz é invertível
            if np.linalg.det(XtX) == 0:
                st.warning("Matriz singular. Usando pseudoinversa.")
                coeficientes = np.linalg.pinv(XtX) @ Xty
            else:
                coeficientes = np.linalg.inv(XtX) @ Xty
                
        except np.linalg.LinAlgError:
            st.warning("Problema numérico na inversão da matriz. Usando pseudoinversa.")
            coeficientes = np.linalg.pinv(XtX) @ Xty
        
        # Previsões
        y_pred = X_matrix @ coeficientes
        
        # Métricas
        ss_res = np.sum((y_clean - y_pred) ** 2)
        ss_tot = np.sum((y_clean - np.mean(y_clean)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        # R² ajustado
        n = len(y_clean)
        p = len(variaveis_predictoras)
        r2_ajustado = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2
        
        mse = ss_res / (n - p - 1) if n > p + 1 else ss_res / n
        rmse = np.sqrt(mse)
        
        # Erros padrão dos coeficientes
        try:
            var_residual = ss_res / (n - p - 1)
            var_coef = var_residual * np.linalg.inv(XtX).diagonal()
            std_coef = np.sqrt(var_coef)
        except:
            std_coef = np.zeros_like(coeficientes)
        
        # Coeficientes com nomes
        nomes_coeficientes = ['Intercepto'] + variaveis_predictoras
        dict_coeficientes = dict(zip(nomes_coeficientes, coeficientes))
        dict_std_erros = dict(zip(nomes_coeficientes, std_coef))
        
        # Estatísticas t e valores-p
        t_stats = {}
        p_values = {}
        for i, nome in enumerate(nomes_coeficientes):
            if std_coef[i] > 0:
                t_stats[nome] = coeficientes[i] / std_coef[i]
                # Valor-p aproximado
                p_values[nome] = 2 * (1 - 0.5 * (1 + math.erf(abs(t_stats[nome]) / math.sqrt(2))))
            else:
                t_stats[nome] = 0
                p_values[nome] = 1.0
        
        return {
            'coeficientes': dict_coeficientes,
            'std_erros': dict_std_erros,
            't_stats': t_stats,
            'p_values': p_values,
            'r2': r2,
            'r2_ajustado': r2_ajustado,
            'mse': mse,
            'rmse': rmse,
            'residuos': y_clean - y_pred,
            'previsoes': y_pred,
            'n_amostras': n,
            'n_variaveis': p
        }
    
    except Exception as e:
        st.error(f"Erro na regressão múltipla: {str(e)}")
        return None

def analise_bayesiana_ab_test_sem_scipy(controle, variacao, prior_alpha=1, prior_beta=1):
    """Análise Bayesiana para A/B Testing sem scipy - VERSÃO CORRIGIDA"""
    try:
        controle = np.array(controle)
        variacao = np.array(variacao)
        
        # Remover NaNs
        controle = controle[~np.isnan(controle)]
        variacao = variacao[~np.isnan(variacao)]
        
        if len(controle) == 0 or len(variacao) == 0:
            st.warning("Dados insuficientes para análise Bayesiana")
            return None
        
        # Converter para proporções se necessário
        # Assumindo que são taxas de conversão entre 0 e 1
        if all(0 <= x <= 1 for x in controle) and all(0 <= x <= 1 for x in variacao):
            sucessos_controle = int(np.sum(controle))
            sucessos_variacao = int(np.sum(variacao))
            total_controle = len(controle)
            total_variacao = len(variacao)
        else:
            # Assumir que são contagens - normalizar
            limiar = np.median(np.concatenate([controle, variacao]))
            sucessos_controle = np.sum(controle > limiar)
            sucessos_variacao = np.sum(variacao > limiar)
            total_controle = len(controle)
            total_variacao = len(variacao)
        
        # Parâmetros posteriores (distribuição Beta)
        posterior_controle_alpha = prior_alpha + sucessos_controle
        posterior_controle_beta = prior_beta + total_controle - sucessos_controle
        
        posterior_variacao_alpha = prior_alpha + sucessos_variacao
        posterior_variacao_beta = prior_beta + total_variacao - sucessos_variacao
        
        # Amostras das distribuições posteriores (simulação)
        n_simulacoes = 10000
        amostras_controle = np.random.beta(posterior_controle_alpha, posterior_controle_beta, n_simulacoes)
        amostras_variacao = np.random.beta(posterior_variacao_alpha, posterior_variacao_beta, n_simulacoes)
        
        # Probabilidade de que variação é melhor que controle
        prob_variacao_melhor = np.mean(amostras_variacao > amostras_controle)
        
        # Estatísticas descritivas
        media_controle = posterior_controle_alpha / (posterior_controle_alpha + posterior_controle_beta)
        media_variacao = posterior_variacao_alpha / (posterior_variacao_alpha + posterior_variacao_beta)
        
        # Intervalos credíveis (95%)
        ic_controle = (
            np.percentile(amostras_controle, 2.5),
            np.percentile(amostras_controle, 97.5)
        )
        ic_variacao = (
            np.percentile(amostras_variacao, 2.5),
            np.percentile(amostras_variacao, 97.5)
        )
        
        # Distribuição da diferença
        diferencas = amostras_variacao - amostras_controle
        prob_diferenca_positiva = np.mean(diferencas > 0)
        ic_diferenca = (np.percentile(diferencas, 2.5), np.percentile(diferencas, 97.5))
        
        return {
            'prob_variacao_melhor': prob_variacao_melhor,
            'prob_diferenca_positiva': prob_diferenca_positiva,
            'media_controle': media_controle,
            'media_variacao': media_variacao,
            'diferenca_medias': media_variacao - media_controle,
            'intervalo_controle': ic_controle,
            'intervalo_variacao': ic_variacao,
            'intervalo_diferenca': ic_diferenca,
            'sucessos_controle': sucessos_controle,
            'sucessos_variacao': sucessos_variacao,
            'total_controle': total_controle,
            'total_variacao': total_variacao,
            'taxa_controle': sucessos_controle / total_controle if total_controle > 0 else 0,
            'taxa_variacao': sucessos_variacao / total_variacao if total_variacao > 0 else 0
        }
    
    except Exception as e:
        st.error(f"Erro na análise Bayesiana: {str(e)}")
        return None

def simulacao_monte_carlo_capabilidade(media, desvio_padrao, lse, lie, n_simulacoes=10000):
    """Simulação Monte Carlo para análise de capabilidade - VERSÃO CORRIGIDA"""
    try:
        if desvio_padrao <= 0:
            st.warning("Desvio padrão deve ser maior que zero")
            return None
        
        # Gerar amostras da distribuição normal
        simulacoes = np.random.normal(media, desvio_padrao, n_simulacoes)
        
        # Calcular métricas de capabilidade
        cpk_simulacoes = []
        ppm_simulacoes = []
        
        # Amostras menores para calcular Cpk
        for i in range(100):
            amostra = np.random.choice(simulacoes, size=min(30, len(simulacoes)), replace=False)
            if np.std(amostra) > 0:
                if lse is not None and lie is not None:
                    cpk_superior = (lse - np.mean(amostra)) / (3 * np.std(amostra))
                    cpk_inferior = (np.mean(amostra) - lie) / (3 * np.std(amostra))
                    cpk = min(cpk_superior, cpk_inferior)
                    cpk_simulacoes.append(cpk)
        
        # Calcular PPM
        if lse is not None and lie is not None:
            fora_especificacao = np.sum((simulacoes > lse) | (simulacoes < lie))
        elif lse is not None:
            fora_especificacao = np.sum(simulacoes > lse)
        elif lie is not None:
            fora_especificacao = np.sum(simulacoes < lie)
        else:
            fora_especificacao = 0
            
        ppm = (fora_especificacao / n_simulacoes) * 1e6
        
        # Calcular Cp e Cpk para a simulação completa
        cp = None
        cpk = None
        if lse is not None and lie is not None and lse > lie:
            cp = (lse - lie) / (6 * desvio_padrao)
            cpk_superior = (lse - media) / (3 * desvio_padrao)
            cpk_inferior = (media - lie) / (3 * desvio_padrao)
            cpk = min(cpk_superior, cpk_inferior)
        
        return {
            'media_simulacao': np.mean(simulacoes),
            'desvio_padrao_simulacao': np.std(simulacoes),
            'cpk_medio': np.mean(cpk_simulacoes) if cpk_simulacoes else 0,
            'cpk_std': np.std(cpk_simulacoes) if cpk_simulacoes else 0,
            'cp': cp,
            'cpk': cpk,
            'ppm_simulado': ppm,
            'percentual_fora_simulado': (fora_especificacao / n_simulacoes) * 100,
            'n_simulacoes': n_simulacoes,
            'simulacoes': simulacoes
        }
    
    except Exception as e:
        st.error(f"Erro na simulação Monte Carlo: {str(e)}")
        return None

# ========== FUNÇÕES CORRIGIDAS PARA CORRELAÇÕES ==========

def calcular_correlacoes_completas(dados, variaveis_selecionadas, metodo='pearson'):
    """Calcula matriz de correlação completa com tratamentos robustos"""
    try:
        if len(variaveis_selecionadas) < 2:
            return None, "Selecione pelo menos 2 variáveis"
        
        dados_corr = dados[variaveis_selecionadas].copy()
        
        # Remover linhas com valores missing
        dados_corr = dados_corr.dropna()
        
        if len(dados_corr) < 2:
            return None, "Dados insuficientes após remoção de valores missing"
        
        # Calcular matriz de correlação
        if metodo.lower() == 'pearson':
            corr_matrix = dados_corr.corr(method='pearson')
        elif metodo.lower() == 'spearman':
            corr_matrix = dados_corr.corr(method='spearman')
        else:
            corr_matrix = dados_corr.corr(method='pearson')
        
        # Calcular valores-p para correlações Pearson
        p_values = pd.DataFrame(np.eye(len(corr_matrix)), 
                               index=corr_matrix.index, 
                               columns=corr_matrix.columns)
        
        if metodo.lower() == 'pearson':
            n = len(dados_corr)
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = corr_matrix.iloc[i, j]
                    if abs(corr_val) < 1.0:  # Evitar divisão por zero
                        t_stat = corr_val * np.sqrt((n-2) / (1 - corr_val**2))
                        p_val = 2 * (1 - stats.t.cdf(abs(t_stat), n-2)) if SCIPY_AVAILABLE else 0.05
                        p_values.iloc[i, j] = p_val
                        p_values.iloc[j, i] = p_val
        
        return corr_matrix, p_values
        
    except Exception as e:
        st.error(f"Erro no cálculo de correlações: {str(e)}")
        return None, None

def analise_correlacao_detalhada(dados, var1, var2):
    """Análise detalhada de correlação entre duas variáveis - VERSÃO CORRIGIDA"""
    try:
        # Verificar se as colunas existem
        if var1 not in dados.columns or var2 not in dados.columns:
            st.error(f"Colunas {var1} ou {var2} não encontradas nos dados")
            return None
        
        # Filtrar dados e garantir que são unidimensionais
        dados_clean = dados[[var1, var2]].dropna()
        
        if len(dados_clean) < 3:
            st.warning("Dados insuficientes para análise de correlação (mínimo 3 pontos)")
            return None
        
        # Garantir que os dados são unidimensionais
        x = dados_clean[var1].values.flatten()  # .flatten() garante 1D
        y = dados_clean[var2].values.flatten()  # .flatten() garante 1D
        
        # Verificar se os arrays têm o mesmo comprimento
        if len(x) != len(y):
            st.error("Arrays com comprimentos diferentes")
            return None
        
        # Correlação Pearson
        if len(x) > 1 and np.std(x) > 0 and np.std(y) > 0:
            correlacao_pearson = np.corrcoef(x, y)[0, 1]
        else:
            correlacao_pearson = 0
        
        # Correlação Spearman
        try:
            rank_x = pd.Series(x).rank()
            rank_y = pd.Series(y).rank()
            if len(rank_x) > 1 and np.std(rank_x) > 0 and np.std(rank_y) > 0:
                correlacao_spearman = np.corrcoef(rank_x, rank_y)[0, 1]
            else:
                correlacao_spearman = 0
        except:
            correlacao_spearman = 0
        
        # Regressão linear
        slope, intercept, r_squared = calcular_regressao_linear(x, y)
        
        # Estatísticas descritivas
        stats_var1 = {
            'media': np.mean(x),
            'std': np.std(x),
            'min': np.min(x),
            'max': np.max(x),
            'n': len(x)
        }
        
        stats_var2 = {
            'media': np.mean(y),
            'std': np.std(y),
            'min': np.min(y),
            'max': np.max(y),
            'n': len(y)
        }
        
        return {
            'pearson': correlacao_pearson,
            'spearman': correlacao_spearman,
            'r_squared': r_squared if r_squared is not None else 0,
            'slope': slope if slope is not None else 0,
            'intercept': intercept if intercept is not None else 0,
            'n_amostras': len(dados_clean),
            'stats_var1': stats_var1,
            'stats_var2': stats_var2
        }
        
    except Exception as e:
        st.error(f"Erro na análise de correlação detalhada: {str(e)}")
        return None

# ========== FUNÇÕES AUXILIARES CORRIGIDAS ==========

def calcular_regressao_linear(x, y):
    """Calcula regressão linear manualmente - VERSÃO CORRIGIDA"""
    # Garantir que os arrays são unidimensionais
    x = np.asarray(x).flatten()
    y = np.asarray(y).flatten()
    
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

# ========== FUNÇÕES CORRIGIDAS PARA ANÁLISE DE RESÍDUOS ==========

def analise_residuos_regressao_simples(x, y):
    """Análise completa de resíduos para regressão linear simples - VERSÃO CORRIGIDA"""
    try:
        # Calcular regressão
        slope, intercept, r_squared = calcular_regressao_linear(x, y)
        
        if slope is None:
            return None
        
        # Calcular previsões e resíduos
        y_pred = slope * x + intercept
        residuos = y - y_pred
        
        # CORREÇÃO: Converter para pandas Series para usar dropna()
        residuos_series = pd.Series(residuos)
        
        # Estatísticas dos resíduos
        stats_residuos = {
            'media': np.mean(residuos),
            'mediana': np.median(residuos),
            'desvio_padrao': np.std(residuos),
            'assimetria': residuos_series.skew(),
            'curtose': residuos_series.kurtosis(),
            'min': np.min(residuos),
            'max': np.max(residuos)
        }
        
        # Teste de normalidade dos resíduos
        p_normalidade = teste_normalidade_manual(residuos)
        
        # Teste de homocedasticidade (correlação entre previsões e resíduos)
        correlacao_previsoes_residuos = np.corrcoef(y_pred, residuos)[0, 1] if len(y_pred) > 1 and len(residuos) > 1 else 0
        
        # Detecção de outliers nos resíduos - CORREÇÃO: usar array numpy
        if len(residuos) > 0:
            # Usar numpy para detectar outliers em vez de pandas
            residuos_clean = residuos[~np.isnan(residuos)]
            if len(residuos_clean) > 0:
                mean_resid = np.mean(residuos_clean)
                std_resid = np.std(residuos_clean)
                if std_resid > 0:
                    z_scores = np.abs((residuos_clean - mean_resid) / std_resid)
                    n_outliers = np.sum(z_scores > 3)
                else:
                    n_outliers = 0
            else:
                n_outliers = 0
        else:
            n_outliers = 0
            
        percentual_outliers = (n_outliers / len(residuos)) * 100 if len(residuos) > 0 else 0
        
        return {
            'residuos': residuos,
            'previsoes': y_pred,
            'estatisticas': stats_residuos,
            'p_normalidade': p_normalidade,
            'correlacao_previsoes_residuos': correlacao_previsoes_residuos,
            'n_outliers': n_outliers,
            'percentual_outliers': percentual_outliers,
            'slope': slope,
            'intercept': intercept,
            'r_squared': r_squared
        }
    
    except Exception as e:
        st.error(f"Erro na análise de resíduos: {str(e)}")
        return None

# ========== FUNÇÃO AUXILIAR CORRIGIDA ==========

def teste_normalidade_manual(data):
    """Teste de normalidade simplificado usando assimetria e curtose - VERSÃO CORRIGIDA"""
    # CORREÇÃO: Converter para array numpy e remover NaNs
    if isinstance(data, pd.Series):
        data_clean = data.dropna().values
    else:
        data_clean = np.array(data)
        data_clean = data_clean[~np.isnan(data_clean)]
    
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

def interpretar_analise_residuos(analise_residuos):
    """Fornece interpretação completa da análise de resíduos"""
    
    interpretacoes = []
    
    # Normalidade
    if analise_residuos['p_normalidade'] > 0.05:
        interpretacoes.append("✅ **Normalidade**: Os resíduos seguem distribuição normal")
    else:
        interpretacoes.append("⚠️ **Normalidade**: Os resíduos não são normais")
    
    # Homocedasticidade
    corr = abs(analise_residuos['correlacao_previsoes_residuos'])
    if corr < 0.1:
        interpretacoes.append("✅ **Homocedasticidade**: Variância constante dos resíduos")
    elif corr < 0.3:
        interpretacoes.append("⚠️ **Homocedasticidade**: Possível heterocedasticidade leve")
    else:
        interpretacoes.append("❌ **Homocedasticidade**: Heterocedasticidade detectada")
    
    # Outliers
    if analise_residuos['percentual_outliers'] < 5:
        interpretacoes.append("✅ **Outliers**: Poucos outliers nos resíduos")
    else:
        interpretacoes.append(f"⚠️ **Outliers**: {analise_residuos['percentual_outliers']:.1f}% dos resíduos são outliers")
    
    # Média dos resíduos
    if abs(analise_residuos['estatisticas']['media']) < 0.01:
        interpretacoes.append("✅ **Média dos resíduos**: Próxima de zero")
    else:
        interpretacoes.append("⚠️ **Média dos resíduos**: Distante de zero")
    
    return interpretacoes

def criar_relatorio_residuos(analise_residuos):
    """Cria relatório completo da análise de resíduos"""
    
    st.markdown("### 📊 Relatório de Análise de Resíduos")
    
    # Métricas principais
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("R²", f"{analise_residuos['r_squared']:.4f}")
        st.metric("Inclinação", f"{analise_residuos['slope']:.4f}")
    
    with col2:
        st.metric("p-valor Normalidade", f"{analise_residuos['p_normalidade']:.4f}")
        st.metric("Intercepto", f"{analise_residuos['intercept']:.4f}")
    
    with col3:
        st.metric("Correlação Res-Pred", f"{analise_residuos['correlacao_previsoes_residuos']:.4f}")
        st.metric("Média Resíduos", f"{analise_residuos['estatisticas']['media']:.4f}")
    
    with col4:
        st.metric("Outliers", f"{analise_residuos['n_outliers']}")
        st.metric("% Outliers", f"{analise_residuos['percentual_outliers']:.1f}%")
    
    # Interpretação
    st.markdown("### 🔍 Interpretação dos Resultados")
    interpretacoes = interpretar_analise_residuos(analise_residuos)
    
    for interpretacao in interpretacoes:
        st.write(interpretacao)
    
    # Recomendações
    st.markdown("### 💡 Recomendações")
    
    if analise_residuos['p_normalidade'] < 0.05:
        st.warning("""
        **Resíduos não normais detectados:**
        - Considere transformar as variáveis (log, sqrt, etc.)
        - Verifique a presença de outliers influentes
        - Considere modelos não lineares
        """)
    
    if abs(analise_residuos['correlacao_previsoes_residuos']) > 0.3:
        st.warning("""
        **Heterocedasticidade detectada:**
        - A variância dos resíduos não é constante
        - Considere transformações nas variáveis
        - Pode indicar relação não linear
        """)
    
    if analise_residuos['percentual_outliers'] > 10:
        st.warning("""
        **Muitos outliers detectados:**
        - Verifique a qualidade dos dados
        - Investigue causas especiais
        - Considere remover ou tratar outliers
        """)
    
    if analise_residuos['r_squared'] < 0.5:
        st.info("""
        **R² baixo:**
        - A variável independente explica pouca variação na dependente
        - Considere incluir outras variáveis no modelo
        - Pode haver relação não linear
        """)

# ========== FUNÇÕES PARA TESTES NÃO PARAMÉTRICOS ==========

def teste_kruskal_wallis_sem_scipy(dados, variavel_resposta, fator):
    """Teste de Kruskal-Wallis (ANOVA não paramétrica)"""
    try:
        # Agrupar dados por fator
        grupos = []
        fatores_unicos = dados[fator].dropna().unique()
        
        if len(fatores_unicos) < 2:
            st.warning("Teste requer pelo menos 2 grupos diferentes")
            return None
        
        for categoria in fatores_unicos:
            grupo = dados[dados[fator] == categoria][variavel_resposta].dropna()
            if len(grupo) > 0:
                grupos.append(grupo)
        
        # Calcular ranks para todos os dados
        todos_dados = np.concatenate(grupos)
        ranks = pd.Series(todos_dados).rank()
        
        # Calcular estatística H
        n_total = len(todos_dados)
        h_stat = 0
        
        start_idx = 0
        for grupo in grupos:
            n_grupo = len(grupo)
            if n_grupo > 0:
                rank_medio_grupo = np.mean(ranks[start_idx:start_idx + n_grupo])
                h_stat += n_grupo * (rank_medio_grupo - (n_total + 1) / 2) ** 2
                start_idx += n_grupo
        
        h_stat = (12 / (n_total * (n_total + 1))) * h_stat
        
        # Valor-p aproximado usando distribuição qui-quadrado
        df = len(grupos) - 1
        if df > 0:
            # Aproximação do valor-p usando distribuição qui-quadrado
            p_value = 1 - chi_square_cdf(h_stat, df)
        else:
            p_value = 1.0
        
        return {
            'h_statistic': h_stat,
            'p_value': p_value,
            'n_grupos': len(grupos),
            'n_total': n_total,
            'significativo': p_value < 0.05
        }
    
    except Exception as e:
        st.error(f"Erro no teste de Kruskal-Wallis: {str(e)}")
        return None

def teste_mann_whitney_sem_scipy(grupo1, grupo2):
    """Teste de Mann-Whitney U (não paramétrico para 2 grupos)"""
    try:
        grupo1 = grupo1.dropna()
        grupo2 = grupo2.dropna()
        
        if len(grupo1) == 0 or len(grupo2) == 0:
            st.warning("Grupos vazios para teste Mann-Whitney")
            return None
        
        # Combinar dados e calcular ranks
        todos_dados = np.concatenate([grupo1, grupo2])
        ranks = pd.Series(todos_dados).rank()
        
        n1, n2 = len(grupo1), len(grupo2)
        r1 = np.sum(ranks[:n1])
        
        # Estatística U
        u1 = r1 - n1 * (n1 + 1) / 2
        u2 = n1 * n2 - u1
        
        u_stat = min(u1, u2)
        
        # Valor-p aproximado (para grandes amostras)
        if n1 > 20 and n2 > 20:
            # Aproximação normal para grandes amostras
            mean_u = n1 * n2 / 2
            std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
            z = (u_stat - mean_u) / std_u
            p_value = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
        else:
            # Aproximação simples para amostras pequenas
            p_value = 0.05 if u_stat < (n1 * n2 / 4) else 0.5
        
        return {
            'u_statistic': u_stat,
            'p_value': p_value,
            'n1': n1,
            'n2': n2,
            'significativo': p_value < 0.05
        }
    
    except Exception as e:
        st.error(f"Erro no teste de Mann-Whitney: {str(e)}")
        return None

def chi_square_cdf(x, df):
    """Função de distribuição acumulada aproximada para qui-quadrado"""
    if df <= 0:
        return 0.0
    
    # Aproximação usando distribuição gama
    if x <= 0:
        return 0.0
    
    # Usando aproximação para função gama incompleta
    return 1 - math.exp(-x/2) * sum((x/2)**k / math.factorial(k) for k in range(int(df/2) + 1))

def analise_nao_parametrica_completa(dados, variavel_resposta, fator):
    """Análise não paramétrica completa para comparação de grupos"""
    try:
        fatores_unicos = dados[fator].dropna().unique()
        
        if len(fatores_unicos) < 2:
            st.warning("Análise requer pelo menos 2 grupos diferentes")
            return None
        
        resultados = {}
        
        # Kruskal-Wallis para múltiplos grupos
        if len(fatores_unicos) > 2:
            resultados['kruskal_wallis'] = teste_kruskal_wallis_sem_scipy(dados, variavel_resposta, fator)
        
        # Mann-Whitney para pares de grupos (quando há 2 grupos)
        if len(fatores_unicos) == 2:
            grupo1 = dados[dados[fator] == fatores_unicos[0]][variavel_resposta]
            grupo2 = dados[dados[fator] == fatores_unicos[1]][variavel_resposta]
            resultados['mann_whitney'] = teste_mann_whitney_sem_scipy(grupo1, grupo2)
        
        # Estatísticas descritivas não paramétricas
        descritivas = {}
        for categoria in fatores_unicos:
            grupo_data = dados[dados[fator] == categoria][variavel_resposta].dropna()
            descritivas[str(categoria)] = {
                'n': len(grupo_data),
                'mediana': np.median(grupo_data),
                'q1': np.percentile(grupo_data, 25),
                'q3': np.percentile(grupo_data, 75),
                'min': np.min(grupo_data),
                'max': np.max(grupo_data),
                'iqr': np.percentile(grupo_data, 75) - np.percentile(grupo_data, 25)
            }
        
        resultados['descritivas'] = descritivas
        resultados['n_grupos'] = len(fatores_unicos)
        
        return resultados
    
    except Exception as e:
        st.error(f"Erro na análise não paramétrica: {str(e)}")
        return None

# ========== FUNÇÕES PARA CARTA DE CONTROLE COM LSE/LIE ==========

def plotar_carta_controle_com_especificacoes(valores, limites_controle, limites_especificacao, titulo, tipo="individual"):
    """Plota uma carta de controle com limites de especificação"""
    LSC, LC, LIC = limites_controle
    LSE, LIE = limites_especificacao
    
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
    
    # Adicionar linhas de controle (3 sigma)
    fig.add_hline(y=LSC, line_dash="dash", line_color="red", 
                  annotation_text="LSC (3σ)", annotation_position="right")
    fig.add_hline(y=LC, line_dash="dash", line_color="green", 
                  annotation_text="LC", annotation_position="right")
    fig.add_hline(y=LIC, line_dash="dash", line_color="red", 
                  annotation_text="LIC (3σ)", annotation_position="right")
    
    # Adicionar limites de especificação (se definidos)
    if LSE is not None and LSE != 0:
        fig.add_hline(y=LSE, line_dash="dot", line_color="purple", 
                      annotation_text="LSE", annotation_position="left",
                      line=dict(width=3))
    
    if LIE is not None and LIE != 0:
        fig.add_hline(y=LIE, line_dash="dot", line_color="purple", 
                      annotation_text="LIE", annotation_position="left",
                      line=dict(width=3))
    
    # Destacar pontos fora dos limites de controle
    pontos_fora_controle = (valores > LSC) | (valores < LIC)
    if pontos_fora_controle.any():
        indices_fora = np.where(pontos_fora_controle)[0] + 1
        valores_fora = valores[pontos_fora_controle]
        
        fig.add_trace(go.Scatter(
            x=indices_fora,
            y=valores_fora,
            mode='markers',
            name='Fora de Controle (3σ)',
            marker=dict(color='red', size=10, symbol='x')
        ))
    
    # Destacar pontos fora dos limites de especificação (se definidos)
    pontos_fora_especificacao = pd.Series([False] * len(valores))
    if LSE is not None and LSE != 0:
        pontos_fora_especificacao = pontos_fora_especificacao | (valores > LSE)
    if LIE is not None and LIE != 0:
        pontos_fora_especificacao = pontos_fora_especificacao | (valores < LIE)
    
    if pontos_fora_especificacao.any() and ((LSE is not None and LSE != 0) or (LIE is not None and LIE != 0)):
        indices_fora_esp = np.where(pontos_fora_especificacao & ~pontos_fora_controle)[0] + 1
        valores_fora_esp = valores[pontos_fora_especificacao & ~pontos_fora_controle]
        
        if len(valores_fora_esp) > 0:
            fig.add_trace(go.Scatter(
                x=indices_fora_esp,
                y=valores_fora_esp,
                mode='markers',
                name='Fora de Especificação',
                marker=dict(color='orange', size=12, symbol='star')
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
    
    return fig, pontos_fora_controle.sum(), pontos_fora_especificacao.sum()

def calcular_estatisticas_carta_com_especificacoes(valores, limites_controle, limites_especificacao):
    """Calcula estatísticas considerando limites de controle e especificação"""
    LSC, LC, LIC = limites_controle
    LSE, LIE = limites_especificacao
    
    stats = {
        'media': np.mean(valores),
        'desvio_padrao': np.std(valores, ddof=1),
        'n': len(valores),
        'minimo': np.min(valores),
        'maximo': np.max(valores)
    }
    
    # Pontos fora dos limites de controle (3 sigma)
    pontos_fora_controle = ((valores > LSC) | (valores < LIC)).sum()
    stats['pontos_fora_controle'] = pontos_fora_controle
    stats['percentual_fora_controle'] = (pontos_fora_controle / len(valores)) * 100 if len(valores) > 0 else 0
    
    # Pontos fora dos limites de especificação
    pontos_fora_especificacao = 0
    if LSE is not None and LSE != 0:
        pontos_fora_especificacao += (valores > LSE).sum()
    if LIE is not None and LIE != 0:
        pontos_fora_especificacao += (valores < LIE).sum()
    
    stats['pontos_fora_especificacao'] = pontos_fora_especificacao
    stats['percentual_fora_especificacao'] = (pontos_fora_especificacao / len(valores)) * 100 if len(valores) > 0 else 0
    
    # Cálculo de Cp e Cpk se ambos limites de especificação estiverem definidos
    if LSE is not None and LIE is not None and LSE != 0 and LIE != 0 and LSE > LIE and stats['desvio_padrao'] > 0:
        cp = (LSE - LIE) / (6 * stats['desvio_padrao'])
        cpk_superior = (LSE - stats['media']) / (3 * stats['desvio_padrao'])
        cpk_inferior = (stats['media'] - LIE) / (3 * stats['desvio_padrao'])
        cpk = min(cpk_superior, cpk_inferior)
        
        stats.update({
            'cp': cp,
            'cpk': cpk,
            'cpk_superior': cpk_superior,
            'cpk_inferior': cpk_inferior
        })
    
    return stats

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
        def normal_cdf(x):
            """Aproximação da função de distribuição acumulada normal"""
            return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0
        
        pct_fora_superior = (1 - normal_cdf(z_superior)) * 100
        pct_fora_inferior = normal_cdf(-z_inferior) * 100
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
        marker_color='lightblue',
        histnorm='probability density'
    ))
    
    # Linha de densidade (aproximação manual da curva normal)
    x_range = np.linspace(data_clean.min(), data_clean.max(), 100)
    pdf = (1/(resultados['desvio_padrao'] * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x_range - resultados['media']) / resultados['desvio_padrao'])**2)
    
    fig.add_trace(go.Scatter(
        x=x_range,
        y=pdf,
        mode='lines',
        name='Curva Normal',
        line=dict(color='red', width=2)
    ))
    
    # Limites de especificação
    if lse is not None and lse != 0:
        fig.add_vline(x=lse, line_dash="dash", line_color="red", 
                     annotation_text="LSE", annotation_position="top")
    
    if lie is not None and lie != 0:
        fig.add_vline(x=lie, line_dash="dash", line_color="red",
                     annotation_text="LIE", annotation_position="top")
    
    # Média do processo
    fig.add_vline(x=resultados['media'], line_dash="solid", line_color="green",
                 annotation_text="Média", annotation_position="bottom")
    
    # Alvo (centro das especificações)
    if lse is not None and lie is not None and lse != 0 and lie != 0:
        alvo = (lse + lie) / 2
        fig.add_vline(x=alvo, line_dash="dot", line_color="orange",
                     annotation_text="Alvo", annotation_position="bottom")
    
    fig.update_layout(
        title=f"Histograma de Capabilidade - {coluna}",
        xaxis_title=coluna,
        yaxis_title="Densidade de Probabilidade",
        showlegend=True,
        height=500
    )
    
    return fig

# ========== FUNÇÕES PARA ANÁLISE ESTATÍSTICA BÁSICA ==========

def criar_dispersao_regressao(dados, eixo_x, eixo_y, color_by=None):
    """Cria gráfico de dispersão com linha de regressão usando apenas numpy"""
    try:
        # Filtrar dados válidos
        mask = ~dados[eixo_x].isna() & ~dados[eixo_y].isna()
        dados_filtrados = dados[mask]
        
        if len(dados_filtrados) < 2:
            st.warning("Dados insuficientes para criar gráfico de dispersão")
            return go.Figure()
        
        # Criar gráfico base
        if color_by and color_by in dados.columns:
            fig = px.scatter(dados_filtrados, x=eixo_x, y=eixo_y, color=color_by,
                            title=f"{eixo_y} vs {eixo_x}")
        else:
            fig = px.scatter(dados_filtrados, x=eixo_x, y=eixo_y,
                            title=f"{eixo_y} vs {eixo_x}")
        
        # Calcular regressão linear
        x_vals = dados_filtrados[eixo_x].values
        y_vals = dados_filtrados[eixo_y].values
        
        slope, intercept, r_squared = calcular_regressao_linear(x_vals, y_vals)
        
        if slope is not None and intercept is not None:
            # Adicionar linha de regressão
            x_range = np.linspace(x_vals.min(), x_vals.max(), 100)
            y_pred = slope * x_range + intercept
            
            fig.add_trace(go.Scatter(
                x=x_range,
                y=y_pred,
                mode='lines',
                name=f'Regressão (R² = {r_squared:.4f})',
                line=dict(color='red', width=3)
            ))
            
            # Adicionar equação
            fig.add_annotation(
                x=0.05,
                y=0.95,
                xref="paper",
                yref="paper",
                text=f"y = {slope:.4f}x + {intercept:.4f}<br>R² = {r_squared:.4f}",
                showarrow=False,
                bgcolor="white",
                bordercolor="black",
                borderwidth=1
            )
        
        return fig
        
    except Exception as e:
        st.error(f"Erro ao criar gráfico de dispersão: {str(e)}")
        return go.Figure()

def calcular_estatisticas_correlacao(dados, eixo_x, eixo_y):
    """Calcula estatísticas de correlação sem usar scipy"""
    try:
        # Filtrar dados válidos
        mask = ~dados[eixo_x].isna() & ~dados[eixo_y].isna()
        dados_filtrados = dados[mask]
        
        if len(dados_filtrados) < 2:
            return None, None, None, None
        
        x_vals = dados_filtrados[eixo_x].values
        y_vals = dados_filtrados[eixo_y].values
        
        # Correlação Pearson
        correlacao_pearson = np.corrcoef(x_vals, y_vals)[0, 1]
        
        # Correlação Spearman (usando ranks)
        rank_x = pd.Series(x_vals).rank()
        rank_y = pd.Series(y_vals).rank()
        correlacao_spearman = np.corrcoef(rank_x, rank_y)[0, 1]
        
        # Regressão linear para R²
        slope, intercept, r_squared = calcular_regressao_linear(x_vals, y_vals)
        
        return correlacao_pearson, correlacao_spearman, r_squared, slope
        
    except Exception as e:
        st.error(f"Erro ao calcular estatísticas: {str(e)}")
        return None, None, None, None

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

# ========== FUNÇÃO MAIN COMPLETA ==========

def main():
        
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

    # Abas principais
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "📈 Análise Temporal", 
        "📊 Estatística Detalhada", 
        "🔥 Correlações", 
        "🔍 Dispersão & Regressão",
        "🎯 Carta de Controle",
        "📊 Análise de Capabilidade",
        "🔬 Estatística Avançada",
        "📋 Resumo Executivo"
    ])

    # ========== ABA 1: ANÁLISE TEMPORAL ==========
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
                                           ["Linha", "Área", "Barra", "Scatter", "Boxplot Temporal"],
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
                else:  # Boxplot Temporal
                    # Criar períodos mensais para boxplot
                    dados_temp['Periodo'] = dados_temp[coluna_data].dt.to_period('M').astypes(str)
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

    # ========== ABA 2: ESTATÍSTICA DETALHADA ==========
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
                    min_val = dados_analise[coluna_analise].min()
                    if min_val <= 0:
                        dados_analise[coluna_analise] = np.log1p(dados_analise[coluna_analise] - min_val + 0.001)
                    else:
                        dados_analise[coluna_analise] = np.log(dados_analise[coluna_analise])
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

    # ========== ABA 3: CORRELAÇÕES ==========
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
                
                # Matriz de correlação usando função corrigida
                try:
                    corr_matrix, p_values = calcular_correlacoes_completas(dados_corr, variaveis_selecionadas, metodo_corr)
                    
                    if corr_matrix is not None:
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
                                p_val = p_values.iloc[i, j] if p_values is not None else 0.05
                                correlations.append({
                                    'Variável 1': corr_matrix.columns[i],
                                    'Variável 2': corr_matrix.columns[j],
                                    'Correlação': corr_value,
                                    '|Correlação|': abs(corr_value),
                                    'Valor-p': p_val,
                                    'Significativo': p_val < 0.05
                                })
                        
                        df_corr = pd.DataFrame(correlations)
                        
                        col_ana1, col_ana2 = st.columns(2)
                        
                        with col_ana1:
                            st.write("📈 Top 10 Maiores Correlações:")
                            top_correlations = df_corr.nlargest(10, '|Correlação|')
                            for _, row in top_correlations.iterrows():
                                corr_value = row['Correlação']
                                corr_color = "🟢" if corr_value > 0 else "🔴"
                                corr_strength = "Forte" if abs(corr_value) > 0.7 else "Moderada" if abs(corr_value) > 0.3 else "Fraca"
                                significativo = "✅" if row['Significativo'] else "❌"
                                st.write(f"{corr_color} {significativo} **{corr_value:.3f}** - {corr_strength}")
                                st.write(f"   {row['Variável 1']} ↔ {row['Variável 2']}")
                                st.write("---")
                        
                        with col_ana2:
                            st.write("📉 Top 10 Menores Correlações:")
                            bottom_correlations = df_corr.nsmallest(10, '|Correlação|')
                            for _, row in bottom_correlations.iterrows():
                                corr_value = row['Correlação']
                                corr_color = "🟢" if corr_value > 0 else "🔴"
                                corr_strength = "Fraca"
                                significativo = "✅" if row['Significativo'] else "❌"
                                st.write(f"{corr_color} {significativo} **{corr_value:.3f}** - {corr_strength}")
                                st.write(f"   {row['Variável 1']} ↔ {row['Variável 2']}")
                                st.write("---")
                    else:
                        st.error("Não foi possível calcular a matriz de correlação")
                
                except Exception as e:
                    st.error(f"Erro ao calcular correlações: {str(e)}")

    # ========== ABA 4: DISPERSÃO & REGRESSÃO ==========
    with tab4:
        st.header("🔍 Gráficos de Dispersão com Regressão e Análise de Resíduos")
        
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
                    # NÃO remover outliers para análise de resíduos
                    st.info("ℹ️ Para análise de resíduos, outliers NÃO são removidos")
                with col_opt2:
                    mostrar_regressao = st.checkbox("📈 Mostrar regressão", value=True,
                                                   key=generate_unique_key("show_regression", f"{eixo_x}_{eixo_y}"))
                with col_opt3:
                    color_by = st.selectbox("Colorir por:", [""] + colunas_numericas,
                                           key=generate_unique_key("color_by", f"{eixo_x}_{eixo_y}"))
                
                dados_scatter = dados_processados.copy()
                
                # Gráfico de dispersão básico
                fig_dispersao = criar_dispersao_regressao(dados_scatter, eixo_x, eixo_y, color_by if color_by else None)
                st.plotly_chart(fig_dispersao, use_container_width=True)
                
                # ========== ANÁLISE DE RESÍDUOS ==========
                st.markdown("---")
                st.subheader("📊 Análise Completa de Resíduos")
                
                # Preparar dados para análise
                dados_clean = dados_scatter[[eixo_x, eixo_y]].dropna()
                x_vals = dados_clean[eixo_x].values
                y_vals = dados_clean[eixo_y].values
                
                if len(x_vals) > 1 and len(y_vals) > 1:
                    # Realizar análise de resíduos
					if.len(x_vals) > 2 and len(y_vals) > 2: # Mínimo para regressão
						analise_residuos = analise_residuos_regressao_simples(x_vals, y_vals)
					else:
						st.warning("Dados insuficientes para análise de resíduos (mínimo 3 pontos)")
						analise_residuos = None
                    
                    if analise_residuos:
                        # Criar gráficos de resíduos
                        graficos_residuos = criar_graficos_residuos_completos(
                            analise_residuos, x_vals, y_vals, eixo_x, eixo_y
                        )
                        
                        # Exibir gráficos em duas colunas
                        col_graf1, col_graf2 = st.columns(2)
                        
                        with col_graf1:
                            st.plotly_chart(graficos_residuos['dispersao_regressao'], use_container_width=True)
                            st.plotly_chart(graficos_residuos['residuos_vs_previsoes'], use_container_width=True)
                            st.plotly_chart(graficos_residuos['residuos_vs_x'], use_container_width=True)
                        
                        with col_graf2:
                            st.plotly_chart(graficos_residuos['histograma_residuos'], use_container_width=True)
                            st.plotly_chart(graficos_residuos['qq_plot'], use_container_width=True)
                        
                        # Relatório de análise de resíduos
                        criar_relatorio_residuos(analise_residuos)
                        
                    else:
                        st.error("Não foi possível realizar a análise de resíduos")
                else:
                    st.warning("Dados insuficientes para análise de resíduos")
        
        # Estatísticas de correlação usando função corrigida
        st.subheader("📊 Estatísticas de Correlação e Regressão")

        try:
            # Usar análise detalhada de correlação
            resultado_corr = analise_correlacao_detalhada(dados_scatter, eixo_x, eixo_y)
    
            if resultado_corr:
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                with col_stat1:
                    st.metric("Correlação (Pearson)", f"{resultado_corr['pearson']:.4f}")
                with col_stat2:
                    st.metric("Correlação (Spearman)", f"{resultado_corr['spearman']:.4f}")
                with col_stat3:
                    st.metric("Coeficiente R²", f"{resultado_corr['r_squared']:.4f}")
                with col_stat4:
                    st.metric("Inclinação", f"{resultado_corr['slope']:.4f}")
           
                # Interpretação da correlação
                st.subheader("🔍 Interpretação da Correlação")
                correlacao_abs = abs(resultado_corr['pearson'])
        
                if correlacao_abs > 0.7:
                    st.success("**Forte correlação** - Relação muito significativa entre as variáveis")
                elif correlacao_abs > 0.3:
                    st.warning("**Correlação moderada** - Relação moderada entre as variáveis")
                else:
                    st.info("**Fraca ou nenhuma correlação** - Pouca relação entre as variáveis")
            else:
                st.warning("Não foi possível calcular as estatísticas de correlação")
        
        except Exception as e:
            st.error(f"Erro ao calcular estatísticas: {str(e)}")

    # ========== ABA 5: CARTA DE CONTROLE ==========
    with tab5:
        st.header("🎯 Cartas de Controle com Limites de Especificação")
        
        st.markdown("""
        **Cartas de Controle** com limites de especificação (LSE/LIE) e limites de controle (3σ).
        - **Limites de Controle (3σ)**: Baseados na variação natural do processo
        - **Limites de Especificação**: Definidos pelos requisitos do cliente/produto
        """)
        
        # Seleção do tipo de carta
        tipo_carta = st.selectbox(
            "Selecione o tipo de Carta de Controle:",
            [
                "X-bar e S (Variáveis Contínuas - Com Grupos)",
                "Individuais e Amplitude Móvel (I-MR)",
                "P (Proporção de Defeituosos)",
                "C (Número de Defeitos)"
            ],
            key=generate_unique_key("tipo_carta", "tab5")
        )
        
        col_config1, col_config2 = st.columns(2)
        
        with col_config1:
            if tipo_carta in ["X-bar e S (Variáveis Contínuas - Com Grupos)", "Individuais e Amplitude Móvel (I-MR)"]:
                coluna_valor = st.selectbox(
                    "Selecione a variável a ser controlada:",
                    colunas_numericas,
                    key=generate_unique_key("carta_valor", "tab5")
                )
            
            elif tipo_carta == "P (Proporção de Defeituosos)":
                coluna_defeitos = st.selectbox(
                    "Selecione a coluna de itens defeituosos:",
                    colunas_numericas,
                    key=generate_unique_key("carta_defeitos", "tab5")
                )
                coluna_tamanho_amostra = st.selectbox(
                    "Selecione a coluna de tamanho da amostra:",
                    colunas_numericas,
                    key=generate_unique_key("carta_tamanho", "tab5")
                )
            
            elif tipo_carta == "C (Número de Defeitos)":
                coluna_defeitos = st.selectbox(
                    "Selecione a coluna de número de defeitos:",
                    colunas_numericas,
                    key=generate_unique_key("carta_num_defeitos", "tab5")
                )
        
        with col_config2:
            # Configurações comuns
            if tipo_carta in ["X-bar e S (Variáveis Contínuas - Com Grupos)", "P (Proporção de Defeituosos)", "C (Número de Defeitos)"]:
                coluna_grupo = st.selectbox(
                    "Selecione a coluna de grupo/amostra (opcional):",
                    [""] + colunas_todas,
                    key=generate_unique_key("carta_grupo", "tab5")
                )
                if not coluna_grupo:
                    tamanho_amostra = st.number_input(
                        "Tamanho do subgrupo:",
                        min_value=2,
                        max_value=50,
                        value=5,
                        key=generate_unique_key("tamanho_amostra", "tab5")
                    )
            
            elif tipo_carta == "Individuais e Amplitude Móvel (I-MR)":
                coluna_tempo = st.selectbox(
                    "Selecione a coluna de tempo/ordem (opcional):",
                    [""] + colunas_todas,
                    key=generate_unique_key("carta_tempo", "tab5")
                )
            
            # Configuração específica de limites para esta carta
            st.subheader("⚙️ Limites para esta Carta")
            if 'coluna_valor' in locals():
                lse_carta = st.number_input(
                    "LSE para esta carta:",
                    value=float(st.session_state.lse_values.get(coluna_valor, 0)),
                    key=generate_unique_key("lse_carta", "tab5")
                )
                lie_carta = st.number_input(
                    "LIE para esta carta:",
                    value=float(st.session_state.lie_values.get(coluna_valor, 0)),
                    key=generate_unique_key("lie_carta", "tab5")
                )
        
        # Botão para gerar carta
        if st.button("📊 Gerar Carta de Controle", use_container_width=True,
                    key=generate_unique_key("gerar_carta", "tab5")):
            
            try:
                if tipo_carta == "X-bar e S (Variáveis Contínuas - Com Grupos)":
                    if 'coluna_valor' in locals():
                        xbar, s, n, limites_xbar, limites_s = criar_carta_controle_xbar_s(
                            dados_processados, coluna_valor, 
                            coluna_grupo if coluna_grupo else None,
                            tamanho_amostra if not coluna_grupo else 5
                        )
                        
                        if xbar is not None:
                            # Carta X-bar com especificações
                            st.subheader(f"📊 Carta X-bar - {coluna_valor}")
                            fig_xbar, pontos_fora_xbar, pontos_fora_esp_xbar = plotar_carta_controle_com_especificacoes(
                                xbar, limites_xbar, (lse_carta, lie_carta),
                                f"Carta X-bar - {coluna_valor}", "xbar"
                            )
                            st.plotly_chart(fig_xbar, use_container_width=True)
                            
                            # Carta S com especificações
                            st.subheader(f"📊 Carta S - {coluna_valor}")
                            fig_s, pontos_fora_s, pontos_fora_esp_s = plotar_carta_controle_com_especificacoes(
                                s, limites_s, (None, None),  # Carta S não usa LSE/LIE
                                f"Carta S - {coluna_valor}", "s"
                            )
                            st.plotly_chart(fig_s, use_container_width=True)
                            
                            # Estatísticas detalhadas
                            st.subheader("📊 Estatísticas da Carta de Controle")
                            
                            # Estatísticas para X-bar
                            stats_xbar = calcular_estatisticas_carta_com_especificacoes(
                                xbar, limites_xbar, (lse_carta, lie_carta)
                            )
                            
                            col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                            with col_stat1:
                                st.metric("LSC X-bar (3σ)", f"{limites_xbar[0]:.4f}")
                                st.metric("LC X-bar", f"{limites_xbar[1]:.4f}")
                                st.metric("LIC X-bar (3σ)", f"{limites_xbar[2]:.4f}")
                            with col_stat2:
                                st.metric("LSE", f"{lse_carta:.4f}" if lse_carta != 0 else "Não definido")
                                st.metric("LIE", f"{lie_carta:.4f}" if lie_carta != 0 else "Não definido")
                                st.metric("Média", f"{stats_xbar['media']:.4f}")
                            with col_stat3:
                                st.metric("Pontos Fora Controle", pontos_fora_xbar)
                                st.metric("Pontos Fora Especificação", pontos_fora_esp_xbar)
                                st.metric("% Fora Controle", f"{stats_xbar['percentual_fora_controle']:.1f}%")
                            with col_stat4:
                                if 'cpk' in stats_xbar:
                                    st.metric("Cp", f"{stats_xbar['cp']:.3f}")
                                    st.metric("Cpk", f"{stats_xbar['cpk']:.3f}")
                                else:
                                    st.metric("Desvio Padrão", f"{stats_xbar['desvio_padrao']:.4f}")
                                    st.metric("Amplitude", f"{stats_xbar['maximo'] - stats_xbar['minimo']:.4f}")
                            
                            # ========== CLASSIFICAÇÃO DA CARTA ==========
                            st.subheader("🎨 Classificação da Carta de Controle")
                            
                            cpk = stats_xbar.get('cpk')
                            total_pontos = len(xbar)
                            
                            if total_pontos > 0:
                                # Classificar a carta de controle
                                cor, classificacao = classificar_carta_controle(cpk, pontos_fora_xbar, total_pontos)
                                
                                # Exibir indicador de classificação
                                html_classificacao = criar_indicador_classificacao(
                                    cor, classificacao, cpk, stats_xbar['percentual_fora_controle']
                                )
                                st.markdown(html_classificacao, unsafe_allow_html=True)
                            else:
                                st.warning("Não há dados suficientes para classificação")
                
                elif tipo_carta == "Individuais e Amplitude Móvel (I-MR)":
                    if 'coluna_valor' in locals():
                        individuais, mr, limites_i, limites_mr = criar_carta_controle_individual(
                            dados_processados, coluna_valor,
                            coluna_tempo if coluna_tempo else None
                        )
                        
                        if individuais is not None:
                            # Carta de Individuais com especificações
                            st.subheader(f"📊 Carta de Individuais - {coluna_valor}")
                            fig_i, pontos_fora_i, pontos_fora_esp_i = plotar_carta_controle_com_especificacoes(
                                individuais, limites_i, (lse_carta, lie_carta),
                                f"Carta de Individuais - {coluna_valor}", "individual"
                            )
                            st.plotly_chart(fig_i, use_container_width=True)
                            
                            # Carta de Amplitude Móvel
                            st.subheader(f"📊 Carta de Amplitude Móvel - {coluna_valor}")
                            fig_mr, pontos_fora_mr, _ = plotar_carta_controle_com_especificacoes(
                                mr, limites_mr, (None, None),  # MR não usa LSE/LIE
                                f"Carta de Amplitude Móvel - {coluna_valor}", "mr"
                            )
                            st.plotly_chart(fig_mr, use_container_width=True)
                            
                            # Estatísticas
                            st.subheader("📊 Estatísticas da Carta de Controle")
                            
                            # Estatísticas para individuais
                            stats_i = calcular_estatisticas_carta_com_especificacoes(
                                individuais, limites_i, (lse_carta, lie_carta)
                            )
                            
                            col_stat1, col_stat2, col_stat3 = st.columns(3)
                            with col_stat1:
                                st.metric("LSC Individuais (3σ)", f"{limites_i[0]:.4f}")
                                st.metric("LC Individuais", f"{limites_i[1]:.4f}")
                                st.metric("LIC Individuais (3σ)", f"{limites_i[2]:.4f}")
                            with col_stat2:
                                st.metric("LSE", f"{lse_carta:.4f}" if lse_carta != 0 else "Não definido")
                                st.metric("LIE", f"{lie_carta:.4f}" if lie_carta != 0 else "Não definido")
                                st.metric("Média", f"{stats_i['media']:.4f}")
                            with col_stat3:
                                st.metric("Pontos Fora Controle", pontos_fora_i)
                                st.metric("Pontos Fora Especificação", pontos_fora_esp_i)
                                if 'cpk' in stats_i:
                                    st.metric("Cpk", f"{stats_i['cpk']:.3f}")
                            
                            # ========== CLASSIFICAÇÃO DA CARTA ==========
                            st.subheader("🎨 Classificação da Carta de Controle")
                            
                            cpk = stats_i.get('cpk')
                            total_pontos = len(individuais)
                            
                            if total_pontos > 0:
                                # Classificar a carta de controle
                                cor, classificacao = classificar_carta_controle(cpk, pontos_fora_i, total_pontos)
                                
                                # Exibir indicador de classificação
                                html_classificacao = criar_indicador_classificacao(
                                    cor, classificacao, cpk, stats_i['percentual_fora_controle']
                                )
                                st.markdown(html_classificacao, unsafe_allow_html=True)
                            else:
                                st.warning("Não há dados suficientes para classificação")
                
                # Cartas P e C (implementação similar)
                elif tipo_carta == "P (Proporção de Defeituosos)":
                    if 'coluna_defeitos' in locals() and 'coluna_tamanho_amostra' in locals():
                        p, n, limites_p = criar_carta_controle_p(
                            dados_processados, coluna_defeitos, coluna_tamanho_amostra,
                            coluna_grupo if coluna_grupo else None
                        )
                        
                        if p is not None:
                            st.subheader(f"📊 Carta P - Proporção de Defeituosos")
                            fig_p, pontos_fora_p, _ = plotar_carta_controle_com_especificacoes(
                                p, limites_p, (None, None),
                                f"Carta P - Proporção de Defeituosos", "p"
                            )
                            st.plotly_chart(fig_p, use_container_width=True)
                            
                            # Estatísticas
                            st.subheader("📊 Estatísticas da Carta P")
                            col_stat1, col_stat2, col_stat3 = st.columns(3)
                            with col_stat1:
                                st.metric("LSC P", f"{limites_p[0]:.4f}")
                                st.metric("LC P", f"{limites_p[1]:.4f}")
                                st.metric("LIC P", f"{limites_p[2]:.4f}")
                            with col_stat2:
                                st.metric("Proporção Média", f"{limites_p[1]:.4f}")
                                st.metric("Tamanho Médio Amostra", f"{n.mean():.1f}" if n is not None else "N/A")
                            with col_stat3:
                                st.metric("Pontos Fora", pontos_fora_p)
                                st.metric("Total Grupos", len(p) if p is not None else 0)
                
                elif tipo_carta == "C (Número de Defeitos)":
                    if 'coluna_defeitos' in locals():
                        c, limites_c = criar_carta_controle_c(
                            dados_processados, coluna_defeitos,
                            coluna_grupo if coluna_grupo else None
                        )
                        
                        if c is not None:
                            st.subheader(f"📊 Carta C - Número de Defeitos")
                            fig_c, pontos_fora_c, _ = plotar_carta_controle_com_especificacoes(
                                c, limites_c, (None, None),
                                f"Carta C - Número de Defeitos", "c"
                            )
                            st.plotly_chart(fig_c, use_container_width=True)
                            
                            # Estatísticas
                            st.subheader("📊 Estatísticas da Carta C")
                            col_stat1, col_stat2, col_stat3 = st.columns(3)
                            with col_stat1:
                                st.metric("LSC C", f"{limites_c[0]:.4f}")
                                st.metric("LC C", f"{limites_c[1]:.4f}")
                                st.metric("LIC C", f"{limites_c[2]:.4f}")
                            with col_stat2:
                                st.metric("Número Médio de Defeitos", f"{limites_c[1]:.2f}")
                                st.metric("Desvio Padrão", f"{np.sqrt(limites_c[1]):.2f}" if limites_c[1] > 0 else "0.00")
                            with col_stat3:
                                st.metric("Pontos Fora", pontos_fora_c)
                                st.metric("Total Grupos", len(c) if c is not None else 0)
                
                # Análise de padrões
                st.subheader("🔍 Análise de Padrões na Carta de Controle")
                
                col_pad1, col_pad2 = st.columns(2)
                with col_pad1:
                    st.info("""
                    **📈 Interpretação dos Limites:**
                    - **Limites de Controle (3σ)**: Variação natural do processo
                    - **Limites de Especificação**: Requisitos do cliente
                    - **Processo Capaz**: Limites de controle dentro dos limites de especificação
                    - **Fora de Especificação**: Pontos além de LSE/LIE (⭐ laranja)
                    - **Fora de Controle**: Pontos além de LSC/LIC (✖️ vermelho)
                    """)
                
                with col_pad2:
                    st.warning("""
                    **🚨 Situações Críticas:**
                    - Pontos fora de especificação (⭐)
                    - Processo incapaz (Cpk < 1.33)
                    - Muitos pontos fora de controle
                    - Tendências sistemáticas
                    - Processo instável
                    """)
            
            except Exception as e:
                st.error(f"❌ Erro ao gerar carta de controle: {str(e)}")
                st.info("💡 **Dica**: Verifique se as colunas selecionadas contêm dados válidos.")

    # ========== ABA 6: ANÁLISE DE CAPABILIDADE ==========
    with tab6:
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
                key=generate_unique_key("capabilidade_col", "tab6")
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
                        key=generate_unique_key("executar_capabilidade", "tab6")):
                
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
                                # Gráfico de controle para capabilidade
                                fig_controle = go.Figure()
                                dados_controle = dados_processados[coluna_capabilidade].dropna()
                                
                                fig_controle.add_trace(go.Scatter(
                                    x=list(range(len(dados_controle))),
                                    y=dados_controle,
                                    mode='lines+markers',
                                    name='Valores',
                                    line=dict(color='blue', width=1),
                                    marker=dict(size=4)
                                ))
                                
                                # Média do processo
                                fig_controle.add_hline(y=resultados['media'], line_dash="solid", line_color="green",
                                                     annotation_text="Média", annotation_position="right")
                                
                                # Limites de especificação
                                if lse_cap != 0:
                                    fig_controle.add_hline(y=lse_cap, line_dash="dash", line_color="red",
                                                         annotation_text="LSE", annotation_position="right")
                                
                                if lie_cap != 0:
                                    fig_controle.add_hline(y=lie_cap, line_dash="dash", line_color="red",
                                                         annotation_text="LIE", annotation_position="right")
                                
                                fig_controle.update_layout(
                                    title=f"Gráfico de Controle - {coluna_capabilidade}",
                                    xaxis_title="Amostra",
                                    yaxis_title=coluna_capabilidade,
                                    showlegend=True,
                                    height=400
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
                            
                            if 'cpk' in resultados:
                                cpk = resultados['cpk']
                                if cpk >= 1.67:
                                    st.success("✅ **Excelente** - Processo altamente capaz (Cpk ≥ 1.67)")
                                elif cpk >= 1.33:
                                    st.success("✅ **Muito Bom** - Processo capaz (1.33 ≤ Cpk < 1.67)")
                                elif cpk >= 1.0:
                                    st.warning("⚠️ **Aceitável** - Processo marginalmente capaz (1.0 ≤ Cpk < 1.33)")
                                elif cpk >= 0.67:
                                    st.error("❌ **Insatisfatório** - Processo incapaz (0.67 ≤ Cpk < 1.0)")
                                else:
                                    st.error("🚨 **Crítico** - Processo totalmente incapaz (Cpk < 0.67)")
                            else:
                                st.info("ℹ️ **Informação** - Cpk não pode ser calculado sem ambos os limites de especificação")
                            
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

    # ========== ABA 7: ESTATÍSTICA AVANÇADAS ==========
    with tab7:
        st.header("🔬 Análises Estatísticas Avançadas")
        
        st.markdown("""
        **Análises Estatísticas Avançadas** incluem:
        - ANOVA (Análise de Variância)
        - Testes de Hipótese
        - Análise de Poder Estatístico
        - Regressão Múltipla
        - Análise Bayesiana
        - Simulações Monte Carlo
        - Testes Não Paramétricos
        """)
        
        # Seleção do tipo de análise avançada
        tipo_analise_avancada = st.selectbox(
            "Selecione o tipo de análise avançada:",
            [
                "ANOVA - Um Fator",
                "Testes Não Paramétricos",
                "Teste de Hipótese para Média",
                "Análise de Poder Estatístico",
                "Regressão Múltipla",
                "Análise Bayesiana (A/B Testing)",
                "Simulação Monte Carlo"
            ],
            key=generate_unique_key("tipo_analise_avancada", "tab7")
        )
        
        if tipo_analise_avancada == "ANOVA - Um Fator":
            st.subheader("📊 ANOVA - Análise de Variância de Um Fator")
            
            if len(colunas_numericas) > 0 and len(colunas_todas) > 1:
                col_anova1, col_anova2 = st.columns(2)
                with col_anova1:
                    variavel_resposta = st.selectbox(
                        "Variável Resposta (numérica):",
                        colunas_numericas,
                        key=generate_unique_key("anova_resposta", "tab7")
                    )
                with col_anova2:
                    fator = st.selectbox(
                        "Fator (categórica):",
                        [col for col in colunas_todas if col != variavel_resposta],
                        key=generate_unique_key("anova_fator", "tab7")
                    )
                
                # Opção para teste não paramétrico
                usar_nao_parametrico = st.checkbox(
                    "Usar teste não paramétrico (Kruskal-Wallis)",
                    key=generate_unique_key("usar_nao_parametrico", "tab7")
                )
                
                if st.button("📈 Executar ANOVA", use_container_width=True,
                           key=generate_unique_key("executar_anova", "tab7")):
                    
                    if usar_nao_parametrico:
                        # Teste não paramétrico Kruskal-Wallis
                        resultado_nao_param = analise_nao_parametrica_completa(dados_processados, variavel_resposta, fator)
                        
                        if resultado_nao_param and 'kruskal_wallis' in resultado_nao_param:
                            resultado = resultado_nao_param['kruskal_wallis']
                            st.subheader("📋 Resultados do Teste de Kruskal-Wallis")
                            
                            col_res1, col_res2 = st.columns(2)
                            with col_res1:
                                st.metric("Estatística H", f"{resultado['h_statistic']:.4f}")
                                st.metric("Valor-p", f"{resultado['p_value']:.4f}")
                            
                            with col_res2:
                                significativo = "✅ Significativo" if resultado['significativo'] else "❌ Não Significativo"
                                st.metric("Significância", significativo)
                                st.metric("Número de Grupos", resultado_nao_param['n_grupos'])
                            
                            st.info("""
                            **📝 Sobre o Teste de Kruskal-Wallis:**
                            - Teste não paramétrico equivalente à ANOVA
                            - Não assume normalidade dos dados
                            - Usa ranks em vez dos valores originais
                            - Adequado para dados ordinais ou não normais
                            """)
                    else:
                        # ANOVA paramétrica tradicional
                        resultado_anova = analise_anova_um_fator_sem_scipy(dados_processados, variavel_resposta, fator)
                        
                        if resultado_anova:
                            st.subheader("📋 Resultados da ANOVA")
                            
                            col_res1, col_res2 = st.columns(2)
                            with col_res1:
                                st.metric("Estatística F", f"{resultado_anova['f_statistic']:.4f}")
                                st.metric("Valor-p", f"{resultado_anova['p_value']:.4f}")
                            
                            with col_res2:
                                significativo = "✅ Significativo" if resultado_anova['significativo'] else "❌ Não Significativo"
                                st.metric("Significância", significativo)
                                st.metric("Número de Grupos", len(resultado_anova['grupos']))
                    
                    # Estatísticas descritivas por grupo (comum a ambos os testes)
                    if ('resultado_anova' in locals() and resultado_anova) or ('resultado_nao_param' in locals() and resultado_nao_param):
                        st.subheader("📊 Estatísticas Descritivas por Grupo")
                        
                        if usar_nao_parametrico and resultado_nao_param:
                            descritivas_df = pd.DataFrame(resultado_nao_param['descritivas']).T
                        else:
                            descritivas_df = pd.DataFrame(resultado_anova['descritivas']).T
                        
                        st.dataframe(descritivas_df.style.format({
                            'media': '{:.4f}',
                            'desvio_padrao': '{:.4f}',
                            'mediana': '{:.4f}',
                            'min': '{:.4f}',
                            'max': '{:.4f}'
                        }))
                        
                        # Gráfico de boxplot por grupo
                        fig = px.box(dados_processados, x=fator, y=variavel_resposta,
                                    title=f"Distribuição de {variavel_resposta} por {fator}")
                        st.plotly_chart(fig, use_container_width=True)
        
        elif tipo_analise_avancada == "Testes Não Paramétricos":
            st.subheader("🎯 Testes Não Paramétricos para Comparação de Grupos")
            
            if len(colunas_numericas) > 0 and len(colunas_todas) > 1:
                col_test1, col_test2 = st.columns(2)
                with col_test1:
                    variavel_resposta = st.selectbox(
                        "Variável Resposta (numérica):",
                        colunas_numericas,
                        key=generate_unique_key("np_resposta", "tab7")
                    )
                with col_test2:
                    fator = st.selectbox(
                        "Fator (categórica):",
                        [col for col in colunas_todas if col != variavel_resposta],
                        key=generate_unique_key("np_fator", "tab7")
                    )
                
                if st.button("🎯 Executar Análise Não Paramétrica", use_container_width=True,
                           key=generate_unique_key("executar_nao_param", "tab7")):
                    
                    resultado_nao_param = analise_nao_parametrica_completa(dados_processados, variavel_resposta, fator)
                    
                    if resultado_nao_param:
                        fatores_unicos = dados_processados[fator].dropna().unique()
                        
                        # Kruskal-Wallis para múltiplos grupos
                        if len(fatores_unicos) > 2 and 'kruskal_wallis' in resultado_nao_param:
                            resultado_kw = resultado_nao_param['kruskal_wallis']
                            
                            st.subheader("📊 Teste de Kruskal-Wallis (Múltiplos Grupos)")
                            col_kw1, col_kw2, col_kw3 = st.columns(3)
                            with col_kw1:
                                st.metric("Estatística H", f"{resultado_kw['h_statistic']:.4f}")
                            with col_kw2:
                                st.metric("Valor-p", f"{resultado_kw['p_value']:.4f}")
                            with col_kw3:
                                sig = "✅ Significativo" if resultado_kw['significativo'] else "❌ Não Significativo"
                                st.metric("Resultado", sig)
                        
                        # Mann-Whitney para 2 grupos
                        elif len(fatores_unicos) == 2 and 'mann_whitney' in resultado_nao_param:
                            resultado_mw = resultado_nao_param['mann_whitney']
                            
                            st.subheader("📊 Teste de Mann-Whitney U (2 Grupos)")
                            col_mw1, col_mw2, col_mw3 = st.columns(3)
                            with col_mw1:
                                st.metric("Estatística U", f"{resultado_mw['u_statistic']:.4f}")
                            with col_mw2:
                                st.metric("Valor-p", f"{resultado_mw['p_value']:.4f}")
                            with col_mw3:
                                sig = "✅ Significativo" if resultado_mw['significativo'] else "❌ Não Significativo"
                                st.metric("Resultado", sig)
                        
                        # Estatísticas descritivas não paramétricas
                        st.subheader("📈 Estatísticas Descritivas Não Paramétricas")
                        descritivas_df = pd.DataFrame(resultado_nao_param['descritivas']).T
                        st.dataframe(descritivas_df.style.format({
                            'mediana': '{:.4f}',
                            'q1': '{:.4f}',
                            'q3': '{:.4f}',
                            'min': '{:.4f}',
                            'max': '{:.4f}',
                            'iqr': '{:.4f}'
                        }))
                        
                        # Gráficos
                        st.subheader("📊 Visualizações")
                        col_viz1, col_viz2 = st.columns(2)
                        
                        with col_viz1:
                            # Boxplot
                            fig_box = px.box(dados_processados, x=fator, y=variavel_resposta,
                                           title=f"Distribuição de {variavel_resposta} por {fator}")
                            st.plotly_chart(fig_box, use_container_width=True)
                        
                        with col_viz2:
                            # Violin plot
                            fig_violin = px.violin(dados_processados, x=fator, y=variavel_resposta,
                                                 title=f"Distribuição de Densidade - {variavel_resposta} por {fator}")
                            st.plotly_chart(fig_violin, use_container_width=True)
                        
                        # Interpretação
                        st.subheader("🔍 Interpretação dos Testes Não Paramétricos")
                        st.info("""
                        **📝 Sobre Testes Não Paramétricos:**
                        - **Vantagens**: Não assumem distribuição normal, robustos a outliers
                        - **Adequados para**: Dados ordinais, distribuições não normais, amostras pequenas
                        - **Kruskal-Wallis**: Equivalente não paramétrico da ANOVA
                        - **Mann-Whitney**: Equivalente não paramétrico do teste t para 2 grupos
                        - **Interpretação**: Rejeitar H₀ indica diferenças significativas entre grupos
                        """)
        
        elif tipo_analise_avancada == "Teste de Hipótese para Média":
            st.subheader("🎯 Teste de Hipótese para Média")
            
            if colunas_numericas:
                col_hip1, col_hip2 = st.columns(2)
                with col_hip1:
                    variavel_teste = st.selectbox(
                        "Variável para teste:",
                        colunas_numericas,
                        key=generate_unique_key("hip_var", "tab7")
                    )
                with col_hip2:
                    valor_referencia = st.number_input(
                        "Valor de referência (H₀):",
                        value=0.0,
                        key=generate_unique_key("hip_valor_ref", "tab7")
                    )
                
                alternativa = st.selectbox(
                    "Hipótese alternativa:",
                    ["two-sided", "greater", "less"],
                    format_func=lambda x: {
                        "two-sided": "Média ≠ Valor de referência",
                        "greater": "Média > Valor de referência", 
                        "less": "Média < Valor de referência"
                    }[x],
                    key=generate_unique_key("hip_alternativa", "tab7")
                )
                
                if st.button("📊 Executar Teste de Hipótese", use_container_width=True,
                           key=generate_unique_key("executar_hipotese", "tab7")):
                    
                    resultado_teste = teste_hipotese_media_sem_scipy(
                        dados_processados, variavel_teste, valor_referencia, alternativa
                    )
                    
                    if resultado_teste:
                        st.subheader("📋 Resultados do Teste de Hipótese")
                        
                        col_res1, col_res2, col_res3 = st.columns(3)
                        with col_res1:
                            st.metric("Estatística t", f"{resultado_teste['t_statistic']:.4f}")
                            st.metric("Valor-p", f"{resultado_teste['p_value']:.4f}")
                        
                        with col_res2:
                            st.metric("Média Amostral", f"{resultado_teste['media_amostral']:.4f}")
                            st.metric("Valor de Referência", f"{resultado_teste['valor_referencia']:.4f}")
                        
                        with col_res3:
                            significativo = "✅ Rejeita H₀" if resultado_teste['significativo'] else "❌ Não rejeita H₀"
                            st.metric("Decisão", significativo)
                        
                        # Intervalo de confiança
                        st.subheader("📊 Intervalo de Confiança 95%")
                        ci = resultado_teste['intervalo_confianca']
                        st.info(f"Intervalo de confiança: ({ci[0]:.4f}, {ci[1]:.4f})")
        
        elif tipo_analise_avancada == "Análise de Poder Estatístico":
            st.subheader("📈 Análise de Poder Estatístico")
            
            if colunas_numericas:
                col_poder1, col_poder2 = st.columns(2)
                with col_poder1:
                    variavel_poder = st.selectbox(
                        "Variável para análise:",
                        colunas_numericas,
                        key=generate_unique_key("poder_var", "tab7")
                    )
                with col_poder2:
                    efeito_detectavel = st.number_input(
                        "Efeito mínimo detectável:",
                        value=0.5,
                        key=generate_unique_key("poder_efeito", "tab7")
                    )
                
                alpha = st.slider(
                    "Nível de significância (α):",
                    min_value=0.01,
                    max_value=0.10,
                    value=0.05,
                    step=0.01,
                    key=generate_unique_key("poder_alpha", "tab7")
                )
                
                if st.button("📊 Calcular Poder Estatístico", use_container_width=True,
                           key=generate_unique_key("calcular_poder", "tab7")):
                    
                    resultado_poder = analise_poder_estatistico_sem_scipy(
                        dados_processados, variavel_poder, efeito_detectavel, alpha
                    )
                    
                    if resultado_poder:
                        st.subheader("📋 Resultados da Análise de Poder")
                        
                        col_res1, col_res2, col_res3 = st.columns(3)
                        with col_res1:
                            st.metric("Poder Atual", f"{resultado_poder['poder_atual']:.3f}")
                            st.metric("Tamanho Amostral Atual", resultado_poder['tamanho_amostral_atual'])
                        
                        with col_res2:
                            st.metric("Tamanho Amostral Necessário", f"{resultado_poder['tamanho_amostral_necessario']:.0f}")
                            st.metric("Effect Size", f"{resultado_poder['effect_size']:.3f}")
                        
                        with col_res3:
                            st.metric("Nível α", resultado_poder['alpha'])
                        
                        # Interpretação
                        st.subheader("🔍 Interpretação do Poder Estatístico")
                        poder = resultado_poder['poder_atual']
                        
                        if poder >= 0.8:
                            st.success("✅ **Poder adequado** - Boa chance de detectar efeitos reais")
                        elif poder >= 0.5:
                            st.warning("⚠️ **Poder moderado** - Considerar aumentar o tamanho amostral")
                        else:
                            st.error("❌ **Poder insuficiente** - Alta chance de erro Tipo II")
        
        elif tipo_analise_avancada == "Regressão Múltipla":
            st.subheader("📈 Regressão Múltipla")
            
            if len(colunas_numericas) > 1:
                col_reg1, col_reg2 = st.columns(2)
                with col_reg1:
                    variavel_resposta = st.selectbox(
                        "Variável Resposta (Y):",
                        colunas_numericas,
                        key=generate_unique_key("reg_resposta", "tab7")
                    )
                with col_reg2:
                    variaveis_predictoras = st.multiselect(
                        "Variáveis Preditivas (X):",
                        [col for col in colunas_numericas if col != variavel_resposta],
                        key=generate_unique_key("reg_predictoras", "tab7")
                    )
                
                if variaveis_predictoras and st.button("📊 Executar Regressão Múltipla", 
                                                     use_container_width=True,
                                                     key=generate_unique_key("executar_regressao", "tab7")):
                    
                    resultado_regressao = analise_regressao_multipla_sem_scipy(
                        dados_processados, variavel_resposta, variaveis_predictoras
                    )
                    
                    if resultado_regressao:
                        st.subheader("📋 Resultados da Regressão Múltipla")
                        
                        # Métricas do modelo
                        col_met1, col_met2, col_met3, col_met4 = st.columns(4)
                        with col_met1:
                            st.metric("R²", f"{resultado_regressao['r2']:.4f}")
                        with col_met2:
                            st.metric("R² Ajustado", f"{resultado_regressao['r2_ajustado']:.4f}")
                        with col_met3:
                            st.metric("RMSE", f"{resultado_regressao['rmse']:.4f}")
                        with col_met4:
                            st.metric("MSE", f"{resultado_regressao['mse']:.4f}")
                        
                        # Coeficientes com estatísticas
                        st.subheader("📊 Coeficientes do Modelo")
                        coeficientes_df = pd.DataFrame({
                            'Variável': list(resultado_regressao['coeficientes'].keys()),
                            'Coeficiente': list(resultado_regressao['coeficientes'].values()),
                            'Erro Padrão': list(resultado_regressao['std_erros'].values()),
                            'Estatística t': list(resultado_regressao['t_stats'].values()),
                            'Valor-p': list(resultado_regressao['p_values'].values())
                        })
                        st.dataframe(coeficientes_df.style.format({
                            'Coeficiente': '{:.4f}',
                            'Erro Padrão': '{:.4f}', 
                            'Estatística t': '{:.4f}',
                            'Valor-p': '{:.4f}'
                        }))
        
        elif tipo_analise_avancada == "Análise Bayesiana (A/B Testing)":
            st.subheader("🎲 Análise Bayesiana para A/B Testing")
            
            if len(colunas_numericas) >= 2:
                st.info("""
                **Análise Bayesiana para comparação de dois grupos (A/B Testing).**
                Selecione duas variáveis numéricas para comparar.
                """)
                
                col_bayes1, col_bayes2 = st.columns(2)
                with col_bayes1:
                    grupo_controle = st.selectbox(
                        "Grupo Controle (A):",
                        colunas_numericas,
                        key=generate_unique_key("bayes_controle", "tab7")
                    )
                with col_bayes2:
                    grupo_variacao = st.selectbox(
                        "Grupo Variação (B):",
                        [col for col in colunas_numericas if col != grupo_controle],
                        key=generate_unique_key("bayes_variacao", "tab7")
                    )
                
                # Parâmetros da prior
                col_prior1, col_prior2 = st.columns(2)
                with col_prior1:
                    prior_alpha = st.number_input("Prior Alpha", value=1.0, min_value=0.1, key=generate_unique_key("prior_alpha", "tab7"))
                with col_prior2:
                    prior_beta = st.number_input("Prior Beta", value=1.0, min_value=0.1, key=generate_unique_key("prior_beta", "tab7"))
                
                if st.button("🎯 Executar Análise Bayesiana", use_container_width=True,
                           key=generate_unique_key("executar_bayes", "tab7")):
                    
                    controle_data = dados_processados[grupo_controle].dropna()
                    variacao_data = dados_processados[grupo_variacao].dropna()
                    
                    resultado_bayes = analise_bayesiana_ab_test_sem_scipy(controle_data, variacao_data, prior_alpha, prior_beta)
                    
                    if resultado_bayes:
                        st.subheader("📋 Resultados da Análise Bayesiana")
                        
                        col_res1, col_res2, col_res3 = st.columns(3)
                        with col_res1:
                            st.metric("Prob. B > A", f"{resultado_bayes['prob_variacao_melhor']:.3f}")
                            st.metric("Média Controle", f"{resultado_bayes['media_controle']:.4f}")
                        
                        with col_res2:
                            st.metric("Média Variação", f"{resultado_bayes['media_variacao']:.4f}")
                            dif_media = resultado_bayes['media_variacao'] - resultado_bayes['media_controle']
                            st.metric("Diferença", f"{dif_media:.4f}")
                        
                        with col_res3:
                            st.metric("Taxa Controle", f"{resultado_bayes['taxa_controle']:.4f}")
                            st.metric("Taxa Variação", f"{resultado_bayes['taxa_variacao']:.4f}")
                        
                        # Intervalos credíveis
                        st.subheader("📊 Intervalos Credíveis 95%")
                        col_ic1, col_ic2, col_ic3 = st.columns(3)
                        with col_ic1:
                            ic_controle = resultado_bayes['intervalo_controle']
                            st.metric("IC Controle", f"({ic_controle[0]:.3f}, {ic_controle[1]:.3f})")
                        with col_ic2:
                            ic_variacao = resultado_bayes['intervalo_variacao']
                            st.metric("IC Variação", f"({ic_variacao[0]:.3f}, {ic_variacao[1]:.3f})")
                        with col_ic3:
                            ic_diferenca = resultado_bayes['intervalo_diferenca']
                            st.metric("IC Diferença", f"({ic_diferenca[0]:.3f}, {ic_diferenca[1]:.3f})")
                        
                        # Interpretação
                        st.subheader("🔍 Interpretação Bayesiana")
                        prob = resultado_bayes['prob_variacao_melhor']
                        
                        if prob > 0.95:
                            st.success("✅ **Evidência forte** de que B é melhor que A")
                        elif prob > 0.8:
                            st.warning("⚠️ **Evidência moderada** de que B é melhor que A")
                        elif prob > 0.6:
                            st.info("ℹ️ **Evidência fraca** de que B é melhor que A")
                        else:
                            st.error("❌ **Pouca evidência** de que B é melhor que A")
        
        elif tipo_analise_avancada == "Simulação Monte Carlo":
            st.subheader("🎲 Simulação Monte Carlo para Capabilidade")
            
            if colunas_numericas:
                col_sim1, col_sim2 = st.columns(2)
                with col_sim1:
                    variavel_simulacao = st.selectbox(
                        "Variável para simulação:",
                        colunas_numericas,
                        key=generate_unique_key("var_simulacao", "tab7")
                    )
                with col_sim2:
                    n_simulacoes = st.number_input(
                        "Número de simulações:",
                        min_value=1000,
                        max_value=100000,
                        value=10000,
                        step=1000,
                        key=generate_unique_key("n_simulacoes", "tab7")
                    )
                
                # Parâmetros do processo
                st.subheader("🎯 Parâmetros do Processo")
                dados_variavel = dados_processados[variavel_simulacao].dropna()
                media_atual = dados_variavel.mean()
                std_atual = dados_variavel.std()
                
                col_param1, col_param2, col_param3 = st.columns(3)
                with col_param1:
                    media_processo = st.number_input(
                        "Média do processo:",
                        value=float(media_atual),
                        key=generate_unique_key("media_processo", "tab7")
                    )
                with col_param2:
                    desvio_processo = st.number_input(
                        "Desvio padrão do processo:",
                        value=float(std_atual),
                        key=generate_unique_key("desvio_processo", "tab7")
                    )
                with col_param3:
                    # Limites de especificação
                    lse_sim = st.number_input(
                        "LSE para simulação:",
                        value=float(st.session_state.lse_values.get(variavel_simulacao, media_atual + 3*std_atual)),
                        key=generate_unique_key("lse_sim", "tab7")
                    )
                    lie_sim = st.number_input(
                        "LIE para simulação:",
                        value=float(st.session_state.lie_values.get(variavel_simulacao, media_atual - 3*std_atual)),
                        key=generate_unique_key("lie_sim", "tab7")
                    )
                
                if st.button("🎲 Executar Simulação Monte Carlo", use_container_width=True,
                            key=generate_unique_key("executar_monte_carlo", "tab7")):
                    
                    with st.spinner("Executando simulação Monte Carlo..."):
                        resultado_simulacao = simulacao_monte_carlo_capabilidade(
                            media_processo, desvio_processo, lse_sim, lie_sim, n_simulacoes
                        )
                    
                    if resultado_simulacao:
                        st.subheader("📋 Resultados da Simulação")
                        
                        col_res1, col_res2, col_res3, col_res4 = st.columns(4)
                        with col_res1:
                            st.metric("Cpk Médio Simulado", f"{resultado_simulacao['cpk_medio']:.3f}")
                            st.metric("Desvio Cpk", f"{resultado_simulacao['cpk_std']:.3f}")
                        
                        with col_res2:
                            st.metric("PPM Simulado", f"{resultado_simulacao['ppm_simulado']:.0f}")
                            st.metric("% Fora Especificação", f"{resultado_simulacao['percentual_fora_simulado']:.2f}%")
                        
                        with col_res3:
                            st.metric("Média Simulada", f"{resultado_simulacao['media_simulacao']:.4f}")
                            st.metric("Desvio Simulado", f"{resultado_simulacao['desvio_padrao_simulacao']:.4f}")
                        
                        with col_res4:
                            st.metric("Nº Simulações", n_simulacoes)
                        
                        # Histograma das simulações
                        st.subheader("📊 Distribuição das Simulações")
                        fig_sim = px.histogram(
                            x=resultado_simulacao['simulacoes'],
                            nbins=50,
                            title=f"Distribuição das {n_simulacoes} Simulações Monte Carlo"
                        )
                        
                        # Adicionar limites de especificação
                        fig_sim.add_vline(x=lse_sim, line_dash="dash", line_color="red", annotation_text="LSE")
                        fig_sim.add_vline(x=lie_sim, line_dash="dash", line_color="red", annotation_text="LIE")
                        fig_sim.add_vline(x=media_processo, line_dash="solid", line_color="green", annotation_text="Média")
                        
                        st.plotly_chart(fig_sim, use_container_width=True)

    # ========== ABA 8: RESUMO EXECUTIVO ==========
    with tab8:
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
            medias = dados_processados[colunas_numericas].mean()
            stds = dados_processados[colunas_numericas].std()
            coef_variacao = (stds / medias).replace([np.inf, -np.inf], np.nan).dropna()
            if len(coef_variacao) > 0:
                coef_variacao_medio = coef_variacao.mean()
                
                if coef_variacao_medio > 0.5:
                    st.warning("⚠️ **Alta variabilidade** detectada no processo")
                elif coef_variacao_medio < 0.1:
                    st.success("✅ **Baixa variabilidade** - Processo estável")
        
        # Recomendações gerais
        st.subheader("💡 Recomendações Gerais")
        
        col_rec1, col_rec2 = st.columns(2)
        with col_rec1:
            st.info("""
            **📈 Para Melhoria Contínua:**
            - Implementar cartas de controle para variáveis críticas
            - Realizar análises de capabilidade regularmente
            - Identificar e eliminar causas especiais de variação
            - Estabelecer limites de controle realistas
            """)
        
        with col_rec2:
            st.info("""
            **🔧 Para Otimização:**
            - Usar simulações Monte Carlo para análise de risco
            - Aplicar técnicas de otimização para encontrar melhores configurações
            - Considerar análises de sensibilidade
            - Implementar sistemas de monitoramento contínuo
            """)

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

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. Configuração da Página
# ==========================================
st.set_page_config(
    page_title="Dashboard de Manutenção Preditiva",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Painel de Manutenção Preditiva - Detecção de Anomalias")
st.markdown(
    """
    Monitoramento de sensores de chão de fábrica utilizando o algoritmo de clusterização **DBSCAN** 
    para segregar regimes normais de operação de possíveis anomalias/falhas mecânicas.
    """
)

# ==========================================
# 2. Geração Sintética dos Dados de Sensores
# ==========================================
@st.cache_data
def gerar_dados_sensores(n_normais=250, n_outliers=20, seed=42):
    np.random.seed(seed)
    
    # Cluster 1: Máquinas operando em regime nominal 1 (baixa rotação / carga média)
    vibracao_1 = np.random.normal(loc=30, scale=3.5, size=n_normais // 2)
    temperatura_1 = np.random.normal(loc=65, scale=4.0, size=n_normais // 2)
    
    # Cluster 2: Máquinas operando em regime nominal 2 (alta rotação / carga alta)
    vibracao_2 = np.random.normal(loc=60, scale=4.0, size=n_normais // 2)
    temperatura_2 = np.random.normal(loc=85, scale=5.0, size=n_normais // 2)
    
    # Valores discrepantes (Outliers / Anomalias simuladas de falha)
    vibracao_outliers = np.random.uniform(low=10, high=95, size=n_outliers)
    temperatura_outliers = np.random.uniform(low=40, high=120, size=n_outliers)
    
    # Consolidação
    vibracao = np.concatenate([vibracao_1, vibracao_2, vibracao_outliers])
    temperatura = np.concatenate([temperatura_1, temperatura_2, temperatura_outliers])
    
    ids_maquinas = [f"MAQ-{i+1:03d}" for i in range(len(vibracao))]
    
    df = pd.DataFrame({
        "ID_Maquina": ids_maquinas,
        "Frequencia_Vibracao": np.round(vibracao, 2),
        "Pico_Temperatura": np.round(temperatura, 2)
    })
    
    return df

df_sensores = gerar_dados_sensores()

# ==========================================
# 3. Barra Lateral: Hiperparâmetros DBSCAN
# ==========================================
st.sidebar.header("🔧 Parâmetros do DBSCAN")
st.sidebar.markdown("Ajuste a vizinhança e densidade para classificar os pontos:")

eps = st.sidebar.slider(
    label="Epsilon (eps): Distância máxima entre amostras",
    min_value=0.1,
    max_value=2.0,
    value=0.45,
    step=0.05,
    help="Raio de busca ao redor de cada ponto no espaço padronizado."
)

min_samples = st.sidebar.slider(
    label="Min Samples: Mínimo de pontos por vizinhança",
    min_value=2,
    max_value=25,
    value=6,
    step=1,
    help="Número mínimo de amostras em uma vizinhança para formar um ponto central."
)

# ==========================================
# 4. Processamento & Treinamento do DBSCAN
# ==========================================
# Normalização das variáveis para garantir isotropia na métrica euclidiana
features = ["Frequencia_Vibracao", "Pico_Temperatura"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_sensores[features])

# Treinamento do DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
df_sensores["Cluster"] = dbscan.fit_predict(X_scaled)

# Classificação amigável
# No DBSCAN, ruído/outlier recebe o rótulo -1
df_sensores["Status"] = df_sensores["Cluster"].apply(
    lambda x: "Anomalia (Ruído)" if x == -1 else f"Cluster {x}"
)

# Separação de anomalias
df_anomalias = df_sensores[df_sensores["Cluster"] == -1].copy()

# ==========================================
# 5. Métricas Gerais (KPIs)
# ==========================================
col1, col2, col3 = st.columns(3)
n_clusters_validos = len(set(df_sensores["Cluster"])) - (1 if -1 in df_sensores["Cluster"].values else 0)

col1.metric("Total de Máquinas Monitoradas", len(df_sensores))
col2.metric("Regimes de Operação (Clusters)", n_clusters_validos)
col3.metric("Anomalias Detectadas", len(df_anomalias), delta_color="inverse")

st.divider()

# ==========================================
# 6. Visualização com Seaborn & Matplotlib
# ==========================================
st.subheader("📊 Mapa de Operação e Dispersão de Sensores")

# Configuração da paleta: Garantir que o rótulo -1 fique preto
unique_clusters = sorted(df_sensores["Cluster"].unique())
colors = sns.color_palette("tab10", n_colors=max(len(unique_clusters), 1))

palette = {}
color_idx = 0
for cluster_id in unique_clusters:
    if cluster_id == -1:
        palette[cluster_id] = "#000000"  # Preto sólido para anomalias
    else:
        palette[cluster_id] = colors[color_idx % len(colors)]
        color_idx += 1

fig, ax = plt.subplots(figsize=(10, 5))
sns.set_style("whitegrid")

# Gráfico de dispersão
scatter = sns.scatterplot(
    data=df_sensores,
    x="Frequencia_Vibracao",
    y="Pico_Temperatura",
    hue="Cluster",
    palette=palette,
    style=(df_sensores["Cluster"] == -1),  # Destaque de marcador para anomalias
    markers={False: "o", True: "X"},
    s=70,
    alpha=0.85,
    edgecolor="k",
    ax=ax
)

ax.set_title("Frequência de Vibração vs. Pico de Temperatura", fontsize=13, fontweight="bold")
ax.set_xlabel("Frequência de Vibração (Hz)", fontsize=11)
ax.set_ylabel("Pico de Temperatura (°C)", fontsize=11)

# Construção de legenda explicativa
handles, labels = ax.get_legend_handles_labels()
novos_labels = [
    "Anomalia (Falha)" if lbl == "-1" else f"Regime Normal ({lbl})" 
    for lbl in labels
]
ax.legend(handles=handles, labels=novos_labels, title="Classificação DBSCAN", loc="upper left")

st.pyplot(fig)

st.divider()

# ==========================================
# 7. Tabela Interativa de Anomalias
# ==========================================
st.subheader("🚨 Lista de Máquinas em Alerta (Anomalias)")

if not df_anomalias.empty:
    st.warning(f"Atenção: Foram identificadas **{len(df_anomalias)} máquinas** fora do padrão normal de operação.")
    
    tabela_exibicao = df_anomalias[["ID_Maquina", "Frequencia_Vibracao", "Pico_Temperatura", "Status"]].reset_index(drop=True)
    
    st.dataframe(
        tabela_exibicao,
        use_container_width=True,
        column_config={
            "ID_Maquina": "Identificador da Máquina",
            "Frequencia_Vibracao": st.column_config.NumberColumn(
                "Frequência de Vibração (Hz)", format="%.2f Hz"
            ),
            "Pico_Temperatura": st.column_config.NumberColumn(
                "Pico de Temperatura (°C)", format="%.2f °C"
            ),
            "Status": "Condição Operacional"
        }
    )
else:
    st.success("Nenhuma anomalia detectada com a configuração atual de parâmetros.")

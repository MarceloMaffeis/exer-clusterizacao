import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

# ==========================================
# 1. Configuração da Página
# ==========================================
st.set_page_config(
    page_title="Dashboard Preditivo 3D",
    page_icon="⚙️",
    layout="wide"
)

st.title("⚙️ Monitoramento Preditivo 3D - Detecção de Anomalias")
st.markdown(
    """
    Agrupamento e detecção de anomalias com **DBSCAN** em espaço tridimensional de sensores.
    Gire, aproxime e passe o cursor sobre os pontos para inspecionar os dados individuais.
    """
)

# ==========================================
# 2. Geração Sintética dos Dados (3 Variáveis)
# ==========================================
@st.cache_data
def gerar_dados_sensores(n_normais=260, n_outliers=25, seed=42):
    np.random.seed(seed)
    
    # Regime 1: Baixa rotação / Carga média
    vib_1 = np.random.normal(loc=28, scale=3.5, size=n_normais // 2)
    temp_1 = np.random.normal(loc=62, scale=4.0, size=n_normais // 2)
    ruido_1 = np.random.normal(loc=45, scale=3.0, size=n_normais // 2)
    
    # Regime 2: Alta rotação / Carga alta
    vib_2 = np.random.normal(loc=65, scale=4.5, size=n_normais // 2)
    temp_2 = np.random.normal(loc=88, scale=5.0, size=n_normais // 2)
    ruido_2 = np.random.normal(loc=75, scale=4.0, size=n_normais // 2)
    
    # Outliers / Falhas mecânicas
    vib_out = np.random.uniform(low=10, high=95, size=n_outliers)
    temp_out = np.random.uniform(low=40, high=120, size=n_outliers)
    ruido_out = np.random.uniform(low=20, high=110, size=n_outliers)
    
    # Concatenação dos vetores
    vibracao = np.concatenate([vib_1, vib_2, vib_out])
    temperatura = np.concatenate([temp_1, temp_2, temp_out])
    ruido = np.concatenate([ruido_1, ruido_2, ruido_out])
    
    ids_maquinas = [f"MAQ-{i+1:03d}" for i in range(len(vibracao))]
    
    df = pd.DataFrame({
        "ID_Maquina": ids_maquinas,
        "Frequencia_Vibracao": np.round(vibracao, 2),
        "Pico_Temperatura": np.round(temperatura, 2),
        "Nivel_Ruido_dB": np.round(ruido, 2)
    })
    
    return df

df_sensores = gerar_dados_sensores()

# ==========================================
# 3. Barra Lateral: Hiperparâmetros DBSCAN
# ==========================================
st.sidebar.header("🔧 Parâmetros do DBSCAN")

eps = st.sidebar.slider(
    label="Epsilon (eps)",
    min_value=0.1,
    max_value=2.5,
    value=0.55,
    step=0.05,
    help="Distância máxima euclidiana entre duas amostras no espaço normalizado."
)

min_samples = st.sidebar.slider(
    label="Min Samples",
    min_value=2,
    max_value=25,
    value=6,
    step=1,
    help="Número mínimo de vizinhos para compor uma região densa."
)

# ==========================================
# 4. Processamento & Algoritmo DBSCAN
# ==========================================
features = ["Frequencia_Vibracao", "Pico_Temperatura", "Nivel_Ruido_dB"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_sensores[features])

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
df_sensores["Cluster_Raw"] = dbscan.fit_predict(X_scaled)

# Classificação textual para exibição e filtro
df_sensores["Classificacao"] = df_sensores["Cluster_Raw"].apply(
    lambda x: "Anomalia" if x == -1 else f"Normal (Cluster {x})"
)

# Rótulo de texto exibido diretamente em cima de cada marcador no gráfico
df_sensores["Rotulo_Ponto"] = (
    df_sensores["ID_Maquina"] + "<br>" +
    df_sensores["Frequencia_Vibracao"].astype(str) + "Hz | " +
    df_sensores["Pico_Temperatura"].astype(str) + "°C"
)

# ==========================================
# 5. Métricas Operacionais
# ==========================================
col_m1, col_m2, col_m3 = st.columns(3)
total_anomalias = (df_sensores["Cluster_Raw"] == -1).sum()
total_clusters = len(set(df_sensores["Cluster_Raw"])) - (1 if -1 in df_sensores["Cluster_Raw"].values else 0)

col_m1.metric("Total de Máquinas", len(df_sensores))
col_m2.metric("Regimes de Operação", total_clusters)
col_m3.metric("Anomalias Detectadas", total_anomalias, delta_color="inverse")

st.divider()

# ==========================================
# 6. Gráfico de Dispersão 3D (Plotly)
# ==========================================
st.subheader("🌐 Espaço Tridimensional de Operação")

# Configuração de paleta: anomalias em preto fixo (#000000)
clusters_unicos = sorted(df_sensores["Cluster_Raw"].unique())
cores_disponiveis = px.colors.qualitative.Plotly

color_map = {}
idx_cor = 0
for cid in clusters_unicos:
    if cid == -1:
        color_map[cid] = "#000000"  # Preto para anomalias
    else:
        color_map[cid] = cores_disponiveis[idx_cor % len(cores_disponiveis)]
        idx_cor += 1

fig = px.scatter_3d(
    df_sensores,
    x="Frequencia_Vibracao",
    y="Pico_Temperatura",
    z="Nivel_Ruido_dB",
    color="Cluster_Raw",
    color_discrete_map=color_map,
    text="Rotulo_Ponto",  # Mostra valores e ID no ponto
    hover_name="ID_Maquina",
    hover_data={
        "Frequencia_Vibracao": ":.2f Hz",
        "Pico_Temperatura": ":.2f °C",
        "Nivel_Ruido_dB": ":.2f dB",
        "Classificacao": True,
        "Cluster_Raw": False,
        "Rotulo_Ponto": False
    }
)

# Ajuste visual: sem legenda, marcações nítidas e rótulos posicionados
fig.update_traces(
    mode="markers+text",
    textposition="top center",
    textfont=dict(size=9, color="#444444"),
    marker=dict(size=5, opacity=0.85, line=dict(width=0.5, color="#222222"))
)

fig.update_layout(
    showlegend=False,  # Oculta legenda conforme solicitado
    margin=dict(l=0, r=0, b=0, t=20),
    height=650,
    scene=dict(
        xaxis_title="Vibração (Hz)",
        yaxis_title="Temperatura (°C)",
        zaxis_title="Ruído (dB)",
        camera=dict(eye=dict(x=1.6, y=1.6, z=1.2))
    )
)

st.plotly_chart(fig, use_container_width=True)

st.divider()

# ==========================================
# 7. Tabela com Filtro Interativo
# ==========================================
st.subheader("📋 Tabela Operacional de Equipamentos")

col_filtro1, col_filtro2 = st.columns([2, 1])

with col_filtro1:
    filtro_status = st.radio(
        label="Filtrar equipamentos por condição:",
        options=["Todos", "Apenas Anomalias", "Apenas Normais"],
        horizontal=True
    )

with col_filtro2:
    busca_id = st.text_input("Filtrar por ID (ex: MAQ-015):", "")

# Aplicação dos filtros no dataframe
df_filtrado = df_sensores.copy()

if filtro_status == "Apenas Anomalias":
    df_filtrado = df_filtrado[df_filtrado["Cluster_Raw"] == -1]
elif filtro_status == "Apenas Normais":
    df_filtrado = df_filtrado[df_filtrado["Cluster_Raw"] != -1]

if busca_id:
    df_filtrado = df_filtrado[df_filtrado["ID_Maquina"].str.contains(busca_id.strip(), case=False)]

# Exibição dos dados
st.caption(f"Mostrando **{len(df_filtrado)}** registros.")

colunas_tabela = ["ID_Maquina", "Classificacao", "Frequencia_Vibracao", "Pico_Temperatura", "Nivel_Ruido_dB"]

st.dataframe(
    df_filtrado[colunas_tabela].reset_index(drop=True),
    use_container_width=True,
    column_config={
        "ID_Maquina": st.column_config.TextColumn("Máquina"),
        "Classificacao": st.column_config.TextColumn("Status Operacional"),
        "Frequencia_Vibracao": st.column_config.NumberColumn("Vibração", format="%.2f Hz"),
        "Pico_Temperatura": st.column_config.NumberColumn("Temperatura", format="%.2f °C"),
        "Nivel_Ruido_dB": st.column_config.NumberColumn("Ruído", format="%.2f dB")
    }
)

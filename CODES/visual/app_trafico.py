import streamlit as st
import json
import numpy as np
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import matplotlib.pyplot as plt 
import plotly.express as px 
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.sql.functions import sin, cos, col, dayofweek, month, avg, desc, when, lit

# Config basica de la pag
st.set_page_config(page_title="Traffic AI Master", layout="wide", initial_sidebar_state="expanded")

# Inicializacion de variables de estado para persistir los puntos seleccionados en el mapa tras recargar la pagina
if 'start_point' not in st.session_state: st.session_state.start_point = None
if 'end_point' not in st.session_state: st.session_state.end_point = None

# Funciones auxiliares
def encontrar_region_cercana(lat, lon, df_ref):
    if df_ref.empty: return "Desconocida"
    coords_ref = df_ref[['latitude', 'longitude']].values
    punto = np.array([lat, lon])
    
    # Vectorizacion para calcular distancias a todos los puntos de referencia
    deltas = coords_ref - punto
    
    # Usamos suma de Einstein  para calcular la distancia euclidiana al cuadrado de forma eficiente, solo es para la direccion
    dist_sq = np.einsum('ij,ij->i', deltas, deltas)
    
    # Obtenemos el indice de la distancia minima
    idx_min = np.argmin(dist_sq)
    return df_ref.iloc[idx_min]['region_name']

def calcular_direccion_cardinal(p1, p2):
    if not p1 or not p2: return "N"
    lat1, lon1 = p1
    lat2, lon2 = p2
    dLat = lat2 - lat1
    dLon = lon2 - lon1
    
    # Determina la direccion dominante comparando el desplazamiento vertical vs horizontal, ya que como tal no usamos la ubi final, sino solo la direccion, igual el mapa se saco de gugul
    if abs(dLat) >= abs(dLon):
        return "N" if dLat > 0 else "S"
    else:
        return "E" if dLon > 0 else "W"

# Cacheamos el recurso para no reiniciar Spark ni recargar el modelo en cada interaccion del usuario
@st.cache_resource
def iniciar_sistema():
    # Iniciamos Spark en modo local usando todos los nucleos disponibles
    spark = SparkSession.builder \
        .appName("Traffic_App_Red") \
        .master("local[*]") \
        .config("spark.driver.memory", "6g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    
    try:
        # -------------- ACA SE CAMBIA EL MODELO SI SE QUIERE OTROOOO
        with open("model_config_RF.json", "r") as f: config = json.load(f)
        model = PipelineModel.load("model_traffic_classifier_RF_Final") 
    except:
        st.error("Error: No se encuentra el modelo.")
        st.stop()

    df_spark = spark.read.csv("datasetLimpio", header=True, inferSchema=False)
    
    numeric_cols = [
        "all_motor_vehicles", "latitude", "longitude", "hour", "all_hgvs",
        "pedal_cycles", "two_wheeled_motor_vehicles", 
        "cars_and_taxis", "buses_and_coaches"
    ]
    
    # Limpieza: convierte strings nulos ('NULL', 'NaN') a nulos reales y luego castea a double, solo por si acaso, ya que el dataset ya viene limpio solo que me daba error pese a que todo estaba bien ya que igual hice el conteo de registros
    for c in numeric_cols:
        df_spark = df_spark.withColumn(c, when(col(c).isin("NULL", "null", "NaN"), lit(None)).otherwise(col(c)).cast("double"))
    
    # Filtramos filas invalidas para el calc
    df_clean = df_spark.filter(
        col("latitude").isNotNull() & 
        col("longitude").isNotNull() & 
        col("all_motor_vehicles").isNotNull()
    )

    # Tomamos una muestra pequena (1%) convertida a Pandas para operaciones rapidas en la UI (mapas)
    df_puntos = df_clean.select("latitude", "longitude", "region_name") \
                        .na.drop() \
                        .sample(False, 0.01, seed=42) \
                        .toPandas()
    
    return spark, model, config, df_puntos, df_clean

# Cacheamos los datos del dashboard para evitar re-calcular las agregaciones costosas
@st.cache_data
def obtener_estadisticas_dashboard(_df_spark):
    # Agregacion para grafico de dispersion (Bicis vs Motos)
    two_wheels = _df_spark.groupBy("region_name") \
                          .agg(avg("pedal_cycles").alias("avg_bicis"), 
                               avg("two_wheeled_motor_vehicles").alias("avg_motos")) \
                          .na.drop().toPandas()

    # Agregacion para estadisticas de vias, excluyendo nombres de calles desconocidos 
    roads_stats = _df_spark.filter(
            (col("road_name") != "U") & (col("road_name").isNotNull())
        ) \
        .groupBy("road_name") \
        .agg(
            avg("buses_and_coaches").alias("avg_bus"),
            avg("cars_and_taxis").alias("avg_car")
        ) \
        .orderBy(desc("avg_bus")) \
        .limit(20) \
        .toPandas()

    # Agregacion jerarquica para el TreeMap
    authority_stats = _df_spark.groupBy("local_authority_name", "region_name") \
                               .agg(avg("all_motor_vehicles").alias("total_trafico")) \
                               .orderBy(desc("total_trafico")).limit(30).toPandas()

    # Preparacion de datos para el Mapa de Calor (Heatmap)
    heatmap_raw = _df_spark.select("latitude", "longitude", "all_motor_vehicles") \
                           .filter("all_motor_vehicles > 0") \
                           .sample(False, 0.05, seed=42) \
                           .toPandas()
    
    heatmap_raw = heatmap_raw.dropna()
    
    if not heatmap_raw.empty:
        # Normalizacion Min-Max para que el peso del heatmap este entre 0 y 1
        max_val = heatmap_raw["all_motor_vehicles"].max()
        if max_val == 0: max_val = 1
        heatmap_raw["weight"] = heatmap_raw["all_motor_vehicles"] / max_val
        heatmap_list = heatmap_raw[['latitude', 'longitude', 'weight']].values.tolist()
    else:
        heatmap_list = []
    
    return two_wheels, roads_stats, authority_stats, heatmap_list

# Carga inicial del sistema
spark, model, config, df_puntos, df_spark = iniciar_sistema()

# Configuracion del menu lateral
st.sidebar.title("Traffic AI Master")
st.sidebar.markdown("### Sistema de Análisis de Tráfico")
opcion = st.sidebar.radio("", ["PLANIFICADOR DE RUTAS", "ANÁLISIS Y ESTADÍSTICAS"])
st.sidebar.markdown("---")
st.sidebar.caption("Machine Learning Traffic Predictor")

# Logica de la Vista 1: Planificador
if opcion == "PLANIFICADOR DE RUTAS":
    st.title("PLANIFICADOR DE RUTAS")
    st.markdown("### Seleccione origen y destino en el mapa")
    
    col_map, col_ctrl = st.columns([2,1])
    
    with col_map:
        m = folium.Map(location=[54.5, -3.0], zoom_start=6)
        
        # Dibuja los marcadores y la linea si existen puntos en el estado
        if st.session_state.start_point:
            folium.Marker(st.session_state.start_point, icon=folium.Icon(color="green", icon="play")).add_to(m)
        if st.session_state.end_point:
            folium.Marker(st.session_state.end_point, icon=folium.Icon(color="red", icon="stop")).add_to(m)
            if st.session_state.start_point:
                folium.PolyLine([st.session_state.start_point, st.session_state.end_point], color="#FF6B35", weight=3).add_to(m)
        
        # Renderiza el mapa y captura los clicks del usuario
        map_data = st_folium(m, height=600, width=None)
        
        # Logica para alternar entre establecer punto de inicio y punto final al hacer clic
        if map_data.get("last_clicked"):
            clicked = (map_data["last_clicked"]["lat"], map_data["last_clicked"]["lng"])
            if not st.session_state.start_point: st.session_state.start_point = clicked; st.rerun()
            elif not st.session_state.end_point: st.session_state.end_point = clicked; st.rerun()
            else: st.session_state.start_point = clicked; st.session_state.end_point = None; st.rerun()

    with col_ctrl:
        if st.button("↻ Reiniciar Puntos"): 
            st.session_state.start_point = None
            st.session_state.end_point = None
            st.rerun()
        
        if st.session_state.start_point and st.session_state.end_point:
            start = st.session_state.start_point
            
            # Calculos automaticos basados en coordenadas
            direccion_calc = calcular_direccion_cardinal(st.session_state.start_point, st.session_state.end_point)
            region_detectada = encontrar_region_cercana(start[0], start[1], df_puntos)
            
            st.info(f"▸ Región: {region_detectada} | Dirección: {direccion_calc}")
            
            with st.form("pred"):
                st.markdown("### Parámetros del Viaje")
                fecha = st.date_input("Fecha")
                hora = st.slider("Hora del día", 0, 23, 8)
                hgvs = st.slider("Cantidad de camiones", 0, 300, 50)
                submit = st.form_submit_button("▶ Predecir Tráfico", type="primary")
            
            if submit:
                try:
                    # Crear DataFrame de una sola fila para PySpark
                    data = [{
                        "latitude": float(start[0]), "longitude": float(start[1]), 
                        "hour": float(hora), "direction_of_travel": direccion_calc,
                        "region_name": region_detectada, "all_hgvs": float(hgvs),
                        "count_date": str(fecha)
                    }]
                    df_in = spark.createDataFrame(data)
                    
                    # Ingenieria de caracteristicas: Transformacion ciclica de la hora (Seno/Coseno)
                    # Esto ayuda al modelo entender que las 23:00 y las 00:00 estan cerca entre si ya que es la representacion ciclica de las horas, me eche un video de eso luego te lo mando
                    df_in = df_in.withColumn("day_of_week", dayofweek(col("count_date"))) \
                                 .withColumn("month", month(col("count_date"))) \
                                 .withColumn("hour_sin", sin(2 * np.pi * col("hour") / 24)) \
                                 .withColumn("hour_cos", cos(2 * np.pi * col("hour") / 24))
                    
                    # Inferencia con el modelo cargado
                    pred = model.transform(df_in)
                    row = pred.select("probability", "prediction").collect()[0]
                    probs = row["probability"].toArray()
                    clase = int(row["prediction"])
                    
                    niveles = ["Muy Bajo", "Bajo", "Medio", "Alto", "Muy Alto"]
                    colores = ["#FFD700", "#FFA500", "#FF8C00", "#FF6347", "#DC143C"]
                    
                    st.markdown(f"<h2 style='text-align: center; color:{colores[clase]}'>{niveles[clase]}</h2>", unsafe_allow_html=True)
                    
                    # Grafico de Matplotlib para mostrar probabilidades por clase
                    fig, ax = plt.subplots(figsize=(6, 3))
                    bars = ax.bar(niveles, probs, color=colores)
                    
                    # Limpieza visual del grafico
                    ax.set_ylim(0, 1.15) 
                    ax.spines['top'].set_visible(False)
                    ax.spines['right'].set_visible(False)
                    ax.spines['left'].set_visible(False)
                    ax.set_yticks([])
                    
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01, 
                                f'{height:.1%}', ha='center', va='bottom', fontsize=11, fontweight='bold')
                    
                    st.pyplot(fig)
                    
                    df_probs = pd.DataFrame([probs], columns=niveles)
                    st.caption("Detalle de probabilidades:")
                    st.dataframe(df_probs.style.format("{:.1%}"))
                    
                except Exception as e:
                    st.error(f"Error: {e}")

# Logica de la Vista 2: Analytics
elif opcion == "ANÁLISIS Y ESTADÍSTICAS":
    st.title("ANÁLISIS Y ESTADÍSTICAS")
    st.markdown("### Panel de control de datos históricos")
    
    with st.spinner("Cargando datos de tráfico..."):
        two_wheels, roads_stats, authority_stats, heatmap_list = obtener_estadisticas_dashboard(df_spark)
        
        tab1, tab2, tab3, tab4 = st.tabs(["▣ MAPA DE CALOR", "▣ MOVILIDAD LIGERA", "▣ TRANSPORTE", "▣ AUTORIDADES"])
        
        with tab1:
            st.subheader("Densidad de Tráfico por Zona")
            m_heat = folium.Map(location=[54.5, -2.5], zoom_start=6) 
            
            if len(heatmap_list) > 0:
                gradient_warm = {0.3: '#FFD700', 0.5: '#FF8C00', 0.7: '#FF4500', 1.0: '#8B0000'}
                
                # Renderizado del mapa de calor usando los datos normalizados
                HeatMap(
                    heatmap_list, 
                    radius=12, 
                    blur=15, 
                    min_opacity=0.4,
                    gradient=gradient_warm
                ).add_to(m_heat)
                st.success(f"Visualizando {len(heatmap_list)} zonas de alta densidad")
            else:
                st.warning("Sin datos suficientes para el mapa de calor")
            
            st_folium(m_heat, height=600, width=None)

        with tab2:
            st.subheader("Bicicletas vs Motocicletas por Región")
            # Grafico de dispersion interactivo con Plotly
            fig = px.scatter(
                two_wheels, 
                x="avg_bicis", 
                y="avg_motos", 
                color="region_name",
                size="avg_bicis", 
                title="Análisis de Movilidad Ligera"
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with tab3:
            st.subheader("Comparativa de Transporte Público y Privado")
            c1, c2 = st.columns(2)
            
            with c1:
                st.caption("► Top Rutas - Autobuses")
                fig_bus = px.bar(
                    roads_stats.head(10), 
                    x="avg_bus", 
                    y="road_name", 
                    orientation='h',
                    color="avg_bus",
                    color_continuous_scale=["#FFE135", "#FF8C00", "#FF6347"]
                )
                fig_bus.update_layout(showlegend=False)
                st.plotly_chart(fig_bus, use_container_width=True)
                
            with c2:
                st.caption("► Top Rutas - Automóviles")
                fig_car = px.bar(
                    roads_stats.sort_values("avg_car", ascending=False).head(10), 
                    x="avg_car", 
                    y="road_name", 
                    orientation='h',
                    color="avg_car",
                    color_continuous_scale=["#FFA500", "#FF6347", "#DC143C"]
                )
                fig_car.update_layout(showlegend=False)
                st.plotly_chart(fig_car, use_container_width=True)

        with tab4:
            st.subheader("Jerarquía de Tráfico por Autoridad Local")
            # TreeMap para visualizar la proporcion de trafico por region y autoridad
            fig_tree = px.treemap(
                authority_stats, 
                path=['region_name', 'local_authority_name'], 
                values='total_trafico',
                color='total_trafico',
                color_continuous_scale=["#FFD700", "#FFA500", "#FF6347", "#DC143C", "#8B0000"]
            )
            fig_tree.update_layout(margin=dict(l=0, r=0, t=30, b=0))
            st.plotly_chart(fig_tree, use_container_width=True)
import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
from openai import OpenAI
import numpy as np
import re
import asyncio
from datetime import timedelta
import matplotlib.pyplot as plt
from io import BytesIO
import os

# Configuración de la página
st.set_page_config(
    page_title="Katalis Ads Pro",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# --- Configuración DeepSeek Segura ---
def get_deepseek_client():
    """Obtiene el cliente de DeepSeek de forma segura"""
    api_key = os.getenv("DEEPSEEK_API_KEY") or st.secrets.get("DEEPSEEK_API_KEY")
    
    if not api_key:
        st.error("""
        🔒 API Key no configurada. Por favor:
        1. Para desarrollo local: crea `.streamlit/secrets.toml` con `DEEPSEEK_API_KEY="tu_key"`
        2. Para producción: configura los secrets en GitHub/Streamlit
        """)
        return None
    
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1"
    )

# --- Funciones principales ---
def validar_telefono(tel):
    if pd.isna(tel): return False
    tel = re.sub(r'[^0-9]', '', str(tel))
    return len(tel) in (10, 12)

def validar_email(email):
    if pd.isna(email): return False
    return bool(re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', str(email).strip()))

def estimar_empleados(valor):
    rangos_extendidos = {
        '0 a 5': 3, '1 a 5': 3, '6 a 10': 8,
        '11 a 30': 20, '31 a 50': 40, '51 a 100': 75,
        '101 a 250': 175, '251 a 500': 375, '501 a 1000': 750,
        '1001 a 2000': 1500, '2001+': 2500
    }
    try:
        if pd.isna(valor): return 1
        valor = str(valor).lower()
        for k, v in rangos_extendidos.items():
            if k in valor: return v
        return max(1, int(float(valor)))
    except:
        return 1

@st.cache_data(ttl=timedelta(hours=6))
def cargar_datos(archivo):
    if archivo.name.endswith('.csv'):
        df = pd.read_csv(archivo, encoding='latin1', low_memory=False)
    else:
        df = pd.read_excel(archivo)
    
    cols_requeridas = ['nom_estab', 'nombre_act', 'per_ocu', 'telefono', 
                      'correoelec', 'www', 'municipio', 'localidad',
                      'entidad', 'latitud', 'longitud']
    
    df = df[cols_requeridas].dropna(subset=['entidad', 'municipio', 'nombre_act'])
    df['per_ocu_estimado'] = df['per_ocu'].apply(estimar_empleados)
    
    # Optimización de memoria
    for col in ['nombre_act', 'municipio', 'localidad', 'entidad']:
        df[col] = df[col].astype('category')
    
    return df

async def procesar_lote_async(df, lote_size=1000):
    resultados = []
    for i in range(0, len(df), lote_size):
        lote = df.iloc[i:i + lote_size].copy()
        await asyncio.sleep(0.01)
        resultados.append(lote)
    return pd.concat(resultados)

def generar_recomendaciones_ia(df_final):
    client = get_deepseek_client()
    if not client:
        return None

    contexto = {
        "total_empresas": len(df_final),
        "giros_unicos": df_final['Giro'].nunique(),
        "empleo_promedio": df_final['Personal Estimado'].mean(),
        "ubicacion_principal": f"{df_final['Estado'].mode()[0]}, {df_final['Municipio'].mode()[0]}",
        "top_giros": df_final['Giro'].value_counts().nlargest(3).index.tolist()
    }

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "system",
                "content": "Eres un experto en Google Ads para empresas B2B. Genera recomendaciones concretas basadas en datos."
            }, {
                "role": "user",
                "content": f"""
                Datos analizados:
                - {contexto['total_empresas']} empresas
                - Giros principales: {', '.join(contexto['top_giros'])}
                - Tamaño promedio: {contexto['empleo_promedio']:.1f} empleados
                - Ubicación: {contexto['ubicacion_principal']}

                Genera:
                1. 3 estrategias de segmentación
                2. Presupuesto mensual estimado (MXN)
                3. 5 palabras clave con mayor potencial
                4. 2 ejemplos de creatividades
                """
            }],
            temperature=0.7,
            max_tokens=1000
        )
        
        return {
            "analisis": response.choices[0].message.content,
            "contexto": contexto
        }
    except Exception as e:
        st.error(f"Error en DeepSeek API: {str(e)}")
        return None

# --- Interfaz ---
def main():
    st.title("🚀 Katalis Ads Pro - Integración DeepSeek")
    
    archivo = st.file_uploader("📂 Sube tu base DENUE", type=['csv','xlsx'])
    
    if not archivo:
        st.info("Por favor sube un archivo para comenzar")
        return
    
    df = cargar_datos(archivo)
    
    # Sidebar con filtros
    with st.sidebar:
        st.header("🔍 Filtros")
        estado = st.selectbox("Estado", sorted(df['entidad'].unique()))
        df_filtrado = df[df['entidad'] == estado]
        
        municipios = st.multiselect(
            "Municipios",
            options=sorted(df_filtrado['municipio'].unique()),
            default=sorted(df_filtrado['municipio'].unique())[:3]
        )
        
        rango_emp = st.slider(
            "👥 Rango empleados",
            1, 2500, (1, 2000)
        )
        
        if st.button("🔄 Reiniciar filtros"):
            st.rerun()
    
    # Aplicar filtros
    if municipios:
        df_filtrado = df_filtrado[df_filtrado['municipio'].isin(municipios)]
    
    df_filtrado = df_filtrado[
        df_filtrado['per_ocu_estimado'].between(*rango_emp)
    ]
    
    # Mostrar resultados
    st.success(f"✅ {len(df_filtrado)} empresas encontradas")
    
    # Pestañas principales
    tab1, tab2, tab3 = st.tabs(["📊 Datos", "🗺️ Mapa", "🤖 IA Recomendaciones"])
    
    with tab1:
        st.dataframe(df_filtrado.head(300), height=600)
        
        if st.button("Procesar en lote (Async)"):
            with st.spinner("Procesando..."):
                df_procesado = asyncio.run(procesar_lote_async(df_filtrado))
                st.session_state.df_procesado = df_procesado
                st.success("¡Procesamiento completo!")
    
    with tab2:
        if not df_filtrado[['latitud', 'longitud']].dropna().empty:
            mapa = folium.Map(
                location=[
                    df_filtrado['latitud'].mean(),
                    df_filtrado['longitud'].mean()
                ],
                zoom_start=10
            )
            MarkerCluster().add_to(mapa)
            st_folium(mapa, width=1200)
    
    with tab3:
        st.subheader("Recomendaciones con DeepSeek")
        
        if 'df_procesado' in st.session_state:
            if st.button("Generar recomendaciones"):
                with st.spinner("Consultando a DeepSeek..."):
                    resultado = generar_recomendaciones_ia(st.session_state.df_procesado)
                    
                    if resultado:
                        st.markdown("## 📈 Análisis Completo")
                        st.write(resultado['analisis'])
                        
                        st.download_button(
                            "📄 Descargar Reporte",
                            data=resultado['analisis'],
                            file_name="recomendaciones_katalis.txt"
                        )
        else:
            st.warning("Procesa los datos primero en la pestaña 📊 Datos")

if __name__ == "__main__":
    main()

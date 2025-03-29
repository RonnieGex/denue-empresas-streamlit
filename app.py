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
import os

# Configuración de la página
st.set_page_config(
    page_title="Katalis Ads Pro",
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# --- Funciones de apoyo ---
def validar_telefono(tel):
    """Valida formato de teléfono mexicano/internacional"""
    if pd.isna(tel): return False
    tel = re.sub(r'[^0-9]', '', str(tel))
    return len(tel) in (10, 12)  # México: 10 dígitos, internacional: +52

def validar_email(email):
    """Valida formato de email con regex robusto"""
    if pd.isna(email): return False
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, str(email).strip()))

def estimar_empleados(valor):
    """Estimación inteligente con rangos extendidos hasta 2000+"""
    rangos = {
        '0 a 5': 3, '1 a 5': 3, '6 a 10': 8,
        '11 a 30': 20, '31 a 50': 40, '51 a 100': 75,
        '101 a 250': 175, '251 a 500': 375, '501 a 1000': 750,
        '1001 a 2000': 1500, '2001+': 2500
    }
    try:
        if pd.isna(valor): return 1
        valor = str(valor).lower()
        for k, v in rangos.items():
            if k in valor: return v
        return max(1, int(float(valor)))
    except:
        return 1

# --- Configuración DeepSeek Segura ---
@st.cache_resource
def get_deepseek_client():
    """Obtiene el cliente configurado para DeepSeek"""
    api_key = os.getenv("DEEPSEEK_API_KEY") or st.secrets.get("DEEPSEEK_API_KEY")
    
    if not api_key:
        st.error("""
        🔒 API Key no configurada. Por favor:
        1. Para desarrollo: crea `.streamlit/secrets.toml` con `DEEPSEEK_API_KEY="tu_key"`
        2. Para producción: configura los secrets en GitHub/Streamlit
        """)
        return None
    
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1"
    )

# --- Procesamiento de datos ---
@st.cache_data(ttl=timedelta(hours=6))
def cargar_datos(archivo):
    """Carga y optimiza el dataframe"""
    if archivo.name.endswith('.csv'):
        df = pd.read_csv(archivo, encoding='latin1', low_memory=False)
    else:
        df = pd.read_excel(archivo)
    
    cols_requeridas = [
        'nom_estab', 'nombre_act', 'per_ocu', 'telefono', 
        'correoelec', 'www', 'municipio', 'localidad',
        'entidad', 'latitud', 'longitud'
    ]
    
    df = df[cols_requeridas].dropna(subset=['entidad', 'municipio', 'nombre_act'])
    df['per_ocu_estimado'] = df['per_ocu'].apply(estimar_empleados)
    
    # Optimización de memoria
    for col in ['nombre_act', 'municipio', 'localidad', 'entidad']:
        df[col] = df[col].astype('category')
    
    return df

async def procesar_lote_async(df, lote_size=1000):
    """Procesamiento asíncrono para grandes volúmenes"""
    resultados = []
    for i in range(0, len(df), lote_size):
        lote = df.iloc[i:i + lote_size].copy()
        await asyncio.sleep(0.01)  # Simula procesamiento
        resultados.append(lote)
    return pd.concat(resultados)

# --- Generación de recomendaciones ---
def generar_recomendaciones_ia(df):
    """Genera recomendaciones personalizadas con DeepSeek"""
    client = get_deepseek_client()
    if not client:
        return None

    # Análisis estadístico
    stats = {
        "total": len(df),
        "giros_unicos": df['Giro'].nunique(),
        "empleo_promedio": df['Personal Estimado'].mean(),
        "ubicacion": f"{df['Estado'].mode()[0]}, {df['Municipio'].mode()[0]}",
        "top_giros": df['Giro'].value_counts().nlargest(3).index.tolist(),
        "contactabilidad": {
            "telefono": df['Teléfono'].apply(validar_telefono).mean(),
            "email": df['Correo'].apply(validar_email).mean(),
            "web": df['Web'].notna().mean()
        }
    }

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{
                "role": "system",
                "content": "Eres un consultor senior de Google Ads especializado en campañas B2B para PYMES."
            }, {
                "role": "user",
                "content": f"""
                Datos analizados:
                - {stats['total']} empresas objetivo
                - Giros principales: {', '.join(stats['top_giros'])}
                - Tamaño promedio: {stats['empleo_promedio']:.1f} empleados
                - Ubicación: {stats['ubicacion']}
                - Contactabilidad: 
                  * Teléfonos válidos: {stats['contactabilidad']['telefono']*100:.1f}%
                  * Emails válidos: {stats['contactabilidad']['email']*100:.1f}%
                  * Sitios web: {stats['contactabilidad']['web']*100:.1f}%

                Genera recomendaciones para:
                1. Segmentación de campañas (máx 3)
                2. Presupuesto mensual estimado (MXN)
                3. 5-10 palabras clave estratégicas
                4. 2 ejemplos de creatividades
                5. Sugerencias de extensiones de anuncio

                Formato: Markdown con secciones claras y datos específicos.
                """
            }],
            temperature=0.7,
            max_tokens=1500
        )
        
        return {
            "analisis": response.choices[0].message.content,
            "stats": stats
        }
    except Exception as e:
        st.error(f"Error en DeepSeek API: {str(e)}")
        return None

# --- Interfaz de usuario ---
def main():
    st.title("🚀 Katalis Ads Pro - Analizador DENUE")
    
    with st.expander("ℹ️ Instrucciones rápidas", expanded=False):
        st.write("""
        1. Sube tu archivo DENUE (CSV/Excel)
        2. Aplica filtros en la barra lateral
        3. Procesa los datos (opcional)
        4. Genera recomendaciones con IA
        """)
    
    # Carga de archivo
    archivo = st.file_uploader("📂 Sube tu base DENUE", type=['csv','xlsx'])
    
    if not archivo:
        st.info("Por favor sube un archivo para comenzar")
        return
    
    try:
        df = cargar_datos(archivo)
        
        # Sidebar con filtros
        with st.sidebar:
            st.header("🔍 Filtros Avanzados")
            
            estado = st.selectbox(
                "📍 Estado",
                options=sorted(df['entidad'].cat.categories),
                key="estado_select"
            )
            
            df_estado = df[df['entidad'] == estado]
            
            municipios_opciones = sorted(df_estado['municipio'].cat.categories)
            municipios_sel = st.multiselect(
                "🏘️ Municipios",
                options=municipios_opciones,
                default=municipios_opciones[:3] if len(municipios_opciones) > 3 else municipios_opciones,
                key="municipios_multiselect"
            )
            
            rango_emp = st.slider(
                "👥 Rango de empleados estimado",
                1, 6000, (1, 2000),
                key="rango_emp_slider"
            )
            
            giros_opciones = sorted(df_estado['nombre_act'].cat.categories)
            giros_sel = st.multiselect(
                "🏢 Giros económicos",
                options=giros_opciones,
                key="giros_multiselect"
            )
            
            st.markdown("---")
            contacto_cols = st.multiselect(
                "📞 Contacto requerido",
                options=['Teléfono', 'Email', 'Sitio Web'],
                default=['Teléfono'],
                key="contacto_multiselect"
            )
            
            if st.button("🔄 Reiniciar Filtros", use_container_width=True):
                st.rerun()
        
        # Aplicar filtros
        df_filtrado = df_estado.copy()
        
        if municipios_sel:
            df_filtrado = df_filtrado[df_filtrado['municipio'].isin(municipios_sel)]
        
        df_filtrado = df_filtrado[
            df_filtrado['per_ocu_estimado'].between(*rango_emp)
        ]
        
        if giros_sel:
            df_filtrado = df_filtrado[df_filtrado['nombre_act'].isin(giros_sel)]
        
        # Validación de contactos
        if 'Teléfono' in contacto_cols:
            df_filtrado = df_filtrado[df_filtrado['telefono'].apply(validar_telefono)]
        if 'Email' in contacto_cols:
            df_filtrado = df_filtrado[df_filtrado['correoelec'].apply(validar_email)]
        if 'Sitio Web' in contacto_cols:
            df_filtrado = df_filtrado[df_filtrado['www'].notna()]
        
        # Preparar datos finales
        columnas_exportar = [
            'nom_estab', 'nombre_act', 'per_ocu', 'per_ocu_estimado',
            'telefono', 'correoelec', 'www', 'municipio', 'localidad',
            'entidad', 'latitud', 'longitud'
        ]
        
        df_final = df_filtrado[columnas_exportar].copy()
        df_final.columns = [
            'Nombre', 'Giro', 'Personal (texto)', 'Personal Estimado',
            'Teléfono', 'Correo', 'Web', 'Municipio', 'Localidad',
            'Estado', 'Latitud', 'Longitud'
        ]
        
        # Mostrar resultados
        st.success(f"✅ {len(df_final)} empresas encontradas")
        
        # Pestañas principales
        tab1, tab2, tab3 = st.tabs(["📊 Datos", "🗺️ Mapa", "🧠 IA Recomendaciones"])
        
        with tab1:
            st.dataframe(
                df_final.head(300),
                height=600,
                use_container_width=True,
                column_config={
                    "Web": st.column_config.LinkColumn(),
                    "Teléfono": st.column_config.TextColumn(width="medium"),
                    "Personal Estimado": st.column_config.ProgressColumn(
                        format="%d",
                        min_value=0,
                        max_value=df_final['Personal Estimado'].max()
                    )
                }
            )
            
            csv = df_final.to_csv(index=False).encode('utf-8')
            st.download_button(
                "💾 Descargar CSV optimizado",
                data=csv,
                file_name="katalis_ads_export.csv",
                mime="text/csv",
                use_container_width=True
            )
            
            if st.button("⚡ Procesar en lote (Async)", use_container_width=True):
                with st.spinner("Procesando datos..."):
                    df_procesado = asyncio.run(procesar_lote_async(df_final))
                    st.session_state.df_procesado = df_procesado
                    st.success("¡Datos listos para análisis!")
        
        with tab2:
            if not df_final[['Latitud', 'Longitud']].dropna().empty:
                max_puntos = st.slider(
                    "🔘 Máximo de puntos en mapa",
                    10, 1000, 300,
                    key="max_puntos_mapa"
                )
                
                mapa = folium.Map(
                    location=[
                        df_final['Latitud'].astype(float).mean(),
                        df_final['Longitud'].astype(float).mean()
                    ],
                    zoom_start=10,
                    tiles='CartoDB positron'
                )
                
                cluster = MarkerCluster().add_to(mapa)
                
                for _, row in df_final.dropna(subset=['Latitud', 'Longitud']).head(max_puntos).iterrows():
                    folium.Marker(
                        location=[float(row['Latitud']), float(row['Longitud'])],
                        popup=f"<b>{row['Nombre']}</b><br>{row['Giro']}",
                        icon=folium.Icon(color='blue', icon='info-sign')
                    ).add_to(cluster)
                
                st_folium(mapa, width=1200)
            else:
                st.warning("No hay suficientes datos geográficos para mostrar el mapa")
        
        with tab3:
            st.subheader("🔑 Configuración de IA")
            
            if 'df_procesado' not in st.session_state:
                st.warning("Primero procesa los datos en la pestaña 📊 Datos")
            else:
                if st.button("✨ Generar Recomendaciones Avanzadas", 
                           type="primary",
                           use_container_width=True):
                    with st.spinner("Analizando datos con IA..."):
                        resultado = generar_recomendaciones_ia(st.session_state.df_procesado)
                        
                        if resultado:
                            st.markdown("## 📈 Análisis Personalizado")
                            st.markdown(resultado['analisis'])
                            
                            # Gráficos de apoyo
                            st.plotly_chart(
                                px.bar(
                                    st.session_state.df_procesado['Giro'].value_counts().head(5),
                                    title="Top 5 Giros para Segmentación",
                                    labels={'value': 'N° Empresas', 'index': 'Giro'}
                                ),
                                use_container_width=True
                            )
                            
                            # Exportar reporte
                            st.download_button(
                                "📄 Descargar Reporte Completo",
                                data=resultado['analisis'].encode('utf-8'),
                                file_name="recomendaciones_katalis.txt",
                                mime="text/plain",
                                use_container_width=True
                            )
    
    except Exception as e:
        st.error(f"❌ Error crítico: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()

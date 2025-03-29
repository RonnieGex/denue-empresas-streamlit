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

# Configuración optimizada para caché en memoria
@st.cache_resource(show_spinner=False)
def get_deepseek_client(api_key):
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1"
    )

# Validación mejorada de contactos
def validar_telefono(tel):
    if pd.isna(tel): return False
    tel = re.sub(r'[^0-9]', '', str(tel))
    return len(tel) in (10, 12)  # México: 10 dígitos, internacional: +52

def validar_email(email):
    if pd.isna(email): return False
    return bool(re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', str(email).strip()))

# Procesamiento asíncrono optimizado
async def procesar_lote_async(df, lote_size=1000):
    resultados = []
    for i in range(0, len(df), lote_size):
        lote = df.iloc[i:i + lote_size].copy()
        await asyncio.sleep(0.01)  # Simula procesamiento
        resultados.append(lote)
    return pd.concat(resultados)

# Generación de recomendaciones mejorada
def generar_recomendaciones_avanzadas(df):
    # Análisis estadístico
    stats = {
        "total_empresas": len(df),
        "giros_unicos": df['Giro'].nunique(),
        "empleo_promedio": df['Personal Estimado'].mean(),
        "contactabilidad": {
            "telefono": df['Teléfono'].apply(validar_telefono).mean(),
            "email": df['Correo'].apply(validar_email).mean(),
            "web": df['Web'].notna().mean()
        }
    }
    
    # Gráficos interactivos
    fig1 = px.histogram(df, x='Personal Estimado', 
                       title='Distribución de Tamaños de Empresa',
                       nbins=20,
                       labels={'Personal Estimado': 'Núm. Empleados'})
    
    top_giros = df['Giro'].value_counts().nlargest(5)
    fig2 = px.pie(top_giros, 
                 names=top_giros.index,
                 title='Top 5 Giros Económicos',
                 hole=0.3)
    
    # Recomendación estratégica mejorada
    recomendacion = f"""
    ## 🚀 Recomendaciones Avanzadas para Google Ads
    
    **Análisis de Datos:**
    - 📊 **Empresas objetivo:** {stats['total_empresas']}
    - 🏭 **Giros económicos únicos:** {stats['giros_unicos']}
    - 👥 **Tamaño promedio:** {stats['empleo_promedio']:.1f} empleados
    
    **Contactabilidad:**
    - 📞 Con teléfono válido: {stats['contactabilidad']['telefono']*100:.1f}%
    - ✉️ Con email válido: {stats['contactabilidad']['email']*100:.1f}%
    - 🌐 Con sitio web: {stats['contactabilidad']['web']*100:.1f}%
    
    **Estrategia Recomendada:**
    1. **Segmentación:**
       - Crear {min(5, stats['giros_unicos'])} campañas por giro principal
       - 3 grupos de tamaño: Pequeñas (1-50), Medianas (51-200), Grandes (201+)
    
    2. **Inversión:**
       - Presupuesto mensual estimado: ${stats['total_empresas']*0.75:.0f}-${stats['total_empresas']*2:.0f} MXN
       - CPC objetivo: $10-15 MXN
    
    3. **Palabras clave:**
       - "{[giro.replace(' ', ' +') for giro in top_giros.index[:3]]}"
       - "proveedores + {df['Estado'].mode()[0]}"
       - "empresas + {df['Municipio'].mode()[0]}"
    """
    
    return {
        "stats": stats,
        "figures": [fig1, fig2],
        "recommendation": recomendacion
    }

# Configuración principal optimizada
st.set_page_config(
    page_title="Katalis Ads Pro", 
    layout="wide",
    page_icon="🚀"
)

# Carga de datos con caché mejorado
@st.cache_data(ttl=timedelta(hours=6), show_spinner="Optimizando base de datos...")
def cargar_datos(archivo):
    if archivo.name.endswith('.csv'):
        df = pd.read_csv(archivo, encoding='latin1', low_memory=False)
    else:
        df = pd.read_excel(archivo)
    
    # Limpieza y optimización de memoria
    cols = ['nom_estab', 'nombre_act', 'per_ocu', 'telefono', 
            'correoelec', 'www', 'municipio', 'localidad', 
            'entidad', 'latitud', 'longitud']
    df = df[cols].dropna(subset=['entidad', 'municipio', 'nombre_act'])
    
    # Convertir a categorías para ahorrar memoria
    for col in ['nombre_act', 'municipio', 'localidad', 'entidad']:
        df[col] = df[col].astype('category')
    
    return df

# Estimación de empleados extendida (hasta 2000+)
def estimar_empleados(valor):
    try:
        if pd.isna(valor): return 1
        
        valor = str(valor).lower().strip()
        rangos_extendidos = {
            '0 a 5': 3, '1 a 5': 3, '6 a 10': 8,
            '11 a 30': 20, '31 a 50': 40, '51 a 100': 75,
            '101 a 250': 175, '251 a 500': 375, '501 a 1000': 750,
            '1001 a 2000': 1500, '2001 a 5000': 3500, '5000+': 6000
        }
        
        for k, v in rangos_extendidos.items():
            if k in valor: return v
        
        return max(1, int(float(valor)))
    except:
        return 1

# Interfaz de usuario mejorada
def main():
    st.title("🚀 Katalis Ads DB Optimizer Pro")
    
    with st.expander("ℹ️ Instrucciones rápidas", expanded=False):
        st.write("""
        1. Sube tu archivo DENUE (CSV o Excel)
        2. Aplica los filtros necesarios
        3. Procesa los datos (opcional en lote)
        4. Genera recomendaciones para Google Ads
        """)
    
    archivo = st.file_uploader("📂 Sube tu base DENUE", type=['csv','xlsx'])
    
    if not archivo:
        st.info("Por favor sube un archivo para comenzar")
        return
    
    try:
        df = cargar_datos(archivo)
        df['per_ocu_estimado'] = df['per_ocu'].apply(estimar_empleados)
        
        # Filtros avanzados
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
                default=municipios_opciones[:3],
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
            
            if st.button("🔄 Reiniciar Filtros"):
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
        
        # Mostrar resultados
        st.success(f"✅ {len(df_filtrado)} empresas encontradas")
        
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
        
        # Pestañas para organización
        tab1, tab2, tab3 = st.tabs(["📊 Resultados", "🗺️ Mapa", "🧠 Recomendaciones IA"])
        
        with tab1:
            st.dataframe(
                df_final.head(300),
                height=600,
                use_container_width=True,
                hide_index=True
            )
            
            csv = df_final.to_csv(index=False).encode('utf-8')
            st.download_button(
                "💾 Descargar CSV optimizado",
                data=csv,
                file_name="katalis_ads_export.csv",
                mime="text/csv",
                key="download_csv"
            )
        
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
                
                st_folium(mapa, height=500, width=900)
            else:
                st.warning("No hay suficientes datos geográficos para mostrar el mapa")
        
        with tab3:
            if st.button("✨ Generar Recomendaciones Avanzadas", type="primary"):
                with st.spinner("Analizando datos y generando estrategia..."):
                    if len(df_final) > 10:  # Mínimo para análisis
                        resultados = generar_recomendaciones_avanzadas(df_final)
                        
                        st.markdown(resultados['recommendation'])
                        
                        for fig in resultados['figures']:
                            st.plotly_chart(fig, use_container_width=True)
                        
                        # Exportar reporte
                        with BytesIO() as buffer:
                            plt.savefig(buffer, format='pdf')
                            st.download_button(
                                "📄 Descargar Reporte Completo (PDF)",
                                data=buffer.getvalue(),
                                file_name="katalis_recomendaciones.pdf",
                                mime="application/pdf"
                            )
                    else:
                        st.warning("Se necesitan más de 10 registros para generar recomendaciones")
    
    except Exception as e:
        st.error(f"❌ Error crítico: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()

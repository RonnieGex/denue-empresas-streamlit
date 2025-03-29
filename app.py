import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px
from openai import OpenAI
import numpy as np

# Configuración optimizada para DeepSeek
@st.cache_resource(show_spinner=False)
def get_deepseek_client(api_key):
    return OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com/v1"
    )

def get_recommendations_deepseek(api_key, prompt):
    try:
        client = get_deepseek_client(api_key)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Eres un experto en giros de negocio y optimización B2B para campañas de Google Ads."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=500
        )
        return response.choices[0].message.content if response.choices else "❌ No se recibió respuesta válida."
    except Exception as e:
        return f"❌ Error al conectar con DeepSeek: {str(e)}"

# Optimización: Uso de tipos categóricos para reducir memoria
@st.cache_data(show_spinner="Cargando y optimizando datos...", ttl=3600)
def cargar_datos(archivo):
    if archivo.name.endswith(".csv"):
        df = pd.read_csv(archivo, encoding="latin1", 
                        usecols=lambda col: col.strip() in [
                            "nom_estab", "nombre_act", "per_ocu", 
                            "telefono", "correoelec", "www", 
                            "municipio", "localidad", "entidad", 
                            "latitud", "longitud"
                        ],
                        dtype={
                            'nom_estab': 'category',
                            'nombre_act': 'category',
                            'municipio': 'category',
                            'localidad': 'category',
                            'entidad': 'category'
                        })
    else:
        df = pd.read_excel(archivo, 
                          usecols=[
                              "nom_estab", "nombre_act", "per_ocu", 
                              "telefono", "correoelec", "www", 
                              "municipio", "localidad", "entidad", 
                              "latitud", "longitud"
                          ])
    
    # Limpieza inicial optimizada
    df = df.dropna(subset=["entidad", "municipio", "nombre_act"])
    df.columns = df.columns.str.strip()
    
    return df

# Vectorización de la estimación de empleados
@st.cache_data
def estimar_empleados_vectorizado(serie):
    def estimar(valor):
        if pd.isna(valor):
            return np.nan
        valor = str(valor).lower()
        if "a" in valor:
            partes = valor.split("a")
            try:
                return (int(partes[0].strip()) + int(partes[1].strip().split()[0])) // 2
            except:
                return np.nan
        elif "menos de" in valor:
            try:
                return int(valor.split("menos de")[1].strip().split()[0]) - 1
            except:
                return np.nan
        elif "más de" in valor:
            try:
                return int(valor.split("más de")[1].strip().split()[0]) + 1
            except:
                return np.nan
        else:
            try:
                return int(valor.strip())
            except:
                return np.nan
    
    return serie.apply(estimar)

# Mapa optimizado con límite estricto de puntos
@st.cache_data(show_spinner=False)
def crear_mapa(df_mapa, max_puntos=300):
    df_mapa = df_mapa.dropna(subset=["Latitud", "Longitud"]).head(max_puntos)
    if df_mapa.empty:
        return None
    
    mapa = folium.Map(
        location=[
            df_mapa["Latitud"].astype(float).mean(),
            df_mapa["Longitud"].astype(float).mean()
        ], 
        zoom_start=11,
        tiles='CartoDB positron'  # Más ligero que OpenStreetMap
    )
    
    cluster = MarkerCluster(
        name="Empresas",
        max_cluster_radius=50,
        disable_clustering_at_zoom=12
    ).add_to(mapa)
    
    for _, row in df_mapa.iterrows():
        folium.Marker(
            location=[float(row["Latitud"]), float(row["Longitud"])],
            popup=f"{row['Nombre']}<br>{row['Giro']}",
            icon=folium.Icon(color='blue', icon='info-sign')
        ).add_to(cluster)
    
    folium.LayerControl().add_to(mapa)
    return mapa

def main():
    st.set_page_config(
        page_title="Katalis Ads DB Optimizer Pro", 
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    st.title("🚀 Katalis Ads DB Optimizer Pro")
    
    # Carga de datos optimizada
    archivo = st.file_uploader(
        "📂 Sube tu archivo del sistema (.csv o .xlsx)", 
        type=["csv", "xlsx"],
        key="file_uploader"
    )
    
    if not archivo:
        st.info("Por favor sube un archivo para comenzar")
        return
    
    try:
        df = cargar_datos(archivo)
        df["per_ocu_estimado"] = estimar_empleados_vectorizado(df["per_ocu"])
        
        # Filtros optimizados
        with st.form(key="form_filtros"):
            col1, col2 = st.columns(2)
            
            with col1:
                estado = st.selectbox(
                    "📍 Estado",
                    options=sorted(df["entidad"].dropna().unique()),
                    index=0,
                    key="estado_select"
                )
                
                municipios_opciones = sorted(
                    df[df["entidad"] == estado]["municipio"].dropna().unique()
                )
                municipios_sel = st.multiselect(
                    "🏘️ Municipios",
                    options=municipios_opciones,
                    default=municipios_opciones[:3],
                    key="municipios_multiselect"
                )
                
            with col2:
                tipo_negocio = st.text_input(
                    "✍️ Describe tu tipo de negocio",
                    key="tipo_negocio_input"
                )
                
                api_key = st.text_input(
                    "🔐 Ingresa tu API Key de DeepSeek",
                    type="password",
                    key="api_key_input"
                )
            
            if st.form_submit_button("🤖 Obtener giros sugeridos"):
                if tipo_negocio and api_key:
                    with st.spinner("Generando sugerencias..."):
                        prompt = f"Dame 5 giros económicos ideales para un negocio que se dedica a: {tipo_negocio}. Devuélvelos en una lista simple."
                        sugerencias = get_recommendations_deepseek(api_key, prompt)
                        st.session_state.sugerencias = sugerencias
        
        if 'sugerencias' in st.session_state:
            st.markdown("### 🎯 Sugerencias de giros:")
            st.write(st.session_state.sugerencias)
        
        # Filtrado principal
        st.subheader("🗂️ Aplicar filtros y ver resultados")
        
        df_filtrado = df[df["entidad"] == estado]
        if municipios_sel:
            df_filtrado = df_filtrado[df_filtrado["municipio"].isin(municipios_sel)]
        
        col1, col2 = st.columns(2)
        
        with col1:
            giros_sel = st.multiselect(
                "🏢 Giros económicos",
                options=sorted(df_filtrado["nombre_act"].dropna().unique()),
                key="giros_multiselect"
            )
            
            nombre_busqueda = st.text_input(
                "🔎 Buscar palabra clave en nombre del negocio",
                key="nombre_busqueda_input"
            )
        
        with col2:
            emp_validos = df_filtrado["per_ocu_estimado"].dropna()
            min_emp = int(emp_validos.min()) if not emp_validos.empty else 0
            max_emp = int(emp_validos.max()) if not emp_validos.empty else 100
            rango_emp = st.slider(
                "👥 Rango de empleados (estimado)",
                min_emp, max_emp, (min_emp, max_emp),
                key="rango_emp_slider"
            )
            
            col3, col4, col5 = st.columns(3)
            con_tel = col3.checkbox("📞 Teléfono", key="con_tel_check")
            con_mail = col4.checkbox("📧 Correo", key="con_mail_check")
            con_web = col5.checkbox("🌐 Web", key="con_web_check")
        
        # Aplicar filtros
        df_filtrado = df_filtrado[
            df_filtrado["per_ocu_estimado"].between(rango_emp[0], rango_emp[1])
        ]
        
        if giros_sel:
            df_filtrado = df_filtrado[df_filtrado["nombre_act"].isin(giros_sel)]
        if nombre_busqueda:
            df_filtrado = df_filtrado[
                df_filtrado["nom_estab"].str.lower().str.contains(nombre_busqueda.lower())
            ]
        if con_tel:
            df_filtrado = df_filtrado[df_filtrado["telefono"].notna()]
        if con_mail:
            df_filtrado = df_filtrado[df_filtrado["correoelec"].notna()]
        if con_web:
            df_filtrado = df_filtrado[df_filtrado["www"].notna()]
        
        # Resultados
        columnas_exportar = [
            "nom_estab", "nombre_act", "per_ocu", "per_ocu_estimado",
            "telefono", "correoelec", "www", "municipio", "localidad",
            "entidad", "latitud", "longitud"
        ]
        
        df_final = df_filtrado[columnas_exportar].copy()
        df_final.columns = [
            "Nombre", "Giro", "Personal (texto)", "Personal Estimado",
            "Teléfono", "Correo", "Web", "Municipio", "Localidad",
            "Estado", "Latitud", "Longitud"
        ]
        
        st.success(f"✅ Empresas encontradas: {len(df_final)}")
        
        # Mostrar resultados con paginación
        limite = st.slider(
            "🔢 ¿Cuántos resultados mostrar?",
            10, min(500, len(df_final)), 50,
            key="limite_slider"
        )
        
        st.dataframe(
            df_final.head(limite),
            use_container_width=True,
            height=600
        )
        
        # Descarga optimizada
        csv = df_final.to_csv(index=False).encode("utf-8")
        st.download_button(
            "📥 Descargar CSV optimizado",
            data=csv,
            file_name="denue_opt.csv",
            mime="text/csv",
            key="download_csv"
        )
        
        # Mapa interactivo con carga diferida
        if not df_final[["Latitud", "Longitud"]].dropna().empty:
            st.subheader("🗺️ Mapa interactivo")
            max_puntos = st.slider(
                "🔘 Máximo de puntos en mapa",
                10, 500, 300,
                key="max_puntos_slider"
            )
            
            with st.spinner("Generando mapa..."):
                mapa = crear_mapa(df_final, max_puntos)
                if mapa:
                    st_folium(
                        mapa,
                        height=500,
                        width=900,
                        returned_objects=[]
                    )
                else:
                    st.warning("No hay suficientes datos geográficos para mostrar el mapa")
        else:
            st.info("No hay coordenadas disponibles para mostrar el mapa.")
    
    except Exception as e:
        st.error(f"❌ Error crítico: {str(e)}")
        st.stop()

if __name__ == "__main__":
    main()

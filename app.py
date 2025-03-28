import streamlit as st
import pandas as pd
import folium
import requests
import json
import re
from openai import OpenAI
from typing import Dict, Optional, List
from streamlit_folium import st_folium

# ------------------ Configuración de la App ------------------
st.set_page_config(
    page_title="Katalis Movistar - Prospectador B2B",
    layout="wide",
    page_icon="📡"
)

# ------------------ Constantes ------------------
DENUE_API_URL = "https://www.inegi.org.mx/app/api/denue/v1/consulta/BuscarAreaActEstr/{entidad}/0/0/0/0/{sector}/0/0/0/0/1/1000/0/{estrato}/{token}"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1"
DEFAULT_DESCRIPTION = "Empresas de tecnología en CDMX y EdoMex con +50 empleados"
TIMEOUT = 30  # segundos

# ------------------ Función IA Optimizada ------------------
def interpretar_prompt_con_ia(descripcion: str, api_key: str) -> Optional[Dict]:
    """Transforma una descripción textual en parámetros estructurados para DENUE usando IA."""
    if not api_key or len(api_key) < 20:
        st.error("❌ API Key inválida o demasiado corta")
        return None

    try:
        client = OpenAI(
            api_key=api_key,
            base_url=DEEPSEEK_API_URL,
            timeout=TIMEOUT
        )

        prompt = f"""
        ACTÚA COMO UN EXPERTO EN CLASIFICACIÓN DENUE. 
        Analiza esta descripción y devuelve EXCLUSIVAMENTE un JSON VÁLIDO SIN texto adicional:

        "{descripcion}"

        FORMATO REQUERIDO (usa códigos CLAE reales):
        {{
            "sectores_clae": ["code1", "code2"],
            "estrato": ["value1", "value2"],
            "ubicaciones": ["code1", "code2"],
            "palabras_clave": ["keyword1", "keyword2"]
        }}

        REGLAS ESTRICTAS:
        1. Solo JSON válido (sin ```json```)
        2. Usa códigos oficiales DENUE
        3. No incluyas explicaciones
        4. Si no estás seguro de un valor, usa valores por defecto para telecomunicaciones
        """

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,  # Más determinista
            max_tokens=500,
            timeout=TIMEOUT,
            response_format={"type": "json_object"}  # Fuerza formato JSON
        )
        
        if not response.choices:
            st.error("No se recibió respuesta de la API")
            return None
            
        raw_response = response.choices[0].message.content.strip()
        
        # Procesamiento ultra-robusto de la respuesta
        try:
            # Extraer JSON aunque venga con o sin marcadores
            json_str = re.sub(r'^```json|```$', '', raw_response, flags=re.IGNORECASE).strip()
            
            # Limpieza agresiva de caracteres problemáticos
            json_str = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
            json_str = re.sub(r'^\s*$\n', '', json_str, flags=re.MULTILINE)
            
            # Parseo estricto
            parsed = json.loads(json_str)
            
            # Validación de estructura
            required_keys = ["sectores_clae", "estrato", "ubicaciones"]
            if not all(key in parsed for key in required_keys):
                raise ValueError("Estructura JSON incompleta")
                
            # Validación de tipos
            if not all(isinstance(parsed[key], list) for key in required_keys):
                raise ValueError("Tipos de datos incorrectos")
                
            return parsed
            
        except Exception as je:
            st.error(f"Error procesando respuesta. Texto recibido:\n{raw_response[:300]}")
            st.error(f"Error técnico: {str(je)}")
            return None
            
    except Exception as e:
        st.error(f"❌ Error de conexión: {str(e)}")
        return None

# ------------------ Consulta DENUE Optimizada ------------------
def consultar_denue(entidad: str, sector: str, estrato: str, token: str) -> pd.DataFrame:
    """Consulta la API DENUE con manejo robusto de errores."""
    try:
        response = requests.get(
            DENUE_API_URL.format(entidad=entidad, sector=sector, estrato=estrato, token=token),
            timeout=TIMEOUT
        )
        response.raise_for_status()
        
        empresas = [{
            "Nombre": e[2] if len(e) > 2 else "N/A",
            "Sector": e[4] if len(e) > 4 else "N/A",
            "Estrato": e[5] if len(e) > 5 else "N/A",
            "Teléfono": e[14] if len(e) > 14 else "N/A",
            "Email": e[15] if len(e) > 15 else "N/A",
            "Dirección": f"{e[8] if len(e) > 8 else ''} {e[9] if len(e) > 9 else ''}, {e[11] if len(e) > 11 else ''}, {e[13] if len(e) > 13 else ''}",
            "Latitud": float(e[19]) if len(e) > 19 and e[19] else 0.0,
            "Longitud": float(e[18]) if len(e) > 18 and e[18] else 0.0
        } for e in response.json()]
        
        return pd.DataFrame(empresas).replace("", "N/A")
    
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error en la API DENUE: {str(e)}")
        return pd.DataFrame()
    except (IndexError, ValueError, KeyError, TypeError) as e:
        st.error(f"❌ Error procesando datos: {str(e)}")
        return pd.DataFrame()

# ------------------ Visualización Mejorada ------------------
def mapa_empresas(df: pd.DataFrame) -> None:
    """Muestra un mapa interactivo con las empresas encontradas."""
    if df.empty:
        st.warning("No hay datos para mostrar en el mapa")
        return
    
    try:
        # Filtrar coordenadas válidas
        df = df[(df['Latitud'] != 0.0) & (df['Longitud'] != 0.0)]
        if df.empty:
            st.warning("No hay ubicaciones válidas para mostrar")
            return
        
        avg_lat = df['Latitud'].mean()
        avg_lon = df['Longitud'].mean()
        
        mapa = folium.Map(
            location=[avg_lat, avg_lon],
            zoom_start=11,
            tiles='cartodbpositron',
            control_scale=True
        )
        
        marker_cluster = folium.plugins.MarkerCluster().add_to(mapa)
        
        for _, row in df.iterrows():
            folium.Marker(
                location=[row['Latitud'], row['Longitud']],
                popup=folium.Popup(
                    f"<b>{row['Nombre']}</b><br>"
                    f"Sector: {row['Sector']}<br>"
                    f"Tel: {row['Teléfono']}<br>"
                    f"Email: {row['Email']}<br>"
                    f"Dirección: {row['Dirección']}",
                    max_width=300
                ),
                icon=folium.Icon(color='blue', icon='building', prefix='fa')
            ).add_to(marker_cluster)
        
        st_folium(mapa, width=1200, height=600, returned_objects=[])
    except Exception as e:
        st.error(f"❌ Error generando mapa: {str(e)}")

# ------------------ Interfaz Principal ------------------
def main():
    st.title("🧠 Prospectador Inteligente B2B")
    st.markdown("**Encuentra clientes ideales usando IA + datos DENUE**")
    
    with st.sidebar:
        st.header("🔑 Credenciales API")
        denue_token = st.text_input("Token DENUE", type="password", help="Obtenlo en https://www.inegi.org.mx/servicios/api_denue.html")
        deepseek_key = st.text_input("API Key DeepSeek", type="password", help="Clave para la API de DeepSeek")
        
        st.markdown("---")
        st.header("⚙️ Configuración")
        mostrar_raw = st.checkbox("Mostrar respuesta cruda de IA", False)
    
    descripcion = st.text_area(
        "**Describe tu cliente ideal:**",
        DEFAULT_DESCRIPTION,
        height=150,
        help="Ejemplo: 'Hoteles de 4-5 estrellas en Quintana Roo con más de 50 habitaciones'"
    )
    
    if st.button("🔍 Buscar Prospectos", type="primary", use_container_width=True):
        if not denue_token or not deepseek_key:
            st.warning("⚠️ Ingresa ambas credenciales API")
            return

        with st.spinner("🔍 Analizando con IA..."):
            parametros = interpretar_prompt_con_ia(descripcion, deepseek_key)
            if mostrar_raw and parametros:
                with st.expander("Respuesta cruda de IA"):
                    st.write(parametros)

        if not parametros:
            st.error("No se pudieron generar parámetros válidos")
            return
            
        st.success("✅ Parámetros generados correctamente")
        
        with st.expander("📋 Ver parámetros detallados", expanded=False):
            st.json(parametros)
        
        # Proceso de búsqueda
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_df = pd.DataFrame()
        
        ubicaciones = parametros.get('ubicaciones', [])
        sectores = parametros.get('sectores_clae', [])
        estratos = parametros.get('estrato', [])
        
        total_combinaciones = len(ubicaciones) * len(sectores) * len(estratos)
        if total_combinaciones == 0:
            st.error("No hay combinaciones válidas para buscar")
            return
            
        try:
            combinacion_actual = 0
            for estado in ubicaciones:
                for sector in sectores:
                    for estrato in estratos:
                        combinacion_actual += 1
                        status_text.text(f"🔍 Buscando: Estado {estado}, Sector {sector}, Estrato {estrato} ({combinacion_actual}/{total_combinaciones})")
                        
                        parcial = consultar_denue(estado, sector, estrato, denue_token)
                        if not parcial.empty:
                            total_df = pd.concat([total_df, parcial], ignore_index=True)
                            
                        progress_bar.progress(combinacion_actual / total_combinaciones)
            
            if not total_df.empty:
                st.success(f"✅ Encontradas {len(total_df)} empresas")
                
                # Mostrar resultados
                col1, col2 = st.columns([2, 1])
                with col1:
                    st.markdown("### 🗺 Mapa de Prospectos")
                    mapa_empresas(total_df)
                
                with col2:
                    st.markdown("### 📊 Resumen")
                    st.metric("Total Empresas", len(total_df))
                    st.metric("Sectores Únicos", total_df['Sector'].nunique())
                    st.metric("Ubicaciones Únicas", total_df['Dirección'].nunique())
                
                # Datos detallados
                st.markdown("### 📋 Detalle Completo")
                st.dataframe(
                    total_df,
                    column_config={
                        "Teléfono": st.column_config.TextColumn(width="medium"),
                        "Email": st.column_config.TextColumn(width="medium"),
                        "Dirección": st.column_config.TextColumn(width="large")
                    },
                    hide_index=True,
                    use_container_width=True
                )
                
                # Exportar datos
                csv = total_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📤 Descargar CSV",
                    data=csv,
                    file_name="prospectos_movistar.csv",
                    mime="text/csv",
                    type="primary",
                    use_container_width=True
                )
            else:
                st.warning("No se encontraron empresas con estos criterios")
                
        except Exception as e:
            st.error(f"Error en la búsqueda: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()

if __name__ == "__main__":
    main()

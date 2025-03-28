import streamlit as st
import pandas as pd
import folium
import requests
import json
from openai import OpenAI
from io import BytesIO
from streamlit_folium import st_folium
from typing import Dict, List, Optional, Tuple

# Configuración de la app
st.set_page_config(
    page_title="Katalis Movistar - Prospectador B2B",
    layout="wide",
    page_icon="📡"
)

# ------------------ Constantes y Configuraciones ------------------
DENUE_API_URL = "https://www.inegi.org.mx/app/api/denue/v1/consulta/BuscarAreaActEstr/{entidad}/0/0/0/0/{sector}/0/0/0/0/1/1000/0/{estrato}/{token}"
DEEPSEEK_API_URL = "https://api.deepseek.com/v1"
DEFAULT_DESCRIPTION = "Empresas de tecnología en CDMX y EdoMex con +50 empleados"
TIMEOUT = 30  # segundos

# ------------------ IA: Traductor de descripción a parámetros DENUE ------------------
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
        Eres un experto en inteligencia comercial B2B para empresas de telecomunicaciones.
        Interpreta esta descripción de cliente ideal:

        "{descripcion}"

        Devuelve un JSON válido como este:
        {{
            "sectores_clae": ["54", "72"],
            "estrato": ["5", "6"],
            "ubicaciones": ["09", "15"],
            "palabras_clave": ["internet", "call center", "servicios digitales"]
        }}
        """

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500,
            timeout=TIMEOUT
        )
        
        if not response.choices:
            st.error("No se recibió respuesta de la API")
            return None
            
        texto = response.choices[0].message.content.strip()
        try:
            # Limpieza robusta del JSON
            texto_limpio = texto.replace("```json", "").replace("```", "").strip()
            return json.loads(texto_limpio)
        except json.JSONDecodeError as je:
            st.error(f"Error decodificando JSON: {je}")
            return None
            
    except Exception as e:
        st.error(f"❌ Error en la conexión con DeepSeek: {str(e)}")
        return None

# ------------------ DENUE: Solicitud a API optimizada ------------------
def consultar_denue(entidad: str, sector: str, estrato: str, token: str) -> pd.DataFrame:
    """Consulta la API DENUE con manejo robusto de errores."""
    try:
        response = requests.get(
            DENUE_API_URL.format(entidad=entidad, sector=sector, estrato=estrato, token=token),
            timeout=TIMEOUT
        )
        response.raise_for_status()
        
        empresas = [{
            "Nombre": e[2],
            "Sector": e[4],
            "Estrato": e[5],
            "Teléfono": e[14],
            "Email": e[15],
            "Dirección": f"{e[8]} {e[9]}, {e[11]}, {e[13]}",
            "Latitud": float(e[19]),
            "Longitud": float(e[18])
        } for e in response.json() if len(e) > 19]  # Validación de campos
        
        return pd.DataFrame(empresas)
    
    except requests.exceptions.RequestException as e:
        st.error(f"❌ Error al consultar DENUE: {str(e)}")
        return pd.DataFrame()
    except (IndexError, ValueError, KeyError) as e:
        st.error(f"❌ Error procesando datos de DENUE: {str(e)}")
        return pd.DataFrame()

# ------------------ Visualización optimizada ------------------
def mapa_empresas(df: pd.DataFrame) -> None:
    """Muestra un mapa interactivo con las empresas encontradas."""
    if df.empty:
        st.warning("No hay datos para mostrar en el mapa")
        return
    
    try:
        # Calcular centro del mapa
        avg_lat = df['Latitud'].mean()
        avg_lon = df['Longitud'].mean()
        
        mapa = folium.Map(
            location=[avg_lat, avg_lon],
            zoom_start=11,
            tiles='cartodbpositron'
        )
        
        # Agrupación de marcadores para mejor rendimiento
        marker_cluster = folium.plugins.MarkerCluster().add_to(mapa)
        
        for _, row in df.iterrows():
            folium.Marker(
                location=[row['Latitud'], row['Longitud']],
                popup=folium.Popup(
                    f"<b>{row['Nombre']}</b><br>"
                    f"Sector: {row['Sector']}<br>"
                    f"Tel: {row['Teléfono']}<br>"
                    f"Dirección: {row['Dirección']}",
                    max_width=300
                ),
                icon=folium.Icon(color='blue', icon='building', prefix='fa')
            ).add_to(marker_cluster)
        
        st_folium(mapa, width=1200, height=500, returned_objects=[])
    except Exception as e:
        st.error(f"❌ Error generando el mapa: {str(e)}")

# ------------------ Streamlit UI ------------------
def main():
    st.title("🧠 Asistente Comercial Inteligente")
    st.markdown("Encuentra a tus prospectos ideales usando IA + datos oficiales DENUE")

    # Sidebar con configuración
    with st.sidebar:
        st.header("🔐 Configuración")
        denue_token = st.text_input("Token de DENUE", type="password", help="Obtenlo en https://www.inegi.org.mx/servicios/api_denue.html")
        deepseek_key = st.text_input("API Key de DeepSeek", type="password", help="Clave para acceder a la API de DeepSeek")
    
    # Entrada principal
    descripcion = st.text_area(
        "Describe tu cliente ideal:", 
        DEFAULT_DESCRIPTION,
        help="Ejemplo: 'Restaurantes de lujo en Guadalajara con más de 20 empleados'"
    )

    if st.button("🔍 Buscar Prospectos", type="primary"):
        if not denue_token or not deepseek_key:
            st.warning("⚠️ Debes ingresar ambas claves en el panel lateral")
            return

        with st.spinner("🔍 Analizando descripción con IA..."):
            parametros = interpretar_prompt_con_ia(descripcion, deepseek_key)

        if not parametros:
            st.error("No se pudieron generar parámetros de búsqueda")
            return

        st.success("✅ Parámetros generados correctamente")
        with st.expander("Ver parámetros generados"):
            st.json(parametros)

        # Validar parámetros recibidos
        required_keys = ['sectores_clae', 'estrato', 'ubicaciones']
        if not all(key in parametros for key in required_keys):
            st.error("Los parámetros generados no tienen el formato correcto")
            return

        # Procesamiento en lotes con feedback visual
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_df = pd.DataFrame()
        
        ubicaciones = parametros['ubicaciones']
        sectores = parametros['sectores_clae']
        estratos = parametros['estrato']
        total_combinaciones = len(ubicaciones) * len(sectores) * len(estratos)
        combinacion_actual = 0
        
        try:
            for estado in ubicaciones:
                for sector in sectores:
                    for estrato in estratos:
                        combinacion_actual += 1
                        status_text.text(f"Buscando: Estado {estado}, Sector {sector}, Estrato {estrato} ({combinacion_actual}/{total_combinaciones})")
                        parcial = consultar_denue(estado, sector, estrato, denue_token)
                        total_df = pd.concat([total_df, parcial], ignore_index=True)
                        progress_bar.progress(combinacion_actual / total_combinaciones)
            
            total_empresas = len(total_df)
            if total_empresas > 0:
                st.success(f"✅ Búsqueda completada: {total_empresas} empresas encontradas")
                
                st.markdown(f"### 🗺 Mapa de Resultados ({total_empresas} empresas)")
                mapa_empresas(total_df)

                st.markdown("### 📋 Detalle de Empresas")
                st.dataframe(
                    total_df,
                    column_config={
                        "Teléfono": st.column_config.TextColumn(width="medium"),
                        "Email": st.column_config.TextColumn(width="medium")
                    },
                    hide_index=True,
                    use_container_width=True
                )

                # Botón de descarga
                csv = total_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Descargar CSV",
                    data=csv,
                    file_name="prospectos_katalis.csv",
                    mime="text/csv",
                    type="primary"
                )
            else:
                st.warning("No se encontraron empresas con los criterios especificados")
                
        except Exception as e:
            st.error(f"Error durante la búsqueda: {str(e)}")
        finally:
            progress_bar.empty()
            status_text.empty()

if __name__ == "__main__":
    main()

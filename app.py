import streamlit as st
import pandas as pd
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import openai
import anthropic
import plotly.express as px

# Función para obtener recomendaciones desde distintos proveedores de IA
def get_recommendations_from_ai(api_key, provider, prompt):
    if provider == "OpenAI (GPT-4)":
        openai.api_key = api_key
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Eres un experto en segmentación de clientes para campañas B2B en Google Ads."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']
    elif provider == "Anthropic (Claude)":
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=300,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content
    elif provider == "DeepSeek":
        openai.api_key = api_key  # DeepSeek usa la API de OpenAI
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "Eres un experto en análisis de mercado."},
                {"role": "user", "content": prompt}
            ]
        )
        return response['choices'][0]['message']['content']
    else:
        return "Proveedor no válido"

# Función para mostrar métricas y gráficos de la base filtrada
def mostrar_metricas(df_filtrado):
    st.markdown("## 📊 Métricas y análisis de la base filtrada")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total de empresas", len(df_filtrado))
    if "per_ocu_estimado" in df_filtrado.columns and not df_filtrado["per_ocu_estimado"].empty:
        col2.metric("Promedio de empleados", round(df_filtrado["per_ocu_estimado"].mean(), 2))
    else:
        col2.metric("Promedio de empleados", "N/A")
    col3.metric("Empresas con contacto", df_filtrado[["telefono", "correoelec"]].dropna(how="all").shape[0])
    if "nombre_act" in df_filtrado.columns:
        top_giros = df_filtrado["nombre_act"].value_counts().nlargest(5).reset_index()
        top_giros.columns = ["Giro", "Cantidad"]
        fig = px.bar(top_giros, x="Giro", y="Cantidad", title="📈 Top 5 Giros Económicos")
        st.plotly_chart(fig, use_container_width=True)

# Guardamos el archivo
Path("/mnt/data/KatalisAds_DB_Optimizer_Pro_FINAL_IA_Metricas.py").write_text(codigo_actualizado)

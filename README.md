# Crear el archivo requirements.txt para la app profesional de Streamlit
contenido_reqs = '''
streamlit
pandas
folium
streamlit-folium
openpyxl
'''

path_reqs = "/mnt/data/requirements.txt"
with open(path_reqs, "w", encoding="utf-8") as f:
    f.write(contenido_reqs)

path_reqs

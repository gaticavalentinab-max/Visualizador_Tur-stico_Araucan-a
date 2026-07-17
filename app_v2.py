import streamlit as st
import geopandas as gpd
import pandas as pd
import folium
from folium.plugins import Draw
from streamlit_folium import st_folium
from pathlib import Path
import numpy as np
import rasterio
from rasterio.warp import calculate_default_transform, reproject, Resampling
import io, base64
from PIL import Image
import matplotlib.colors as mcolors
import plotly.graph_objects as go
from shapely.ops import linemerge
from shapely.geometry import LineString

# ─────────────────────────────────────────────────────────────
# Configuración general
# ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="Rutas Patrimoniales de La Araucanía", layout="wide", page_icon="🥾")
st.title("🥾 GeoVisualizador Turístico – Rutas Patrimoniales de La Araucanía")
st.write(
    "Explora la red vial, las rutas y hitos patrimoniales, y las áreas silvestres "
    "protegidas de la Región de La Araucanía. Activa capas, filtra por atributos y "
    "genera el perfil de elevación de una ruta."
)

DATA = Path("data")
CRS_METRICO = 32719  # UTM 19S: CRS proyectado usado para calcular áreas, largos y perfiles

# ─────────────────────────────────────────────────────────────
# PASO 1: Paletas y estilos cartográficos
# ─────────────────────────────────────────────────────────────

# Zonas protegidas (SNASPE): paleta azul/púrpura, deliberadamente alejada de
# la rampa verde-café-blanco del DEM para que ambas capas no se confundan
# cuando están activas al mismo tiempo.
ESTILO_CATEGORIA_SNASPE = {
    "Parque Nacional":   {"color": "#0D47A1", "fillColor": "#1565C0"},  # azul fuerte
    "Reserva Nacional":  {"color": "#4A148C", "fillColor": "#7B1FA2"},  # púrpura
    "Monumento Natural": {"color": "#AD1457", "fillColor": "#EC407A"},  # magenta
    "default":           {"color": "#37474F", "fillColor": "#607D8B"},
}

# Red vial: jerarquía por tipo de carpeta (capa de rodadura)
ESTILO_CARPETA = {
    "Pavimento Doble Calzada": {"color": "#B71C1C", "weight": 4.5},
    "Pavimento":               {"color": "#D84315", "weight": 3.0},
    "Pavimento Básico":        {"color": "#EF6C00", "weight": 2.2},
    "Ripio":                   {"color": "#DAA520", "weight": 1.6},
    "Suelo Natural":           {"color": "#8B6914", "weight": 1.2},
    "Sin Información":         {"color": "#9E9E9E", "weight": 1.0},
    "default":                 {"color": "#9E9E9E", "weight": 1.0},
}

# Rutas patrimoniales: una paleta vistosa, una por ruta temática
PALETA_RUTAS = [
    "#8E44AD", "#2980B9", "#27AE60", "#E67E22",
    "#C0392B", "#16A085", "#D35400", "#2C3E50",
]

# Dificultad de los hitos
ESTILO_DIFICULTAD = {
    "Fácil":              "#2ECC71",
    "Medianamente Fácil": "#F1C40F",
    "Bajo":               "#3498DB",
    "Sin información":    "#95A5A6",
}

# DEM: rampa hipsométrica
COLORMAP_DEM = [
    (0.00, "#1a4314"), (0.15, "#3f7d20"), (0.30, "#9ACD32"),
    (0.45, "#DAA520"), (0.60, "#CD853F"), (0.75, "#8B4513"),
    (0.88, "#D2B48C"), (1.00, "#FFFAFA"),
]

# ─────────────────────────────────────────────────────────────
# PASO 2: Funciones auxiliares de estilo y color
# ─────────────────────────────────────────────────────────────

def construir_mapa_colores(serie, paleta):
    """Asigna un color de la paleta a cada valor único de una columna."""
    valores = sorted(serie.dropna().unique().tolist())
    return {str(v): paleta[i % len(paleta)] for i, v in enumerate(valores)}


def normalizar_dificultad(valor):
    """Los hitos traen textos de dificultad poco homogéneos (ej. con notas
    sobre el clima). Se agrupan en 4 categorías legibles para el mapa."""
    if pd.isna(valor):
        return "Sin información"
    v = str(valor).lower()
    if "medianamente" in v:
        return "Medianamente Fácil"
    if "bajo" in v:
        return "Bajo"
    if "fácil" in v or "facil" in v:
        return "Fácil"
    return "Sin información"


def estilo_categorico_poligono(color_map_dict, col, opacidad=0.55, weight=1.5):
    def style_fn(feature):
        val = feature["properties"].get(col, "default")
        vals = color_map_dict.get(val, color_map_dict["default"])
        return {"color": vals["color"], "weight": weight, "fillColor": vals["fillColor"], "fillOpacity": opacidad}
    return style_fn


def estilo_carpeta_linea(feature):
    carpeta = feature["properties"].get("CARPETA", "default")
    vals = ESTILO_CARPETA.get(carpeta, ESTILO_CARPETA["default"])
    return {"color": vals["color"], "weight": vals["weight"], "opacity": 0.85}


def estilo_ruta_patrimonial(color_map_rutas):
    def style_fn(feature):
        nom = str(feature["properties"].get("NOM_RUTA", ""))
        color = color_map_rutas.get(nom, "#555555")
        return {"color": color, "weight": 4, "opacity": 0.9, "dashArray": "6,4"}
    return style_fn


def leyenda_categorica_html(titulo, color_map, icono="🔲", top="10px", right="10px"):
    items = "".join(
        f"""<div style="display:flex;align-items:center;margin:3px 0;">
              <div style="background:{c};width:16px;height:16px;border:1px solid #555;
                          margin-right:7px;border-radius:2px;flex-shrink:0;"></div>
              <span style="font-size:11px;color:#222;">{etq}</span></div>"""
        for etq, c in color_map.items()
    )
    return f"""
    <div style="position:fixed;top:{top};right:{right};z-index:1000;background:rgba(255,255,255,0.93);
        padding:10px 14px;border-radius:8px;border:1px solid #bbb;box-shadow:2px 2px 6px rgba(0,0,0,0.25);
        max-height:260px;overflow-y:auto;min-width:170px;font-family:Arial, sans-serif;">
      <b style="font-size:12px;">{icono} {titulo}</b><hr style="margin:5px 0;border-color:#ddd;">{items}
    </div>"""


def leyenda_dem_html(dem_min, dem_max, top="10px", right="10px"):
    stops = ", ".join([f"{color} {int(pct*100)}%" for pct, color in COLORMAP_DEM])
    gradient = f"linear-gradient(to top, {stops})"
    return f"""
    <div style="position:fixed;top:{top};right:{right};z-index:1000;background:rgba(255,255,255,0.93);
        padding:10px 14px;border-radius:8px;border:1px solid #bbb;box-shadow:2px 2px 6px rgba(0,0,0,0.25);
        min-width:130px;font-family:Arial, sans-serif;">
      <b style="font-size:12px;">🏔️ Elevación (m)</b><hr style="margin:5px 0;border-color:#ddd;">
      <div style="display:flex;align-items:stretch;gap:8px;">
        <div style="width:22px;height:150px;background:{gradient};border:1px solid #888;border-radius:3px;flex-shrink:0;"></div>
        <div style="display:flex;flex-direction:column;justify-content:space-between;font-size:11px;color:#333;">
          <span><b>{int(dem_max)} m</b></span>
          <span>{int(dem_min + (dem_max - dem_min) * 0.75)} m</span>
          <span>{int(dem_min + (dem_max - dem_min) * 0.50)} m</span>
          <span>{int(dem_min + (dem_max - dem_min) * 0.25)} m</span>
          <span><b>{int(dem_min)} m</b></span>
        </div>
      </div>
    </div>"""

# ─────────────────────────────────────────────────────────────
# PASO 3: Raster DEM -> overlay de imagen para folium
# ─────────────────────────────────────────────────────────────

def aplicar_colormap_dem(band, nodata):
    posiciones = [p for p, _ in COLORMAP_DEM]
    colores = [c for _, c in COLORMAP_DEM]
    cmap = mcolors.LinearSegmentedColormap.from_list("dem", list(zip(posiciones, colores)))
    mascara = (band == nodata) if nodata is not None else np.zeros_like(band, dtype=bool)
    valid = band[~mascara]
    dem_min = float(valid.min()) if len(valid) > 0 else 0
    dem_max = float(valid.max()) if len(valid) > 0 else 1
    norm = mcolors.Normalize(vmin=dem_min, vmax=dem_max)
    rgba = cmap(norm(band))
    rgba[mascara, 3] = 0
    rgba[~mascara, 3] = 0.80
    return (rgba * 255).astype(np.uint8), dem_min, dem_max


@st.cache_data(show_spinner=False)
def raster_a_overlay(raster_path):
    with rasterio.open(raster_path) as src:
        if src.crs and src.crs.to_epsg() != 4326:
            transform, width, height = calculate_default_transform(
                src.crs, "EPSG:4326", src.width, src.height, *src.bounds
            )
            data = np.zeros((1, height, width), dtype=np.float32)
            reproject(
                source=rasterio.band(src, 1), destination=data[0],
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=transform, dst_crs="EPSG:4326",
                resampling=Resampling.bilinear,
            )
            bounds_wgs84 = rasterio.transform.array_bounds(height, width, transform)
        else:
            data = src.read().astype(np.float32)
            bounds_wgs84 = src.bounds

        nodata = src.nodata
        img_array, dem_min, dem_max = aplicar_colormap_dem(data[0], nodata)
        img_pil = Image.fromarray(img_array)
        buf = io.BytesIO()
        img_pil.save(buf, format="PNG")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
        bounds = [[bounds_wgs84[1], bounds_wgs84[0]], [bounds_wgs84[3], bounds_wgs84[2]]]
        return img_b64, bounds, dem_min, dem_max

# ─────────────────────────────────────────────────────────────
# PASO 4: Perfil de elevación (ruta existente o transecto dibujado)
# ─────────────────────────────────────────────────────────────

def geometria_a_linea_simple(geom):
    """Convierte MultiLineString a una única LineString (fusiona tramos
    contiguos; si quedan varios trozos separados, se toma el más largo)."""
    if geom.geom_type == "MultiLineString":
        fusion = linemerge(geom)
        if fusion.geom_type == "MultiLineString":
            fusion = max(fusion.geoms, key=lambda g: g.length)
        return fusion
    return geom


def perfil_elevacion(geom_wgs84, dem_path, crs_origen="EPSG:4326", n_puntos=250):
    """Muestrea el DEM cada cierta distancia a lo largo de una línea.
    Se reproyecta la línea a la CRS métrica del DEM para que las
    distancias del eje X del gráfico sean kilómetros reales, no grados."""
    linea_m = gpd.GeoSeries([geom_wgs84], crs=crs_origen).to_crs(CRS_METRICO).iloc[0]
    linea_m = geometria_a_linea_simple(linea_m)
    largo_m = linea_m.length
    if largo_m == 0:
        return None, None
    distancias_m = np.linspace(0, largo_m, n_puntos)
    puntos = [linea_m.interpolate(d) for d in distancias_m]
    coords = [(p.x, p.y) for p in puntos]
    with rasterio.open(dem_path) as src:
        nodata = src.nodata
        elevs = [v[0] for v in src.sample(coords)]
    elevs = [np.nan if (nodata is not None and e == nodata) else float(e) for e in elevs]
    return distancias_m / 1000.0, elevs

# ─────────────────────────────────────────────────────────────
# PASO 5: Carga y preprocesamiento de datos
# ─────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=True)
def cargar_datos():
    hitos = gpd.read_file(DATA / "hitos_patrimoniales.gpkg").to_crs(4326)
    rutas = gpd.read_file(DATA / "rutas_patrimoniales_araucania.gpkg").to_crs(4326)
    vial = gpd.read_file(DATA / "red_vial_araucania.gpkg")
    zonas = gpd.read_file(DATA / "zonas_protegidas.gpkg")
    limite = gpd.read_file(DATA / "limite_araucania.gpkg").to_crs(4326)

    # Decisión de diseño: la capa de red vial regional trae tramos de las
    # regiones vecinas (Biobío, Los Ríos). Primero se filtra por el atributo
    # REGION (más rápido) y luego se recorta espacialmente con gpd.clip()
    # contra el polígono límite real. El recorte espacial es más robusto que
    # solo confiar en el atributo, porque corta exactamente en el borde del
    # territorio en vez de incluir/excluir el tramo completo según a qué
    # región esté asignado administrativamente.
    vial = vial[vial["REGION"] == "Región de La Araucanía"].copy()
    vial = vial.to_crs(4326)
    vial = gpd.clip(vial, limite)

    # SUPERFICIE viene como texto con coma decimal (formato chileno) -> float
    zonas["AREA_HA"] = zonas["SUPERFICIE"].astype(str).str.replace(",", ".").astype(float)
    zonas = zonas.to_crs(4326)

    # Dificultad de hitos: se normaliza a 4 categorías legibles
    hitos["DIFICULTAD_NORM"] = hitos["DIFICULTAD"].apply(normalizar_dificultad)

    # Largo real de cada tramo de ruta patrimonial y de cada tramo vial,
    # calculado en CRS métrica. El atributo SHAPE_LENG/SHAPE_Leng original
    # resultó inconsistente para varias rutas (valores en 0 o fuera de
    # escala), por lo que se optó por recalcularlo geométricamente.
    rutas["LARGO_KM"] = rutas.to_crs(CRS_METRICO).geometry.length / 1000
    vial["LARGO_KM"] = vial.to_crs(CRS_METRICO).geometry.length / 1000

    return hitos, rutas, vial, zonas, limite


try:
    hitos, rutas, vial, zonas, limite = cargar_datos()
except Exception as e:
    st.error(f"No fue posible cargar los datos desde la carpeta 'data/': {e}")
    st.stop()

DEM_PATH = DATA / "dem_araucania.tif"

# ─────────────────────────────────────────────────────────────
# PASO 6: Sidebar — capas, filtros y estadísticas
# ─────────────────────────────────────────────────────────────

st.sidebar.title("🗂️ Capas y filtros")

st.sidebar.subheader("Capas visibles")
mostrar_limite = st.sidebar.checkbox("🗺️ Límite regional", value=True)
mostrar_zonas = st.sidebar.checkbox("🌲 Áreas silvestres protegidas", value=True)
mostrar_vial = st.sidebar.checkbox("🛣️ Red vial", value=True)
mostrar_rutas = st.sidebar.checkbox("🥾 Rutas patrimoniales", value=True)
mostrar_hitos = st.sidebar.checkbox("📍 Hitos patrimoniales", value=True)
mostrar_dem = st.sidebar.checkbox("🏔️ DEM (elevación)", value=False)

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Filtro interactivo")

categorias_disp = sorted(zonas["CATEGORIA"].unique().tolist())
categorias_sel = st.sidebar.multiselect(
    "Categoría de área protegida", categorias_disp, default=categorias_disp
)

carpetas_disp = sorted(vial["CARPETA"].unique().tolist())
carpetas_sel = st.sidebar.multiselect(
    "Tipo de carpeta (red vial)", carpetas_disp, default=carpetas_disp
)

rutas_disp = ["Todas"] + sorted(rutas["NOM_RUTA"].unique().tolist())
ruta_sel = st.sidebar.selectbox("Ruta patrimonial", rutas_disp)

# Aplicación de filtros
zonas_f = zonas[zonas["CATEGORIA"].isin(categorias_sel)] if categorias_sel else zonas.iloc[0:0]
vial_f = vial[vial["CARPETA"].isin(carpetas_sel)] if carpetas_sel else vial.iloc[0:0]
if ruta_sel == "Todas":
    rutas_f, hitos_f = rutas, hitos
else:
    rutas_f = rutas[rutas["NOM_RUTA"] == ruta_sel]
    hitos_f = hitos[hitos["NOM_RUTA"] == ruta_sel]

st.sidebar.markdown("---")
st.sidebar.subheader("📊 Panel de estadísticas")
st.sidebar.metric("Área protegida (filtrada)", f"{zonas_f['AREA_HA'].sum():,.0f} ha")
st.sidebar.metric("Largo rutas patrimoniales", f"{rutas_f['LARGO_KM'].sum():,.1f} km")
st.sidebar.metric("Largo red vial (filtrada)", f"{vial_f['LARGO_KM'].sum():,.0f} km")
st.sidebar.metric("Hitos patrimoniales", f"{len(hitos_f)}")

with st.sidebar.expander("Ver desglose por categoría / ruta"):
    st.write("**Áreas protegidas (ha):**")
    st.dataframe(
        zonas_f.groupby("CATEGORIA")["AREA_HA"].sum().sort_values(ascending=False).round(0),
        width='stretch',
    )
    st.write("**Rutas patrimoniales (km):**")
    st.dataframe(
        rutas_f.groupby("NOM_RUTA")["LARGO_KM"].sum().sort_values(ascending=False).round(1),
        width='stretch',
    )

st.sidebar.markdown("---")
st.sidebar.subheader("📈 Perfil de elevación")
modo_perfil = st.sidebar.radio(
    "¿Cómo generar el perfil?",
    ["Ruta patrimonial existente", "Dibujar transecto en el mapa"],
)

tramo_sel = None
if modo_perfil == "Ruta patrimonial existente":
    opciones_tramo = rutas.apply(
        lambda r: f"{r['NOM_RUTA']} — {r['CIRCUITO']} ({r['LARGO_KM']:.1f} km)", axis=1
    )
    idx_sel = st.sidebar.selectbox(
        "Selecciona un tramo", options=opciones_tramo.index, format_func=lambda i: opciones_tramo[i]
    )
    tramo_sel = rutas.loc[idx_sel]
else:
    st.sidebar.info(
        "Usa la herramienta de dibujo (ícono de línea) en la esquina "
        "superior izquierda del mapa para trazar un transecto sobre el DEM."
    )

# ─────────────────────────────────────────────────────────────
# PASO 7: Construcción del mapa
# ─────────────────────────────────────────────────────────────

centroide = limite.geometry.iloc[0].centroid
centro = [centroide.y, centroide.x]
minx, miny, maxx, maxy = limite.total_bounds
m = folium.Map(location=centro, zoom_start=8, tiles=None)
m.fit_bounds([[miny, minx], [maxy, maxx]])
folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)
folium.TileLayer("CartoDB positron", name="Mapa claro (CartoDB)").add_to(m)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri World Imagery", name="Satélite (Esri)",
).add_to(m)

leyendas_html = []
offset_top = 10

# ── Límite regional (solo contorno, para dar contexto territorial) ─
if mostrar_limite:
    folium.GeoJson(
        limite,
        name="🗺️ Límite regional",
        style_function=lambda f: {"color": "#E91E63", "weight": 3.5, "fill": False, "dashArray": "10,5"},
        tooltip=folium.GeoJsonTooltip(fields=["Region"], aliases=["Región"]),
    ).add_to(m)

# ── DEM ─────────────────────────────────────────────────────
if mostrar_dem:
    try:
        with st.spinner("Cargando DEM..."):
            img_b64, bounds, dem_min, dem_max = raster_a_overlay(str(DEM_PATH))
        folium.raster_layers.ImageOverlay(
            image=f"data:image/png;base64,{img_b64}", bounds=bounds, opacity=0.75, name="🏔️ DEM elevación",
        ).add_to(m)
        leyendas_html.append(leyenda_dem_html(dem_min, dem_max, top=f"{offset_top}px"))
        offset_top += 220
    except Exception as e:
        st.warning(f"No fue posible cargar el DEM: {e}")

# ── Zonas protegidas (polígonos) ───────────────────────────
if mostrar_zonas and len(zonas_f) > 0:
    folium.GeoJson(
        zonas_f,
        name="🌲 Áreas protegidas",
        style_function=estilo_categorico_poligono(ESTILO_CATEGORIA_SNASPE, "CATEGORIA", opacidad=0.7, weight=2.5),
        tooltip=folium.GeoJsonTooltip(fields=["NOMBRE_TOT", "CATEGORIA", "AREA_HA"],
                                       aliases=["Nombre", "Categoría", "Superficie (ha)"]),
    ).add_to(m)
    leyendas_html.append(leyenda_categorica_html(
        "Áreas protegidas",
        {k: v["fillColor"] for k, v in ESTILO_CATEGORIA_SNASPE.items() if k != "default"},
        icono="🌲", top=f"{offset_top}px",
    ))
    offset_top += 130

# ── Red vial (líneas) ──────────────────────────────────────
if mostrar_vial and len(vial_f) > 0:
    folium.GeoJson(
        vial_f,
        name="🛣️ Red vial",
        style_function=estilo_carpeta_linea,
        tooltip=folium.GeoJsonTooltip(fields=["NOMBRE_CAM", "CARPETA", "CLASIFICAC"],
                                       aliases=["Camino", "Carpeta", "Clasificación"]),
    ).add_to(m)
    leyendas_html.append(leyenda_categorica_html(
        "Red vial (carpeta)",
        {k: v["color"] for k, v in ESTILO_CARPETA.items() if k not in ("default",)},
        icono="🛣️", top=f"{offset_top}px",
    ))
    offset_top += 190

# ── Rutas patrimoniales (líneas) ───────────────────────────
if mostrar_rutas and len(rutas_f) > 0:
    color_map_rutas = construir_mapa_colores(rutas["NOM_RUTA"], PALETA_RUTAS)
    folium.GeoJson(
        rutas_f,
        name="🥾 Rutas patrimoniales",
        style_function=estilo_ruta_patrimonial(color_map_rutas),
        tooltip=folium.GeoJsonTooltip(fields=["NOM_RUTA", "CIRCUITO", "LARGO_KM"],
                                       aliases=["Ruta", "Tramo", "Largo (km)"]),
    ).add_to(m)
    leyendas_html.append(leyenda_categorica_html(
        "Rutas patrimoniales", color_map_rutas, icono="🥾", top=f"{offset_top}px",
    ))
    offset_top += min(60 + len(color_map_rutas) * 23, 260) + 10

# ── Hitos patrimoniales (puntos) ───────────────────────────
if mostrar_hitos and len(hitos_f) > 0:
    for _, row in hitos_f.iterrows():
        color = ESTILO_DIFICULTAD.get(row["DIFICULTAD_NORM"], "#95A5A6")
        popup_html = (
            f"<b>{row['NOM_HITO']}</b><br>"
            f"Ruta: {row['NOM_RUTA']}<br>"
            f"Circuito: {row['NOM_CIRCUI']}<br>"
            f"Dificultad: {row['DIFICULTAD_NORM']}"
        )
        folium.CircleMarker(
            location=[row.geometry.y, row.geometry.x],
            radius=5, color="#333333", weight=1, fillColor=color, fillOpacity=0.9,
            tooltip=folium.Tooltip(f"{row['NOM_HITO']} — {row['DIFICULTAD_NORM']}"),
            popup=folium.Popup(popup_html, max_width=250),
        ).add_to(m)
    leyendas_html.append(leyenda_categorica_html(
        "Dificultad del hito", ESTILO_DIFICULTAD, icono="📍", top=f"{offset_top}px",
    ))
    offset_top += 130

# ── Herramienta de dibujo (solo si se eligió ese modo) ─────
if modo_perfil == "Dibujar transecto en el mapa":
    Draw(
        export=False,
        draw_options={"polyline": True, "polygon": False, "circle": False,
                      "rectangle": False, "marker": False, "circlemarker": False},
        edit_options={"edit": True, "remove": True},
    ).add_to(m)

for html in leyendas_html:
    m.get_root().html.add_child(folium.Element(html))

folium.LayerControl(collapsed=False).add_to(m)

salida_mapa = st_folium(m, width=1200, height=650, key="mapa_principal")

# ─────────────────────────────────────────────────────────────
# PASO 8: Perfil de elevación (se dibuja debajo del mapa)
# ─────────────────────────────────────────────────────────────

st.subheader("📈 Perfil de elevación")

geom_perfil = None
etiqueta_perfil = ""

if modo_perfil == "Ruta patrimonial existente" and tramo_sel is not None:
    geom_perfil = tramo_sel.geometry
    etiqueta_perfil = f"{tramo_sel['NOM_RUTA']} — {tramo_sel['CIRCUITO']}"
elif modo_perfil == "Dibujar transecto en el mapa" and salida_mapa.get("last_active_drawing"):
    dibujo = salida_mapa["last_active_drawing"]
    if dibujo and dibujo["geometry"]["type"] == "LineString":
        geom_perfil = LineString(dibujo["geometry"]["coordinates"])
        etiqueta_perfil = "Transecto dibujado por el usuario"

if geom_perfil is not None:
    try:
        dist_km, elev = perfil_elevacion(geom_perfil, str(DEM_PATH))
        if dist_km is not None:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dist_km, y=elev, mode="lines", fill="tozeroy",
                line=dict(color="#8B4513", width=2), name=etiqueta_perfil,
            ))
            fig.update_layout(
                title=etiqueta_perfil,
                xaxis_title="Distancia (km)", yaxis_title="Elevación (m s.n.m.)",
                height=350, margin=dict(l=10, r=10, t=40, b=10),
            )
            st.plotly_chart(fig, width='stretch')
            elev_validas = [e for e in elev if not np.isnan(e)]
            if elev_validas:
                c1, c2, c3 = st.columns(3)
                c1.metric("Elevación mínima", f"{min(elev_validas):.0f} m")
                c2.metric("Elevación máxima", f"{max(elev_validas):.0f} m")
                c3.metric("Desnivel total", f"{max(elev_validas) - min(elev_validas):.0f} m")
        else:
            st.info("La geometría seleccionada tiene largo cero; no se puede generar el perfil.")
    except Exception as e:
        st.warning(f"No fue posible generar el perfil de elevación: {e}")
else:
    st.info(
        "Selecciona un tramo de ruta patrimonial en la barra lateral, o dibuja un "
        "transecto sobre el mapa (modo 'Dibujar transecto'), para ver aquí su perfil de elevación."
    )

# ─────────────────────────────────────────────────────────────
# Pie de página
# ─────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Fuentes: Red vial y rutas/hitos patrimoniales, IDE MOP/SUBDERE; áreas silvestres "
    "protegidas, CONAF; DEM, elaboración propia a partir de datos de elevación regional. "
    "Desarrollado con Python, GeoPandas, Folium y Streamlit."
)

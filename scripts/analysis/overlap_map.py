import os
import sys

# PROJ fix
if not os.environ.get("PROJ_DATA") and not os.environ.get("PROJ_LIB"):
    _proj_dir = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "share", "proj")
    if os.path.isdir(_proj_dir):
        os.environ["PROJ_DATA"] = _proj_dir

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import MarkerCluster

# ── paths ──────────────────────────────────────────────────────────────────
OV_PATH   = 'data/interm/over_calc/over_com_buffer_0.gpkg'
MINE_PATH = 'data/dcrm_cluster_data/dcrm_cluster_data/mine_polygons.gpkg'
OUT_PATH  = 'fig/overlap_map.html'

CRS_PROJ  = 'EPSG:6933'   # equal-area metres for area / simplification
CRS_WEB   = 'EPSG:4326'   # WGS84 for Folium

SIMPLIFY_TOL_M = 500       # metres — controls polygon complexity reduction


COM_SELECT = ['Lithium', 'Iron', 'Copper', 'Aluminium', 'Gold', 'Silver', 'Zinc',
              'Nickel', 'Cobalt', 'Unknown', 'Coal', 'Rare Earth', 'Spodumene', 'Magnesium']

PALETTE = {
    'Lithium':    '#e6194b', 'Iron':      '#3cb44b', 'Copper':    '#4363d8',
    'Aluminium':  '#f58231', 'Gold':      '#FFD700', 'Silver':    '#aaaaaa',
    'Zinc':       '#42d4f4', 'Nickel':    '#f032e6', 'Cobalt':    '#469990',
    'Unknown':    '#bcbd22', 'Coal':      '#555555', 'Rare Earth':'#8c564b',
    'Spodumene':  '#e377c2', 'Magnesium': '#17becf', 'Other':     '#888888',
}


# ── data loading & preparation ─────────────────────────────────────────────

def load_data():
    print('  loading overlaps ...')
    ov = gpd.read_file(OV_PATH).to_crs(CRS_PROJ)

    # ── per-overlap stats ──────────────────────────────────────────────────
    # total overlap area per ov_id
    total_area = ov.groupby('ov_id')['alloc_area'].sum().rename('total_area_km2')

    # commodity list per overlap (all commodities, comma-separated)
    com_list = (ov.groupby('ov_id')['commodity_id']
                  .apply(lambda x: ', '.join(sorted(x.unique())))
                  .rename('commodities'))

    # dominant commodity (highest alloc_area) for colour coding
    dom_com = (ov.sort_values('alloc_area', ascending=False)
                 .drop_duplicates('ov_id')[['ov_id', 'commodity_id']]
                 .rename(columns={'commodity_id': 'dom_commodity'}))

    # unique overlap polygons
    unique_ov = (ov.drop_duplicates('ov_id')
                   [['ov_id', 'mine_id', 'indig_id', 'geometry']]
                   .join(total_area, on='ov_id')
                   .merge(com_list.reset_index(), on='ov_id')
                   .merge(dom_com, on='ov_id'))

    # ── mine area for share calculation ───────────────────────────────────
    print('  loading mine areas ...')
    mines = gpd.read_file(MINE_PATH).to_crs(CRS_PROJ)
    mines['mine_area_km2'] = mines.geometry.area * 1e-6
    mine_area = mines.set_index('id')['mine_area_km2']
    unique_ov['mine_area_km2'] = unique_ov['mine_id'].map(mine_area)
    unique_ov['share_pct'] = (
        unique_ov['total_area_km2'] / unique_ov['mine_area_km2'] * 100
    ).clip(upper=100).round(1)

    # ── colour per overlap ─────────────────────────────────────────────────
    unique_ov['com_plot'] = unique_ov['dom_commodity'].where(
        unique_ov['dom_commodity'].isin(COM_SELECT), 'Other')
    unique_ov['color'] = unique_ov['com_plot'].map(PALETTE)

    return unique_ov


def make_popup(row):
    """Return an HTML popup string for one overlap."""
    share_str = f"{row['share_pct']:.1f} %" if pd.notna(row['share_pct']) else 'n/a'
    mine_str  = f"{row['mine_area_km2']:.3f}" if pd.notna(row['mine_area_km2']) else 'n/a'
    html = f"""
    <div style="font-family:Arial,sans-serif; font-size:12px; min-width:220px">
      <b style="font-size:13px">Overlap ID {row['ov_id']}</b>
      <table style="margin-top:6px; border-collapse:collapse; width:100%">
        <tr><td style="color:#666;padding:2px 6px 2px 0">Mine ID</td>
            <td><b>{row['mine_id']}</b></td></tr>
        <tr><td style="color:#666;padding:2px 6px 2px 0">Territory</td>
            <td><b>{row['indig_id']}</b></td></tr>
        <tr><td style="color:#666;padding:2px 6px 2px 0">Commodities</td>
            <td>{row['commodities']}</td></tr>
        <tr><td style="color:#666;padding:2px 6px 2px 0">Overlap area</td>
            <td><b>{row['total_area_km2']:.3f} km²</b></td></tr>
        <tr><td style="color:#666;padding:2px 6px 2px 0">Mine area</td>
            <td>{mine_str} km²</td></tr>
        <tr><td style="color:#666;padding:2px 6px 2px 0">Share</td>
            <td><b>{share_str}</b></td></tr>
      </table>
    </div>"""
    return folium.Popup(html, max_width=280)


# ── map building ───────────────────────────────────────────────────────────

def build_map(unique_ov):
    # reproject to WGS84
    gdf = unique_ov.to_crs(CRS_WEB).copy()

    # ── simplified polygon layer (complexity-reduced, toggle off by default) ─
    print('  simplifying polygons ...')
    gdf_simp = unique_ov.copy()
    gdf_simp['geometry'] = gdf_simp.geometry.simplify(
        SIMPLIFY_TOL_M, preserve_topology=True)
    gdf_simp = gdf_simp.to_crs(CRS_WEB)
    print(f'  polygon layer: {len(gdf_simp):,} features')

    # ── centroids ──────────────────────────────────────────────────────────
    gdf_cen = gdf.copy()
    gdf_cen['geometry'] = gdf.geometry.centroid
    print(f'  centroid layer: {len(gdf_cen):,} markers')

    # ── initialise map ─────────────────────────────────────────────────────
    m = folium.Map(
        location=[20, 10],
        zoom_start=3,
        tiles=None,           # add tiles manually for layer control
    )
    folium.TileLayer('CartoDB positron', name='CartoDB Positron').add_to(m)
    folium.TileLayer('CartoDB dark_matter', name='CartoDB Dark').add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/'
              'World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri &mdash; Source: Esri, Maxar, Earthstar Geographics',
        name='ESRI Satellite',
    ).add_to(m)
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/'
              'World_Street_Map/MapServer/tile/{z}/{y}/{x}',
        attr='Esri &mdash; Source: Esri, HERE, Garmin, USGS',
        name='ESRI Street Map',
    ).add_to(m)

    # ── polygon layer (off by default) ────────────────────────────────────
    print('  adding polygon layer ...')
    poly_layer = folium.FeatureGroup(name='Overlap polygons (simplified)',
                                     show=False)

    def poly_style(feature):
        color = feature['properties'].get('color', '#888888')
        return {
            'fillColor':   color,
            'color':       '#333333',
            'weight':      0.5,
            'fillOpacity': 0.45,
        }

    folium.GeoJson(
        gdf_simp[['ov_id', 'mine_id', 'indig_id', 'commodities',
                  'total_area_km2', 'share_pct', 'color', 'geometry']]
               .__geo_interface__,
        style_function=poly_style,
        tooltip=folium.GeoJsonTooltip(
            fields=['ov_id', 'commodities', 'total_area_km2', 'share_pct'],
            aliases=['Overlap ID', 'Commodities', 'Area (km²)', 'Share (%)'],
            localize=True,
        ),
        name='polygons',
    ).add_to(poly_layer)
    poly_layer.add_to(m)

    # ── centroid marker cluster (on by default) ────────────────────────────
    print('  adding marker cluster ...')
    cluster_layer = folium.FeatureGroup(name='Overlap centroids (clustered)',
                                        show=True)
    cluster = MarkerCluster(
        options={'maxClusterRadius': 40, 'disableClusteringAtZoom': 8}
    ).add_to(cluster_layer)

    for _, row in gdf_cen.iterrows():
        lon, lat = row.geometry.x, row.geometry.y
        folium.CircleMarker(
            location=[lat, lon],
            radius=5,
            color=row['color'],
            fill=True,
            fill_color=row['color'],
            fill_opacity=0.85,
            weight=0.8,
            popup=make_popup(row),
            tooltip=f"ID {row['ov_id']} · {row['com_plot']} · {row['total_area_km2']:.2f} km²",
        ).add_to(cluster)

    cluster_layer.add_to(m)

    # ── commodity colour legend ────────────────────────────────────────────
    legend_html = """
    <div style="position:fixed; bottom:30px; left:30px; z-index:9999;
                background:white; padding:10px 14px; border-radius:6px;
                border:1px solid #ccc; font-family:Arial,sans-serif;
                font-size:11px; line-height:1.7; box-shadow:2px 2px 6px rgba(0,0,0,0.15)">
      <b style="font-size:12px">Dominant commodity</b><br>
    """
    for com, color in PALETTE.items():
        if com == 'Other':
            continue
        legend_html += (f'<span style="display:inline-block;width:12px;height:12px;'
                        f'border-radius:50%;background:{color};'
                        f'margin-right:6px;vertical-align:middle"></span>{com}<br>')
    legend_html += ('  <span style="display:inline-block;width:12px;height:12px;'
                    'border-radius:50%;background:#888888;'
                    'margin-right:6px;vertical-align:middle"></span>Other<br>')
    legend_html += '</div>'
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ── main ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading data ...')
    unique_ov = load_data()
    print(f'  {len(unique_ov):,} unique overlaps ready')

    print('Building map ...')
    m = build_map(unique_ov)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    m.save(OUT_PATH)
    print(f'  -> {OUT_PATH}')
    print('Done.')

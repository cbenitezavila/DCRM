import os
import sys

# PROJ fix — works whether or not conda env is activated
if not os.environ.get("PROJ_DATA") and not os.environ.get("PROJ_LIB"):
    _proj_dir = os.path.join(os.path.dirname(os.path.dirname(sys.executable)), "share", "proj")
    if os.path.isdir(_proj_dir):
        os.environ["PROJ_DATA"] = _proj_dir

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import shapely
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

# ── paths & constants ──────────────────────────────────────────────────────
MINE_PATH        = 'data/dcrm_cluster_data/dcrm_cluster_data/mine_polygons.gpkg'
OV_PATH          = 'data/interm/over_calc/over_com_buffer_0.gpkg'
WORLD_BOUND_PATH = '/home/flhuber/Projects/DCRM/data/world_bound/world-administrative-boundaries.shp'
FIG_PATH         = 'fig/share'
CACHE_DIR        = 'data/interm/share_cache'
CRS_PROJ         = 'EPSG:6933'   # equal-area metres

# keep in sync with spatial.py
COM_SELECT = ['Lithium', 'Iron', 'Copper', 'Aluminium', 'Gold', 'Silver', 'Zinc',
              'Nickel', 'Cobalt', 'Unknown', 'Coal', 'Rare Earth', 'Spodumene', 'Magnesium']

PALETTE = {
    'Lithium':    '#e6194b', 'Iron':      '#3cb44b', 'Copper':    '#4363d8',
    'Aluminium':  '#f58231', 'Gold':      '#ffe119', 'Silver':    '#aaaaaa',
    'Zinc':       '#42d4f4', 'Nickel':    '#f032e6', 'Cobalt':    '#469990',
    'Unknown':    '#bcbd22', 'Coal':      '#333333', 'Rare Earth':'#8c564b',
    'Spodumene':  '#e377c2', 'Magnesium': '#17becf', 'Other':     '#cccccc',
}

NON_COMMODITY = {"not relevant"}

os.makedirs(FIG_PATH, exist_ok=True)


# ── helpers ────────────────────────────────────────────────────────────────

def parse_materials(s):
    """Same logic as M_to_to_commodity: NaN field → ['Unknown'], empty parts dropped."""
    if pd.isna(s):
        return ['Unknown']
    result = []
    for c in [p.strip() for p in str(s).split(',')]:
        if c == '':
            continue
        result.append('Unknown' if c.lower() in NON_COMMODITY else c)
    return result if result else ['Unknown']


def savefig(name):
    path = os.path.join(FIG_PATH, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  -> {path}')


def clip_areas(polys, world):
    """
    Clip polygons to country boundaries and return area sums per country/region.

    Instead of assigning each polygon to one country by centroid (which breaks
    when a polygon straddles a border), this splits every polygon along country
    boundaries and attributes each piece to the country it physically lies in.
    Numerator and denominator are then computed on the same geographic basis,
    making shares > 100 % geometrically impossible.

    Returns a DataFrame with columns: country, region, area_km2
    """
    w = world[['name', 'continent', 'geometry']].copy()
    clipped = gpd.overlay(polys[['geometry']], w, how='intersection', keep_geom_type=False)
    clipped['area_km2'] = clipped.geometry.area * 1e-6
    return (clipped.groupby(['name', 'continent'])['area_km2']
                   .sum()
                   .reset_index()
                   .rename(columns={'name': 'country', 'continent': 'region'}))


# ── data loading ───────────────────────────────────────────────────────────

def load_data():
    mines = gpd.read_file(MINE_PATH).to_crs(CRS_PROJ)
    ov    = gpd.read_file(OV_PATH).to_crs(CRS_PROJ)
    world = gpd.read_file(WORLD_BOUND_PATH).to_crs(CRS_PROJ)

    # total overlap area per ov_id (sum alloc_area over commodities = overlay_area)
    ov_total = ov.groupby('ov_id')['alloc_area'].sum().rename('total_area')
    ov = ov.join(ov_total, on='ov_id')
    unique_ov = ov.drop_duplicates('ov_id')[['ov_id', 'total_area', 'geometry']].copy()

    # clip both datasets to country boundaries — no centroid mismatch possible
    print('  clipping mine polygons to country boundaries ...')
    mine_by_country = clip_areas(mines, world).rename(columns={'area_km2': 'mine_area_km2'})

    print('  clipping overlap polygons to country boundaries ...')
    ov_by_country = clip_areas(unique_ov, world).rename(columns={'area_km2': 'overlap_km2'})

    mines['mine_area_km2'] = mines.geometry.area * 1e-6

    # global share = total overlap / total mine area  (aggregate ratio)
    total_mine_km2    = mine_by_country['mine_area_km2'].sum()
    total_overlap_km2 = ov_by_country['overlap_km2'].sum()
    global_share = (total_overlap_km2 / total_mine_km2) * 100   # %

    return mines, ov, unique_ov, mine_by_country, ov_by_country, global_share, world


# ── A. Top 10 countries by absolute overlap, displayed as relative share ───

def plot_A_country_share(mine_by_country, ov_by_country, global_share):
    df_all = (mine_by_country[['country', 'mine_area_km2']]
              .merge(ov_by_country[['country', 'overlap_km2']], on='country', how='outer')
              .fillna(0))
    df_all = df_all[df_all['mine_area_km2'] > 0].copy()
    df_all['share'] = df_all['overlap_km2'] / df_all['mine_area_km2']
    country_avg = df_all['share'].mean() * 100   # mean of per-country shares

    # select by absolute overlap, display relative share
    df = df_all.set_index('country').nlargest(10, 'overlap_km2').sort_values('share', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("flare", len(df))
    bars = ax.barh(df.index, df['share'] * 100,
                   color=palette, edgecolor='white', linewidth=0.4)
    for bar, val in zip(bars, df['share'] * 100):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}%', va='center', fontsize=9)
    ax.axvline(country_avg, color='steelblue', linewidth=1.2, linestyle='--',
               label=f'Per-country avg  {country_avg:.2f}%')
    ax.axvline(global_share, color='crimson', linewidth=1.2, linestyle=':',
               label=f'Global share  {global_share:.2f}%')
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_xlabel('Overlap area / Total mine area  (%)')
    ax.set_title('A  -  Top 10 Countries by Absolute Overlap (share displayed)', fontsize=13)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    sns.despine(ax=ax, left=True)
    ax.tick_params(left=False)
    plt.tight_layout()
    savefig('A_country_share.png')


# ── C. World regions by relative overlap share ────────────────────────────

def plot_C_region_share(mine_by_country, ov_by_country, global_share):
    mine_by_region = mine_by_country.groupby('region')['mine_area_km2'].sum()
    ov_by_region   = ov_by_country.groupby('region')['overlap_km2'].sum()

    df = pd.concat([mine_by_region, ov_by_region], axis=1).fillna(0)
    df['share'] = df['overlap_km2'] / df['mine_area_km2']
    df = (df[df['mine_area_km2'] > 0]
            .sort_values('share', ascending=False))
    region_avg = df['share'].mean() * 100   # mean of per-region shares

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("flare", len(df))[::-1]
    bars = ax.barh(df.index[::-1], df['share'][::-1] * 100,
                   color=palette, edgecolor='white', linewidth=0.4)
    for bar, val in zip(bars, df['share'][::-1] * 100):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=9)
    ax.axvline(region_avg, color='steelblue', linewidth=1.2, linestyle='--',
               label=f'Per-region avg  {region_avg:.1f}%')
    ax.axvline(global_share, color='crimson', linewidth=1.2, linestyle=':',
               label=f'Global share  {global_share:.1f}%')
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_xlabel('Overlap area / Total mine area  (%)')
    ax.set_title('C  -  World Regions: Mine–Indigenous Land Overlap Share', fontsize=13)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    sns.despine(ax=ax, left=True)
    ax.tick_params(left=False)
    plt.tight_layout()
    savefig('C_region_share.png')


# ── B. Top 10 countries by relative overlap share ─────────────────────────

def plot_B_country_relative(mine_by_country, ov_by_country, global_share):
    df_all = (mine_by_country[['country', 'mine_area_km2']]
              .merge(ov_by_country[['country', 'overlap_km2']], on='country', how='outer')
              .fillna(0))
    df_all = df_all[df_all['mine_area_km2'] > 0].copy()
    df_all['share'] = df_all['overlap_km2'] / df_all['mine_area_km2']
    country_avg = df_all['share'].mean() * 100   # mean of per-country shares

    df = (df_all.set_index('country')
                .sort_values('share', ascending=False)
                .head(10))

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("flare", len(df))[::-1]
    bars = ax.barh(df.index[::-1], df['share'][::-1] * 100,
                   color=palette, edgecolor='white', linewidth=0.4)
    for bar, val in zip(bars, df['share'][::-1] * 100):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=9)
    ax.axvline(country_avg, color='steelblue', linewidth=1.2, linestyle='--',
               label=f'Per-country avg  {country_avg:.2f}%')
    ax.axvline(global_share, color='crimson', linewidth=1.2, linestyle=':',
               label=f'Global share  {global_share:.2f}%')
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_xlabel('Overlap area / Total mine area  (%)')
    ax.set_title('B  -  Top 10 Countries: Relative Mine–Indigenous Land Overlap Share', fontsize=13)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    sns.despine(ax=ax, left=True)
    ax.tick_params(left=False)
    plt.tight_layout()
    savefig('B_country_relative_share.png')


# ── D. Commodity overlap share ─────────────────────────────────────────────

def plot_D_commodity_share(mines, ov, global_share):
    # allocated mine area per commodity (equal split across commodities per mine)
    mc = mines[['id', 'mine_area_km2', 'materials_list']].copy()
    mc['commodities'] = mc['materials_list'].apply(parse_materials)
    mc['n']           = mc['commodities'].str.len()
    mc = mc.explode('commodities').rename(columns={'commodities': 'commodity_id'})
    mc['alloc_mine_km2'] = mc['mine_area_km2'] / mc['n']
    mc['com_plot'] = mc['commodity_id'].where(mc['commodity_id'].isin(COM_SELECT), 'Other')

    mine_by_com = (mc.groupby('com_plot')['alloc_mine_km2']
                     .sum().rename('mine_area_km2'))

    ov_c = ov.copy()
    ov_c['com_plot'] = ov_c['commodity_id'].where(ov_c['commodity_id'].isin(COM_SELECT), 'Other')
    ov_by_com = (ov_c.groupby('com_plot')['alloc_area']
                     .sum().rename('overlap_km2'))

    df = pd.concat([mine_by_com, ov_by_com], axis=1).fillna(0)
    df['share'] = df['overlap_km2'] / df['mine_area_km2']
    df = (df[df['mine_area_km2'] > 0]
            .sort_values('share', ascending=False))
    commodity_avg = df['share'].mean() * 100   # mean of per-commodity shares

    colors = [PALETTE.get(c, PALETTE['Other']) for c in df.index[::-1]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df.index[::-1], df['share'][::-1] * 100,
                   color=colors, edgecolor='white', linewidth=0.4)
    for bar, val in zip(bars, df['share'][::-1] * 100):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}%', va='center', fontsize=9)
    ax.axvline(commodity_avg, color='steelblue', linewidth=1.2, linestyle='--',
               label=f'Per-commodity avg  {commodity_avg:.2f}%')
    ax.axvline(global_share, color='crimson', linewidth=1.2, linestyle=':',
               label=f'Global share  {global_share:.2f}%')
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_xlabel('Allocated overlap area / Allocated mine area  (%)')
    ax.set_title('D  -  Mine–Indigenous Land Overlap Share by Commodity', fontsize=13)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
    sns.despine(ax=ax, left=True)
    ax.tick_params(left=False)
    plt.tight_layout()
    savefig('D_commodity_share.png')


# ── E. Choropleth: country-level overlap share ────────────────────────────

def plot_E_choropleth(mine_by_country, ov_by_country, world):
    # build per-country share
    df = (mine_by_country[['country', 'mine_area_km2']]
          .merge(ov_by_country[['country', 'overlap_km2']], on='country', how='outer')
          .fillna(0))
    df['share_pct'] = np.where(df['mine_area_km2'] > 0,
                               df['overlap_km2'] / df['mine_area_km2'] * 100,
                               np.nan)

    # join share onto world geometry; reproject to Robinson for visual clarity
    CRS_ROBIN = '+proj=robin +lon_0=0 +datum=WGS84'
    gdf = (world[['name', 'geometry']]
           .merge(df[['country', 'share_pct']], left_on='name', right_on='country', how='left')
           .to_crs(CRS_ROBIN))

    # cap colorscale at 95th percentile so outliers don't compress the palette
    valid = gdf['share_pct'].dropna()
    vmax  = float(valid.quantile(0.95))
    cmap  = sns.color_palette("flare", as_cmap=True)

    fig, ax = plt.subplots(figsize=(18, 9))

    # background: countries with no mining data
    gdf[gdf['share_pct'].isna()].plot(
        ax=ax, color='#d9d9d9', edgecolor='white', linewidth=0.3)

    # foreground: countries with data, colored by share
    gdf[gdf['share_pct'].notna()].plot(
        column='share_pct', ax=ax, cmap=cmap,
        vmin=0, vmax=vmax,
        edgecolor='white', linewidth=0.3)

    # colorbar
    norm = plt.Normalize(vmin=0, vmax=vmax)
    sm   = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.018, pad=0.02, shrink=0.6)
    cbar.set_label('Overlap / Mine area  (%)', fontsize=10)
    cbar.ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    cbar.ax.text(0.5, 1.04, f'(capped at {vmax:.0f}%)', transform=cbar.ax.transAxes,
                 ha='center', fontsize=7, color='#555555')

    ax.set_title('E  -  Mine–Indigenous Land Overlap Share by Country', fontsize=13, pad=10)
    ax.axis('off')
    plt.tight_layout()
    savefig('E_choropleth_share.png')


# ── F. Commodity exposure quadrant: overlap area vs share ─────────────────

def plot_F_commodity_quadrant(mines, ov, global_share):
    # ── per-commodity totals — ALL commodities, no bucketing ───────────────
    mc = mines[['id', 'mine_area_km2', 'materials_list']].copy()
    mc['commodities'] = mc['materials_list'].apply(parse_materials)
    mc['n']           = mc['commodities'].str.len()
    mc = mc.explode('commodities').rename(columns={'commodities': 'commodity_id'})
    mc['alloc_mine_km2'] = mc['mine_area_km2'] / mc['n']
    mine_by_com = mc.groupby('commodity_id')['alloc_mine_km2'].sum().rename('mine_area_km2')

    ov_by_com = ov.groupby('commodity_id')['alloc_area'].sum().rename('overlap_km2')

    df = (pd.concat([mine_by_com, ov_by_com], axis=1)
            .fillna(0)
            .reset_index()
            .rename(columns={'commodity_id': 'commodity'}))
    df['share_pct'] = df['overlap_km2'] / df['mine_area_km2'] * 100
    df = df[df['mine_area_km2'] > 0]

    # ── select which commodities get a text label ──────────────────────────
    # restrict candidate pool to commodities that appear in the overlap layer
    df_ov = df[df['overlap_km2'] > 0].copy()
    df_ov['rank_share']   = df_ov['share_pct'].rank(pct=True)
    df_ov['rank_overlap'] = df_ov['overlap_km2'].rank(pct=True)
    df_ov['rank_both']    = df_ov['rank_share'] + df_ov['rank_overlap']

    top_share   = set(df_ov.nlargest(3, 'share_pct')['commodity'])
    top_overlap = set(df_ov.nlargest(3, 'overlap_km2')['commodity'])
    top_both    = set(df_ov[~df_ov['commodity'].isin(top_share | top_overlap)]
                         .nlargest(3, 'rank_both')['commodity'])
    # ── quadrant thresholds: median overlap among overlapping (x), global share (y)
    x_med = df_ov['overlap_km2'].median()
    y_ref = global_share

    in_top_right = set(df_ov[(df_ov['overlap_km2'] >  x_med) &
                             (df_ov['share_pct']   >  y_ref)]['commodity'])
    in_top_left  = set(df_ov[(df_ov['overlap_km2'] <= x_med) &
                             (df_ov['share_pct']   >  y_ref)]['commodity'])
    to_label = top_share | top_overlap | top_both | in_top_right | in_top_left

    # ── axis bounds ────────────────────────────────────────────────────────
    xlo = df_ov['overlap_km2'].min() * 0.3
    xhi = df_ov['overlap_km2'].max() * 3.0
    yhi = df['share_pct'].max() * 1.22

    fig, ax = plt.subplots(figsize=(13, 8))
    ax.set_xlim(xlo, xhi)
    ax.set_ylim(0,   yhi)

    # quadrant corner labels (no background shading)
    ax.text(xlo * 1.8,  yhi * 0.97,  'Q2  High share · Low overlap',
            fontsize=8, color='#888', va='top')
    ax.text(x_med * 1.6, yhi * 0.97, 'Q1  High share · High overlap\n→ exposure reduction priority',
            fontsize=8, color='#c0392b', va='top', fontweight='bold')
    ax.text(xlo * 1.8,  y_ref * 0.08, 'Q3  Low share · Low overlap',
            fontsize=8, color='#888', va='bottom')
    ax.text(x_med * 1.6, y_ref * 0.08, 'Q4  Low share · High overlap',
            fontsize=8, color='#27ae60', va='bottom')

    # ── reference lines ────────────────────────────────────────────────────
    ax.axvline(x_med, color='#999', linewidth=1.0, linestyle='--', zorder=2)
    ax.axhline(y_ref, color='crimson', linewidth=1.8, linestyle=':', zorder=2)

    # ── bubbles: all commodities ───────────────────────────────────────────
    max_area = df['mine_area_km2'].max()
    sizes  = (df['mine_area_km2'] / max_area).apply(np.sqrt) * 2800 + 80
    # named palette colour for known commodities, muted grey for the rest
    colors = [PALETTE.get(c, '#bbbbbb') for c in df['commodity']]
    ax.scatter(df['overlap_km2'], df['share_pct'], s=sizes, c=colors,
               alpha=0.75, edgecolors='white', linewidth=0.9, zorder=4)

    # ── labels only for the 9 selected commodities ────────────────────────
    for _, row in df[df['commodity'].isin(to_label)].iterrows():
        ax.annotate(row['commodity'],
                    xy=(row['overlap_km2'], row['share_pct']),
                    xytext=(7, 4), textcoords='offset points',
                    fontsize=8.5, fontweight='bold', color='#111', zorder=5)

    ax.set_xscale('log')
    ax.set_xlabel('Total allocated overlap area  (km²)  [log scale]', fontsize=10)
    ax.set_ylabel('Overlap / Mine area  (%)', fontsize=10)
    ax.set_title('F  -  Commodity Exposure Quadrant', fontsize=13)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))

    # ── size legend: fixed marker sizes so text never overlaps ────────────
    size_handles = [
        plt.Line2D([], [], marker='o', linestyle='None', color='#aaa',
                   markeredgecolor='white', markeredgewidth=0.5,
                   markersize=ms, label=lbl)
        for lbl, ms in [('10 k km²', 5), ('100 k km²', 9), ('500 k km²', 13)]
    ]
    share_handle = plt.Line2D([], [], color='crimson', linestyle=':',
                               linewidth=1.8, label=f'Global share  {global_share:.1f}%')
    ax.legend(handles=[share_handle] + size_handles,
              fontsize=8.5, framealpha=0.85, loc='center',
              bbox_to_anchor=(0.18, 0.78), bbox_transform=ax.transAxes,
              title='Mine area  (km²)', labelspacing=0.7)
    sns.despine(ax=ax)
    plt.tight_layout()
    savefig('F_commodity_quadrant.png')


# ── K. Top-10 countries: stacked overlap area by commodity + share line ───

def plot_K_country_commodity(ov, mine_by_country, ov_by_country, world, global_share):
    # assign a country to each unique overlap via centroid sjoin
    pts = ov.drop_duplicates('ov_id').copy()
    pts = pts.set_geometry(pts.geometry.centroid)
    w   = world[['name', 'geometry']].copy()
    joined = gpd.sjoin(pts[['ov_id', 'geometry']], w, how='left', predicate='within')
    ov_c = ov.merge(joined[['ov_id', 'name']].rename(columns={'name': 'country'}),
                    on='ov_id', how='left')

    ov_c['com_plot'] = ov_c['commodity_id'].where(
        ov_c['commodity_id'].isin(COM_SELECT), 'Other')

    # top 10 countries by total absolute overlap
    top10 = (ov_by_country
             .nlargest(15, 'overlap_km2')['country'].tolist())

    ov_top = ov_c[ov_c['country'].isin(top10)]

    # pivot: country × commodity  →  allocated area (km²)
    pivot = (ov_top.groupby(['country', 'com_plot'])['alloc_area']
                   .sum()
                   .unstack(fill_value=0))
    com_order = [c for c in COM_SELECT + ['Other'] if c in pivot.columns]
    pivot = pivot[com_order]
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]

    # share per country (from clipped areas — geographically consistent)
    share_df = (ov_by_country[['country', 'overlap_km2']]
                .merge(mine_by_country[['country', 'mine_area_km2']], on='country')
                .set_index('country'))
    share_df['share_pct'] = share_df['overlap_km2'] / share_df['mine_area_km2'] * 100
    shares = share_df.loc[pivot.index, 'share_pct']

    x = np.arange(len(pivot))
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # stacked bars
    bottom = np.zeros(len(pivot))
    for com in com_order:
        vals = pivot[com].values
        ax1.bar(x, vals, bottom=bottom,
                color=PALETTE.get(com, PALETTE['Other']),
                label=com, edgecolor='white', linewidth=0.4)
        bottom += vals

    # share on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(x, shares.values, color='steelblue', linewidth=1.8,
             linestyle='--', marker='o', markersize=5, zorder=5)
    ax2.axhline(global_share, color='crimson', linewidth=1.8, linestyle=':',
                label=f'Global share  {global_share:.1f}%')
    ax2.set_ylabel('Overlap / Mine area  (%)', fontsize=10)
    ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f'{v:.0f}%'))
    ax2.set_ylim(bottom=0)

    ax1.set_xticks(x)
    ax1.set_xticklabels(pivot.index, rotation=25, ha='right', fontsize=9)
    ax1.set_ylabel('Allocated overlap area  (km²)', fontsize=10)
    ax1.set_title('K  -  Top 10 Countries: Overlap Area by Commodity and Share', fontsize=13)

    # combined legend
    bar_handles, bar_labels = ax1.get_legend_handles_labels()
    share_line  = mpatches.Patch(color='white', label='')  # spacer
    dot_handle  = plt.Line2D([0], [0], color='steelblue', linewidth=1.8,
                              linestyle='--', marker='o', markersize=5,
                              label='Country share  (%)')
    global_line = plt.Line2D([0], [0], color='crimson', linewidth=1.8,
                              linestyle=':', label=f'Global share  {global_share:.1f}%')
    ax1.legend(handles=bar_handles + [dot_handle, global_line],
               loc='upper right', fontsize=8, ncol=2, framealpha=0.85)

    sns.despine(ax=ax1, right=False)
    plt.tight_layout()
    savefig('K_country_commodity.png')


# ── main ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading data ...')
    mines, ov, unique_ov, mine_by_country, ov_by_country, global_share, world = load_data()
    print(f'  mines: {len(mines):,}  |  unique overlaps: {unique_ov["ov_id"].nunique():,}')
    print(f'  global share (total overlap / total mine area): {global_share:.3f}%')

    print('A - Country absolute ...');   plot_A_country_share(mine_by_country, ov_by_country, global_share)
    print('B - Country relative ...');   plot_B_country_relative(mine_by_country, ov_by_country, global_share)
    print('C - Region share ...');       plot_C_region_share(mine_by_country, ov_by_country, global_share)
    print('D - Commodity share ...');    plot_D_commodity_share(mines, ov, global_share)
    print('E - Choropleth ...');         plot_E_choropleth(mine_by_country, ov_by_country, world)
    print('F - Commodity quadrant ...');  plot_F_commodity_quadrant(mines, ov, global_share)
    print('K - Country×commodity ...');  plot_K_country_commodity(ov, mine_by_country, ov_by_country, world, global_share)

    print('\nDone. Figures saved to', FIG_PATH)

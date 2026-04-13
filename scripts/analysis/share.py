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
COM_SELECT = ['Lithium', 'Iron', 'Copper', 'Aluminium', 'Gold',
              'Silver', 'Zinc', 'Lead', 'Nickel', 'Cobalt', 'Unknown', 'Coal']

PALETTE = {
    'Lithium':   '#e6194b', 'Iron':      '#3cb44b', 'Copper':    '#4363d8',
    'Aluminium': '#f58231', 'Gold':      '#ffe119', 'Silver':    '#aaaaaa',
    'Zinc':      '#42d4f4', 'Lead':      '#911eb4', 'Nickel':    '#f032e6',
    'Cobalt':    '#469990', 'Unknown':   '#bcbd22', 'Coal':      '#333333',
    'Other':     '#cccccc',
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

    # global average = mean of per-mine overlap share (share_i = overlap_i / mine_area_i)
    # mines with no overlap get share = 0; included so the average reflects all mines
    mines['mine_area_km2'] = mines.geometry.area * 1e-6
    ov_per_mine = (ov.drop_duplicates('ov_id')[['ov_id', 'mine_id', 'total_area']]
                     .groupby('mine_id')['total_area'].sum())
    mine_shares = mines[['id', 'mine_area_km2']].copy()
    mine_shares = mine_shares.join(ov_per_mine.rename('overlap_km2'), on='id').fillna(0)
    mine_shares['share'] = mine_shares['overlap_km2'] / mine_shares['mine_area_km2']
    global_avg = mine_shares['share'].mean() * 100   # % — same value used in A, B, D

    return mines, ov, unique_ov, mine_by_country, ov_by_country, global_avg, world


# ── A. Top 10 countries by absolute overlap, displayed as relative share ───

def plot_A_country_share(mine_by_country, ov_by_country, global_avg):
    df = (mine_by_country[['country', 'mine_area_km2']]
          .merge(ov_by_country[['country', 'overlap_km2']], on='country', how='outer')
          .fillna(0)
          .set_index('country'))
    df['share'] = df['overlap_km2'] / df['mine_area_km2']

    # select by absolute overlap, display relative share
    df = df.nlargest(10, 'overlap_km2').sort_values('share', ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("flare", len(df))
    bars = ax.barh(df.index, df['share'] * 100,
                   color=palette, edgecolor='white', linewidth=0.4)
    for bar, val in zip(bars, df['share'] * 100):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}%', va='center', fontsize=9)
    ax.axvline(global_avg, color='black', linewidth=1.2, linestyle='--',
               label=f'Global avg  {global_avg:.2f}%')
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_xlabel('Overlap area / Total mine area  (%)')
    ax.set_title('A  -  Top 10 Countries by Absolute Overlap (share displayed)', fontsize=13)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    sns.despine(ax=ax, left=True)
    ax.tick_params(left=False)
    plt.tight_layout()
    savefig('A_country_share.png')


# ── C. World regions by relative overlap share ────────────────────────────

def plot_C_region_share(mine_by_country, ov_by_country):
    mine_by_region = mine_by_country.groupby('region')['mine_area_km2'].sum()
    ov_by_region   = ov_by_country.groupby('region')['overlap_km2'].sum()

    df = pd.concat([mine_by_region, ov_by_region], axis=1).fillna(0)
    df['share'] = df['overlap_km2'] / df['mine_area_km2']
    df = (df[df['mine_area_km2'] > 0]
            .sort_values('share', ascending=False))

    fig, ax = plt.subplots(figsize=(9, 5))
    palette = sns.color_palette("flare", len(df))[::-1]
    bars = ax.barh(df.index[::-1], df['share'][::-1] * 100,
                   color=palette, edgecolor='white', linewidth=0.4)
    for bar, val in zip(bars, df['share'][::-1] * 100):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=9)
    ax.set_xlabel('Overlap area / Total mine area  (%)')
    ax.set_title('C  -  World Regions: Mine–Indigenous Land Overlap Share', fontsize=13)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    sns.despine(ax=ax, left=True)
    ax.tick_params(left=False)
    plt.tight_layout()
    savefig('C_region_share.png')


# ── B. Top 10 countries by relative overlap share ─────────────────────────

def plot_B_country_relative(mine_by_country, ov_by_country, global_avg):
    df = (mine_by_country[['country', 'mine_area_km2']]
          .merge(ov_by_country[['country', 'overlap_km2']], on='country', how='outer')
          .fillna(0)
          .set_index('country'))
    df['share'] = df['overlap_km2'] / df['mine_area_km2']
    df = (df[df['mine_area_km2'] > 0]
            .sort_values('share', ascending=False)
            .head(10))

    fig, ax = plt.subplots(figsize=(10, 6))
    palette = sns.color_palette("flare", len(df))[::-1]
    bars = ax.barh(df.index[::-1], df['share'][::-1] * 100,
                   color=palette, edgecolor='white', linewidth=0.4)
    for bar, val in zip(bars, df['share'][::-1] * 100):
        ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height() / 2,
                f'{val:.1f}%', va='center', fontsize=9)
    ax.axvline(global_avg, color='black', linewidth=1.2, linestyle='--',
               label=f'Global avg  {global_avg:.2f}%')
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_xlabel('Overlap area / Total mine area  (%)')
    ax.set_title('B  -  Top 10 Countries: Relative Mine–Indigenous Land Overlap Share', fontsize=13)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    sns.despine(ax=ax, left=True)
    ax.tick_params(left=False)
    plt.tight_layout()
    savefig('B_country_relative_share.png')


# ── D. Commodity overlap share ─────────────────────────────────────────────

def plot_D_commodity_share(mines, ov, global_avg):
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

    colors = [PALETTE.get(c, PALETTE['Other']) for c in df.index[::-1]]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(df.index[::-1], df['share'][::-1] * 100,
                   color=colors, edgecolor='white', linewidth=0.4)
    for bar, val in zip(bars, df['share'][::-1] * 100):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                f'{val:.2f}%', va='center', fontsize=9)
    ax.axvline(global_avg, color='black', linewidth=1.2, linestyle='--',
               label=f'Global avg  {global_avg:.2f}%')
    ax.legend(fontsize=9, framealpha=0.85)
    ax.set_xlabel('Allocated overlap area / Allocated mine area  (%)')
    ax.set_title('D  -  Mine–Indigenous Land Overlap Share by Commodity', fontsize=13)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.1f}%'))
    sns.despine(ax=ax, left=True)
    ax.tick_params(left=False)
    plt.tight_layout()
    savefig('D_commodity_share.png')


# ── main ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading data ...')
    mines, ov, unique_ov, mine_by_country, ov_by_country, global_avg, world = load_data()
    print(f'  mines: {len(mines):,}  |  unique overlaps: {unique_ov["ov_id"].nunique():,}')
    print(f'  global avg per-mine share: {global_avg:.3f}%')

    print('A - Country absolute ...');   plot_A_country_share(mine_by_country, ov_by_country, global_avg)
    print('B - Country relative ...');   plot_B_country_relative(mine_by_country, ov_by_country, global_avg)
    print('C - Region share ...');       plot_C_region_share(mine_by_country, ov_by_country)
    print('D - Commodity share ...');    plot_D_commodity_share(mines, ov, global_avg)

    print('\nDone. Figures saved to', FIG_PATH)

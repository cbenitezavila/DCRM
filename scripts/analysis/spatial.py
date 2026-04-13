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
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.stats import gaussian_kde
from shapely.geometry import box
import seaborn as sns

# ── paths & constants ──────────────────────────────────────────────────────
OV_PATH          = 'data/interm/over_calc/over_com_buffer_0.gpkg'
FIG_PATH         = 'fig/spatial'
WORLD_BOUND_PATH = '/home/flhuber/Projects/DCRM/data/world_bound/world-administrative-boundaries.shp'
INDIG_PATH       = 'data/IPL_IndigenousPeoplesLands_2017/01_Data/IPL_IndigenousPeoplesLands_2017/IPL_2017.shp'
MINE_PATH        = 'data/dcrm_cluster_data/dcrm_cluster_data/mine_polygons.gpkg'
COM_SELECT = ['Lithium', 'Iron', 'Copper', 'Aluminium', 'Gold',
              'Silver', 'Zinc', 'Lead', 'Nickel', 'Cobalt', 'Unknown', 'Coal', 'Phosphate']

CRS      = 'EPSG:4326'
CRS_PROJ = 'EPSG:6933'   # equal-area metres
GRID_KM  = 50

os.makedirs(FIG_PATH, exist_ok=True)

# ── colour palette ─────────────────────────────────────────────────────────
PALETTE = {
    'Lithium':   '#e6194b',
    'Iron':      '#3cb44b',
    'Copper':    '#4363d8',
    'Aluminium': '#f58231',
    'Gold':      '#ffe119',
    'Silver':    '#aaaaaa',
    'Zinc':      '#42d4f4',
    'Phosphate':      '#911eb4',
    'Nickel':    '#f032e6',
    'Cobalt':    '#469990',
    'Unknown':   '#bcbd22',
    'Coal':      '#333333',
    'Other':     '#cccccc',
}

# ── helpers ────────────────────────────────────────────────────────────────

def load_data():
    ov    = gpd.read_file(OV_PATH).to_crs(CRS)
    world = gpd.read_file(WORLD_BOUND_PATH).to_crs(CRS)
    # anything outside COM_SELECT -> "Other"
    ov['com_plot'] = ov['commodity_id'].where(ov['commodity_id'].isin(COM_SELECT), 'Other')
    # total area per overlap (sum alloc_area over all commodities = original overlay_area)
    ov_area = ov.groupby('ov_id')['alloc_area'].sum().rename('total_area')
    ov = ov.join(ov_area, on='ov_id')
    return ov, world


def savefig(name):
    path = os.path.join(FIG_PATH, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  -> {path}')


def legend_patches(labels):
    return [mpatches.Patch(color=PALETTE.get(c, PALETTE['Other']), label=c) for c in labels]


# ── A. Global KDE ──────────────────────────────────────────────────────────

def plot_A_kde(ov, world):
    pts = ov.drop_duplicates('ov_id').copy()
    cx  = pts.geometry.centroid.x.values
    cy  = pts.geometry.centroid.y.values

    kde = gaussian_kde(np.vstack([cx, cy]), bw_method='scott')
    xx, yy = np.meshgrid(np.linspace(-180, 180, 720),
                         np.linspace( -90,  90, 360))
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(16, 8))
    world.plot(ax=ax, color='#e8e8e8', edgecolor='none', linewidth=0, zorder=1)
    im = ax.imshow(zz, origin='lower', extent=[-180, 180, -90, 90],
                   cmap='YlOrRd', alpha=0.85, aspect='auto', zorder=2,
                   vmin=0, vmax=np.percentile(zz, 99))
    # country boundaries on top of the KDE so they remain visible
    world.boundary.plot(ax=ax, linewidth=0.4, color='#555555', zorder=3)
    plt.colorbar(im, ax=ax, orientation='horizontal', fraction=0.03, pad=0.02,
                 label='Kernel density estimate')
    ax.set_title('A  -  Global KDE: Concentration of Mine-Indigenous Land Overlaps', fontsize=13)
    ax.set_axis_off()
    savefig('A_global_kde.png')


# ── B. Country choropleth ──────────────────────────────────────────────────

def plot_B_choropleth(ov, world):
    unique_ov = ov.drop_duplicates('ov_id')[['ov_id', 'total_area', 'geometry']].copy()
    unique_ov = unique_ov.set_geometry(unique_ov.geometry.centroid)

    joined = gpd.sjoin(unique_ov, world[['name', 'geometry']],
                       how='left', predicate='within')
    country_sum = (joined.groupby('name')['total_area']
                         .sum()
                         .reset_index()
                         .rename(columns={'total_area': 'total_overlap_km2'}))

    world_plot = world.merge(country_sum, on='name', how='left')

    flare = sns.color_palette("flare", as_cmap=True)

    fig, ax = plt.subplots(figsize=(18, 9))
    # no-data countries — greyed out
    world_plot[world_plot['total_overlap_km2'].isna()].plot(
        ax=ax, color='#bdbdbd', edgecolor='#999999', linewidth=0.3)
    # countries with overlap — quantile classification + flare
    world_plot[world_plot['total_overlap_km2'].notna()].plot(
        column='total_overlap_km2', ax=ax,
        cmap=flare, scheme='quantiles', k=6,
        legend=True,
        legend_kwds={'title': 'Overlap area (km2)', 'loc': 'lower left',
                     'fontsize': 8, 'title_fontsize': 9, 'framealpha': 0.85},
    )
    world.boundary.plot(ax=ax, linewidth=0.3, color='#555555')
    ax.set_title('B  -  Total Mine-Indigenous Land Overlap by Country (km2)', fontsize=13)
    ax.set_axis_off()
    savefig('B_country_choropleth.png')


# ── C. 50 km grid – dominant commodity among unique overlaps ──────────────

def plot_C_unique_polygons(ov, world):
    # dominant commodity per ov_id = highest alloc_area
    dom = (ov.sort_values('alloc_area', ascending=False)
             .drop_duplicates('ov_id')[['ov_id', 'com_plot', 'alloc_area', 'geometry']])

    dom_proj = dom.to_crs(CRS_PROJ)
    step = GRID_KM * 1000
    cx = dom_proj.geometry.centroid.x
    cy = dom_proj.geometry.centroid.y

    dom_proj = dom_proj.copy()
    dom_proj['cell_x'] = (cx // step) * step
    dom_proj['cell_y'] = (cy // step) * step

    # dominant commodity per cell (by alloc_area)
    cell_com = (dom_proj.groupby(['cell_x', 'cell_y', 'com_plot'])['alloc_area']
                        .sum().reset_index())
    idx      = cell_com.groupby(['cell_x', 'cell_y'])['alloc_area'].idxmax()
    dominant = cell_com.loc[idx].reset_index(drop=True)

    dominant['geometry'] = dominant.apply(
        lambda r: box(r['cell_x'], r['cell_y'],
                      r['cell_x'] + step, r['cell_y'] + step), axis=1)
    dom_gdf = gpd.GeoDataFrame(dominant, geometry='geometry', crs=CRS_PROJ).to_crs(CRS)

    fig, ax = plt.subplots(figsize=(18, 9))
    world.plot(ax=ax, color='#e8e8e8', edgecolor='#aaaaaa', linewidth=0.3)
    for com in dom_gdf['com_plot'].unique():
        dom_gdf[dom_gdf['com_plot'] == com].plot(
            ax=ax, color=PALETTE.get(com, PALETTE['Other']), alpha=0.85, linewidth=0)

    present = [c for c in COM_SELECT + ['Other'] if c in dom_gdf['com_plot'].values]
    ax.legend(handles=legend_patches(present), loc='lower left',
              fontsize=7, ncol=2, framealpha=0.85)
    ax.set_title(f'C  -  Dominant Commodity per {GRID_KM}x{GRID_KM} km Cell (unique overlaps)', fontsize=13)
    ax.set_axis_off()
    savefig('C_unique_overlaps.png')


# ── D. 50 km grid – dominant commodity ────────────────────────────────────

def plot_D_grid(ov, world):
    ov_proj = ov.to_crs(CRS_PROJ)
    step    = GRID_KM * 1000

    cx = ov_proj.geometry.centroid.x
    cy = ov_proj.geometry.centroid.y

    grid_df           = ov_proj.copy()
    grid_df['cell_x'] = (cx // step) * step
    grid_df['cell_y'] = (cy // step) * step

    cell_com = (grid_df.groupby(['cell_x', 'cell_y', 'com_plot'])['alloc_area']
                       .sum().reset_index())
    idx      = cell_com.groupby(['cell_x', 'cell_y'])['alloc_area'].idxmax()
    dominant = cell_com.loc[idx].reset_index(drop=True)

    dominant['geometry'] = dominant.apply(
        lambda r: box(r['cell_x'], r['cell_y'],
                      r['cell_x'] + step, r['cell_y'] + step), axis=1)
    dom_gdf = gpd.GeoDataFrame(dominant, geometry='geometry', crs=CRS_PROJ).to_crs(CRS)

    fig, ax = plt.subplots(figsize=(18, 9))
    world.plot(ax=ax, color='#e8e8e8', edgecolor='#aaaaaa', linewidth=0.3)
    for com, grp in dom_gdf.groupby('com_plot'):
        gpd.GeoDataFrame(grp, geometry='geometry', crs=CRS).plot(
            ax=ax, color=PALETTE.get(com, PALETTE['Other']), alpha=0.85, linewidth=0)

    present = [c for c in COM_SELECT + ['Other'] if c in dom_gdf['com_plot'].values]
    ax.legend(handles=legend_patches(present), loc='lower left',
              fontsize=7, ncol=2, framealpha=0.85)
    ax.set_title(f'D  -  Dominant Commodity per {GRID_KM}x{GRID_KM} km Grid Cell', fontsize=13)
    ax.set_axis_off()
    savefig('D_grid_dominant_commodity.png')


# ── E. Total overlap area by commodity (bar) ──────────────────────────────

def plot_E_commodity_bar(ov):
    by_com = (ov.groupby('com_plot')['alloc_area']
                .sum()
                .sort_values(ascending=False)
                .reset_index())

    fig, ax = plt.subplots(figsize=(11, 5))
    colors = [PALETTE.get(c, PALETTE['Other']) for c in by_com['com_plot']]
    bars   = ax.bar(by_com['com_plot'], by_com['alloc_area'],
                    color=colors, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, by_com['alloc_area']):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                f'{val:,.0f}', ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('Allocated overlap area (km2)')
    ax.set_title('E  -  Total Mine-Indigenous Land Overlap Area by Commodity', fontsize=13)
    ax.tick_params(axis='x', rotation=30)
    plt.tight_layout()
    savefig('E_commodity_area_bar.png')


# ── F. Top 25 indigenous territories by total overlap ─────────────────────

def plot_F_top_indig(ov):
    by_indig = (ov.drop_duplicates('ov_id')
                  .groupby('indig_id')['total_area']
                  .sum()
                  .sort_values(ascending=False)
                  .head(25)
                  .reset_index())

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(by_indig['indig_id'][::-1], by_indig['total_area'][::-1],
            color='#4363d8', edgecolor='white', linewidth=0.4)
    ax.set_xlabel('Total overlap area (km2)')
    ax.set_title('F  -  Top 25 Indigenous Territories by Mine Overlap Area', fontsize=13)
    plt.tight_layout()
    savefig('F_top_indig_territories.png')


# ── G. Overlap area distribution by commodity (box, log scale) ────────────

def plot_G_area_dist(ov):
    order  = (ov.groupby('com_plot')['alloc_area']
                .median()
                .sort_values(ascending=False)
                .index.tolist())
    data   = [ov.loc[ov['com_plot'] == c, 'alloc_area'].values for c in order]
    colors = [PALETTE.get(c, PALETTE['Other']) for c in order]

    fig, ax = plt.subplots(figsize=(12, 5))
    bp = ax.boxplot(data, patch_artist=True,
                    flierprops=dict(marker='.', markersize=2, alpha=0.3),
                    medianprops=dict(color='black', linewidth=1.5))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
    ax.set_yscale('log')
    ax.set_xticks(range(1, len(order) + 1))
    ax.set_xticklabels(order, rotation=30)
    ax.set_ylabel('Allocated overlap area (km2)  [log scale]')
    ax.set_title('G  -  Distribution of Overlap Area by Commodity', fontsize=13)
    plt.tight_layout()
    savefig('G_area_distribution.png')


# ── H. Stacked bar: overlap by continent and commodity ────────────────────

def plot_H_continent(ov, world):
    unique_ov = ov.drop_duplicates('ov_id')[['ov_id', 'total_area', 'geometry']].copy()
    unique_ov = unique_ov.set_geometry(unique_ov.geometry.centroid)

    joined = gpd.sjoin(unique_ov, world[['continent', 'geometry']],
                       how='left', predicate='within')
    joined['continent'] = joined['continent'].fillna('Unknown')

    ov_cont = ov.merge(joined[['ov_id', 'continent']], on='ov_id', how='left')
    pivot   = (ov_cont.groupby(['continent', 'com_plot'])['alloc_area']
                      .sum()
                      .unstack(fill_value=0))
    pivot   = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).index]
    com_order = [c for c in COM_SELECT + ['Other'] if c in pivot.columns]
    pivot   = pivot[com_order]

    fig, ax = plt.subplots(figsize=(12, 6))
    bottom = np.zeros(len(pivot))
    for com in com_order:
        vals = pivot[com].values
        ax.bar(pivot.index, vals, bottom=bottom,
               color=PALETTE.get(com, PALETTE['Other']),
               label=com, edgecolor='white', linewidth=0.4)
        bottom += vals

    ax.legend(loc='upper right', fontsize=8, ncol=2, framealpha=0.85)
    ax.set_ylabel('Total allocated overlap area (km2)')
    ax.set_title('H  -  Overlap Area by Continent and Commodity', fontsize=13)
    ax.tick_params(axis='x', rotation=20)
    plt.tight_layout()
    savefig('H_continent_commodity.png')


# ── I. Mine polygons — commodity assigned vs not, 50x50 km grid ───────────

def _mine_grid(mines_proj, step, status_col):
    """Return a GeoDataFrame of 50 km cells coloured by dominant status."""
    cx = mines_proj.geometry.centroid.x
    cy = mines_proj.geometry.centroid.y
    df = mines_proj[[status_col]].copy()
    df['cell_x'] = (cx // step) * step
    df['cell_y'] = (cy // step) * step

    # majority status per cell
    cell_counts = (df.groupby(['cell_x', 'cell_y', status_col])
                     .size().rename('n').reset_index())
    idx      = cell_counts.groupby(['cell_x', 'cell_y'])['n'].idxmax()
    dominant = cell_counts.loc[idx].reset_index(drop=True)

    dominant['geometry'] = dominant.apply(
        lambda r: box(r['cell_x'], r['cell_y'],
                      r['cell_x'] + step, r['cell_y'] + step), axis=1)
    return gpd.GeoDataFrame(dominant, geometry='geometry', crs=CRS_PROJ).to_crs(CRS)


def plot_I_mine_assignment(ov, world):
    mines = gpd.read_file(MINE_PATH).to_crs(CRS_PROJ)

    # Assigned = materials_list is not NaN (any value including "Not relevant")
    # Not assigned = materials_list is NaN
    mines['status'] = mines['materials_list'].apply(
        lambda v: 'Not assigned' if pd.isna(v) else 'Assigned'
    )
    overlap_ids = set(ov['mine_id'].unique())
    mines_in_ov = mines[mines['id'].isin(overlap_ids)].copy()

    col_assigned = '#2196F3'
    col_none     = '#FF7043'
    step         = GRID_KM * 1000

    fig, axes = plt.subplots(1, 2, figsize=(22, 8))

    for ax, gdf, title in [
        (axes[0], mines,       f'All mines  ({len(mines):,})'),
        (axes[1], mines_in_ov, f'Overlapping indig. land  ({len(mines_in_ov):,})'),
    ]:
        grid = _mine_grid(gdf, step, 'status')
        world.to_crs(CRS).plot(ax=ax, color='#f0f0f0', edgecolor='#cccccc', linewidth=0.3)
        for status, color in [('Not assigned', col_none), ('Assigned', col_assigned)]:
            sub = grid[grid['status'] == status]
            if not sub.empty:
                sub.plot(ax=ax, color=color, alpha=0.85, linewidth=0)
        world.to_crs(CRS).boundary.plot(ax=ax, linewidth=0.3, color='#888888')

        n_asgn = (gdf['status'] == 'Assigned').sum()
        n_none = (gdf['status'] == 'Not assigned').sum()
        patches = [
            mpatches.Patch(color=col_assigned, label=f'Assigned  (n={n_asgn:,})'),
            mpatches.Patch(color=col_none,     label=f'Not assigned  (n={n_none:,})'),
        ]
        ax.legend(handles=patches, loc='lower left', fontsize=8, framealpha=0.85)
        ax.set_title(title, fontsize=12)
        ax.set_axis_off()

    fig.suptitle(f'I  -  Mine Assignment Status from materials_list  ({GRID_KM}x{GRID_KM} km grid, majority rule)',
                 fontsize=14, y=1.01)
    plt.tight_layout()
    savefig('I_mine_commodity_assignment.png')


# ── J. Indigenous territories ─────────────────────────────────────────────

def plot_J_indig_territories(world):
    iland = gpd.read_file(INDIG_PATH).to_crs(CRS)

    # flag territories that have at least one mine overlap
    ov_raw       = gpd.read_file(OV_PATH)   # raw overlap (before commodity split)
    overlap_names = set(ov_raw['indig_id'].unique())
    iland['has_overlap'] = iland['Name_'].isin(overlap_names)

    col_overlap  = '#E53935'   # red — has mine overlap
    col_pristine = '#43A047'   # green — no mine overlap

    fig, ax = plt.subplots(figsize=(18, 9))
    world.plot(ax=ax, color='#f5f5f5', edgecolor='#dddddd', linewidth=0.3)

    for flag, color, label in [
        (False, col_pristine, f'No mine overlap  (n={(~iland["has_overlap"]).sum():,})'),
        (True,  col_overlap,  f'Mine overlap     (n={iland["has_overlap"].sum():,})'),
    ]:
        sub = iland[iland['has_overlap'] == flag]
        if not sub.empty:
            sub.plot(ax=ax, color=color, alpha=0.75, linewidth=0.1,
                     edgecolor='white', label=label)

    world.boundary.plot(ax=ax, linewidth=0.3, color='#888888')
    ax.legend(loc='lower left', fontsize=9, framealpha=0.85)
    ax.set_title('J  -  Indigenous Territories: mine overlap status', fontsize=13)
    ax.set_axis_off()
    savefig('J_indig_territories.png')


# ── main ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Loading data ...')
    ov, world = load_data()
    print(f'  {len(ov):,} rows | {ov["ov_id"].nunique():,} unique overlaps '
          f'| {ov["commodity_id"].nunique()} commodities')

    print('A - KDE ...');               plot_A_kde(ov, world)
    print('B - Choropleth ...');        plot_B_choropleth(ov, world)
    print('C - Unique polygons ...');   plot_C_unique_polygons(ov, world)
    print('D - Grid ...');              plot_D_grid(ov, world)
    print('E - Commodity bar ...');     plot_E_commodity_bar(ov)
    print('F - Top territories ...');   plot_F_top_indig(ov)
    print('G - Area distribution ...');  plot_G_area_dist(ov)
    print('H - Continent breakdown ...'); plot_H_continent(ov, world)
    print('I - Mine assignment ...');   plot_I_mine_assignment(ov, world)
    print('J - Indig territories ...'); plot_J_indig_territories(world)

    print('\nDone. All figures saved to', FIG_PATH)

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
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy import stats

OV_PATH  = 'data/interm/over_calc/over_com_buffer_0.gpkg'
FIG_PATH = 'fig/share'

COM_SELECT = ['Lithium', 'Iron', 'Copper', 'Aluminium', 'Gold', 'Silver', 'Zinc',
              'Nickel', 'Cobalt', 'Unknown', 'Coal', 'Rare Earth', 'Spodumene', 'Magnesium']

PALETTE = {
    'Lithium':    '#e6194b', 'Iron':      '#3cb44b', 'Copper':    '#4363d8',
    'Aluminium':  '#f58231', 'Gold':      '#ffe119', 'Silver':    '#aaaaaa',
    'Zinc':       '#42d4f4', 'Nickel':    '#f032e6', 'Cobalt':    '#469990',
    'Unknown':    '#bcbd22', 'Coal':      '#333333', 'Rare Earth':'#8c564b',
    'Spodumene':  '#e377c2', 'Magnesium': '#17becf', 'Other':     '#bbbbbb',
}


def savefig(name):
    path = os.path.join(FIG_PATH, name)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  -> {path}')


def load_overlaps():
    ov = gpd.read_file(OV_PATH)

    # total_area per overlap = sum of alloc_area across all commodity rows
    total = ov.groupby('ov_id')['alloc_area'].sum().rename('total_area')

    # one row per unique overlap
    unique = (ov.drop_duplicates('ov_id')[['ov_id', 'mine_id', 'indig_id']]
                .join(total, on='ov_id'))
    unique = unique[unique['total_area'] > 0].copy()

    # dominant commodity per overlap (highest alloc_area)
    dom = (ov.sort_values('alloc_area', ascending=False)
             .drop_duplicates('ov_id')[['ov_id', 'commodity_id']])
    unique = unique.merge(dom, on='ov_id', how='left')
    unique['com_plot'] = unique['commodity_id'].where(
        unique['commodity_id'].isin(COM_SELECT), 'Other')

    return unique


def print_summary(areas):
    """Print summary statistics to stdout."""
    log_a = np.log10(areas)
    pcts  = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print('\n── Summary: overlap area per unique overlap (km²) ──────────────')
    print(f'  N              : {len(areas):>10,}')
    print(f'  Total          : {areas.sum():>12,.1f} km²')
    print(f'  Mean           : {areas.mean():>12,.3f} km²')
    print(f'  Median         : {areas.median():>12,.3f} km²')
    print(f'  Std dev        : {areas.std():>12,.3f} km²')
    print(f'  Min            : {areas.min():>12,.4f} km²')
    print(f'  Max            : {areas.max():>12,.1f} km²')
    print(f'  Geometric mean : {10**log_a.mean():>12,.3f} km²')
    print(f'  Log10 skewness : {stats.skew(log_a):>12,.3f}')
    print(f'  Log10 kurtosis : {stats.kurtosis(log_a):>12,.3f}')
    print('  Percentiles:')
    for p, v in zip(pcts, np.percentile(areas, pcts)):
        print(f'    p{p:>2}          : {v:>12,.4f} km²')


def plot_overlap_distribution(unique):
    areas    = unique['total_area'].values
    log_a    = np.log10(areas)
    geo_mean = 10 ** log_a.mean()
    median   = np.median(areas)
    p25, p75 = np.percentile(areas, [25, 75])

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ── left: log-scale histogram with KDE ────────────────────────────────
    ax = axes[0]
    bins = np.linspace(log_a.min(), log_a.max(), 50)
    ax.hist(log_a, bins=bins, color='steelblue', alpha=0.75,
            edgecolor='white', linewidth=0.4, density=True, label='Histogram')

    kde_x = np.linspace(log_a.min(), log_a.max(), 400)
    kde   = stats.gaussian_kde(log_a, bw_method='scott')
    ax.plot(kde_x, kde(kde_x), color='#c0392b', linewidth=2.0, label='KDE')

    # reference lines
    for val, lbl, ls in [
        (np.log10(median),   f'Median  {median:.2f} km²',        '--'),
        (np.log10(geo_mean), f'Geom. mean  {geo_mean:.2f} km²',  ':'),
        (np.log10(p25),      f'P25  {p25:.3f} km²',              '-.'),
        (np.log10(p75),      f'P75  {p75:.2f} km²',              '-.'),
    ]:
        ax.axvline(val, color='#555', linewidth=1.2, linestyle=ls, label=lbl)

    ax.set_xlabel('log₁₀  Overlap area  (km²)', fontsize=10)
    ax.set_ylabel('Density', fontsize=10)
    ax.set_title('Distribution of Overlap Area (log₁₀ scale)', fontsize=12)
    ax.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f'{10**v:.3g}'))
    ax.legend(fontsize=8, framealpha=0.85)
    sns.despine(ax=ax)

    # summary stats box
    stats_txt = (
        f'N = {len(areas):,}\n'
        f'Total = {areas.sum():,.0f} km²\n'
        f'Mean = {areas.mean():.2f} km²\n'
        f'Median = {median:.3f} km²\n'
        f'Geo. mean = {geo_mean:.3f} km²\n'
        f'Std = {areas.std():.2f} km²\n'
        f'P5 = {np.percentile(areas, 5):.4f} km²\n'
        f'P95 = {np.percentile(areas, 95):.2f} km²'
    )
    ax.text(0.97, 0.97, stats_txt, transform=ax.transAxes,
            fontsize=7.5, va='top', ha='right',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#cccccc', alpha=0.9))

    # ── right: boxplot by dominant commodity (log scale) ──────────────────
    ax2 = axes[1]
    order  = (unique.groupby('com_plot')['total_area']
                    .median()
                    .sort_values(ascending=False)
                    .index.tolist())
    data   = [np.log10(unique.loc[unique['com_plot'] == c, 'total_area'].values)
              for c in order]
    colors = [PALETTE.get(c, PALETTE['Other']) for c in order]

    bp = ax2.boxplot(data, patch_artist=True, vert=True,
                     flierprops=dict(marker='.', markersize=2.5,
                                     alpha=0.35, markeredgewidth=0),
                     medianprops=dict(color='black', linewidth=1.5),
                     whiskerprops=dict(linewidth=0.8),
                     capprops=dict(linewidth=0.8))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.80)

    ax2.set_xticks(range(1, len(order) + 1))
    ax2.set_xticklabels(order, rotation=35, ha='right', fontsize=8.5)
    ax2.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f'{10**v:.3g}'))
    ax2.set_ylabel('Overlap area  (km²)  [log scale]', fontsize=10)
    ax2.set_title('Overlap Area by Dominant Commodity', fontsize=12)
    sns.despine(ax=ax2)

    plt.suptitle('Mine–Indigenous Land Overlap: Area Distribution', fontsize=13, y=1.01)
    plt.tight_layout()
    savefig('G_overlap_distribution.png')


if __name__ == '__main__':
    print('Loading overlaps ...')
    unique = load_overlaps()
    print(f'  {len(unique):,} unique overlaps  |  '
          f'{unique["com_plot"].nunique()} commodity groups')

    print_summary(unique['total_area'])

    print('\nPlotting ...')
    plot_overlap_distribution(unique)
    print('Done.')

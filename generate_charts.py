import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False

DARK_BG   = '#0f1117'
CARD_BG   = '#1a1d27'
ACCENT1   = '#4f8ef7'  # blue
ACCENT2   = '#f7724f'  # orange
ACCENT3   = '#4fdb8e'  # green
ACCENT4   = '#f7c94f'  # yellow
MUTED     = '#555a6e'
TEXT      = '#e0e4f0'
SUBTEXT   = '#8890a8'

def styled_fig(figsize, title):
    fig = plt.figure(figsize=figsize, facecolor=DARK_BG)
    fig.suptitle(title, color=TEXT, fontsize=15, fontweight='bold', y=0.97)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# CHART 1 — Pipeline de Extração de Dados
# ─────────────────────────────────────────────────────────────────────────────
def chart_extraction():
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=90)

    # Raw odometer deltas (noisy)
    raw_km = np.abs(np.random.normal(180, 80, 90))
    raw_km[10:13] = [2800, 3100, 2950]   # outliers / reset odometer
    raw_km[35:38] = 0                     # stopped period
    raw_km[60] = np.nan                   # missing

    # Cleaned
    clean_km = raw_km.copy()
    clean_km[10:13] = np.random.normal(180, 30, 3)
    clean_km[35:38] = 0
    clean_km = np.clip(clean_km, 0, 400)
    clean_km = pd.Series(clean_km).ffill().values

    # Rolling mean 7d
    roll7 = pd.Series(clean_km).rolling(7, min_periods=1).mean().values

    fig = styled_fig((14, 8), 'Etapa 1 — Extração e Limpeza de Dados por Veículo')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.08, right=0.97, top=0.90, bottom=0.08)

    # --- Raw data ---
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.set_facecolor(CARD_BG)
    ax0.plot(dates, raw_km, color=ACCENT2, lw=0.9, alpha=0.8, label='km_dia_raw')
    ax0.axhspan(400, 3200, color=ACCENT2, alpha=0.08)
    ax0.annotate('resets/outliers\ndetectados', xy=(dates[11], 3000),
                 xytext=(dates[20], 2500), color=ACCENT2, fontsize=8,
                 arrowprops=dict(arrowstyle='->', color=ACCENT2, lw=0.8))
    ax0.set_title('Dado Bruto (km/dia por odômetro)', color=TEXT, fontsize=9)
    ax0.tick_params(colors=SUBTEXT, labelsize=7)
    ax0.set_ylabel('km', color=SUBTEXT, fontsize=8)
    for s in ax0.spines.values(): s.set_color(MUTED)

    # --- Cleaned data ---
    ax1 = fig.add_subplot(gs[0, 1])
    ax1.set_facecolor(CARD_BG)
    ax1.fill_between(dates, clean_km, alpha=0.2, color=ACCENT3)
    ax1.plot(dates, clean_km, color=ACCENT3, lw=1.2, label='km_dia_clean')
    ax1.plot(dates, roll7, color=ACCENT1, lw=1.8, ls='--', label='média 7d')
    ax1.axvspan(dates[35], dates[37], color=ACCENT4, alpha=0.15, label='período parado')
    ax1.set_title('Dado Limpo + Média Móvel 7d', color=TEXT, fontsize=9)
    ax1.legend(fontsize=7, labelcolor=TEXT, facecolor=CARD_BG, edgecolor=MUTED)
    ax1.tick_params(colors=SUBTEXT, labelsize=7)
    ax1.set_ylabel('km', color=SUBTEXT, fontsize=8)
    for s in ax1.spines.values(): s.set_color(MUTED)

    # --- Accumulated KM ---
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor(CARD_BG)
    accum = np.cumsum(clean_km)
    ax2.plot(dates, accum, color=ACCENT1, lw=2)
    ax2.fill_between(dates, accum, alpha=0.15, color=ACCENT1)
    target = 12000
    idx_reach = np.argmax(accum >= target)
    ax2.axhline(target, color=ACCENT4, ls='--', lw=1.2, label=f'meta: {target:,.0f} km')
    ax2.axvline(dates[idx_reach], color=ACCENT4, ls=':', lw=1)
    ax2.annotate(f'atingido em\n{dates[idx_reach].strftime("%d/%m/%Y")}',
                 xy=(dates[idx_reach], target),
                 xytext=(dates[idx_reach-20], target*0.7),
                 color=ACCENT4, fontsize=8,
                 arrowprops=dict(arrowstyle='->', color=ACCENT4, lw=0.8))
    ax2.set_title('KM Acumulado', color=TEXT, fontsize=9)
    ax2.legend(fontsize=7, labelcolor=TEXT, facecolor=CARD_BG, edgecolor=MUTED)
    ax2.tick_params(colors=SUBTEXT, labelsize=7)
    ax2.set_ylabel('km total', color=SUBTEXT, fontsize=8)
    for s in ax2.spines.values(): s.set_color(MUTED)

    # --- Features por dia da semana ---
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor(CARD_BG)
    days = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
    means = [195, 210, 205, 198, 185, 90, 40]
    stds  = [45, 50, 48, 42, 55, 30, 20]
    x = np.arange(7)
    bars = ax3.bar(x, means, color=ACCENT1, alpha=0.7, width=0.55)
    ax3.errorbar(x, means, yerr=stds, fmt='none', color=TEXT, capsize=4, lw=1.2)
    ax3.set_xticks(x)
    ax3.set_xticklabels(days, color=SUBTEXT, fontsize=8)
    ax3.set_title('Perfil por Dia da Semana (km_dia_mean)', color=TEXT, fontsize=9)
    ax3.tick_params(colors=SUBTEXT, labelsize=7)
    ax3.set_ylabel('km médio', color=SUBTEXT, fontsize=8)
    for s in ax3.spines.values(): s.set_color(MUTED)

    plt.savefig('/home/user/apimovias/chart_01_extracao.png', dpi=150, bbox_inches='tight',
                facecolor=DARK_BG)
    plt.close()
    print('chart_01_extracao.png saved')

# ─────────────────────────────────────────────────────────────────────────────
# CHART 2 — Perfis de Veículos (features + segmentação)
# ─────────────────────────────────────────────────────────────────────────────
def chart_profiles():
    np.random.seed(7)
    n = 80

    # Simulated fleet
    segments = np.random.choice([0, 1, 2], n, p=[0.4, 0.35, 0.25])
    per_day  = [np.random.normal(v, s, n) for v, s in
                [(50, 15), (180, 40), (320, 60)]]
    km_per_day = np.array([per_day[seg][i] for i, seg in enumerate(segments)])
    km_per_day = np.abs(km_per_day)
    active_rate_by_seg = [np.random.uniform(0.4, 0.7, n),
                          np.random.uniform(0.7, 0.9, n),
                          np.random.uniform(0.85, 1.0, n)]
    active_rate = np.array([active_rate_by_seg[seg][i] for i, seg in enumerate(segments)])

    fig = styled_fig((14, 9), 'Etapa 3/4 — Perfis e Segmentação de Veículos')
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38,
                           left=0.07, right=0.97, top=0.90, bottom=0.08)

    colors_seg = [ACCENT1, ACCENT3, ACCENT2]
    labels_seg = ['Segmento 0\n(baixo uso)', 'Segmento 1\n(uso médio)', 'Segmento 2\n(alto uso)']

    # --- Scatter: km_per_day vs active_rate ---
    ax0 = fig.add_subplot(gs[0, :2])
    ax0.set_facecolor(CARD_BG)
    for seg in [0, 1, 2]:
        mask = segments == seg
        ax0.scatter(km_per_day[mask], active_rate[mask],
                    c=colors_seg[seg], label=labels_seg[seg],
                    alpha=0.75, s=55, edgecolors='none')
    ax0.set_xlabel('km / dia (média)', color=SUBTEXT, fontsize=9)
    ax0.set_ylabel('taxa dias ativos', color=SUBTEXT, fontsize=9)
    ax0.set_title('Segmentação da Frota (KMeans sobre 15 features)', color=TEXT, fontsize=9)
    ax0.legend(fontsize=8, labelcolor=TEXT, facecolor=CARD_BG, edgecolor=MUTED, loc='upper left')
    ax0.tick_params(colors=SUBTEXT, labelsize=7)
    for s in ax0.spines.values(): s.set_color(MUTED)

    # --- Pie: category distribution ---
    ax1 = fig.add_subplot(gs[0, 2])
    ax1.set_facecolor(CARD_BG)
    cats = [53, 27]
    ax1.pie(cats, labels=['KM (53%)', 'H (27%)'],
            colors=[ACCENT1, ACCENT4], startangle=90,
            wedgeprops={'edgecolor': DARK_BG, 'linewidth': 2},
            textprops={'color': TEXT, 'fontsize': 9})
    ax1.set_title('Distribuição por Categoria', color=TEXT, fontsize=9)

    # --- Heatmap: weekday features ---
    ax2 = fig.add_subplot(gs[1, :2])
    ax2.set_facecolor(CARD_BG)
    days = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
    features_wk = ['mean', 'std', 'median', 'prob_active', 'cv']
    data_hm = np.array([
        [195, 210, 205, 198, 185, 90, 40],   # mean
        [45, 50, 48, 42, 55, 30, 20],         # std
        [185, 200, 195, 190, 175, 80, 30],    # median
        [0.88, 0.90, 0.89, 0.87, 0.82, 0.45, 0.20],  # prob_active
        [0.23, 0.24, 0.23, 0.21, 0.30, 0.33, 0.50],  # cv
    ])
    # normalize rows
    data_norm = (data_hm - data_hm.min(axis=1, keepdims=True)) / \
                (data_hm.max(axis=1, keepdims=True) - data_hm.min(axis=1, keepdims=True) + 1e-9)
    im = ax2.imshow(data_norm, aspect='auto', cmap='Blues', vmin=0, vmax=1)
    ax2.set_xticks(range(7)); ax2.set_xticklabels(days, color=SUBTEXT, fontsize=8)
    ax2.set_yticks(range(5)); ax2.set_yticklabels(features_wk, color=SUBTEXT, fontsize=8)
    ax2.set_title('Heatmap — 70 Features por Dia da Semana (veículo #1316)', color=TEXT, fontsize=9)
    for r in range(5):
        for c in range(7):
            ax2.text(c, r, f'{data_hm[r,c]:.2f}', ha='center', va='center',
                     color=TEXT if data_norm[r,c] < 0.7 else DARK_BG, fontsize=7)
    for s in ax2.spines.values(): s.set_color(MUTED)

    # --- Bar: segment sizes ---
    ax3 = fig.add_subplot(gs[1, 2])
    ax3.set_facecolor(CARD_BG)
    seg_counts = [np.sum(segments == i) for i in range(3)]
    ax3.barh(['Seg 0', 'Seg 1', 'Seg 2'], seg_counts,
             color=colors_seg, alpha=0.8, height=0.5)
    for i, v in enumerate(seg_counts):
        ax3.text(v+0.5, i, str(v), va='center', color=TEXT, fontsize=9)
    ax3.set_xlabel('veículos', color=SUBTEXT, fontsize=8)
    ax3.set_title('Veículos por Segmento', color=TEXT, fontsize=9)
    ax3.tick_params(colors=SUBTEXT, labelsize=8)
    for s in ax3.spines.values(): s.set_color(MUTED)

    plt.savefig('/home/user/apimovias/chart_02_perfis.png', dpi=150, bbox_inches='tight',
                facecolor=DARK_BG)
    plt.close()
    print('chart_02_perfis.png saved')

# ─────────────────────────────────────────────────────────────────────────────
# CHART 3 — Predições
# ─────────────────────────────────────────────────────────────────────────────
def chart_predictions():
    np.random.seed(99)

    fig = styled_fig((15, 9), 'Etapa 5 — Engine de Predições')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35,
                           left=0.08, right=0.97, top=0.90, bottom=0.08)

    # ---- date-to-reach: histórico + predição + intervalo ----
    ax0 = fig.add_subplot(gs[0, :])
    ax0.set_facecolor(CARD_BG)

    hist_days = 120
    future_days = 90
    all_days = hist_days + future_days
    base = datetime(2024, 1, 1)
    dates_hist = [base + timedelta(days=i) for i in range(hist_days)]
    dates_fut  = [base + timedelta(days=hist_days+i) for i in range(future_days)]

    km_hist = np.abs(np.random.normal(185, 40, hist_days))
    km_hist[50:54] = 0
    km_hist = np.clip(km_hist, 0, 380)
    accum_hist = np.cumsum(km_hist)

    trend = np.mean(km_hist[-30:])
    noise = np.random.normal(0, 15, future_days)
    km_pred = np.clip(trend + noise + np.linspace(0, 5, future_days), 0, 400)
    accum_pred_base = accum_hist[-1] + np.cumsum(km_pred)
    upper_factor = np.random.uniform(1.05, 1.20, future_days).cumprod()
    lower_factor = np.random.uniform(0.80, 0.95, future_days).cumprod()
    accum_upper = accum_hist[-1] + np.cumsum(km_pred * upper_factor)
    accum_lower = accum_hist[-1] + np.cumsum(km_pred * lower_factor)

    target1 = 28000
    idx_reach = np.argmax(accum_pred_base >= target1)
    target_date = dates_fut[idx_reach]

    ax0.plot(dates_hist, accum_hist, color=ACCENT1, lw=2, label='histórico acumulado')
    ax0.plot(dates_fut, accum_pred_base, color=ACCENT3, lw=2, ls='--', label='predição (ensemble)')
    ax0.fill_between(dates_fut, accum_lower, accum_upper, color=ACCENT3, alpha=0.15,
                     label='intervalo de confiança')
    ax0.axhline(target1, color=ACCENT4, lw=1.5, ls=':', label=f'meta: {target1:,} km')
    ax0.axvline(target_date, color=ACCENT4, lw=1.2, ls=':')
    ax0.annotate(f'data prevista:\n{target_date.strftime("%d/%m/%Y")}\n({idx_reach} dias)',
                 xy=(target_date, target1),
                 xytext=(dates_fut[idx_reach+10], target1 * 0.88),
                 color=ACCENT4, fontsize=9,
                 arrowprops=dict(arrowstyle='->', color=ACCENT4))
    ax0.axvline(base + timedelta(days=hist_days), color=MUTED, lw=1.5, ls='--', alpha=0.8)
    ax0.text(base + timedelta(days=hist_days+1), accum_hist[-1]*0.97,
             'hoje →', color=MUTED, fontsize=8)
    ax0.set_title('date-to-reach — Veículo #1316 (KM, Segmento 2)', color=TEXT, fontsize=9)
    ax0.legend(fontsize=8, labelcolor=TEXT, facecolor=CARD_BG, edgecolor=MUTED)
    ax0.tick_params(colors=SUBTEXT, labelsize=7)
    ax0.set_ylabel('km acumulado', color=SUBTEXT, fontsize=8)
    for s in ax0.spines.values(): s.set_color(MUTED)

    # ---- accumulated-at-step: múltiplos veículos ----
    ax1 = fig.add_subplot(gs[1, 0])
    ax1.set_facecolor(CARD_BG)
    vehicles = ['#1316', '#2201', '#3045', '#4102', '#5887']
    steps = [30, 60, 90]
    bar_width = 0.18
    x = np.arange(len(vehicles))
    colors_step = [ACCENT1, ACCENT3, ACCENT2]
    for i, (step, color) in enumerate(zip(steps, colors_step)):
        vals = np.random.normal([5500, 4200, 7800, 3100, 6400], 300)
        vals = vals * (1 + i * 0.35)
        ax1.bar(x + i*bar_width - bar_width, vals, bar_width*0.9,
                label=f'+{step} dias', color=color, alpha=0.8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(vehicles, color=SUBTEXT, fontsize=8)
    ax1.set_title('accumulated-at-step\n(km acumulado por veículo)', color=TEXT, fontsize=9)
    ax1.legend(fontsize=7, labelcolor=TEXT, facecolor=CARD_BG, edgecolor=MUTED)
    ax1.tick_params(colors=SUBTEXT, labelsize=7)
    ax1.set_ylabel('km acumulado previsto', color=SUBTEXT, fontsize=8)
    for s in ax1.spines.values(): s.set_color(MUTED)

    # ---- model comparison ----
    ax2 = fig.add_subplot(gs[1, 1])
    ax2.set_facecolor(CARD_BG)
    models = ['DLinear', 'NLinear', 'NBEATS', 'NHiTS', 'LightGBM', 'XGBoost']
    maes   = [312, 298, 275, 285, 260, 268]
    colors_m = [ACCENT1, ACCENT1, ACCENT3, ACCENT3, ACCENT2, ACCENT2]
    bars = ax2.barh(models, maes, color=colors_m, alpha=0.75, height=0.55)
    for bar, val in zip(bars, maes):
        ax2.text(val+3, bar.get_y() + bar.get_height()/2,
                 f'{val}', va='center', color=TEXT, fontsize=8)
    ax2.set_xlabel('MAE (km/dia)', color=SUBTEXT, fontsize=8)
    ax2.set_title('Comparação de Modelos\n(erro médio absoluto)', color=TEXT, fontsize=9)
    ax2.tick_params(colors=SUBTEXT, labelsize=8)
    # best model marker
    best_idx = np.argmin(maes)
    ax2.get_children()[best_idx].set_edgecolor(ACCENT4)
    ax2.get_children()[best_idx].set_linewidth(2)
    ax2.text(maes[best_idx]+3, best_idx + 0.3, '★ melhor', color=ACCENT4, fontsize=7)
    for s in ax2.spines.values(): s.set_color(MUTED)

    plt.savefig('/home/user/apimovias/chart_03_predicoes.png', dpi=150, bbox_inches='tight',
                facecolor=DARK_BG)
    plt.close()
    print('chart_03_predicoes.png saved')

# ─────────────────────────────────────────────────────────────────────────────
# CHART 4 — Pipeline end-to-end (visual flow)
# ─────────────────────────────────────────────────────────────────────────────
def chart_pipeline():
    fig = styled_fig((16, 6), 'Arquitetura End-to-End — Movias')
    ax = fig.add_subplot(111)
    ax.set_facecolor(DARK_BG)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 6)
    ax.axis('off')

    steps = [
        (1.0,  3.0, 'PostgreSQL\n(viagens brutas)',       ACCENT2,  '①'),
        (3.8,  3.0, 'Extractor\nSQL (14 CTEs)',            ACCENT4,  '②'),
        (6.6,  3.0, 'CSV\n(27 colunas)',                   MUTED,    '③'),
        (9.4,  3.0, 'AI Service\nImport/Perfis',           ACCENT1,  '④'),
        (12.2, 3.0, 'SQLite\n85 features',                 ACCENT3,  '⑤'),
        (14.8, 3.0, 'Predições\nAPI',                      ACCENT3,  '⑥'),
    ]

    box_w, box_h = 1.8, 1.2
    for (x, y, label, color, num) in steps:
        rect = mpatches.FancyBboxPatch((x - box_w/2, y - box_h/2),
                                        box_w, box_h,
                                        boxstyle='round,pad=0.08',
                                        facecolor=color, alpha=0.18,
                                        edgecolor=color, linewidth=2)
        ax.add_patch(rect)
        ax.text(x, y + 0.15, num, ha='center', va='center',
                color=color, fontsize=14, fontweight='bold')
        ax.text(x, y - 0.28, label, ha='center', va='center',
                color=TEXT, fontsize=8, linespacing=1.4)

    # Arrows
    arrow_xs = [(1.9, 2.9), (4.7, 5.7), (7.5, 8.5), (10.3, 11.3), (13.1, 14.1)]
    arrow_labels = ['query SQL', 'limpeza\n+ features', 'import\n+ classify', 'KMeans\nsegment', 'inference\nML']
    for (x0, x1), lbl in zip(arrow_xs, arrow_labels):
        ax.annotate('', xy=(x1, 3.0), xytext=(x0, 3.0),
                    arrowprops=dict(arrowstyle='->', color=MUTED, lw=2.0))
        ax.text((x0+x1)/2, 3.6, lbl, ha='center', va='center',
                color=SUBTEXT, fontsize=7, linespacing=1.3)

    # Sub-labels below
    sublabels = [
        (1.0,  1.8, 'trips.alltrips\northogonal DB'),
        (3.8,  1.8, 'paralelo\n+ atômico CSV'),
        (6.6,  1.8, 'volume\ncompartilhado'),
        (9.4,  1.8, 'KM / H\n+ segmento'),
        (12.2, 1.8, '15 summary\n70 weekday'),
        (14.8, 1.8, 'NBEATS / LightGBM\n+ ensemble'),
    ]
    for (x, y, lbl) in sublabels:
        ax.text(x, y, lbl, ha='center', va='center', color=SUBTEXT,
                fontsize=7, linespacing=1.4,
                bbox=dict(facecolor=CARD_BG, edgecolor=MUTED, boxstyle='round,pad=0.3', alpha=0.6))

    # Title tags on top
    ax.text(8.0, 5.2, 'Gateway :8080  ←→  AI Service :8000  ←→  Extractor :8000',
            ha='center', va='center', color=SUBTEXT, fontsize=9,
            bbox=dict(facecolor=CARD_BG, edgecolor=MUTED, boxstyle='round,pad=0.4', alpha=0.8))

    plt.savefig('/home/user/apimovias/chart_04_pipeline.png', dpi=150, bbox_inches='tight',
                facecolor=DARK_BG)
    plt.close()
    print('chart_04_pipeline.png saved')

# Run all
chart_extraction()
chart_profiles()
chart_predictions()
chart_pipeline()
print('\nAll charts generated!')

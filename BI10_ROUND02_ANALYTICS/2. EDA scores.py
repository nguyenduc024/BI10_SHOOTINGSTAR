# %% [markdown]
# # Phase 2 — EDA and two scores
# Input: `data_clean.parquet`. Run top-to-bottom. Tasks 1–2 only.
# %%
from datetime import datetime, date
from pathlib import Path
import hashlib
import json
import sys

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / 'data_clean.parquet'
TABLES, FIGURES, OUTPUTS, REPORTS = (ROOT / name for name in ('tables', 'figures', 'outputs', 'reports'))
for folder in (TABLES, OUTPUTS, REPORTS, FIGURES / 'eda', FIGURES / 'financial_health', FIGURES / 'engagement'):
    folder.mkdir(parents=True, exist_ok=True)
CUTOFF = datetime(2025, 6, 5)  # exclusive
REFERENCE_DATE = date(2025, 6, 4)
with SOURCE.open('rb') as handle:
    source_hash = hashlib.file_digest(handle, 'sha256').hexdigest()

# One visual language: ink = population, blue = finance/calls, teal =
# engagement/behavior, amber = leads/uncertainty, coral = adverse signal.
NAVY, BLUE, TEAL = '#14243A', '#3478C0', '#168E84'
AMBER, CORAL, SLATE, BG = '#BB8125', '#BD5148', '#596B7E', '#F7F9FC'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 12,
    'axes.titlesize': 19, 'axes.titleweight': 'semibold',
    'axes.labelcolor': NAVY, 'text.color': NAVY,
    'xtick.color': SLATE, 'ytick.color': SLATE,
    'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.spines.left': False, 'axes.spines.bottom': False,
    'grid.color': '#DBE4EC', 'grid.linewidth': .7})

# %% [markdown]
# ## Input check
# Confirm customer grain and cutoff. Contract status has no dated history.
# %%
scan = pl.scan_parquet(SOURCE)
schema = scan.collect_schema()
rows, unique, missing_id = scan.select(pl.len().alias('rows'),
    pl.col('SKP_CLIENT').n_unique().alias('unique'),
    pl.col('SKP_CLIENT').null_count().alias('missing_id')).collect().row(0)
if rows == 0 or rows != unique or missing_id:
    raise ValueError('Parquet must contain one non-null row per customer')
missing = scan.select(pl.all().null_count()).collect().row(0)
pl.DataFrame({'column': schema.names(), 'dtype': [str(t) for t in schema.dtypes()],
    'null_count': missing}).with_columns((pl.col('null_count') / rows).alias('null_share')).write_csv(TABLES / 'column_missingness.csv')
indicators = ['LATEST_INCOME_MAIN', 'FIRST_DUE_TOTAL', 'SECOND_DUE_TOTAL', 'THIRD_DUE_TOTAL',
    'FOURTH_DUE_TOTAL', 'CONTRACT_WRITTEN_OFF_COUNT', 'CALL_COUNT_90D_ASOF_2025_06_04',
    'BEHAVIOR_EVENTS_90D_ASOF_2025_06_04', 'LEAD_COUNT_90D_ASOF_2025_06_04']
distributions = []
for name in indicators:
    distributions.append({'column': name, **scan.select(pl.col(name).min().alias('minimum'),
        pl.col(name).median().alias('median'), pl.col(name).quantile(.90).alias('p90'),
        pl.col(name).quantile(.99).alias('p99'), pl.col(name).max().alias('maximum'),
        (pl.col(name) == 0).sum().alias('zero_count'),
        pl.col(name).null_count().alias('null_count')).collect().to_dicts()[0]})
pl.DataFrame(distributions).write_csv(TABLES / 'indicator_distribution.csv')
date_fields = ['LAST_CONTRACT_APPLICATION_AT', 'LAST_CONTRACT_SIGNED_AT', 'LAST_CALL_AT',
    'LAST_CALL_ASOF_2025_06_04', 'LAST_LEAD_AT', 'LAST_BEHAVIOR_AT']
date_ranges = {name: [str(v) if v is not None else None for v in
    scan.select(pl.col(name).min().alias('earliest'),
                pl.col(name).max().alias('latest')).collect().row(0)] for name in date_fields}
if any(scan.select(pl.col(name).max()).collect().item() >= CUTOFF
       for name in ('LAST_CALL_ASOF_2025_06_04', 'LAST_LEAD_AT', 'LAST_BEHAVIOR_AT')):
    raise ValueError('Activity feature exceeds the June-2025 cutoff')
inspection = {'rows': rows, 'unique_clients': unique, 'columns': len(schema),
    'source_file': SOURCE.name, 'date_ranges': date_ranges, 'cutoff_exclusive': str(CUTOFF)}
(TABLES / 'inspection.json').write_text(json.dumps(inspection, indent=2), encoding='utf-8')
print(f'{rows:,} unique customers; {len(schema)} fields. Source SHA-256: {source_hash}')

# %% [markdown]
# ## EDA
# Measure coverage, sparse financial signals and 90-day activity.
# %%
fields = ['SKP_CLIENT', 'CONTRACT_CASE_COUNT', 'CONTRACT_ACTIVE_COUNT', 'CONTRACT_WRITTEN_OFF_COUNT',
    'CONTRACT_INCOME_MAIN_CONFLICT_CASE_COUNT', 'LATEST_INCOME_MAIN',
    'FIRST_DUE_TOTAL', 'SECOND_DUE_TOTAL', 'THIRD_DUE_TOTAL', 'FOURTH_DUE_TOTAL',
    'CALL_COUNT_ASOF_2025_06_04', 'CALL_COUNT_90D_ASOF_2025_06_04',
    'CALL_INBOUND_ASOF_2025_06_04', 'LAST_CALL_ASOF_2025_06_04',
    'LEAD_COUNT', 'LEAD_COUNT_90D_ASOF_2025_06_04', 'LAST_LEAD_AT',
    'BEHAVIOR_EVENT_COUNT', 'BEHAVIOR_EVENTS_90D_ASOF_2025_06_04',
    'BEHAVIOR_PRODUCTS_90D_ASOF_2025_06_04', 'LAST_BEHAVIOR_AT']
df = scan.select(fields).with_columns(
    (pl.col('FIRST_DUE_TOTAL').fill_null(0) + 2 * pl.col('SECOND_DUE_TOTAL').fill_null(0)
     + 3 * pl.col('THIRD_DUE_TOTAL').fill_null(0) + 4 * pl.col('FOURTH_DUE_TOTAL').fill_null(0)).alias('due_weighted'),
    ((pl.col('LATEST_INCOME_MAIN') > 0) & pl.col('LATEST_INCOME_MAIN').is_finite())
        .fill_null(False).alias('income_usable'),
    pl.max_horizontal('LAST_CALL_ASOF_2025_06_04', 'LAST_LEAD_AT', 'LAST_BEHAVIOR_AT')
        .alias('last_customer_activity_at')).with_columns(
    (pl.lit(REFERENCE_DATE) - pl.col('last_customer_activity_at').dt.date())
        .dt.total_days().clip(lower_bound=0).alias('days_since_activity')).collect()
eda = df.select(pl.len().alias('customers'),
    (pl.col('CONTRACT_ACTIVE_COUNT') > 0).sum().alias('customers_with_active_contract'),
    (pl.col('due_weighted') > 0).sum().alias('customers_with_overdue'),
    (pl.col('CONTRACT_WRITTEN_OFF_COUNT') > 0).sum().alias('customers_with_writeoff'),
    pl.col('income_usable').sum().alias('customers_with_unambiguous_income'),
    (pl.col('CONTRACT_INCOME_MAIN_CONFLICT_CASE_COUNT') > 0).sum().alias('customers_with_case_income_conflict'),
    (pl.col('CALL_COUNT_ASOF_2025_06_04') > 0).sum().alias('customers_with_pre_cutoff_call'),
    (pl.col('LEAD_COUNT') > 0).sum().alias('customers_with_lead'),
    (pl.col('BEHAVIOR_EVENT_COUNT') > 0).sum().alias('customers_with_behavior'),
    (pl.col('CALL_COUNT_90D_ASOF_2025_06_04') > 0).sum().alias('recent_calls'),
    (pl.col('LEAD_COUNT_90D_ASOF_2025_06_04') > 0).sum().alias('recent_leads'),
    (pl.col('BEHAVIOR_EVENTS_90D_ASOF_2025_06_04') > 0).sum().alias('recent_behavior'),
    pl.col('CALL_COUNT_90D_ASOF_2025_06_04').sum().alias('recent_call_events'),
    pl.col('LEAD_COUNT_90D_ASOF_2025_06_04').sum().alias('recent_lead_events'),
    pl.col('BEHAVIOR_EVENTS_90D_ASOF_2025_06_04').sum().alias('recent_behavior_events'),
    pl.col('days_since_activity').null_count().alias('no_observed_activity')).to_dicts()[0]
eda['correlations'] = {}
for x, y, name in [('LATEST_INCOME_MAIN', 'due_weighted', 'income_vs_overdue'),
    ('CALL_COUNT_90D_ASOF_2025_06_04', 'BEHAVIOR_EVENTS_90D_ASOF_2025_06_04', 'recent_calls_vs_behavior'),
    ('BEHAVIOR_PRODUCTS_90D_ASOF_2025_06_04', 'BEHAVIOR_EVENTS_90D_ASOF_2025_06_04', 'products_vs_behavior')]:
    subset = df.select(x, y).drop_nulls()
    eda['correlations'][name] = {'spearman_r': subset.select(pl.corr(x, y, method='spearman')).item(),
                                  'n': subset.height}
(TABLES / 'eda_summary.json').write_text(json.dumps(eda, indent=2), encoding='utf-8')
focus = [('Any overdue history', 'customers_with_overdue'), ('Any written-off case', 'customers_with_writeoff'),
    ('Any active case', 'customers_with_active_contract'), ('Usable latest income', 'customers_with_unambiguous_income'),
    ('Income conflict in any case', 'customers_with_case_income_conflict'), ('90d calls', 'recent_calls'),
    ('90d behavior', 'recent_behavior'), ('90d leads', 'recent_leads'), ('No observed activity', 'no_observed_activity')]
pl.DataFrame([{'indicator': label, 'customer_count': eda[key], 'percent': round(100 * eda[key] / rows, 2)}
    for label, key in focus]).write_csv(TABLES / 'eda_focus.csv')

# %% [markdown]
# ## Task 1 — Financial Health
# Overdue 45%, write-off 25%, income 30%. Descriptive, not default risk.
# %%
def scale(column, low, high):
    return (((pl.col(column) - low) / (high - low)) * 100).clip(0, 100) if high > low else pl.lit(50.0)

positive = df.filter(pl.col('income_usable')).select(pl.col('LATEST_INCOME_MAIN').log1p().alias('log_income'))
income_lo, income_hi = positive.select(pl.col('log_income').quantile(.05).alias('lo'),
    pl.col('log_income').quantile(.95).alias('hi')).row(0)
due_hi = df.filter(pl.col('due_weighted') > 0).select(pl.col('due_weighted').log1p().quantile(.99)).item() or 1.0
df = df.with_columns(pl.when(pl.col('income_usable')).then(pl.col('LATEST_INCOME_MAIN').log1p())
    .otherwise(None).alias('log_income')).with_columns(
    (100 * (1 - (pl.col('due_weighted').log1p() / due_hi).clip(0, 1))).alias('health_overdue_component'),
    pl.when(pl.col('CONTRACT_WRITTEN_OFF_COUNT') > 0).then(0.0).otherwise(100.0)
        .alias('health_written_off_component'),
    pl.when(pl.col('income_usable')).then(scale('log_income', income_lo, income_hi))
        .otherwise(50.0).alias('health_income_component'))
h, w, i = (pl.col(name) for name in ('health_overdue_component',
    'health_written_off_component', 'health_income_component'))
df = df.with_columns(
    (h * .45 + w * .25 + i * .30).round(2).clip(0, 100).alias('financial_health_score'),
    pl.when(pl.col('income_usable')).then(1.0).otherwise(.7).alias('financial_evidence_weight'),
    (h * .55 + w * .15 + i * .30).round(2).alias('financial_weight_sensitivity'),
    (h * .45 + w * .25 + pl.when(pl.col('income_usable')).then(i).otherwise(0.0) * .30)
        .round(2).alias('financial_missing_income_low_bound'),
    (h * .45 + w * .25 + pl.when(pl.col('income_usable')).then(i).otherwise(100.0) * .30)
        .round(2).alias('financial_missing_income_high_bound'))
financial_lo, financial_hi = df.filter(pl.col('income_usable')).select(
    pl.col('financial_health_score').quantile(.25).alias('lo'),
    pl.col('financial_health_score').quantile(.75).alias('hi')).row(0)
if not financial_lo < financial_hi:
    raise ValueError('Financial score quantiles are degenerate')
df = df.with_columns(pl.when(~pl.col('income_usable')).then(pl.lit('Evidence-limited'))
    .when(pl.col('financial_health_score') < financial_lo).then(pl.lit('Lower observed signals'))
    .when(pl.col('financial_health_score') < financial_hi).then(pl.lit('Middle observed signals'))
    .otherwise(pl.lit('Higher observed signals')).alias('financial_health_group'))
financial_params = {'weights': {'overdue': .45, 'written_off': .25, 'income': .30},
    'income_log_p05': income_lo, 'income_log_p95': income_hi,
    'due_log_positive_p99': due_hi, 'group_p25': financial_lo, 'group_p75': financial_hi,
    'missing_income_policy': 'neutral 50, evidence-limited group, 0-100 income-component bounds'}

# %% [markdown]
# ## Task 2 — Engagement
# Pre-cutoff activity: recency 45%, frequency 30%, other signals 25%.
# %%
behavior, calls, inbound, leads, products = (
    'BEHAVIOR_EVENTS_90D_ASOF_2025_06_04', 'CALL_COUNT_90D_ASOF_2025_06_04',
    'CALL_INBOUND_ASOF_2025_06_04', 'LEAD_COUNT_90D_ASOF_2025_06_04',
    'BEHAVIOR_PRODUCTS_90D_ASOF_2025_06_04')
caps = {}
for name in (behavior, calls, inbound, leads, products):
    value = df.filter(pl.col(name) > 0).select(pl.col(name).log1p().quantile(.95)).item()
    caps[name] = float(value) if value is not None else 0.0

def capped_log(name):
    return (pl.col(name).clip(lower_bound=0).log1p() / caps[name]).clip(0, 1) * 100 if caps[name] > 0 else pl.lit(0.0)

df = df.with_columns(
    pl.when(pl.col('days_since_activity').is_null()).then(0.0)
        .otherwise(100 * (-pl.col('days_since_activity') / 90).exp()).alias('engagement_recency_component'),
    capped_log(behavior).alias('_behavior_scaled'), capped_log(calls).alias('_calls_scaled'),
    capped_log(inbound).alias('engagement_interaction_component'),
    capped_log(leads).alias('engagement_lead_component'),
    capped_log(products).alias('engagement_product_component'))
df = df.with_columns((.75 * pl.col('_behavior_scaled') + .25 * pl.col('_calls_scaled'))
    .alias('engagement_frequency_component'))
r, f, q, l, p = (pl.col(name) for name in ('engagement_recency_component',
    'engagement_frequency_component', 'engagement_interaction_component',
    'engagement_lead_component', 'engagement_product_component'))
df = df.with_columns(
    (r * .45 + f * .30 + q * .10 + l * .10 + p * .05).round(2).clip(0, 100).alias('engagement_score'),
    (r * .35 + f * .40 + q * .10 + l * .10 + p * .05).round(2).alias('engagement_weight_sensitivity'),
    (r * .45 + pl.col('_behavior_scaled') * .30 + q * .10 + l * .10 + p * .05)
        .round(2).alias('engagement_no_call_proxy_sensitivity'))
engagement_lo, engagement_hi = df.filter(pl.col('engagement_score') > 0).select(
    pl.col('engagement_score').quantile(1 / 3).alias('lo'),
    pl.col('engagement_score').quantile(2 / 3).alias('hi')).row(0)
if not engagement_lo < engagement_hi:
    raise ValueError('Engagement score quantiles are degenerate')
df = df.with_columns(pl.when(pl.col('days_since_activity').is_null()).then(pl.lit('No observed activity'))
    .when(pl.col('engagement_score') == 0).then(pl.lit('Dormant observed history'))
    .when(pl.col('engagement_score') < engagement_lo).then(pl.lit('Lower observed engagement'))
    .when(pl.col('engagement_score') < engagement_hi).then(pl.lit('Middle observed engagement'))
    .otherwise(pl.lit('Higher observed engagement')).alias('engagement_group'))
engagement_params = {'weights': {'recency': .45, 'frequency': .30, 'interaction': .10,
    'leads': .10, 'products': .05}, 'positive_log_p95_caps': caps,
    'recency_decay_days': 90, 'active_tertile_1': engagement_lo,
    'active_tertile_2': engagement_hi, 'cutoff_exclusive': str(CUTOFF)}

# %% [markdown]
# ## QA
# Check bounds, groups and sensitivity. No outcome validation is possible.
# %%
if df.height != rows or df['SKP_CLIENT'].n_unique() != rows:
    raise ValueError('Customer keys changed during scoring')
validation = {'population': rows, 'independent_financial_outcome_available': False,
    'independent_engagement_outcome_available': False, 'scores': {}}
for name, score, alternative, group, components, weights in (
    ('financial', 'financial_health_score', 'financial_weight_sensitivity', 'financial_health_group',
     ['health_overdue_component', 'health_written_off_component', 'health_income_component'], financial_params['weights']),
    ('engagement', 'engagement_score', 'engagement_weight_sensitivity', 'engagement_group',
     ['engagement_recency_component', 'engagement_frequency_component', 'engagement_interaction_component',
      'engagement_lead_component', 'engagement_product_component'], engagement_params['weights'])):
    if df.select(pl.col(score).null_count()).item() or df.select((~pl.col(score).is_between(0, 100)).sum()).item():
        raise ValueError(f'Invalid {score}')
    if df.select(pl.col(group).null_count()).item():
        raise ValueError(f'Unassigned {group}')
    metric = df.select(pl.col(score).min().alias('min'), pl.col(score).quantile(.10).alias('p10'),
        pl.col(score).median().alias('median'), pl.col(score).quantile(.90).alias('p90'),
        pl.col(score).max().alias('max'), (pl.col(score) <= 1).sum().alias('floor_at_1'),
        (pl.col(score) >= 99).sum().alias('ceiling_at_99'),
        (pl.col(score) - pl.col(alternative)).abs().mean().alias('weight_mean_absolute_delta'),
        (pl.col(score) - pl.col(alternative)).abs().quantile(.95).alias('weight_p95_absolute_delta'),
        pl.corr(score, alternative, method='spearman').alias('weight_rank_spearman')).to_dicts()[0]
    metric['components_mean'] = {col: df.select(pl.col(col).mean()).item() for col in components}
    metric['components_weighted_mean'] = {col: metric['components_mean'][col] * weight
        for col, weight in zip(components, weights.values())}
    groups = df.group_by(group).agg(pl.len().alias('count'), pl.col(score).median().alias('median_score'),
        pl.col(score).min().alias('min_score'), pl.col(score).max().alias('max_score')).with_columns(
        (pl.col('count') / rows * 100).round(2).alias('percent')).sort(group)
    groups.write_csv(TABLES / f'{name}_groups.csv')
    metric['groups'] = groups.to_dicts()
    validation['scores'][name] = metric
validation['financial_missing_income'] = {'count': rows - eda['customers_with_unambiguous_income'],
    'average_uncertainty_width': df.filter(~pl.col('income_usable')).select(
        (pl.col('financial_missing_income_high_bound') - pl.col('financial_missing_income_low_bound')).mean()).item()}
validation['engagement_no_call_proxy'] = {'mean_absolute_delta': df.select(
    (pl.col('engagement_score') - pl.col('engagement_no_call_proxy_sensitivity')).abs().mean()).item(),
    'rank_spearman': df.select(pl.corr('engagement_score', 'engagement_no_call_proxy_sensitivity', method='spearman')).item()}
due_p95 = df.filter(pl.col('due_weighted') > 0).select(pl.col('due_weighted').log1p().quantile(.95)).item()
finance_alt = 100 * (1 - (pl.col('due_weighted').log1p() / due_p95).clip(0, 1)) * .45 + w * .25 + i * .30
recency_alt = pl.when(pl.col('days_since_activity').is_null()).then(0.0).otherwise(
    100 * (-pl.col('days_since_activity') / 60).exp())
engagement_alt = recency_alt * .45 + f * .30 + q * .10 + l * .10 + p * .05
df = df.with_columns(finance_alt.clip(0, 100).alias('_financial_alt'),
    engagement_alt.clip(0, 100).alias('_engagement_alt'))
for label, col, alt in [('financial_normalization_sensitivity', 'financial_health_score', '_financial_alt'),
                         ('engagement_recency_sensitivity', 'engagement_score', '_engagement_alt')]:
    validation[label] = {'mean_absolute_delta': df.select((pl.col(col) - pl.col(alt)).abs().mean()).item(),
        'rank_spearman': df.select(pl.corr(col, alt, method='spearman')).item()}
validation['financial_normalization_sensitivity']['due_positive_log_p95'] = due_p95
validation['engagement_recency_sensitivity']['alternative_decay_days'] = 60
(TABLES / 'validation_metrics.json').write_text(json.dumps(validation, indent=2), encoding='utf-8')
for filename, content in [('financial_parameters.json', financial_params),
    ('engagement_parameters.json', engagement_params)]:
    (TABLES / filename).write_text(json.dumps(content, indent=2), encoding='utf-8')

# %% [markdown]
# ## Export
# Keep scores separate; verify the source remains unchanged.
# %%
score_cols = ['SKP_CLIENT', 'financial_health_score', 'financial_health_group', 'financial_evidence_weight',
    'income_usable', 'CONTRACT_INCOME_MAIN_CONFLICT_CASE_COUNT', 'health_overdue_component',
    'health_written_off_component', 'health_income_component', 'financial_missing_income_low_bound',
    'financial_missing_income_high_bound', 'engagement_score', 'engagement_group',
    'engagement_recency_component', 'engagement_frequency_component', 'engagement_interaction_component',
    'engagement_lead_component', 'engagement_product_component', 'days_since_activity',
    'CALL_COUNT_90D_ASOF_2025_06_04', 'BEHAVIOR_EVENTS_90D_ASOF_2025_06_04',
    'LEAD_COUNT_90D_ASOF_2025_06_04', 'BEHAVIOR_PRODUCTS_90D_ASOF_2025_06_04']
out = df.select(score_cols).sort('SKP_CLIENT')
out.write_parquet(OUTPUTS / 'customer_scores.parquet', compression='zstd')
out.write_csv(OUTPUTS / 'customer_scores.csv', float_precision=2)
with SOURCE.open('rb') as handle:
    if hashlib.file_digest(handle, 'sha256').hexdigest() != source_hash:
        raise ValueError('Input Parquet changed during analysis')
(TABLES / 'source_provenance.json').write_text(json.dumps({'source': SOURCE.name,
    'sha256': source_hash, 'rows': rows, 'cutoff_exclusive': str(CUTOFF),
    'software': {'python': sys.version.split()[0], 'polars': pl.__version__,
                 'numpy': np.__version__, 'matplotlib': plt.matplotlib.__version__},
    'scope': 'EDA, Financial Health, Engagement only'}, indent=2), encoding='utf-8')
print('Financial / Engagement medians:', validation['scores']['financial']['median'],
      validation['scores']['engagement']['median'])

# %% [markdown]
# ## Figures
# Shared style. One chart per cell; interpretation follows.
# %%
def frame(title, subtitle, footer):
    fig, ax = plt.subplots(figsize=(12.8, 7.2), layout='constrained')
    ax.set_title(title, loc='left', pad=34, color=NAVY)
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=11, color=SLATE, va='bottom')
    fig.text(.03, .025, footer, fontsize=10, color=SLATE)
    ax.set_axisbelow(True)
    return fig, ax

def finish(fig, destination):
    fig.savefig(destination, dpi=180, facecolor=BG)
    if 'get_ipython' in globals():
        display(fig)
    plt.close(fig)

def count_axis(ax, axis='x'):
    formatter = FuncFormatter(lambda value, _: f'{value / 1e6:.1f}M' if abs(value) >= 1e6
        else f'{value / 1e3:.0f}k' if abs(value) >= 1e3 else f'{value:.0f}')
    (ax.xaxis if axis == 'x' else ax.yaxis).set_major_formatter(formatter)

def bar_chart(path, title, subtitle, labels, values, colors, footer):
    fig, ax = frame(title, subtitle, footer)
    ax.grid(axis='x')
    pos = np.arange(len(labels))
    ax.barh(pos, values, color=colors, height=.58)
    ax.set_yticks(pos, labels)
    ax.invert_yaxis()
    limit = max(values) * 1.25
    ax.set_xlim(0, limit)
    ax.set_xlabel('Customers (categories can overlap)')
    count_axis(ax)
    for index, value in enumerate(values):
        ax.text(value + limit * .012, index, f'{value:,}  ({value / rows:.1%})',
            va='center', color=NAVY, fontsize=11)
    finish(fig, path)

def score_chart(path, title, subtitle, values, color, median, edge_note, footer):
    fig, ax = frame(title, subtitle, footer)
    ax.grid(axis='y')
    edges = np.arange(0, 105, 5)
    ax.hist(values, bins=edges, color=color, rwidth=.94)
    ax.axvline(median, color=CORAL, linewidth=2, label=f'Median {median:.1f}')
    ax.legend(frameon=False, loc='upper right')
    ax.text(.98, .84, edge_note, transform=ax.transAxes, ha='right', color=NAVY)
    ax.set(xlim=(0, 100), xlabel='Score (0–100)', ylabel='Customers per five-point bin')
    count_axis(ax, 'y')
    finish(fig, path)

def sensitivity_chart(path, title, subtitle, score, alternate, color, metrics, footer):
    fig, ax = frame(title, subtitle, footer)
    ax.grid(axis='y')
    bands = df.select(score, alternate).with_columns(
        ((pl.col(score) / 10).floor() * 10).clip(0, 90).alias('band')).group_by('band').agg(
        pl.col(score).median().alias('primary'), pl.col(alternate).median().alias('alternative'))
    bands = bands.sort('band')
    x = bands['band'].to_numpy() + 5
    ax.plot(x, bands['primary'].to_numpy(), color=color, linewidth=3, marker='o', label='Primary median')
    ax.plot(x, bands['alternative'].to_numpy(), color=AMBER, linewidth=2.5,
        marker='s', linestyle='--', label='Alternative median')
    ax.set(xlim=(0, 100), ylim=(0, 105), xlabel='Primary-score band (10 points)',
        ylabel='Median score within band')
    ax.legend(loc='upper left', frameon=False)
    ax.text(.99, .96, f"Rank ρ {metrics['weight_rank_spearman']:.3f}  |  mean |Δ| {metrics['weight_mean_absolute_delta']:.1f}  |  p95 |Δ| {metrics['weight_p95_absolute_delta']:.1f}",
        transform=ax.transAxes, ha='right', color=NAVY)
    ax.text(.99, .03, 'All customers; lines show band medians', transform=ax.transAxes,
        ha='right', color=SLATE, fontsize=10)
    finish(fig, path)

# %% [markdown]
# ### 1. Coverage
# %%
bar_chart(FIGURES / 'eda' / '01_coverage.png', 'Every customer remains in the analysis',
    'Coverage differs by source; categories overlap',
    ['All customers', 'Usable latest income', 'Call before cutoff', 'Any lead', 'Any behavior'],
    [rows, eda['customers_with_unambiguous_income'], eda['customers_with_pre_cutoff_call'],
     eda['customers_with_lead'], eda['customers_with_behavior']],
    [NAVY, SLATE, BLUE, AMBER, TEAL], 'Source: data_clean.parquet | no customer excluded')
# %% [markdown]
# All customers are retained. Income and activity coverage differ.

# %%
bar_chart(FIGURES / 'eda' / '02_financial_signals.png', 'Adverse financial records are sparse',
    'Status and arrears may reflect extraction-time state',
    ['Income conflict in any case', 'Any active case', 'Any overdue history', 'Any written-off case'],
    [eda['customers_with_case_income_conflict'], eda['customers_with_active_contract'],
     eda['customers_with_overdue'], eda['customers_with_writeoff']],
    [SLATE, BLUE, AMBER, CORAL], 'Source: data_clean.parquet | flags are not default probabilities')
# %% [markdown]
# Conflicts are common; overdue and write-off flags are sparse.

# %%
bar_chart(FIGURES / 'eda' / '03_recent_activity.png', 'Calls reach the most customers in 90 days',
    'Distinct customers with events from 7 March through 4 June 2025',
    ['Any call', 'Any behavior event', 'Any lead'],
    [eda['recent_calls'], eda['recent_behavior'], eda['recent_leads']],
    [BLUE, TEAL, AMBER], 'Source: data_clean.parquet | 90-day window; 5 June excluded')
# %% [markdown]
# Calls are most widespread. Outreach does not prove customer intent.

# %%
score_chart(FIGURES / 'financial_health' / '04_financial_score.png',
    'Financial scores cluster high', 'Observed-signal index, not risk probability',
    df['financial_health_score'].to_numpy(), BLUE,
    validation['scores']['financial']['median'],
    f"Ceiling ≥99: {validation['scores']['financial']['ceiling_at_99']:,}",
    'Source: data_clean.parquet | missing income receives neutral 50 and an evidence flag')
# %% [markdown]
# Median: 85/100. 306,050 customers have limited income evidence.

# %%
sensitivity_chart(FIGURES / 'financial_health' / '05_financial_sensitivity.png',
    'Financial ranking changes little under a weight shift',
    'Overdue 45→55%; written-off 25→15%; income stays 30%',
    'financial_health_score', 'financial_weight_sensitivity', BLUE,
    validation['scores']['financial'],
    'Source: data_clean.parquet | internal robustness, not outcome validation')
# %% [markdown]
# Weight sensitivity tests ranking stability, not predictive value.

# %%
score_chart(FIGURES / 'engagement' / '06_engagement_score.png',
    'Engagement concentrates at the low end', 'Observed activity through 4 June 2025',
    df['engagement_score'].to_numpy(), TEAL,
    validation['scores']['engagement']['median'],
    f"Floor ≤1: {validation['scores']['engagement']['floor_at_1']:,}",
    'Source: data_clean.parquet | zero can mean absent or remote recorded history')
# %% [markdown]
# Median: 9.93/100. Low scores mean little recorded activity.

# %%
sensitivity_chart(FIGURES / 'engagement' / '07_engagement_sensitivity.png',
    'Engagement ranks hold; individual scores can move',
    'Recency 45→35%; frequency 30→40%; other weights unchanged',
    'engagement_score', 'engagement_weight_sensitivity', TEAL,
    validation['scores']['engagement'],
    'Source: data_clean.parquet | internal robustness, not outcome validation')
# %% [markdown]
# Rankings are stable overall; individual scores can shift.

# %% [markdown]
# ## Reports
# Methods, results and limits.
# %%
fin, eng = validation['scores']['financial'], validation['scores']['engagement']
financial_report = f'''# Financial Health

**Score:** 45% overdue + 25% write-off + 30% reported income. Severity-weighted
overdue counts use a log-p99 cap. Positive latest income uses log-p05–p95 scaling.
Missing income gets neutral 50, an evidence flag and score bounds.

**Result:** median {fin['median']:.2f}/100; {validation['financial_missing_income']['count']:,}
customers have limited income evidence. Not a default probability. Income is
unverified; contract status and arrears may be extraction-time.
'''
engagement_report = f'''# Engagement

**Score:** 45% recency + 30% 90-day frequency + 10% inbound calls + 10% leads
+ 5% product breadth. Activity ends 4 June 2025; count inputs use log-p95 caps.

**Result:** median {eng['median']:.2f}/100; {eda['no_observed_activity']:,} customers
have no observed activity. Measures recorded activity, not intent or satisfaction.
'''
validation_report = f'''# Validation

All {rows:,} customers retained; scores are within [0,100]. Weight-shift rank
correlation: Financial {fin['weight_rank_spearman']:.3f}; Engagement
{eng['weight_rank_spearman']:.3f}. Sensitivity checks only; no independent
future outcome is available for validation.

Source SHA-256: `{source_hash}`.
'''
summary = f'''# BI10 — EDA and Tasks 1–2

**Customers:** {rows:,} | **Usable latest income:** {eda['customers_with_unambiguous_income']:,}
**Income conflicts:** {eda['customers_with_case_income_conflict']:,} | **Recorded overdue:** {eda['customers_with_overdue']:,}

**Financial Health median:** {fin['median']:.2f}/100
**Engagement median:** {eng['median']:.2f}/100

Descriptive indexes, not lending or conversion predictions. Contract status may
be extraction-time. No independent future outcome is available.
'''
analysis_report = summary + '\n## Methods\n\n' + financial_report + '\n' + engagement_report + '\n' + validation_report
for name, text in [('EXECUTIVE_SUMMARY.md', summary), ('ANALYTICAL_REPORT.md', analysis_report),
    ('FINANCIAL_HEALTH_METHODOLOGY.md', financial_report),
    ('ENGAGEMENT_METHODOLOGY.md', engagement_report), ('VALIDATION_REPORT.md', validation_report)]:
    (REPORTS / name).write_text(text, encoding='utf-8')
print(f'Exported {out.height:,} scores, 7 figures, 5 reports; source unchanged.')

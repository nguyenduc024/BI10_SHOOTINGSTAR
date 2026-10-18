# BI10 colour palette

One palette for every chart, slide and dashboard in the Round 02 submission, so the same colour always means the same thing.

| File | Use it for |
|---|---|
| `bi10_colors.py` | Python notebooks: `BI_COLORS`, band colours, `apply_style()`, `style_axes()`, `highlight_colors()` |
| `bi10_colors.json` | Any other tool or script (JavaScript dashboards, Streamlit, Excel notes) |
| `bi10_colors.css` | HTML dashboards and prototypes (CSS variables `--bi-*`) |
| `bi10_powerbi_theme.json` | Power BI: *View → Themes → Browse for themes* |

## Base colours

| Name | Hex | Meaning |
|---|---|---|
| navy | `#16324F` | Titles, benchmark lines, "Very high" engagement |
| blue | `#0072B2` | Neutral focus; "Stable" health, "Moderate" engagement |
| teal | `#009E73` | Healthy, engaged, good outcome |
| orange | `#D55E00` | Risk, group of interest, bad outcome |
| gold | `#E69F00` | Watch, warning |
| slate | `#667085` | Neutral data series; "No credit history" |
| context | `#C3CAD4` | Bars that are only context |
| grid | `#D8E1E8` | Gridlines and axis lines |
| text | `#1F2933` | Body text and value labels |
| muted | `#667085` | Tick labels and secondary text |
| surface | `#FFFFFF` | Background |

## Segment colours (fixed)

| Financial Health band | Colour | Engagement band | Colour |
|---|---|---|---|
| Healthy | teal `#009E73` | Very high | navy `#16324F` |
| Stable | blue `#0072B2` | High | teal `#009E73` |
| Watchlist | gold `#E69F00` | Moderate | blue `#0072B2` |
| Stressed | orange `#D55E00` | Low | gold `#E69F00` |
| No credit history | slate `#667085` | Dormant | orange `#D55E00` |

## Rules

1. **Grey for context, one accent for the point.** Most bars are `context`; only the group the chart is about gets orange (risk) or teal (good).
2. **A colour keeps its meaning.** Never use orange for a good result or teal for a risk.
3. **Short descriptive titles** ("Bad rate by product"); the insight goes in the text or slide headline.
4. **Bars start at zero**; label values on the bars instead of relying on gridlines.
5. **Benchmarks** (average, base rate) are dashed navy lines with a short label such as `avg 2.2%`.

## Python example

```python
import matplotlib.pyplot as plt
from style.bi10_colors import BI_COLORS, apply_style, style_axes, highlight_colors

apply_style()
products = ["TW", "CW", "CC", "BNPL", "CL", "CD"]
bad_rate = [4.9, 4.7, 2.3, 2.1, 2.1, 2.1]

fig, ax = plt.subplots(figsize=(6.5, 3.0), layout="constrained")
ax.barh(products, bad_rate, color=highlight_colors(products, {"TW", "CW"}))
ax.axvline(2.2, color=BI_COLORS["navy"], linestyle="--", linewidth=1)
ax.invert_yaxis()
ax.set_title("Bad rate by product")
style_axes(ax, grid_axis="x")
```

"""Visual design for the Streamlit dashboard: colours, CSS, small HTML components, chart style."""

PALETTE = {
    "ink": "#14213D", "paper": "#F6F7F9", "teal": "#0F8B8D", "amber": "#E0A100",
    "coral": "#D1495B", "slate": "#5C677D", "mist": "#A9B4C6", "line": "#DDE2EA",
}
RISK_COLORS = {"Low": PALETTE["teal"], "Medium": PALETTE["amber"], "High": PALETTE["coral"]}
FONT = "Schibsted Grotesk, system-ui, -apple-system, Segoe UI, sans-serif"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Schibsted+Grotesk:wght@400;500;600;700&display=swap');
.stApp, .stApp p, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5,
.stApp button, .stApp input, .stApp textarea, .stMarkdown { font-family: 'Schibsted Grotesk', system-ui, sans-serif; }
.block-container {  padding-top: 5rem !important; padding-bottom: 3rem; max-width: 1180px; }
#MainMenu, footer { visibility: hidden; }
.stApp h5 { font-size: 0.95rem; font-weight: 600; color: #14213D; margin: 1.1rem 0 0.2rem; }

.ciq-title { font-size: 2.1rem; font-weight: 700; letter-spacing: -0.02em; color: #14213D; line-height: 1.1; margin: 0; }
.ciq-sub { color: #5C677D; font-size: 1rem; margin: 0.25rem 0 1rem; }
.ciq-lead { border-left: 4px solid #0F8B8D; padding: 0.35rem 0 0.35rem 0.9rem; font-size: 1.08rem;
           color: #14213D; margin: 0.6rem 0 1.2rem; max-width: 62ch; }

.ciq-kpi { background: #FFFFFF; border: 1px solid #DDE2EA; border-radius: 10px; padding: 14px 16px; height: 100%; }
.ciq-kpi .l { font-size: 0.84rem; color: #5C677D; }
.ciq-kpi .v { font-size: 1.85rem; font-weight: 700; color: #14213D; line-height: 1.2; font-variant-numeric: tabular-nums; }
.ciq-kpi .n { font-size: 0.78rem; color: #5C677D; margin-top: 2px; }
.ciq-kpi.high .v { color: #D1495B; }

.ciq-result { background: #FFFFFF; border: 1px solid #DDE2EA; border-radius: 10px; padding: 18px 20px; margin-bottom: 0.6rem; }
.ciq-result .top { display: flex; align-items: flex-end; gap: 20px; }
.ciq-result .pct { font-size: 3rem; font-weight: 700; line-height: 1; font-variant-numeric: tabular-nums; }
.ciq-result .lvl { font-size: 1.05rem; font-weight: 600; margin-top: 4px; }
.ciq-result .note { color: #5C677D; font-size: 0.84rem; margin-top: 12px; }
.sig { display: flex; align-items: flex-end; gap: 5px; height: 54px; }
.sig i { display: block; width: 15px; border-radius: 3px; background: #DDE2EA; }
.sig i:nth-child(1) { height: 16px; }
.sig i:nth-child(2) { height: 34px; }
.sig i:nth-child(3) { height: 52px; }
</style>
"""


def kpi(label, value, note="", tone=""):
    return (f'<div class="ciq-kpi {tone}"><div class="l">{label}</div>'
            f'<div class="v">{value}</div><div class="n">{note}</div></div>')


def signal_bars(level):
    """Phone-signal style meter: strong signal = low churn risk, weak signal = high churn risk."""
    filled = {"Low": 3, "Medium": 2, "High": 1}[level]
    color = RISK_COLORS[level]
    bars = "".join(f'<i style="background:{color}"></i>' if n < filled else "<i></i>" for n in range(3))
    return f'<div class="sig" title="Retention signal">{bars}</div>'


def result_card(prob, level, threshold):
    color = RISK_COLORS[level]
    shown = "<1%" if prob < 0.01 else f"{prob:.0%}"
    return (f'<div class="ciq-result"><div class="top">{signal_bars(level)}'
            f'<div><div class="pct" style="color:{color}">{shown}</div>'
            f'<div class="lvl" style="color:{color}">{level} risk</div></div></div>'
            f'<div class="note">Estimated chance this customer leaves. Customers at or above '
            f'{threshold:.0%} are flagged for follow-up. This is an estimate, not a guarantee.</div></div>')


def style_fig(fig, height=330):
    p = PALETTE
    fig.update_layout(
        template="simple_white", height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT, size=13, color=p["ink"]), margin=dict(l=8, r=12, t=48, b=8),
        title=dict(x=0, xanchor="left", font=dict(size=15, color=p["ink"])),
        legend=dict(orientation="h", y=-0.18, x=0, title=None),
    )
    fig.update_xaxes(showgrid=False, linecolor=p["line"], title=None)
    fig.update_yaxes(gridcolor=p["line"], zeroline=False, title=None)
    return fig

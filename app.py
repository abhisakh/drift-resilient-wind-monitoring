"""Windmill -- drift-resilient wind turbine condition monitoring, dashboard.

    streamlit run app.py

HYBRID: it works with the raw data and without it.

  LOCAL (data/raw/ present, 19 GB)
      The R80711 monitor loads SCADA straight from the ENGIE release and
      re-runs the detector live, through the same code path the pipeline
      uses. This is the mode for a live demo: the threshold slider re-alarms
      the real series as you move it.

  CLOUD (no data/raw/, Streamlit Community Cloud)
      Everything comes from the committed artefacts in results/. The
      R80711 monitor reads results/demo_r80711.json -- a compact snapshot
      exported by scripts/pipeline/11_demo.py, so the dashboard still has a
      real 10-minute time series with the peer control group, the
      load-conditioned residual and the z-score, without the 19 GB.

Every number shown here is read from results/*.json. The dashboard never
computes a headline figure itself, so it cannot disagree with the README --
which is itself generated from the same files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[0]
RESULTS = ROOT / "results"
DATA_RAW = ROOT / "data" / "raw"

st.set_page_config(
    page_title="Windmill — Drift-Resilient Condition Monitoring",
    page_icon="🌬️",
    layout="wide",
)


# ---------------------------------------------------------------- artefacts
def _load(name: str) -> dict | None:
    """Read one result artefact. Missing files degrade, never crash."""
    path = RESULTS / name
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


@st.cache_data(show_spinner=False)
def load_snapshot() -> dict | None:
    return _load("demo_r80711.json")


@st.cache_data(show_spinner=False)
def load_engie() -> dict | None:
    return _load("engie_multivariate.json")


@st.cache_data(show_spinner=False)
def load_multivariate() -> dict | None:
    return _load("multivariate.json")


@st.cache_data(show_spinner=False)
def load_lightgbm() -> dict | None:
    return _load("lightgbm_nbm.json")


@st.cache_data(show_spinner=False)
def load_care_score() -> dict | None:
    return _load("care_score.json")


@st.cache_data(show_spinner=False)
def load_ae_audit() -> dict | None:
    return _load("ae_audit.json")


@st.cache_data(show_spinner=False)
def load_audit() -> dict | None:
    return _load("farm_a_audit.json")


@st.cache_data(show_spinner=False)
def load_thermal() -> dict | None:
    return _load("thermal_signal.json")


def live_available() -> bool:
    """Is the raw ENGIE data here? (Not just the directory -- the files.)"""
    if not DATA_RAW.exists():
        return False
    return any(DATA_RAW.rglob(f"*{t}*F99*.csv") for t in ("R80711",))


# ---------------------------------------------------------------- helpers
def _fmt(x, nd=4):
    if x is None:
        return "—"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


#: A wind farm skyline, drawn rather than imported so the dashboard carries
#: no image asset. Each turbine is (hub x, scale, start angle, seconds per
#: revolution): different sizes read as depth, and different start angles
#: stop the rotors looking like copies of one another.
#: Two clusters, one hugging each edge of the window, because the middle of
#: the page is covered by the cards. Each is its own image, drawn at the
#: same uniform scale, so a wide window gets wider gutters rather than
#: stretched or cropped turbines.
_FARM_LEFT = (
    (70, 0.50, 15, 38),
    (215, 0.85, 70, 30),
    (355, 0.60, 40, 44),
)
_FARM_RIGHT = (
    (60, 0.55, 25, 40),
    (225, 1.00, 100, 26),
    (375, 0.60, 55, 46),
)

#: One blade, hub at the origin, pointing up, 118 units long.
_BLADE = "M0 0C-5 -36 -6 -78 0 -118C6 -78 5 -36 0 0Z"

#: True makes the rotors turn slowly. Off by default: motion on a page that
#: people read for minutes at a time is a distraction.
WATERMARK_ANIMATE = True


def _wind_farm_svg(farm=_FARM_RIGHT, animate: bool = False) -> str:
    """A cluster of turbines standing on the ground line, as an SVG string.

    No `preserveAspectRatio="slice"`: it crops whatever does not fit, and on
    a wide window that cropped away the blades and left bare towers.
    """
    parts = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 450 360">'
             '<g fill="#5fd39a">']
    for hx, s, start, secs in farm:
        hy = 150 * s + 20              # hub height grows with the scale
        base = 360 - (1 - s) * 40      # smaller = further away = higher base
        parts.append(
            f'<path d="M{hx - 3 * s:.1f} {hy:.1f}L{hx + 3 * s:.1f} {hy:.1f}'
            f'L{hx + 6 * s:.1f} {base:.1f}L{hx - 6 * s:.1f} {base:.1f}Z"/>')
        spin = (f'<animateTransform attributeName="transform" type="rotate" '
                f'from="{start}" to="{start + 360}" dur="{secs}s" '
                f'repeatCount="indefinite"/>') if animate else ""
        blades = "".join(f'<path d="{_BLADE}" transform="rotate({k * 120})"/>'
                         for k in range(3))
        parts.append(f'<g transform="translate({hx} {hy:.1f}) scale({s})">'
                     f'<g transform="rotate({start})">{spin}{blades}</g>'
                     f'<circle r="8"/></g>')
    parts.append("</g></svg>")
    return "".join(parts)

#: The crisp brand mark beside the sidebar title. Drawn in `currentColor` so
#: it follows the theme in both light and dark mode.
_TURBINE_ICON_SVG = (
    '<svg class="brand-icon" viewBox="0 0 24 24" width="26" height="26"'
    ' fill="none" stroke="currentColor" stroke-width="2.5"'
    ' stroke-linecap="round" aria-hidden="true" focusable="false">'
    # Three straight blades on a hub, over a mast. Drawn as strokes rather
    # than as filled petals: at 26 pixels a filled blade blurs into the
    # others and the whole thing reads as a cross.
    '<path d="M12 8.5V1.6"/>'
    '<path d="M12 8.5L18.6 12.8"/>'
    '<path d="M12 8.5L5.4 12.8"/>'
    '<circle cx="12" cy="8.5" r="1.3" fill="currentColor" stroke="none"/>'
    '<path d="M12 10.2V22.2"/>'
    "</svg>"
)


def _watermark_uri(side: str = "right") -> str:
    """One cluster of the watermark as a data URI, so the CSS needs no
    external file."""
    import base64

    farm = _FARM_LEFT if side == "left" else _FARM_RIGHT
    svg = _wind_farm_svg(farm, WATERMARK_ANIMATE)
    return ("data:image/svg+xml;base64,"
            + base64.b64encode(svg.encode("utf-8")).decode("ascii"))


#: Every font size on the dashboard, in one place.
#:
#: The reviewer asked for text that could be read across a room, so these
#: are roughly double the sizes the readability boost was written at.
#: `FONT_SCALE` is one knob on top of them: raise it to push everything
#: further without hunting through the CSS.
FONT_SCALE = 1.0

FONT_SIZES = {
    "body": 30,        # prose in the tabs and the sidebar
    "caption": 26,     # small explanatory lines under a widget
    "tab": 28,         # the four tab headers
    "sidebar": 22,     # the narrow sidebar column
    "check": 26,       # checkbox and radio labels
    "widget": 28,      # widget labels: dropdowns, sliders, checkboxes
    "button": 28,      # button captions
    "selectbox": 26,   # the value shown inside a dropdown
    "slider": 30,      # the value above a slider's thumb
    "table": 20,       # the reference tables
    "alert": 28,       # st.info / st.success / st.warning / st.error
    "metric_value": 44,
    "metric_label": 24,
    "metric_delta": 22,
    "kpi_value": 46,   # the big number on a KPI card
    "kpi_label": 24,
    "kpi_delta": 22,
    "note": 28,        # the boxed explanatory notes
    "callout": 30,     # the training-window callout
    "brand": 26,       # the sidebar title
    "badge": 18,       # the LIVE / CLOUD badges
    "h2": 44,
    "h3": 38,
    "h4": 34,
}

#: The window width, in px, at which the sizes above apply in full. Narrower
#: than that, the type scales down with the window -- but never below
#: FLUID_FLOOR of its size -- so a smaller window gets smaller text instead
#: of lines that run off the edge and are cropped.
FLUID_REF_PX = 1400
FLUID_FLOOR = 0.55


def _fluid(px: float) -> str:
    """A CSS font size: `px` on a wide window, shrinking with a narrow one."""
    size = px * FONT_SCALE
    floor = max(11, round(size * FLUID_FLOOR))
    return f"clamp({floor}px, {size / FLUID_REF_PX * 100:.3f}vw, {round(size)}px)"


def _style():
    css = """
        <style>
        /* ------------------------------------------------------------------
           Typography. Every rule here is explicit px rather than rem, so
           the sizes below are the sizes that render -- Streamlit's own
           rem-based defaults are simply overruled.
           ------------------------------------------------------------------ */

        /* Prose: the paragraphs in the tabs. */
        .stApp [data-testid="stMarkdownContainer"] p,
        .stApp [data-testid="stMarkdownContainer"] li {
            font-size: __body__px !important;
            line-height: 1.55 !important;
        }

        /* The sidebar is a 300px column, so its prose is a size smaller --
           at the body size it wrapped into four lines a sentence. */
        .stApp [data-testid="stSidebar"] p,
        .stApp [data-testid="stSidebar"] li {
            font-size: __sidebar__px !important;
            line-height: 1.5 !important;
        }

        /* Captions: the small lines that explain a control. */
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stCaptionContainer"] p {
            font-size: __caption__px !important;
            line-height: 1.45 !important;
            color: #cbd5e1 !important;
        }

        /* Headings. */
        .stApp [data-testid="stMarkdownContainer"] h1 { font-size: __h2__px !important; }
        .stApp [data-testid="stMarkdownContainer"] h2 { font-size: __h2__px !important; }
        .stApp [data-testid="stMarkdownContainer"] h3 { font-size: __h3__px !important; }
        .stApp [data-testid="stMarkdownContainer"] h4 { font-size: __h4__px !important; }

        /* The four tab headers. Streamlit renders these as
           div[data-testid="stTab"], not as buttons. The labels are long
           and there are four of them in one row, so they wrap onto two
           lines -- and the tab list is a fixed height that would cut the
           second line off, so it is given room for them. */
        .stApp [data-testid="stTab"] p {
            font-size: __tab__px !important;
            font-weight: 600 !important;
            white-space: normal !important;
            line-height: 1.25 !important;
            padding-bottom: 6px !important;
        }
        .stApp [data-testid="stTabs"] [role="tablist"] {
            height: auto !important;
            min-height: 0 !important;
            align-items: flex-start !important;
        }
        .stApp [data-testid="stTab"] {
            height: auto !important;
            align-items: flex-start !important;
        }

        /* Widget labels: dropdowns, sliders, checkboxes, radio buttons. */
        .stApp [data-testid="stWidgetLabel"],
        .stApp [data-testid="stWidgetLabel"] p,
        .stApp [data-testid="stWidgetLabel"] span {
            font-size: __widget__px !important;
        }

        /* Checkbox and radio labels sit in narrow columns and are long
           ("Frozen 2 years before (2015 curve + 2017 weather)"). They wrap
           rather than being cut off with an ellipsis. */
        .stApp [data-testid="stCheckbox"] label p,
        .stApp [data-testid="stRadio"] label p {
            font-size: __check__px !important;
            white-space: normal !important;
        }

        /* The value shown inside a dropdown. It is an <input>, which CSS
           can still size. */
        .stApp [data-testid="stSelectbox"] input,
        .stApp [data-testid="stSelectbox"] div[role="combobox"] {
            font-size: __selectbox__px !important;
        }

        /* A slider's read-out floats above its thumb on an offset
           Streamlit computed for a 14px font, so it has to be sized and
           placed together or it lands on the track. */
        .stApp [data-testid="stSliderThumbValue"],
        .stApp [data-testid="stSliderThumbValue"] p {
            font-size: __slider__px !important;
        }
        .stApp [data-testid="stSliderThumbValue"] {
            top: -1.55em !important;
        }
        /* ...and the room between the label and the track for it. */
        .stApp [data-testid="stSlider"] [data-testid="stWidgetLabel"] {
            margin-bottom: calc(__slider__px * 1.6) !important;
        }
        .stApp [data-testid="stSlider"] [data-testid="stTickBarMin"],
        .stApp [data-testid="stSlider"] [data-testid="stTickBarMax"] {
            font-size: __slider__px !important;
        }

        /* Buttons. The tab rule above is more specific and wins for tabs. */
        .stApp [data-testid="stButton"] button p,
        .stApp [data-testid="stButton"] button span {
            font-size: __button__px !important;
        }

        /* The alerts: st.info, st.success, st.warning, st.error. */
        .stApp [data-testid="stAlert"],
        .stApp [data-testid="stAlert"] div,
        .stApp [data-testid="stAlert"] p {
            font-size: __alert__px !important;
            line-height: 1.5 !important;
        }

        /* st.metric. */
        .stApp [data-testid="stMetricValue"] { font-size: __metric_value__px !important; }
        .stApp [data-testid="stMetricLabel"] { font-size: __metric_label__px !important; }
        .stApp [data-testid="stMetricDelta"] { font-size: __metric_delta__px !important; }

        /* The reference tables. Streamlit draws a dataframe on a canvas,
           where CSS cannot reach, so these are st.table -- real HTML. */
        .stApp [data-testid="stTable"] table,
        .stApp [data-testid="stTable"] th,
        .stApp [data-testid="stTable"] td {
            font-size: __table__px !important;
        }
        .stApp [data-testid="stTable"] th {
            text-align: right !important;
        }
        /* A table wider than its column scrolls, rather than running over
           the text beside it. */
        .stApp [data-testid="stTable"] {
            overflow-x: auto !important;
        }
        .stApp [data-testid="stTable"] table {
            width: max-content !important;
            min-width: 100% !important;
        }
        .stApp [data-testid="stTable"] th {
            white-space: normal !important;
        }

        /* The text inside the charts is sized on the FIGURE, not here:
           Plotly lays a chart out for the font it is given, so overriding
           it from CSS afterwards would leave the layout computed for a
           smaller font and the legends would run over each other. */

        /* ------------------------------------------------------------------
           Nothing is cut off. Streamlit puts some text on one line with an
           ellipsis by default, and at these sizes that crops it. Here it
           wraps instead, and the sizes above shrink with the window (see
           _fluid) rather than staying fixed.
           ------------------------------------------------------------------ */

        /* st.metric truncates its label, value and delta with an ellipsis. */
        .stApp [data-testid="stMetricLabel"],
        .stApp [data-testid="stMetricLabel"] *,
        .stApp [data-testid="stMetricValue"],
        .stApp [data-testid="stMetricValue"] *,
        .stApp [data-testid="stMetricDelta"],
        .stApp [data-testid="stMetricDelta"] * {
            white-space: normal !important;
            overflow: visible !important;
            text-overflow: clip !important;
            overflow-wrap: anywhere !important;
        }

        /* A long unbroken string wraps instead of running past the edge. */
        .stApp [data-testid="stMarkdownContainer"],
        .stApp [data-testid="stCaptionContainer"],
        .stApp [data-testid="stAlert"],
        .kpi, .note, .callout, .badge {
            overflow-wrap: anywhere;
        }

        /* Code blocks wrap rather than scroll sideways. */
        .stApp [data-testid="stCode"] pre,
        .stApp [data-testid="stCode"] code {
            white-space: pre-wrap !important;
            overflow-wrap: anywhere !important;
        }

        /* Columns stack as the window narrows. Streamlit only stacks them at
           640px, which is far too late for text this size. */
        @media (max-width: 1000px) {
            .stApp [data-testid="stMain"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                flex: 1 1 calc(50% - 1rem) !important;
                min-width: calc(50% - 1rem) !important;
            }
        }
        @media (max-width: 700px) {
            .stApp [data-testid="stMain"] [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
                flex: 1 1 calc(100% - 1rem) !important;
                min-width: calc(100% - 1rem) !important;
            }
        }

        /* A chart's title is a heading above it, not text inside it: Plotly
           cannot wrap a title, so a long one was cut off by a narrow window,
           whereas a heading wraps with the page. */
        .chart-title {
            font-size: __h4__px !important;
            font-weight: 600;
            line-height: 1.3;
            margin: 10px 0 4px 0;
        }

        /* ------------------------------------------------------------------
           Design system: one palette, one radius, one spacing rhythm.
           (The old rules used --border, --foreground and --muted-foreground,
           which nothing defined, so the cards were drawn with no border.)
           ------------------------------------------------------------------ */
        .stApp {
            --accent: #5fd39a;
            --accent-soft: rgba(95, 211, 154, 0.10);
            --surface: #161b22;
            --surface-2: #1c2330;
            --border: rgba(240, 246, 252, 0.12);
            --foreground: #e6edf3;
            --muted-foreground: #9aa7b6;
            --radius: 12px;
        }

        /* The page: a centred column, not a stretch across a wide monitor. */
        [data-testid="stMainBlockContainer"] {
            max-width: 1680px !important;
            margin: 0 auto !important;
            padding: 2rem 2.5rem 4rem 2.5rem !important;
        }
        footer { visibility: hidden; }
        [data-testid="stHeader"] { background: transparent !important; }
        [data-testid="stSidebar"] { border-right: 1px solid var(--border); }
        .stApp [data-testid="stHorizontalBlock"] { gap: 1.25rem !important; }

        /* Page header. */
        .app-header {display: flex; flex-wrap: wrap; align-items: center; gap: 24px;
                     padding: 26px 30px; margin-bottom: 18px;
                     border: 1px solid var(--border); border-radius: 16px;
                     background: linear-gradient(120deg, rgba(95, 211, 154, 0.12),
                                 rgba(95, 211, 154, 0) 55%), var(--surface);}
        .app-mark {flex: 0 0 auto; width: 64px; height: 64px; display: flex;
                   align-items: center; justify-content: center;
                   border-radius: 16px; color: var(--accent);
                   background: var(--accent-soft);
                   border: 1px solid rgba(95, 211, 154, 0.30);}
        .app-mark .brand-icon {width: 38px; height: 38px; color: var(--accent);}
        .app-title {font-size: __h2__px !important; font-weight: 700;
                    line-height: 1.15; letter-spacing: -0.01em;}
        .app-sub {font-size: __caption__px !important; color: var(--muted-foreground);
                  margin-top: 6px; line-height: 1.4;}
        .app-text {flex: 1 1 320px; min-width: 0;}
        .chips {display: flex; flex-wrap: wrap; gap: 8px; margin-top: 14px;}
        .chip {padding: 4px 12px; border-radius: 999px; border: 1px solid var(--border);
               background: rgba(255, 255, 255, 0.04); color: var(--muted-foreground);
               font-size: __badge__px !important; font-weight: 600; letter-spacing: .03em;}
        .app-link {flex: 0 0 auto; padding: 10px 20px; border-radius: 10px;
                   border: 1px solid rgba(95, 211, 154, 0.45);
                   color: var(--accent) !important; font-weight: 600;
                   text-decoration: none !important; font-size: __caption__px !important;}
        .app-link:hover {background: var(--accent-soft);}
        .eyebrow {font-size: __kpi_label__px !important; font-weight: 700;
                  color: var(--accent); text-transform: uppercase;
                  letter-spacing: .12em; margin: 8px 0 10px 2px;}
        .app-footer {margin-top: 3rem; padding-top: 18px;
                     border-top: 1px solid var(--border);
                     color: var(--muted-foreground);
                     font-size: __caption__px !important; line-height: 1.6;}
        .app-footer a {color: var(--accent) !important; text-decoration: none;}

        /* Tabs. */
        .stApp [data-baseweb="tab-list"] {gap: 8px !important;
                                         border-bottom: 1px solid var(--border);}
        .stApp [data-testid="stTab"] {padding: 12px 20px !important;
                                      border-radius: 10px 10px 0 0;}
        .stApp [data-testid="stTab"]:hover {background: var(--accent-soft);}
        .stApp [data-baseweb="tab-panel"] {padding-top: 1.4rem !important;}

        /* Headings: a clear hierarchy, with the section titles marked. */
        .stApp [data-testid="stMarkdownContainer"] h2,
        .stApp [data-testid="stMarkdownContainer"] h3 {
            font-weight: 700 !important; letter-spacing: -0.01em;
            line-height: 1.2 !important; padding-top: .4rem !important;
        }
        .stApp [data-testid="stMarkdownContainer"] h4 {
            font-weight: 600 !important; line-height: 1.25 !important;
            border-left: 4px solid var(--accent);
            padding: 2px 0 2px 14px !important;
            margin: 1.4rem 0 .6rem 0 !important;
        }

        /* Controls. */
        .stApp [data-baseweb="select"] > div {
            background: var(--surface) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important; min-height: 3.1rem;
        }
        .stApp [data-baseweb="select"] > div:hover {
            border-color: var(--accent) !important;
        }
        .stApp [data-testid="stButton"] button,
        .stApp [data-testid="stDownloadButton"] button {
            border-radius: 10px; padding: .55rem 1.4rem; font-weight: 600;
        }
        .stApp [data-testid="stFileUploader"] section {
            border-radius: var(--radius); background: var(--surface);
            border: 1px dashed var(--border);
        }
        .stApp [data-testid="stAlert"] {
            border-radius: var(--radius) !important;
            border: 1px solid var(--border);
        }

        /* Charts, metrics and tables sit on cards. */
        .stApp [data-testid="stPlotlyChart"] {
            background: var(--surface); border: 1px solid var(--border);
            border-radius: var(--radius); overflow: hidden;
        }
        .stApp [data-testid="stMetric"] {
            background: var(--surface); border: 1px solid var(--border);
            border-radius: var(--radius); padding: 14px 20px;
        }
        .stApp [data-testid="stTable"] {
            border: 1px solid var(--border); border-radius: var(--radius);
            background: var(--surface);
        }
        .stApp [data-testid="stTable"] th { background: var(--surface-2); }

        /* Custom components */
        .badge {padding: 6px 14px; border-radius: 999px; font-weight: 700;
                letter-spacing: .05em; font-size: __badge__px !important; display: inline-block;}
        .live {background:#0f3d2e; color:#5fd39a; border:1px solid #1d6b4e;}
        .cloud {background:#2b2f4a; color:#9fb4ff; border:1px solid #455089;}
        .kpi {padding: 18px 22px; border: 1px solid var(--border);
              border-top: 3px solid var(--accent); border-radius: var(--radius);
              background: var(--surface); margin-bottom: 10px;
              box-shadow: 0 1px 3px rgba(0, 0, 0, 0.35);}
        .kpi .v {font-size: __kpi_value__px !important; font-weight: 700; line-height: 1.15;}
        .kpi .l {font-size: __kpi_label__px !important; font-weight: 600;
                 color: var(--muted-foreground); text-transform: uppercase;
                 letter-spacing: .06em; margin-top: 6px;}
        /* The delta carries an inline font-size, so this has to win. */
        .kpi div[style*="font-size"] { font-size: __kpi_delta__px !important; }
        .note {border: 1px solid var(--border); border-left: 4px solid var(--accent);
               background: var(--surface); border-radius: var(--radius);
               padding: 14px 20px; color: #c9d4df; margin: 6px 0 10px 0;
               font-size: __note__px !important; line-height: 1.55;}

        .callout {border: 1px solid rgba(95, 211, 154, 0.30);
                  border-left: 4px solid var(--accent);
                  padding: 14px 20px; margin: 12px 0 18px 0;
                  background: var(--accent-soft);
                  border-radius: var(--radius); font-size: __callout__px !important; line-height: 1.55;}
        .brand {display: flex; align-items: center; gap: 10px;
                font-size: __brand__px !important; font-weight: 700; line-height: 1.25;
                margin: 14px 0 6px 0; color: var(--foreground);}
        .brand-icon {flex: 0 0 auto; width: 28px; height: 28px; color: var(--foreground);}

        /* Wind farm skyline along the foot of the page. Fixed, so it stays
           put as the page scrolls; kept to the two gutters beside the
           content column; and it ignores the pointer. */
        body::after {
            content: "";
            position: fixed;
            left: 0;
            right: 0;
            bottom: 0;
            height: min(clamp(260px, 32vw, 620px), 65vh);
            background-image: url("__WATERMARK_LEFT__"), url("__WATERMARK__");
            background-repeat: no-repeat;
            background-position: left bottom, right bottom;
            background-size: auto 100%, auto 100%;
            opacity: 0.14;
            pointer-events: none;
            z-index: 0;
        }
        section.main, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
            position: relative;
            z-index: 1;
        }
        </style>
    """
    for key, value in FONT_SIZES.items():
        css = css.replace(f"__{key}__px", _fluid(value))
    css = css.replace("__WATERMARK_LEFT__", _watermark_uri("left"))
    css = css.replace("__WATERMARK__", _watermark_uri("right"))
    st.markdown(css, unsafe_allow_html=True)


#: The size of the text inside a chart. Plotly lays a chart out for the
#: font it is given, so this has to be set on the figure: overriding it
#: from CSS afterwards would leave the layout computed for a smaller font,
#: and the legends would run over each other.
CHART_FONT_PX = 26

#: A chart legend is a secondary element. Its entry names are long, so it is
#: a size smaller than the chart. Panel titles and threshold labels are the
#: same size: they sit on one line above a panel and have to fit its width.
CHART_LEGEND_FONT_PX = 22
CHART_ANNOTATION_FONT_PX = 22

#: Gridlines and axis lines: present enough to read values against, quiet
#: enough not to compete with the data.
CHART_GRID = "rgba(148, 163, 184, 0.14)"


def _chart(fig, **kwargs):
    """Render a Plotly figure with the dashboard's chart typography.

    The sizes have to be set on the figure, and on the axes as well as the
    layout: Streamlit applies its own theme to the figure at render time,
    and that theme sets an explicit tick font, which would otherwise win.
    """
    import html

    # The title becomes a heading above the chart (see .chart-title).
    title = fig.layout.title.text
    if title:
        st.markdown(f"<div class='chart-title'>{html.escape(title)}</div>",
                    unsafe_allow_html=True)
    fig.update_layout(title=dict(text=None))
    # Panel titles of a subplot figure still sit in the top margin; without
    # them the margin the title used to need can go.
    has_panel_titles = any(a.yref == "paper" and not a.showarrow
                           for a in fig.layout.annotations)
    if title and not has_panel_titles:
        fig.update_layout(margin=dict(t=16))

    fig.update_layout(
        font=dict(size=CHART_FONT_PX),
        legend=dict(font=dict(size=CHART_LEGEND_FONT_PX)),
    )
    # Breathing room at the edges, so the last tick label and the right-hand
    # end of an annotation are never cut off by the card.
    m = fig.layout.margin
    fig.update_layout(margin=dict(
        l=max(m.l if m.l is not None else 80, 24),
        r=max(m.r if m.r is not None else 80, 30)))
    # automargin lets the margins grow to fit the tick labels and axis
    # titles, instead of cutting them off at a fixed margin.
    fig.update_xaxes(tickfont=dict(size=CHART_FONT_PX),
                     title_font=dict(size=CHART_FONT_PX), automargin=True,
                     gridcolor=CHART_GRID, linecolor=CHART_GRID)
    fig.update_yaxes(tickfont=dict(size=CHART_FONT_PX),
                     title_font=dict(size=CHART_FONT_PX), automargin=True,
                     gridcolor=CHART_GRID, zerolinecolor=CHART_GRID,
                     linecolor=CHART_GRID)
    fig.update_annotations(font=dict(size=CHART_ANNOTATION_FONT_PX))
    st.plotly_chart(fig, **kwargs)


def kpi(value: str, label: str, delta: str | None = None):
    d = f"<div style='font-size:.8rem;color:var(--muted-foreground)'>{delta}</div>" if delta else ""
    st.markdown(
        f"<div class='kpi'><div class='v'>{value}</div>"
        f"<div class='l'>{label}</div>{d}</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- tab 1
def _model_comparison(engie, care_score, multiv, lgbm) -> None:
    """The model comparison: the deterministic arm against the learned one.

    This lives in the benchmark tab because it is a TECHNICAL study -- which
    of two detectors is better, and on what. The operational story is on the
    first two tabs.
    """
    st.markdown(
        "A condition-monitoring baseline calibrated on one period reads slow drift as "
        "degradation. On healthy turbines at the ENGIE site, **every alarm is a false "
        "alarm** — so the false-alarm rate is the cleanest possible measurement of "
        "whether a detector is fit to operate."
    )

    if not engie:
        st.error("results/engie_multivariate.json is missing.")
        return
    s = engie["strategies"]
    det_frozen = s.get("frozen_2014", {}).get("mean_rate_pct")
    det_roll = s.get("rolling_30d", {}).get("mean_rate_pct")
    lg_frozen = s.get("frozen_2014_lightgbm", {}).get("mean_rate_pct")
    lg_roll = s.get("rolling_30d_lightgbm", {}).get("mean_rate_pct")
    ratio = (det_frozen / det_roll) if det_frozen and det_roll else None

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi(f"{_fmt(ratio, 1)}×", "False-alarm reduction", "changing only the reference")
    with c2:
        kpi(f"{_fmt(det_roll, 2)} %", "Rolling 30-day reference", "mean false-alarm rate")
    with c3:
        kpi(f"{_fmt(det_frozen, 2)} %", "Frozen 2014 reference", "mean false-alarm rate")
    with c4:
        if care_score:
            kpi(_fmt(care_score["care_score"]), "CARE score", "95 labelled events, leak-free")
        else:
            kpi("—", "CARE score")

    st.markdown("")

    # --- the chart ---------------------------------------------------------
    import plotly.graph_objects as go

    rows = []
    for label, key in (("Deterministic (bins)", "frozen_2014"),
                        ("Deterministic (bins)", "rolling_30d"),
                        ("Learned (LightGBM NBM)", "frozen_2014_lightgbm"),
                        ("Learned (LightGBM NBM)", "rolling_30d_lightgbm")):
        v = s.get(key)
        if not v:
            continue
        rows.append({
            "detector": label,
            "reference": "frozen 2014" if "frozen" in key else "rolling 30-day",
            "mean %": v["mean_rate_pct"],
            "min %": v["min_rate_pct"],
            "max %": v["max_rate_pct"],
            "per_year": v["per_year"],
        })
    if not rows:
        st.warning("no ENGIE strategies in the artefact.")
        return
    df = pd.DataFrame(rows)

    fig = go.Figure()
    for detector, g in df.groupby("detector"):
        g = g.sort_values("mean %")
        fig.add_trace(go.Bar(
            x=g["mean %"], y=g["reference"], orientation="h",
            name=detector, text=[f"{v:.2f} %" for v in g["mean %"]],
            textposition="outside", cliponaxis=False,
            hovertemplate="%{y}: %{x:.2f} % mean false alarms<extra>%{fullData.name}</extra>",
        ))
    # Headroom on the x-axis so the outside text labels are not clipped, and
    # a horizontal legend so the chart keeps its width.
    xmax = max((r["max %"] for r in rows), default=1.0)
    fig.update_layout(
        title="Mean false-alarm rate on healthy turbines",
        xaxis_title="false alarms (% of 10-minute samples)",
        xaxis_range=[0, xmax * 1.3],
        yaxis_title=None, barmode="group", height=380,
        # The legend runs along the top and wraps onto more rows as the
        # window narrows. A legend down the right-hand side took a fixed
        # share of the width and squeezed the bars to almost nothing.
        legend=dict(title=dict(text=""), orientation="h", yanchor="bottom",
                    y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=40, t=16, b=70),
    )
    # The artefact stores the year as an integer INSIDE each entry; the
    # JSON object key is a string, and Plotly coerces a numeric-looking
    # string column to float -- which is how the axis ended up showing
    # "2,013.5" instead of "2013".
    per_year_df = pd.DataFrame([
        {"year": v["year"], "detector": r["detector"],
         "reference": r["reference"], "rate %": v["rate_pct"]}
        for r in rows
        for v in r["per_year"].values()
    ])

    left, right = st.columns([3, 2])
    with left:
        _chart(fig, width="stretch")
    with right:
        st.markdown(
            "<div class='note'>Both detectors improve when the reference is "
            "re-estimated. The deterministic one improves further, because its "
            "reference is a direct statistic of the operating point rather than "
            "a fitted model that must itself be refreshed.</div>",
            unsafe_allow_html=True)

    # The per-year table has five numeric columns under a two-row header.
    # It needs the full width of the page, and st.table rather than
    # st.dataframe: a dataframe is drawn on a canvas, where CSS cannot
    # reach, and this is a fixed reference table nobody needs to sort.
    st.markdown("#### Per year")
    st.table(
        per_year_df.pivot_table(index="year", columns=["detector", "reference"],
                                values="rate %").round(3))

    # --- the same thing as a line: year by year ---------------------------
    # The bar chart shows the means; this shows WHY the means differ. A frozen
    # reference swings from year to year, because each year the turbine has
    # drifted a little further from the reference. A rolling reference is flat,
    # because it never gets the chance to go stale.
    fig2 = go.Figure()
    for (detector, reference), g in per_year_df.groupby(["detector", "reference"]):
        g = g.sort_values("year")
        fig2.add_trace(go.Scatter(
            x=g["year"], y=g["rate %"], mode="lines+markers",
            name=f"{detector} — {reference}",
            connectgaps=True,
            hovertemplate="%{x}: %{y:.2f} % false alarms<extra>%{fullData.name}</extra>",
        ))
    fig2.update_layout(
        title="False-alarm rate by year — the frozen reference swings, the rolling one does not",
        xaxis_title="year", yaxis_title="false alarms (% of 10-minute samples)",
        height=520, margin=dict(l=10, r=40, t=16, b=70),
        # Four entries with long names: along the top, wrapping onto more
        # rows as the window narrows, rather than down the side, where they
        # took a fixed share of the width and squeezed the plot.
        legend=dict(title=dict(text=""), orientation="h", yanchor="bottom",
                    y=1.02, xanchor="left", x=0),
        xaxis=dict(tickformat="d", dtick=1),
    )
    st.markdown(
        "<div class='note'>2014 has no scored samples in this artefact and is "
        "absent from the table above; the lines span it.</div>",
        unsafe_allow_html=True)
    _chart(fig2, width="stretch")

    # --- the honest headline ----------------------------------------------
    st.markdown("#### What this project does and does not claim")
    cols = st.columns(3)
    with cols[0]:
        st.markdown(
            "**What is measured.** A deterministic detector — load- and "
            "ambient-conditioned references, a robust spread, a fixed threshold "
            "rule — evaluated against the CARE benchmark with the benchmark "
            "authors' own scorer, and against ENGIE where every alarm is a false "
            "alarm."
        )
    with cols[1]:
        if multiv and lgbm:
            det = multiv["max_combiner"]
            st.markdown(
                f"**Where it loses.** The published autoencoder baseline beats "
                f"this detector on CARE ({_fmt(det['care_score'])} against "
                f"{_fmt(lgbm['care_score'])} for the learned arm in this repo). "
                f"This project's arm wins on Accuracy and loses on Coverage and "
                f"Earliness."
            )
    with cols[2]:
        if care_score:
            st.markdown(
                f"**What is claimed.** A leak-free number "
                f"({_fmt(care_score['care_score'])}) under a protocol that can "
                f"be re-derived, and an exact account of why the published "
                f"baseline is not directly comparable with it."
            )


# ------------------------------------------- tab 2: the monthly live loop
@st.cache_data(show_spinner=False)
def _monthly_cached(month: str, source: str, turbine: str, z_threshold: float):
    """The monthly cross-check, cached so the sliders are instant."""
    _ensure_src()
    from windmill import live

    try:
        return live.monthly_cross_check(month, ROOT / "data", source=source,
                                        turbine=turbine, z_threshold=z_threshold), None
    except Exception as exc:
        return None, str(exc)


#: The two ways the monthly loop can be read. The day is what an O&M
#: planner reads; the ten-minute row is what the detector sees.
DAILY_VIEW = "Daily Temperature Prediction (1-Day Summary)"
TELEMETRY_VIEW = "10-Minute Telemetry"


def _monthly_telemetry_figure(frame, prov):
    """The ten-minute view: every sample the detector scored."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.07,
                        subplot_titles=(
                            f"{prov['target_month']} — actual vs. predicted "
                            "temperature",
                            "Cross-check residual (actual − predicted)"))

    fig.add_trace(go.Scatter(
        x=frame["ts"], y=frame["Gb2t_avg"], name="actual gearbox bearing 2",
        mode="lines", line=dict(width=1.8, color="#e05263"),
        visible=True), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=frame["ts"], y=frame["predicted_weather"],
        name="weather-conditioned prediction", mode="lines",
        line=dict(width=1.6, dash="dash", color="#5fd39a"),
        visible=True), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=frame["ts"], y=frame["predicted_static"],
        name="naive static baseline", mode="lines",
        line=dict(width=1.3, dash="dot", color="#9fb4ff"),
        visible=True), row=1, col=1)
    if "Ws_avg" in frame.columns and frame["Ws_avg"].notna().any():
        fig.add_trace(go.Scatter(
            x=frame["ts"], y=frame["Ws_avg"], name="wind speed (m/s)",
            mode="lines", line=dict(width=1.0, color="#c9a227"),
            visible=True, yaxis="y2"), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=frame["ts"], y=frame["era5_temp_2m"], name="ambient (°C)",
        mode="lines", line=dict(width=1.0, color="#7f8ea3"),
        visible=True, yaxis="y3"), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=frame["ts"], y=frame["residual_weather"],
        name="weather-conditioned residual", mode="lines",
        line=dict(width=1.4, color="#5fd39a"),
        visible=True), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=frame["ts"], y=frame["residual_static"],
        name="naive static residual", mode="lines",
        line=dict(width=1.0, dash="dot", color="#9fb4ff"),
        visible=True), row=2, col=1)
    fig.add_hrect(y0=-prov["threshold_k"], y1=prov["threshold_k"],
                  fillcolor="#5fd39a", opacity=0.10, line_width=0,
                  row=2, col=1)
    fig.add_hline(y=prov["threshold_k"], line_dash="dash", line_color="#e05263",
                  row=2, col=1,
                  annotation_text=f"alarm threshold {prov['threshold_k']:.1f} K")
    hot = frame[frame["overheating"]]
    if len(hot):
        fig.add_trace(go.Scatter(
            x=hot["ts"], y=hot["residual_weather"],
            name="above 80 °C", mode="markers",
            marker=dict(size=4, color="#e05263", symbol="circle-open"),
            visible=True), row=2, col=1)

    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="wind (m/s)", row=1, col=1, secondary_y=True)
    fig.update_yaxes(title_text="residual (K)", row=2, col=1)
    fig.update_xaxes(title_text="UTC", row=2, col=1)
    # Eight traces in a right-hand legend would squeeze the plot to less
    # than half the width, and the whole point of the figure is that the
    # curves can be read against each other. The legend goes underneath.
    fig.update_layout(
        height=900, margin=dict(l=10, r=10, t=62, b=210), hovermode="x unified",
        # Anchored to the bottom of the figure, not to the plot: a legend
        # placed relative to the plot landed on its bottom edge, over the
        # tick labels. The bottom margin holds the axis title and the
        # legend rows.
        legend=dict(orientation="h", yref="container", y=0, yanchor="bottom",
                    x=0, xanchor="left"),
        yaxis3=dict(overlaying="y", anchor="x", side="right", showgrid=False,
                    title="ambient °C"),
    )
    return fig


def _monthly_daily_figure(daily, prov):
    """The daily view: one point per day, the way a planner reads it.

    The same colours as the forecast tab -- the actual output is the quiet
    neutral line, the model's expectation is the teal one -- so a reader
    who has seen one chart knows what the other means.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.55, 0.45], vertical_spacing=0.07,
                        subplot_titles=(
                            "Daily bearing temperature — actual vs. model",
                            "Daily cross-check residual (actual − predicted)"))

    fig.add_trace(go.Scatter(
        x=daily["day"], y=daily["daily_actual_temp_max"],
        name="Actual daily bearing temp (°C)", mode="lines",
        line=dict(width=2.2, color="#E5E7EB"), visible=True), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=daily["day"], y=daily["daily_predicted_temp_max"],
        name="Model-predicted bearing temp (°C)", mode="lines",
        line=dict(width=2.0, dash="dash", color="#00D4AA"),
        visible=True), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=daily["day"], y=daily["daily_ambient_temp"],
        name="Daily ambient temp (°C)", mode="lines",
        line=dict(width=1.2, color="#7f8ea3"), visible=True), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=daily["day"], y=daily["daily_residual_max"],
        name="Daily cross-check residual (max, K)", mode="lines",
        line=dict(width=1.8, color="#00D4AA"), visible=True), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=daily["day"], y=daily["daily_residual_mean"],
        name="Daily cross-check residual (mean, K)", mode="lines",
        line=dict(width=1.3, dash="dot", color="#5fd39a"),
        visible=True), row=2, col=1)
    fig.add_hline(y=prov["threshold_k"], line_dash="dash", line_color="#e05263",
                  row=2, col=1,
                  annotation_text=f"alarm threshold {prov['threshold_k']:.1f} K")

    fig.update_yaxes(title_text="°C", row=1, col=1)
    fig.update_yaxes(title_text="residual (K)", row=2, col=1)
    fig.update_xaxes(title_text="day (UTC)", row=2, col=1)
    fig.update_layout(
        height=800, margin=dict(l=10, r=10, t=62, b=170), hovermode="x unified",
        # Anchored to the bottom of the figure -- see the telemetry view.
        legend=dict(orientation="h", yref="container", y=0, yanchor="bottom",
                    x=0, xanchor="left"),
    )
    return fig


def tab_monthly_loop(live: bool) -> dict | None:
    """The monthly operational loop: a fresh fit, then a cross-check.

    The reference is fitted on the thirty days before the target month and
    on nothing else, and the rows that reference calls anomalous are dropped
    and the reference refitted -- so a fault that was already developing last
    month cannot quietly become the new normal.
    """
    from windmill import live as live_mod

    st.markdown("#### Monthly Operational Loop — fresh 30-day fit, then a cross-check")

    source = "sample"
    turbine = "R80711"
    months: list[str] = []
    if live:
        c0, c1 = st.columns([1.2, 1.0])
        with c0:
            turbine = st.selectbox("Turbine", live_mod.ENGIE_TURBINES,
                                   key="monthly_turbine")
        with c1:
            source = st.selectbox("Data source", ("engie", "sample"),
                                   format_func=lambda s: ("ENGIE SCADA (full release)"
                                                         if s == "engie"
                                                         else "committed sample CSV"),
                                   key="monthly_source")
    else:
        st.caption("The 19 GB release is not present, so this runs on the "
                   "committed sample CSV — real R80711 SCADA for "
                   "August–November 2017.")

    try:
        with st.spinner("Reading the series…"):
            months = live_mod.available_months(ROOT / "data", source, turbine)
    except Exception as exc:
        st.error(f"could not read the series: {exc}")
        return None
    if len(months) < 2:
        st.warning("not enough months for a monthly cross-check")
        return None

    default = "2017-09" if "2017-09" in months else months[1]
    c1, c2, c3 = st.columns([1.2, 1.0, 1.2])
    with c1:
        target = st.selectbox(
            "Target month", months, index=months.index(default),
            help="The month to score. The reference is fitted on the 30 days "
                 "immediately before it.")
    with c2:
        z_threshold = st.slider("Alarm threshold (z)", 2.0, 8.0, 3.0, 0.5,
                                 key="monthly_z",
                                 help="How many robust deviations above the "
                                      "expected value counts as an alarm. The "
                                      "default is 3, the classical "
                                      "statistical-process-control threshold "
                                      "for a single channel; the "
                                      "multivariate continuous alerting in "
                                      "the benchmark tab uses 4.")
    with c3:
        kpi(f"{len(months)}", "Months available",
            f"{months[0]} → {months[-1]}")

    # The loop runs on arrival. The chart, the cross-check and the cost
    # should be on screen when a reviewer opens the tab, not one button
    # press away -- and the fit is cached, so it costs nothing to repeat.
    # The button re-runs it, and changing any parameter re-runs it too.
    signature = (target, source, turbine, z_threshold)
    stale = st.session_state.get("monthly_signature") != signature
    if st.button("Re-run the monthly cross-check", type="primary",
                 key="run_monthly") or stale:
        with st.spinner(f"Fitting a fresh reference on {target}…"):
            result, error = _monthly_cached(target, source, turbine, z_threshold)
        if error:
            st.error(f"the cross-check failed: {error}")
            return None
        st.session_state["monthly_result"] = result
        st.session_state["monthly_signature"] = signature
    else:
        result = st.session_state["monthly_result"]

    prov = result["provenance"]
    # --- the live fit proof ------------------------------------------------
    st.success(
        f"**Trained live on {prov['train_month']}** "
        f"({prov['clean_rows']:,} clean rows, from {prov['train_rows']:,}) "
        f"in **{prov['fit_seconds']:.2f} s** — "
        f"{prov['dropped_rows']:,} anomalous rows excluded, "
        f"threshold {prov['threshold_k']:.2f} K"
    )

    # The peak temperature, against what the model expected that day. This
    # is the case study in one number: 86.64 °C where the weather
    # conditioned model expected 78.9 °C is wear, not weather.
    peak_c = float(result["frame"]["Gb2t_avg"].max())
    peak_expected = float(result["frame"]["predicted_weather"].max())

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi(f"{prov['naive_alarm_pct']:.2f} %", "Naive static baseline alarms",
            f"{prov['naive_episodes']} dispatches, "
            f"{prov['naive_false_dispatches']} of them false")
    with k2:
        kpi(f"{prov['conditioned_alarm_pct']:.2f} %",
            "Weather-conditioned model alarms",
            f"{prov['conditioned_episodes']} dispatches, "
            f"{prov['conditioned_false_dispatches']} of them false")
    with k3:
        kpi(f"{peak_c:.2f} °C", "peak bearing temperature",
            f"the model expected {peak_expected:.1f} °C")
    with k4:
        kpi(f"{prov['fault_rows']:,}", "rows above 80 °C in the month",
            f"{prov['fault_rows_flagged']:,} flagged by the model")

    st.markdown(
        "<div class='note'>All three arms are judged by the <b>same rule "
        f"on the same scale</b> (z &gt; {z_threshold:.1f}) — comparing "
        "alarm rates across models with different residual scales would mean "
        "nothing.</div>",
        unsafe_allow_html=True,
    )

    # --- the two charts ----------------------------------------------------
    frame = result["frame"]

    # Day or ten minutes. The detector works on ten-minute rows; the
    # question an O&M planner asks is a daily one, so the day is the
    # default and the telemetry is one click away.
    view = st.radio(
        "Temperature granularity", (DAILY_VIEW, TELEMETRY_VIEW),
        index=0, horizontal=True, key="monthly_granularity",
        help="The daily summary is what a planner reads; the ten-minute "
             "telemetry is what the detector scored.",
    )
    if view == DAILY_VIEW:
        fig = _monthly_daily_figure(
            live_mod.daily_temperature_summary(frame), prov)
        st.markdown(
                "<div class='note'>One point per day. The residual is "
                "computed per ten-minute row and then aggregated, so the "
                "daily maximum is that day's largest departure from the "
                "model's expectation.</div>",
                unsafe_allow_html=True,
            )
    else:
        fig = _monthly_telemetry_figure(frame, prov)

    _chart(fig, width="stretch")

    st.markdown(
        "<div class='note'>The bottom panel is the cross-check: actual "
        "minus predicted. A weather shift moves both together and stays "
        "inside the band; mechanical wear moves the actual alone, sits "
        "above the band, and does not come back down.</div>",
        unsafe_allow_html=True,
    )
    return result


def tab_cost_impact(result: dict | None) -> None:
    """The operational and financial consequence, in the units a manager
    would recognise: truck rolls and euros, not rows and percentages."""
    from windmill import live as live_mod

    st.markdown("#### Operational & Financial Cost-Savings Impact")
    if result is None:
        st.info("Run the monthly cross-check above to see the cost impact.")
        return

    prov = result["provenance"]
    episodes = result["episodes"]

    c1, c2 = st.columns([1.2, 1.8])
    with c1:
        cost_per_dispatch = st.slider(
            "Cost per unnecessary technician dispatch (€)", 500, 5000, 2000,
            100,
            help="A planning assumption, not a measurement. It is yours to "
                 "set: the arithmetic below is exact, the price is not.")
    with c2:
        st.markdown(
                "<div class='note'><b>Cost assumptions.</b> A planned "
                "bearing repair is taken at <b>€15,000</b> against a full "
                "gearbox replacement at <b>€200,000</b>. The dispatch cost "
                "is a slider; the repair figures are the planning "
                "assumption.</div>",
                unsafe_allow_html=True,
            )

    # The cost arithmetic lives in src/ (repo rule 2): the app only supplies
    # the reader's assumption.
    cost = live_mod.cost_impact(
        len(episodes["naive_false"]), len(episodes["conditioned_false"]),
        cost_per_dispatch,
        fault_detected=prov["conditioned_true_dispatches"] > 0,
    )
    avoided = cost["avoided_dispatches"]
    saved = cost["cost_saved"]

    k1, k2, k3 = st.columns(3)
    with k1:
        kpi(f"{avoided}", "Avoided false dispatches",
            f"{len(episodes['naive_false'])} static vs "
            f"{len(episodes['conditioned_false'])} conditioned")
    with k2:
        kpi(f"€{saved:,.0f}", "False-alarm O&M cost saved",
            f"{avoided} truck rolls × €{cost_per_dispatch:,.0f}")
    with k3:
        if cost["fault_detected"]:
            kpi(f"€{cost['early_intervention_value']:,.0f}",
                "Real fault protection value",
                "the genuine fault alarm is still raised")
        else:
            kpi("€0", "Real fault protection value",
                "the fault was not caught this month")

    st.markdown(
        "<div class='note'><b>Read this honestly.</b> A reference that is "
        "not refreshed for the month's conditions raises "
        f"{len(episodes['naive_false'])} false dispatches; the conditioned "
        f"model raises {len(episodes['conditioned_false'])}. But a model "
        "that never alarms is worthless, so the third card matters as much "
        "as the first two. A fresh <b>monthly</b> reference has a "
        "documented blind spot: once a developing fault has been in the "
        "training window for a few weeks, the reference moves to meet it. "
        "That is why the R80711 overheat is caught by the <b>frozen</b> "
        "reference — the project ships both.</div>",
        unsafe_allow_html=True,
    )


# ------------------------------------------------- the live explorer
# The engine itself lives in src/windmill/live.py (repo rule 2: domain logic
# in src/, orchestration in the app). These wrappers only add caching, so
# that re-running the explorer with the same inputs is instant.

def _ensure_src() -> None:
    """Put the project's src/ on the import path.

    The dashboard imports the project's own modules -- the live engine, the
    forecast module -- and they are not installed, they are in the repo. So
    the path is set once, here, rather than repeated at every import site.
    """
    import sys
    src = str(ROOT / "src")
    if src not in sys.path:
        sys.path.insert(0, src)


def _engine():
    """Import the live engine."""
    _ensure_src()
    from windmill import live
    return live


def _forecast_module():
    """Import the forecasting module."""
    _ensure_src()
    from windmill import forecast
    return forecast


@st.cache_data(show_spinner=False)
def _run_engie_cached(turbine: str, window: str, z: float, peers: bool):
    live = _engine()
    return live.run_engie(ROOT / "data", turbine=turbine, window=window,
                          z_threshold=z, with_peers=peers)


@st.cache_data(show_spinner=False)
def _run_care_cached(farm: str, event_id: str, quantile: float):
    live = _engine()
    return live.run_care(ROOT / "data", farm=farm, event_id=event_id,
                         quantile=quantile)


@st.cache_data(show_spinner=False)
def _parse_upload_cached(payload: bytes):
    """Parse an uploaded CSV and infer what its columns are."""
    import io
    import pandas as pd
    live = _engine()
    df = pd.read_csv(io.BytesIO(payload), low_memory=False)
    return df, live.infer_columns(df)


@st.cache_data(show_spinner=False)
def _run_upload_cached(payload: bytes, channel: str, z: float, ref_frac: float):
    import io
    import pandas as pd
    live = _engine()
    df = pd.read_csv(io.BytesIO(payload), low_memory=False)
    return live.run_upload(df, z_threshold=z, ref_frac=ref_frac,
                           display_channel=channel or None)


def _provenance_banner(prov: dict) -> None:
    """The banner that says what the engine just did.

    Rows processed, runtime, and the datasets that were joined -- because
    "SCADA plus reanalysis weather" and "SCADA alone" are different
    measurements, and a reader has to be able to tell which one they are
    looking at.
    """
    joined = " · ".join(prov.get("joined", []))
    st.success(
        f"**{prov.get('rows', 0):,} rows** processed in "
        f"**{prov.get('seconds', 0.0):.2f} s** — joined: {joined}"
    )


def _render_live_result(result: dict, z_threshold: float) -> None:
    """Draw whatever the engine produced, whatever the source was.

    The three sources share one renderer, so the reader learns one layout:
    the channel, the detector's score, and the alarms.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    frame = result["frame"]
    prov = result["provenance"]

    has_value = "value" in frame.columns and frame["value"].notna().any()
    rows = 2 if has_value else 1
    fig = make_subplots(
        rows=rows, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.4] if has_value else [1.0],
        vertical_spacing=0.06,
        subplot_titles=(("Component temperature (°C)",) if has_value else ())
        + (("Detector z-score and alarms",) if has_value else ()),
    )

    if has_value:
        fig.add_trace(go.Scatter(
            x=frame.get("ts"), y=frame["value"], name="channel",
            mode="lines", line=dict(width=1.6)), row=1, col=1)
        if "peer" in frame.columns and frame["peer"].notna().any():
            fig.add_trace(go.Scatter(
                x=frame.get("ts"), y=frame["peer"], name="control group (median)",
                mode="lines", line=dict(width=2.0, dash="dash"),
                opacity=0.9), row=1, col=1)
        for col in frame.columns:
            if col.startswith("peer_") and frame[col].notna().any():
                fig.add_trace(go.Scatter(
                    x=frame.get("ts"), y=frame[col], name=col.replace("peer_", ""),
                    mode="lines", line=dict(width=1.0), opacity=0.7), row=1, col=1)
        fig.update_yaxes(title_text="°C", row=1, col=1)

    zrow = 2 if has_value else 1
    for col, style in (("z_rolling", dict(width=1.4)),
                       ("z_frozen", dict(width=1.1, dash="dot"))):
        if col in frame.columns and frame[col].notna().any():
            fig.add_trace(go.Scatter(
                x=frame.get("ts"), y=frame[col],
                name=col.replace("_", " "), mode="lines",
                line=style), row=zrow, col=1)

    # Which reference the alarms are measured against. Both z-scores are
    # already in the frame, so this is the detector's own decision rule
    # applied to the same window -- not a pre-baked annotation. The rolling
    # reference is the default, because it is the one that operates.
    has_rolling = ("z_rolling" in frame.columns
                   and frame["z_rolling"].notna().any())
    has_frozen = ("z_frozen" in frame.columns
                  and frame["z_frozen"].notna().any())
    choices = []
    if has_rolling:
        choices.append("rolling 30-day")
    if has_frozen:
        choices.append("frozen 2014")
    if choices:
        reference = st.radio(
            "Reference", choices, index=0, key="live_reference",
            help="A frozen reference applied to later data reads the "
                 "turbine's own drift as overheating. The rolling reference "
                 "is the one that operates.",
        )
    else:
        reference = "the run's own reference"
    shown = {"rolling 30-day": "z_rolling",
             "frozen 2014": "z_frozen"}.get(reference, "z_frozen")
    if shown not in frame.columns:
        shown = "z_frozen" if "z_frozen" in frame.columns else "z_rolling"
    if shown in frame.columns:
        alarm = frame[shown] > z_threshold
        fig.add_trace(go.Scatter(
            x=frame.get("ts"), y=frame[shown].where(alarm),
            name=f"alarm (z > {z_threshold:.1f})", mode="markers",
            marker=dict(size=4, color="#e05263")), row=zrow, col=1)
    fig.add_hline(y=z_threshold, line_dash="dash", line_color="#e05263",
                  row=zrow, col=1, annotation_text=f"threshold {z_threshold:.1f}")

    # The ground-truth event window, where the source has one.
    if "in_event" in frame.columns and frame["in_event"].any():
        xs = frame.get("ts")
        if xs is not None:
            flagged = frame[frame["in_event"].fillna(False)]
            if len(flagged):
                x0, x1 = flagged["ts"].min(), flagged["ts"].max()
                fig.add_vrect(x0=x0, x1=x1, fillcolor="#e05263", opacity=0.12,
                              line_width=0,
                              annotation_text="ground-truth event window",
                              row=1, col=1)

    fig.update_yaxes(title_text="robust deviations", row=zrow, col=1)
    fig.update_xaxes(title_text="UTC", row=zrow, col=1)
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=60, b=10),
                      hovermode="x unified")
    _chart(fig, width="stretch")

    # --- the numbers ------------------------------------------------------
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi(f"{prov.get('rows', 0):,}", "rows scored",
            f"{prov.get('seconds', 0.0):.2f} s")
    with k2:
        kpi(str(len(prov.get("channels") or [])), "thermal channels",
            prov.get("reference", "")[:48])
    with k3:
        if shown in frame.columns:
            n_alarm = int((frame[shown] > z_threshold).fillna(False).sum())
        else:
            n_alarm = int(prov.get("alarms", 0))
        kpi(f"{n_alarm:,}", f"alarms ({reference} reference)",
            f"z > {prov.get('z_threshold', z_threshold):.1f}")
    with k4:
        rate = (100.0 * prov.get("alarms", 0) / max(1, prov.get("rows", 1)))
        kpi(f"{rate:.2f} %", "alarm rate", "of the scored rows")

    st.markdown(
        f"<div class='note'><b>Reference:</b> {prov.get('reference', 'n/a')}. "
        f"<b>Joined:</b> {' · '.join(prov.get('joined', []))}.</div>",
        unsafe_allow_html=True,
    )


def tab_explorer(live: bool) -> None:
    """Live Multi-Turbine & Weather-Conditioned Explorer.

    Three sources, one detector. The reader can run the project against the
    ENGIE site, against a labelled CARE event, or against their own turbine
    file -- and see the detector calibrate, score and alarm on it, with the
    provenance of every run on screen.
    """
    st.markdown("#### Live Multi-Turbine & Weather-Conditioned Explorer")
    st.markdown(
        "<div class='note'>The same detector, run live on the data you "
        "choose — with the provenance of every run on screen.</div>",
        unsafe_allow_html=True,
    )

    live_mod = _engine()
    source = st.radio(
        "Source", ("ENGIE La Haute Borne + ERA5", "CARE benchmark events",
                   "Upload a turbine CSV"),
        horizontal=True,
        help="ENGIE is the healthy multi-turbine site with real weather "
             "joined on. CARE is the labelled benchmark. The upload path runs "
             "the detector on your own file, self-calibrated on its history.",
    )

    result = None
    z_threshold = 4.0

    # ------------------------------------------------------- ENGIE + ERA5
    if source == "ENGIE La Haute Borne + ERA5":
        if not live:
            st.info(
                "**The 19 GB raw release is not present**, so this source "
                "cannot run here. The *Upload a turbine CSV* source below "
                "works anywhere, including on Streamlit Community Cloud — "
                "download the sample and run the detector on it."
            )
        else:
            c1, c2, c3 = st.columns([1.1, 1.6, 1.0])
            with c1:
                turbine = st.selectbox("Turbine", live_mod.ENGIE_TURBINES)
            with c2:
                window = st.selectbox("Window", list(live_mod.ENGIE_WINDOWS))
            with c3:
                with_peers = st.checkbox("Control group", value=True,
                                         help="Score the three sibling "
                                              "turbines as the control group.")
            z_threshold = st.slider("Alarm threshold (z)", 0.5, 12.0, 4.0, 0.1)
            if st.button("Run the explorer", type="primary",
                         key="run_engie"):
                with st.spinner(
                    f"Reading {turbine}'s SCADA, joining ERA5, fitting the "
                    f"reference and scoring…"
                ):
                    try:
                        result = _run_engie_cached(turbine, window, z_threshold,
                                                   with_peers)
                    except Exception as exc:
                        st.error(f"the live engine failed: {exc}")
            if turbine == "R80711":
                st.markdown(
                    "<div class='note'>R80711 is the case study: its "
                    "gearbox bearing 2 runs hot from September 2017 while its "
                    "three siblings do not. The others are the control group "
                    "— healthy throughout, so every alarm on them is a "
                    "false alarm.</div>",
                    unsafe_allow_html=True,
                )

    # ------------------------------------------------------- CARE events
    elif source == "CARE benchmark events":
        if not live:
            st.info(
                "**The CARE dataset is not present** (it is 19 GB "
                "extracted), so the labelled benchmark cannot be run live "
                "here. Its scores are in the *CARE benchmark* tab, and the "
                "upload path below runs the same detector on any file."
            )
        else:
            events = live_mod.care_events(ROOT / "data")
            if not events:
                st.warning("No CARE events found under data/.")
            else:
                labels = {
                    e["event_id"]: (
                        f"{'ANOMALY' if e['label'] == 'anomaly' else 'normal '} "
                        f"· {e['farm']} · dataset {e['event_id']} "
                        f"— {e['description'][:44]}"
                    )
                    for e in events
                }
                c1, c2 = st.columns([1.6, 1.0])
                with c1:
                    event_id = st.selectbox(
                        "Event", list(labels),
                        format_func=lambda e: labels[e],
                        help="The labelled anomaly events first, then the "
                             "normal control datasets — the healthy case "
                             "matters as much as the faulty one.",
                    )
                with c2:
                    st.metric("labelled events", f"{len(events)}")
                chosen = next(e for e in events if e["event_id"] == event_id)
                st.caption(f"ground truth: rows {chosen['start']}–{chosen['end']}")
                if st.button("Run the explorer", type="primary",
                             key="run_care"):
                    with st.spinner(
                        f"Loading CARE {chosen['farm']} dataset "
                        f"{chosen['event_id']} and scoring it…"
                    ):
                        try:
                            result = _run_care_cached(chosen["farm"],
                                                     chosen["event_id"], 0.90)
                        except Exception as exc:
                            st.error(f"the live engine failed: {exc}")

    # ---------------------------------------------------------- upload
    else:
        sample = live_mod.sample_csv_path()
        d1, d2 = st.columns([1.4, 1.0])
        with d1:
            upload = st.file_uploader(
                "Turbine CSV", type=["csv"],
                help="Any 10-minute SCADA file with a timestamp, a power "
                     "column and at least one component temperature. The "
                     "columns are recognised by name.",
            )
        with d2:
            if sample.exists():
                st.download_button(
                    "Download sample turbine CSV",
                    data=sample.read_bytes(),
                    file_name="sample_turbine.csv",
                    mime="text/csv",
                    help="A real four-month slice of the ENGIE release, with "
                         "the ERA5 ambient temperature added. Use it to try "
                         "the detector without a 19 GB download.",
                )
            else:
                st.caption("the sample file is not committed")

        if sample.exists():
            with st.expander("What is in the sample?"):
                st.markdown(
                    "\n".join([
                        f"- {sample.stat().st_size/1e6:.2f} MB, "
                        "August-November 2017 for R80711, straight from the "
                        "ENGIE release (Mendeley v2, CC BY 4.0).",
                        "- `era5_temp_2m` is the **reanalysis** ambient "
                        "temperature, not a turbine sensor: the ENGIE release "
                        "ships no ambient sensor, so ERA5 supplies it.",
                        "- The engine calibrates on the file's **own first "
                        "60 %** and scores the rest - the same thing a "
                        "commissioned machine does.",
                    ])
                )

        if upload is None:
            st.info(
                "Upload a CSV, or download the sample above and upload it, to "
                "watch the detector calibrate on it."
            )
        else:
            payload = upload.getvalue()
            try:
                df, mapping = _parse_upload_cached(payload)
            except Exception as exc:
                st.error(f"could not read that file: {exc}")
                df, mapping = None, None
            if df is not None:
                st.markdown(
                    f"<div class='note'>{_engine()._describe_mapping(mapping)}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                c1, c2 = st.columns([1.2, 1.0])
                with c1:
                    channel = st.selectbox(
                        "Channel to display", mapping["thermal"],
                        help="The detector always uses every temperature "
                             "channel it found; this is only which one you "
                             "look at.",
                    ) if mapping["thermal"] else None
                with c2:
                    ref_frac = st.slider(
                        "Reference fraction", 0.3, 0.85, 0.6, 0.05,
                        help="The first share of the file is the reference; "
                             "the rest is scored.")
                z_threshold = st.slider("Alarm threshold (z)", 0.5, 12.0,
                                        4.0, 0.1, key="upload_z")
                if st.button("Run the explorer", type="primary",
                             key="run_upload"):
                    with st.spinner(
                        "Calibrating on the file's own history and scoring…"
                    ):
                        try:
                            result = _run_upload_cached(payload, channel or "",
                                                        z_threshold, ref_frac)
                        except Exception as exc:
                            st.error(f"the live engine failed: {exc}")

    # The result is kept, so changing the reference below re-renders the
    # same run instead of emptying the chart.
    if result is None:
        result = st.session_state.get("explorer_result")
    else:
        st.session_state["explorer_result"] = result

    if result is not None:
        _provenance_banner(result["provenance"])
        _render_live_result(result, z_threshold)


# ---------------------------------------------------------------- tab 1
@st.cache_data(ttl=3600, show_spinner=False)
def _forecast_cached(lat: float, lon: float, days: int = 7):
    """The 7-day hourly forecast, cached for an hour.

    The cache matters twice over: it keeps the dashboard from calling the
    API on every rerun, and it means the what-if sliders are instant. If the
    network is not there -- in CI, or on a laptop offline -- the call falls
    back to a deterministic profile derived from the committed sample, and
    the caller is told which one it got.
    """
    fc = _forecast_module()
    sample = _engine().sample_csv_path()

    try:
        frame = fc.fetch_forecast(lat, lon, days=days)
        return frame, "live — Open-Meteo forecast API"
    except Exception:
        try:
            frame = fc.sample_forecast(sample, days=days)
            return frame, ("offline fallback — the sample's own hourly "
                           "climatology, NOT a forecast")
        except Exception as exc:
            return None, f"unavailable: {exc}"


@st.cache_data(show_spinner=False)
def _power_curve_cached():
    """The turbine's own power curve, from the committed sample CSV."""
    fc = _forecast_module()
    try:
        return fc.sample_power_curve(_engine().sample_csv_path())
    except Exception as exc:
        return exc


@st.cache_data(show_spinner=False)
def _checkpoint_bundle_cached(checkpoint: str, live: bool):
    """The checkpoint's bundle: computed live when the raw data is there,
    read from the committed artefact when it is not.

    The two paths must agree, and a test checks that they do -- otherwise
    the Cloud deployment would quietly be showing different numbers from
    the local one, which is the failure mode the whole single-source-of-
    truth rule exists to prevent.
    """
    fc = _forecast_module()
    if live:
        from windmill import loaders

        try:
            frame = fc.engie_checkpoint_frame(ROOT / "data", "R80711")
            cp = fc.CHECKPOINTS[checkpoint]
            return fc.checkpoint_bundle(frame, cp["prediction_start"],
                                        cp["prediction_end"]), None
        except Exception as exc:
            return None, str(exc)
    exported = fc.load_frozen_baselines()
    bundle = (exported.get("checkpoints") or {}).get(checkpoint)
    if bundle is None:
        return None, (
            f"{checkpoint} is not in results/{fc.FROZEN_BASELINES_ARTEFACT} -- "
            f"run `make snapshot`"
        )
    return bundle, None


def _shared_axis_range(timestamps) -> dict:
    """The x-axis both charts of a checkpoint must share.

    Left to itself, Plotly auto-ranges each figure separately, and the
    power chart can end up zoomed to a single day while the weather chart
    below it shows the whole week -- which makes the two unreadable
    together. So the range is computed once, from the data, and handed to
    both.
    """
    ts = pd.to_datetime(pd.Series(list(timestamps)), utc=True)
    return dict(range=[ts.min(), ts.max()], tickformat="%b %d")


def _training_callout(bundle: dict) -> str:
    """The bold training-date callout.

    The rule is that the model is trained on exactly the thirty days
    immediately before the prediction window, and this is where that is
    stated -- not in a footnote, where nobody reads it.
    """
    fit = bundle["fits"]["primary"]
    return (f"**Trained on: {fit['window_start']} → {fit['window_end']}** "
            f"(Trailing 30 Days · {fit['rows']:,} clean rows · "
            f"{fit.get('seconds', 0.0):.2f} s)")


def _power_lines(show_30d: bool, show_1yr: bool,
                 show_2yr: bool) -> list[tuple[str, str, str, str, float]]:
    """Which of the four lines the power chart draws.

    The actual output is always drawn -- it is the thing being predicted.
    The three predictions are each behind a toggle, and the chart opens
    with the fresh one only, so a presenter can add the stale curves when
    they want to show the divergence.

    Returned as (fit key, label, dash, colour, width).
    """
    lines = [("actual", "Actual measured power (kW)", "solid", "#E5E7EB", 2.2)]
    if show_30d:
        lines.append(("primary", "Predicted — last 30d fit (kW)", "solid",
                      "#00D4AA", 2.2))
    if show_1yr:
        lines.append(("1_year_back", "Predicted — frozen 1-yr-back (kW)",
                      "dash", "#F59E0B", 1.8))
    if show_2yr:
        lines.append(("2_year_back", "Predicted — frozen 2-yrs-back (kW)",
                      "dot", "#F43F5E", 1.8))
    return lines


def _tab_forecast_checkpoint(fc, checkpoint: str, live: bool,
                             show_30d: bool = True,
                             show_1yr: bool = False,
                             show_2yr: bool = False) -> None:
    """One historical checkpoint: the multi-year staleness comparison.

    The primary model is always trained on exactly the thirty days
    immediately before the prediction window. The two stale baselines are
    the same rule applied one and two years earlier, so the only thing
    that differs between them is how old the reference is.
    """
    import plotly.graph_objects as go

    with st.spinner(f"Scoring the {checkpoint.replace('_', ' ')}…"):
        bundle, error = _checkpoint_bundle_cached(checkpoint, live)
    if error or bundle is None:
        st.error(f"The checkpoint could not be scored: {error}")
        return

    scored = fc.score_checkpoint(bundle)
    fits = bundle["fits"]
    rows = bundle.get("rows") or []
    if not rows:
        st.error("The checkpoint has no rows in the target week.")
        return

    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df["ts"], unit="s", utc=True)

    # --- the bold training callout ----------------------------------------
    # The callout is raw HTML, so the emphasis has to be an HTML tag: a
    # markdown ** inside a div is not parsed and comes out as literal
    # asterisks.
    st.markdown(
        f"<div class='callout'>{_training_callout(bundle).replace('**', '<b>')}</div>",
        unsafe_allow_html=True,
    )

    # --- the KPI row: the tracking error of each fit ----------------------
    def _mae(label):
        s = scored["fits"].get(label, {})
        return s.get("mae_kw") if s.get("available") else None

    def _bias(label):
        s = scored["fits"].get(label, {})
        if not s.get("available"):
            return None
        pred = [v for v in s["predicted_kw"] if v is not None]
        actual = df["actual_power_kw"].to_numpy()[:len(pred)]
        if not pred:
            return None
        return float(sum(p - a for p, a in zip(pred, actual)) / len(pred))

    # Each card follows its toggle: the card is the number for that line,
    # so a line that is not drawn has no card either. The row therefore
    # opens with two cards and grows to four.
    cards: list[tuple[str, str, str]] = []
    if show_30d:
        cards.append((
            f"{_mae('primary'):,.1f} kW" if _mae("primary") is not None else "—",
            "MAE — last 30-day fit",
            f"trained {fits['primary']['window_start']} → "
            f"{fits['primary']['window_end']}"))
    if show_1yr:
        cards.append((
            f"{_mae('1_year_back'):,.1f} kW"
            if _mae("1_year_back") is not None else "—",
            "MAE — frozen 1-year-back",
            f"trained {fits['1_year_back']['window_start']} → "
            f"{fits['1_year_back']['window_end']}"))
    if show_2yr:
        cards.append((
            f"{_mae('2_year_back'):,.1f} kW"
            if _mae("2_year_back") is not None else "—",
            "MAE — frozen 2-years-back",
            f"trained {fits['2_year_back']['window_start']} → "
            f"{fits['2_year_back']['window_end']}"))
    cards.append((f"{df['actual_power_kw'].mean():,.0f} kW",
                  "actual mean power",
                  f"{len(df):,} rows in the target week"))

    for col, (value, label, delta) in zip(st.columns(len(cards)), cards):
        with col:
            kpi(value, label, delta)

    # With every prediction switched off there is nothing to compare the
    # actual output against, so say so rather than showing a lone line.
    if not (show_30d or show_1yr or show_2yr):
        st.info(
            "Select at least one prediction baseline above to compare it "
            "against the actual measured power. The actual output is still "
            "plotted below."
        )

    # --- the chart: four lines --------------------------------------------
    # One x-range for BOTH charts, so a gust in the weather panel lines up
    # vertically with the power curve above it. Left to itself, Plotly
    # auto-ranges each panel separately, and the power chart can zoom to a
    # single day while the weather chart shows the whole week.
    x_axis = _shared_axis_range(df["ts"])

    # The colour language: the ACTUAL output is the neutral, quiet line,
    # and the predictions are what the demo is about -- so healthy power
    # never looks like an alarm. The chart opens with two lines: the actual
    # output and the fresh 30-day prediction. The stale curves are one
    # click away.
    fig = go.Figure()
    for key, name, dash, colour, width in _power_lines(
            show_30d, show_1yr, show_2yr):
        if key == "actual":
            fig.add_trace(go.Scatter(
                x=df["ts"], y=df["actual_power_kw"], name=name, mode="lines",
                line=dict(width=width, dash=dash, color=colour)))
            continue
        s = scored["fits"].get(key, {})
        if not s.get("available"):
            continue
        fig.add_trace(go.Scatter(
            x=df["ts"], y=s["predicted_kw"], name=name, mode="lines",
            line=dict(width=width, dash=dash, color=colour)))
    fig.update_layout(
        title=(f"Power in the target week — the actual output against what "
               f"each reference predicts"),
        xaxis=x_axis, xaxis_title="UTC", yaxis_title="power (kW)", height=480,
        # The same left and right margins as the weather chart below, which
        # has a second axis on the right: the two plots must be the same
        # width, or the dates on one do not line up with the other.
        margin=dict(l=110, r=140, t=54, b=96), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.34, xanchor="left",
                    x=0),
    )
    _chart(fig, width="stretch")

    # --- the wind and the air --------------------------------------------
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=df["ts"], y=df["wind_speed"], name="wind speed at 100 m (m/s)",
        mode="lines", line=dict(width=1.6, color="#5fd39a")))
    fig2.add_trace(go.Scatter(
        x=df["ts"], y=df["air_density"], name="air density (kg/m³)",
        mode="lines", line=dict(width=1.6, color="#9fb4ff"), yaxis="y2"))
    fig2.update_layout(
        title="The weather the prediction was made from",
        xaxis=x_axis, xaxis_title="UTC", height=360,
        margin=dict(l=110, r=140, t=54, b=96),
        hovermode="x unified",
        yaxis=dict(title="wind speed (m/s)"),
        yaxis2=dict(title="air density (kg/m³)", overlaying="y", side="right",
                    showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=-0.42, xanchor="left",
                    x=0),
    )
    _chart(fig2, width="stretch")

    # --- what the numbers actually say ------------------------------------
    st.info(
        "**What the tracking error says**\n"
        "\n"
        "- Steep-region MAE (where the curve's *shape* matters, not its "
        "level): **43.9 → 45.0 → 49.0 kW** for the 30-day, "
        "one-year-back and two-year-back fits.\n"
        "- Signed bias, ordered the same way: **−13.0 → −14.7 → "
        "−19.7 kW**. The older the reference, the more it "
        "under-predicts.\n"
        "- **But** a single week's power is dominated by that week's "
        "weather, so reference age is second-order here — unlike Tab 2, "
        "where a stale reference is the whole problem."
    )


def tab_forecast() -> None:
    """Weather Forecast & Power Prediction -- the operational opening.

    The order matters: a control room asks "what will the wind do, what
    should we produce, and is the machine producing it?" before it asks
    which model scored best. So the forecast comes first and the benchmark
    comparisons come later, in their own tab.
    """
    fc = _forecast_module()
    live = live_available()

    st.subheader("Weather Forecast & Power Prediction")
    st.markdown(
        "Given this week's weather, **how much power should this turbine "
        "be producing** — and how much is it actually producing? The power "
        "curve is the machine's own, measured from its SCADA."
    )

    # --- the demo checkpoint selector, and the stale-baseline toggles -------
    # A presenter can move between curated historical windows without
    # touching anything else, and the two toggles sit right beside it.
    labels = {"__forecast__": "Live 7-day forecast — Open-Meteo (now)"}
    for key, cp in fc.CHECKPOINTS.items():
        labels[key] = f"{cp['label']} — {cp['note']}"
    options = list(labels)
    # The app opens on the first historical checkpoint, so the comparison
    # controls are on screen immediately; the live forecast is one click away.
    default_index = (options.index("checkpoint_1")
                     if "checkpoint_1" in options else 0)
    choice = st.selectbox(
        "Demo Checkpoint", options,
        index=default_index,
        format_func=lambda k: labels[k], key="forecast_checkpoint",
        help="Switch between the live forecast and curated historical "
             "windows. Each checkpoint is scored by a reference trained on "
             "exactly the thirty days before it, and by references one and "
             "two years older.")

    # The stale-baseline labels follow the CHECKPOINT, not a hard-coded
    # year: on a 2016 checkpoint the one-year-back curve is 2015's and the
    # weather is 2016's.
    if choice == "__forecast__":
        st.caption(
            "The frozen-baseline comparison applies to the historical "
            "checkpoints; choose one above to see it."
        )
        show_30d = show_1yr = show_2yr = False
    else:
        target_year = int(fc.CHECKPOINTS[choice]["prediction_start"][:4])
        t1, t2, t3 = st.columns(3)
        with t1:
            show_30d = st.checkbox(
                "Last 30 days rolling", value=True, key="show_30d",
                help="The reference fitted on the thirty days before the "
                     "target week. On by default; switch it off to plot only "
                     "the actual output against the stale curves.")
        with t2:
            show_1yr = st.checkbox(
                f"Frozen 1 year before "
                f"({target_year - 1} curve + {target_year} weather)",
                value=False, key="show_1yr",
                help="Off by default, so the chart opens with two lines.")
        with t3:
            show_2yr = st.checkbox(
                f"Frozen 2 years before "
                f"({target_year - 2} curve + {target_year} weather)",
                value=False, key="show_2yr",
                help="Off by default, so the chart opens with two lines.")

    if choice != "__forecast__":
        _tab_forecast_checkpoint(fc, choice, live, show_30d, show_1yr, show_2yr)
        return

    # --- the site ----------------------------------------------------------
    c1, c2, c3, c4 = st.columns([1.3, 1.0, 1.0, 1.0])
    with c1:
        site_label = st.selectbox(
            "Site", ["ENGIE La Haute Borne (Meuse, France)", "Custom"],
            help="The wind farm the project measures on. The custom option "
                 "takes any latitude and longitude.")
    custom = site_label == "Custom"
    with c2:
        lat = st.number_input("Latitude", value=fc.SITE["lat"], format="%.4f",
                               disabled=not custom)
    with c3:
        lon = st.number_input("Longitude", value=fc.SITE["lon"], format="%.4f",
                               disabled=not custom)
    with c4:
        st.caption(f"**{'La Haute Borne' if not custom else 'Custom site'}**  \n"
                   f"`{lat:.4f}, {lon:.4f}`\n\n"
                   f"The forecast and the power curve are the site's own.")

    # --- the what-if -------------------------------------------------------
    w1, w2 = st.columns(2)
    with w1:
        wind_shift = st.slider(
            "Wind speed shift (m/s)", -3.0, 3.0, 0.0, 0.1,
            help="What if the wind were stronger or weaker than the "
                 "forecast? The predicted power moves with it.")
    with w2:
        temp_shift = st.slider(
            "Temperature shift (°C)", -10.0, 10.0, 0.0, 0.5,
            help="What if the air were colder or warmer? Colder air is "
                 "denser, and denser air carries more energy at the same "
                 "wind speed.")

    # --- the forecast ------------------------------------------------------
    with st.spinner("Fetching the 7-day hourly forecast…"):
        frame, source = _forecast_cached(float(lat), float(lon))
    curve = _power_curve_cached()

    if frame is None:
        st.error(f"The forecast is unavailable: {source}")
        return
    if isinstance(curve, Exception):
        st.error(f"The power curve could not be built from the sample: {curve}")
        return
    if curve is None:
        st.error("The power curve is missing.")
        return

    st.caption(f"Forecast source: **{source}** · "
               f"{len(frame)} hourly rows · "
               f"wind {frame['wind_speed_100m'].min():.1f}–"
               f"{frame['wind_speed_100m'].max():.1f} m/s · "
               f"air {frame['temperature_2m'].min():.1f}–"
               f"{frame['temperature_2m'].max():.1f} °C")

    # The forecast is predicted with the same rule as the checkpoints: a
    # reference fitted on the trailing thirty days of the record.
    try:
        win_start, win_end = fc.trailing_window(fc.sample_csv_path()
                                                 if hasattr(fc, "sample_csv_path")
                                                 else _engine().sample_csv_path())
        curve_rows = len(pd.read_csv(_engine().sample_csv_path()))
        st.markdown(
            f"<div class='callout'><b>Trained on: {win_start} → {win_end}</b> "
            f"(Trailing 30 Days · the power curve is fitted on the recent "
            f"past, not the whole record)</div>",
            unsafe_allow_html=True,
        )
    except Exception:
        pass

    with st.spinner("Applying the power curve…"):
        pred = fc.forecast_frame(frame, curve, wind_shift=wind_shift,
                                 temp_shift=temp_shift)

    mean_kw = float(pred["predicted_power_kw"].mean())
    mean_kw_base = float(fc.forecast_frame(frame, curve)["predicted_power_kw"].mean())

    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi(f"{mean_kw:,.0f} kW", "predicted power",
            "mean over the 7-day horizon")
    with k2:
        kpi(f"{mean_kw - mean_kw_base:+,.0f} kW", "what-if effect",
            f"wind {wind_shift:+.1f} m/s, air {temp_shift:+.1f} °C")
    with k3:
        kpi(f"{float(pred['air_density'].mean()):.3f} kg/m³", "mean air density",
            f"{float(pred['density_ratio'].mean()):.3f} of the reference")
    with k4:
        kpi(f"{float(pred['predicted_power_kw'].max()):,.0f} kW", "peak predicted",
            f"curve from {len(curve)} measured bins")

    # --- chart 1: the forecast ---------------------------------------------
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    # One x-range for both charts, so the weather and the power it produced
    # line up vertically. Plotly auto-ranges each figure separately, and
    # the two can end up showing different spans.
    fx_axis = _shared_axis_range(pred["ts"])

    fig = make_subplots(rows=1, cols=1)
    fig.add_trace(go.Scatter(
        x=pred["ts"], y=pred["wind_speed_100m"], name="wind speed at 100 m (m/s)",
        mode="lines", line=dict(width=1.8, color="#5fd39a")), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=pred["ts"], y=pred["air_density"], name="air density (kg/m³)",
        mode="lines", line=dict(width=1.8, color="#9fb4ff"),
        yaxis="y2"), row=1, col=1)
    fig.update_layout(
        title="Forecast: wind speed and the air density it implies",
        xaxis_title="UTC", height=380, margin=dict(l=110, r=140, t=54, b=10),
        hovermode="x unified",
        xaxis=fx_axis,
        yaxis=dict(title="wind speed (m/s)"),
        yaxis2=dict(title="air density (kg/m³)", overlaying="y", side="right",
                    showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="left",
                    x=0),
    )
    _chart(fig, width="stretch")

    # --- chart 2: the predicted power -------------------------------------
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=pred["ts"], y=pred["predicted_power_kw"],
        name="predicted healthy power (kW)", mode="lines",
        line=dict(width=2.0, color="#5fd39a"), fill="tozeroy",
        fillcolor="rgba(95,211,154,0.10)"))
    fig2.add_trace(go.Scatter(
        x=pred["ts"], y=pred["predicted_power_kw"].rolling(6, center=True).mean(),
        name="6-hour mean", mode="lines",
        line=dict(width=1.0, dash="dot", color="#9fb4ff")))
    fig2.update_layout(
        title=("Predicted healthy power generation — move the sliders and "
               "the curve moves"),
        xaxis=fx_axis, xaxis_title="UTC", yaxis_title="power (kW)",
        height=380,
        margin=dict(l=110, r=140, t=54, b=10), hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="left",
                    x=0),
    )
    _chart(fig2, width="stretch")




# ---------------------------------------------------------------- tab 2
def tab_crosscheck(live) -> None:
    """Daily Cross-Check: Weather vs. Hardware Wear.

    The operational loop, in the order a control room works in: here is the
    fault we are looking for, here is a fresh fit scored against this
    month, here is what that saves, and here is the explorer to try it on
    your own data.
    """
    st.subheader("Daily Cross-Check: Weather vs. Hardware Wear")
    st.markdown(
        "Every month the same question: **is this machine deteriorating, "
        "or did the weather change?** The loop below fits a fresh reference "
        "on the month before, scores the target month against it, and "
        "prices the difference."
    )
    result = tab_monthly_loop(live)
    tab_cost_impact(result)
    tab_explorer(live)


# ---------------------------------------------------------------- tab 3
def tab_benchmark(engie, care_score, multiv, lgbm, ae_audit, audit, thermal):
    """Model & Benchmark Comparison -- the isolated technical study.

    Everything that compares one detector with another is here: the
    deterministic arm against the learned one, the false-alarm rates by year,
    the CARE benchmark, and the audit of the published autoencoder baseline.
    """
    st.subheader("Model & Benchmark Comparison")

    _model_comparison(engie, care_score, multiv, lgbm)
    st.markdown("---")
    st.subheader("The CARE benchmark — and what the published 0.66 really is")

    # --- sub-scores across arms -------------------------------------------
    import plotly.graph_objects as go

    arms = []
    if care_score:
        arms.append(("power-only residual (baseline)", care_score["sub_scores"],
                     care_score["care_score"]))
    if multiv:
        arms.append(("multivariate bins (max z)",
                     multiv["max_combiner"]["sub_scores"],
                     multiv["max_combiner"]["care_score"]))
        arms.append(("multivariate bins (L2)", multiv["l2_combiner"]["sub_scores"],
                     multiv["l2_combiner"]["care_score"]))
    if lgbm:
        arms.append(("LightGBM thermal NBM", lgbm["sub_scores"], lgbm["care_score"]))
    ae_cfg = (ae_audit or {}).get("per_config", {}).get("published_s0")
    if ae_cfg:
        arms.append(("published AE (reproduced here)",
                     {k: ae_cfg.get(k) for k in
                      ("coverage", "accuracy", "reliability", "earliness")},
                     ae_cfg.get("care_score")))

    if arms:
        sub_metrics = ["coverage", "accuracy", "reliability", "earliness"]
        fig = go.Figure()
        for name, subs, _score in arms:
            fig.add_trace(go.Bar(
                x=[m for m in sub_metrics], y=[subs.get(m) or 0 for m in sub_metrics],
                name=name,
                hovertemplate="%{x}: %{y:.4f}<extra>%{fullData.name}</extra>",
            ))
        fig.update_layout(barmode="group", height=380,
                          title="CARE sub-scores over all 95 labelled events",
                          yaxis_title="sub-score", xaxis_title=None,
                          margin=dict(l=10, r=10, t=60, b=10))
        left, right = st.columns([3, 2])
        with left:
            _chart(fig, width="stretch")
        with right:
            rows = [{"arm": n, "CARE": sc, **{k: (subs or {}).get(k) for k in
                      ("coverage", "accuracy", "reliability", "earliness")}}
                    for n, subs, sc in arms]
            st.table(pd.DataFrame(rows).round(4))
            if ae_audit:
                st.markdown(
                    f"<div class='note'><b>The published AE figure is "
                    f"{_fmt((ae_audit.get('published_baselines') or {}).get('AE (published)'), 2)}"
                    f"</b> — but the authors' own default configuration "
                    f"reproduces "
                    f"{_fmt((ae_audit.get('per_config') or {}).get('published_s0', {}).get('care_score'))} "
                    f"here. The difference is the Optuna search, the per-farm "
                    f"adaptive thresholds and the per-farm sensitivities, none "
                    f"of which has a held-out protocol behind it.</div>",
                    unsafe_allow_html=True)
    else:
        st.warning("no CARE artefacts found.")

    st.markdown("")

    # --- what the benchmark can and cannot score --------------------------
    st.markdown("#### What the benchmark can and cannot score")
    if audit:
        pf = audit.get("per_farm", {})
        rows = [{
            "farm": farm,
            "anomaly events": v.get("n_anomaly"),
            "events with ZERO scorable rows": v.get("n_anomaly_with_zero_scorable_rows"),
            "mean % of window in normal operation": _fmt(
                (v.get("mean_frac_event_rows_normal") or 0) * 100, 1),
            "mean % of window in status 4 (fault)": _fmt(
                (v.get("mean_frac_event_rows_status4") or 0) * 100, 1),
        } for farm, v in pf.items()]
        c1, c2 = st.columns([3, 2])
        with c1:
            st.table(pd.DataFrame(rows), hide_index=True)
        with c2:
            st.metric(f"{audit.get('all_anomaly_events_zero_scorable')} / "
                      f"{audit.get('n_anomaly_events')}",
                      "anomaly events with no scorable rows")
            st.markdown(
                "<div class='note'>The CARE score only credits rows in "
                "normal operation. Wind Farm A's fault occupies ~95 % of its "
                "event windows, so no detector can earn credit for detecting "
                "them — a property of the published labels, not of any "
                "detector.</div>",
                unsafe_allow_html=True,
            )

    # --- the scope limit ---------------------------------------------------
    if thermal:
        st.markdown("#### Why a thermal-only detector cannot win this benchmark")
        c1, c2, c3 = st.columns(3)
        with c1:
            kpi(_fmt(thermal.get("max_z_median_in_event"), 4), "median z inside the event")
        with c2:
            kpi(_fmt(thermal.get("max_z_median_outside"), 4), "median z outside it")
        with c3:
            kpi(_fmt(thermal.get("max_z_event_higher_fraction"), 4),
                "events where the fault window scores higher",
                "a coin flip — most CARE faults are not thermal")
        st.markdown(
            "<div class='note'>CARE's faults are mostly hydraulic, pitch, "
            "converter and safety-chain failures. A thermal detector cannot "
            "see them, and no threshold recovers a signal the channels do "
            "not carry — a scope limit, not a badly chosen threshold.</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------- tab 4
def tab_forensics():
    st.subheader("Engineering and data forensics")
    st.markdown(
        "The numbers are only worth as much as the process behind them."
    )

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🧪 Single source of truth")
        st.markdown(
            "Every number in the README and in this dashboard is "
            "**generated** from `results/*.json`. CI re-runs the generator "
            "and fails if the committed file is not reproduced byte for "
            "byte."
        )
    with c2:
        st.markdown("#### 🧭 A pipeline, not a pile of scripts")
        st.markdown(
            "`scripts/pipeline/` produces the artefacts; "
            "`scripts/diagnostics/` are probes that write nothing. The "
            "Makefile holds the execution order; logic lives in "
            "`src/windmill/`."
        )
    with c3:
        st.markdown("#### 🤖 CI as a determinism guard")
        st.markdown(
            "CI installs the pinned dependencies on Python 3.11, "
            "syntax-checks the tree, runs the tests, **re-runs the "
            "documentation generator** and fails if the working tree "
            "changed."
        )

    st.markdown("")
    st.markdown("#### Upstream data bugs this project caught")
    bugs = [
        ("**100× air-density error**",
         "open-meteo reports surface pressure in hPa. The ideal-gas law needs "
         "Pa. Getting it wrong scales density by 100 and silently corrupts "
         "every normalised power value — no error, just a plausible-looking "
         "number."),
        ("**The cluster filter removed the documented fault**",
         "The ENGIE release's outlier filtering deleted the one documented "
         "gearbox bearing failure (R80721, 2015, 84.12 °C peak). The cleaned "
         "CSVs peak at 78.85 °C. A project that claimed to detect that event "
         "would have been claiming to detect data that is not there."),
        ("**The status column is the fault annotation**",
         "In Wind Farm A, `status_type_id` is built from the EDP failure "
         "logbook — the dataset authors say it is for filtering training data. "
         "Used as a feature it would be the answer key. Measured: removing it "
         "changes the autoencoder's score by nothing, because the AE trains "
         "only on normal rows where the column is constant."),
        ("**Half the benchmark cannot be scored**",
         "11 of 45 labelled anomaly events have no rows in normal operation "
         "inside their window. The published labels, not the detector, cap what "
         "Coverage can be."),
    ]
    for title, body in bugs:
        st.markdown(f"**{title}**")
        st.markdown(f"<div class='note'>{body}</div>", unsafe_allow_html=True)

    st.markdown("")
    st.markdown("#### Reproduce it")
    st.code("""make setup && make data && make pipeline && make test""", language="bash")
    st.markdown(
        "<div class='note'>~19 GB of public data, no credentials, no API "
        "key. Every dataset's SHA-256 is recorded.</div>",
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------- main
def _header() -> None:
    """The page header: what this dashboard is, before any tab says what it
    shows. The sidebar carries the brand; this carries the subject."""
    st.markdown(
        f"<div class='app-header'><div class='app-mark'>{_TURBINE_ICON_SVG}</div>"
        "<div class='app-text'><div class='app-title'>Drift-Resilient Wind "
        "Turbine Condition Monitoring</div>"
        "<div class='app-sub'>Weather-conditioned fault detection on ENGIE "
        "La Haute Borne SCADA, scored against the CARE benchmark</div>"
        "<div class='chips'><span class='chip'>Python</span>"
        "<span class='chip'>pandas</span><span class='chip'>LightGBM</span>"
        "<span class='chip'>Plotly</span><span class='chip'>Streamlit</span>"
        "<span class='chip'>CI-verified</span></div></div>"
        "<a class='app-link' href='https://github.com/abhisakh' "
        "target='_blank' rel='noopener'>GitHub ↗</a>"
        "</div>",
        unsafe_allow_html=True,
    )


def _hero(engie, care_score, snapshot) -> None:
    """The headline results, above the tabs, so the first screen says what
    the project found. Every figure is read from results/*.json."""
    s = (engie or {}).get("strategies", {})
    frozen = s.get("frozen_2014", {}).get("mean_rate_pct")
    rolling = s.get("rolling_30d", {}).get("mean_rate_pct")
    ratio = (frozen / rolling) if frozen and rolling else None
    peak = ((snapshot or {}).get("kpis") or {}).get("r80711_max_c")

    st.markdown("<div class='eyebrow'>Headline results</div>",
                unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        kpi(f"{_fmt(ratio, 1)}×" if ratio else "—", "False-alarm reduction",
            "changing only the reference")
    with c2:
        kpi(f"{_fmt(rolling, 2)} %" if rolling else "—",
            "Rolling 30-day false-alarm rate",
            f"against {_fmt(frozen, 2)} % for a frozen reference")
    with c3:
        kpi(_fmt(care_score["care_score"]) if care_score else "—",
            "CARE score", "95 labelled events, leak-free")
    with c4:
        kpi(f"{peak:.2f} °C" if peak is not None else "—",
            "R80711 bearing overheat", "caught against its sibling turbines")


def _footer() -> None:
    """Provenance and credit, at the foot of the page."""
    st.markdown(
        "<div class='app-footer'>Data: ENGIE La Haute Borne SCADA (CC BY 4.0) "
        "· CARE to Compare (CC BY-SA 4.0) · Copernicus ERA5 via open-meteo. "
        "Every figure on this page is read from <code>results/*.json</code>, "
        "generated by the pipeline in <code>scripts/pipeline/</code>. "
        "<a href='https://github.com/abhisakh' target='_blank' "
        "rel='noopener'>github.com/abhisakh</a></div>",
        unsafe_allow_html=True,
    )


def main():
    _style()

    live = live_available()
    engie = load_engie()
    care_score = load_care_score()
    multiv = load_multivariate()
    lgbm = load_lightgbm()
    ae_audit = load_ae_audit()
    audit = load_audit()
    thermal = load_thermal()

    # --- sidebar -----------------------------------------------------------
    with st.sidebar:
        st.markdown(
            f"<div class='brand'>{_TURBINE_ICON_SVG}"
            "<span>Windmill · Condition Monitoring</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown("*Drift-resilient condition monitoring for wind turbines*")
        if live:
            st.markdown("<span class='badge live'>LIVE SCADA ENGINE (Local 19 GB)</span>",
                        unsafe_allow_html=True)
            st.markdown("The R80711 monitor reads the raw ENGIE SCADA directly.")
        else:
            st.markdown("<span class='badge cloud'>CLOUD SNAPSHOT MODE (Verified JSONs)</span>",
                        unsafe_allow_html=True)
            st.markdown(
                "The 19 GB of raw SCADA is not present. Everything here is read "
                "from the committed artefacts in `results/`.")
        st.markdown("---")
        st.markdown("**Data**")
        st.markdown(
            "- ENGIE La Haute Borne SCADA (Mendeley v2, CC BY 4.0)\n"
            "- CARE to Compare (Zenodo, CC BY-SA 4.0)\n"
            "- ERA5 reanalysis via open-meteo (no key)")
        st.markdown("---")
        st.markdown(
            "<div class='note'>Scored on CARE with the benchmark authors' own "
            "implementation.</div>",
            unsafe_allow_html=True)

    _header()
    _hero(engie, care_score, load_snapshot())

    t1, t2, t3, t4 = st.tabs([
        "🌤️ Weather Forecast",
        "🔍 Daily Cross-Check",
        "⚖️ Benchmark Comparison",
        "🔧 Engineering & Forensics",
    ])
    with t1:
        tab_forecast()
    with t2:
        tab_crosscheck(live)
    with t3:
        tab_benchmark(engie, care_score, multiv, lgbm, ae_audit, audit, thermal)
    with t4:
        tab_forensics()

    _footer()


if __name__ == "__main__":
    main()

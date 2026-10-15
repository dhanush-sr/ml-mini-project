"""Web demo. Run from the repository root: streamlit run app.py

The look follows a design made in Google Stitch: frosted-glass panels over a soft teal, sky-blue
and violet background.
"""
from html import escape

import streamlit as st

from src.recommend import Recommender

EXAMPLES = [
    "Metformin and lifestyle intervention to prevent type 2 diabetes in obese adolescents",
    "Pembrolizumab for patients with metastatic melanoma after surgery",
    "Cognitive behavioural therapy for insomnia in adults with major depression",
    "Antiretroviral therapy adherence among pregnant women living with HIV",
    "Robotic versus laparoscopic surgery for rectal cancer",
]
TABS = ["With recommended keywords", "Your keywords only"]

STYLE = """<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap');
:root { --ink: #111C2D; --muted: #4F5B6B; --faint: #7A8594; --accent: #0B6E69;
        --tint: rgba(11, 110, 105, 0.08); --tint-line: rgba(11, 110, 105, 0.22); --line: rgba(17, 28, 45, 0.08);
        --mono: 'JetBrains Mono', ui-monospace, monospace; }
.stApp {
  background:
    radial-gradient(38rem 30rem at 6% -4%, rgba(11, 110, 105, 0.20), transparent 70%),
    radial-gradient(34rem 30rem at 102% 30%, rgba(56, 189, 248, 0.18), transparent 70%),
    radial-gradient(40rem 32rem at 18% 96%, rgba(192, 132, 252, 0.16), transparent 70%),
    #F6F8FC;
  background-attachment: fixed;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stMainBlockContainer"] { max-width: 66rem; padding-top: 3.5rem; padding-bottom: 5rem; }
::selection { background: rgba(11, 110, 105, 0.18); }
textarea { caret-color: var(--accent); }

/* frosted glass panels */
.st-key-inputs, .st-key-trials, .glass {
  background: rgba(255, 255, 255, 0.62);
  -webkit-backdrop-filter: blur(20px) saturate(150%); backdrop-filter: blur(20px) saturate(150%);
  border: 1px solid rgba(255, 255, 255, 0.8); border-radius: 24px;
  box-shadow: inset 0 1px 0 rgba(255, 255, 255, 0.7), 0 18px 40px -18px rgba(17, 28, 45, 0.16),
              0 2px 6px rgba(17, 28, 45, 0.04);
}
.st-key-inputs { padding: 1.75rem 1.75rem 1.25rem; }
.st-key-trials { padding: 1.5rem 1.75rem 1.25rem; margin-top: 0.5rem; }
.st-key-inputs [data-testid="stSelectbox"] [data-testid="stWidgetLabel"] p,
.st-key-inputs [data-testid="stTextArea"] [data-testid="stWidgetLabel"] p { text-transform: uppercase; letter-spacing: 0.06em;
  font-size: 0.75rem; font-weight: 600; color: var(--muted); }
.st-key-controls { border-top: 1px solid var(--line); padding-top: 1rem; margin-top: 0.25rem; }
.st-key-controls [data-testid="stSliderThumbValue"], .st-key-controls [data-testid="stSliderTickBar"] { display: none; }
/* checkbox and trial toggle sit at the right end of their rows */
.st-key-controls [data-testid="stColumn"]:last-child [data-testid="stVerticalBlock"],
.st-key-trials [data-testid="stColumn"]:last-child [data-testid="stVerticalBlock"] { align-items: flex-end; }
.st-key-trials [data-testid="stColumn"]:last-child [data-testid="stElementContainer"] { width: fit-content !important; }
.st-key-trials [data-testid="stBaseButton-segmented_control"] { background: rgba(255, 255, 255, 0.6); }
.st-key-trials [data-testid="stBaseButton-segmented_controlActive"] { background: var(--accent);
  border-color: var(--accent); color: #fff; }
.st-key-trials [data-testid="stBaseButton-segmented_controlActive"] p { color: #fff; }

/* header */
h1.title { font-size: 2.5rem; line-height: 1.15; letter-spacing: -0.025em; font-weight: 700; color: var(--ink);
           margin: 0 0 0.75rem; padding: 0; }
.lead { color: var(--muted); font-size: 1.125rem; line-height: 1.6; max-width: 46rem; margin: 0 0 0.75rem; }
.meta { color: var(--faint); font-size: 0.8125rem; font-variant-numeric: tabular-nums; margin: 0 0 1.75rem; }
.inline-label { color: var(--muted); font-size: 0.875rem; font-weight: 500; white-space: nowrap; }
.pill { display: inline-block; font-family: var(--mono); font-size: 0.8125rem; color: var(--accent);
        background: var(--tint); border: 1px solid var(--tint-line); border-radius: 8px; padding: 0.15rem 0.55rem; }

/* results */
.results { display: grid; grid-template-columns: minmax(0, 2fr) minmax(0, 3fr); gap: 1.25rem; margin: 1.25rem 0 0.75rem; }
.results .glass { padding: 1.5rem 1.6rem; }
.panel-head { display: flex; justify-content: space-between; align-items: baseline; margin-bottom: 1.1rem; }
.panel-head h3 { font-size: 1.0625rem; font-weight: 600; color: var(--ink); margin: 0; padding: 0; letter-spacing: -0.01em; }
.panel-head span { font-family: var(--mono); font-size: 0.75rem; color: var(--faint); }
.chips { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.chip { padding: 0.35rem 0.85rem; border-radius: 999px; font-size: 0.875rem; color: var(--accent);
        background: var(--tint); border: 1px solid var(--tint-line); }
.chip.rare { color: var(--faint); background: transparent; border: 1px dashed rgba(17, 28, 45, 0.22); }
.note, .empty { color: var(--muted); font-size: 0.875rem; line-height: 1.55; margin: 0.9rem 0 0; }
.empty { margin: 0; }
.stHtml .ranked { list-style: none; margin: 0; padding: 0; }
.ranked li { display: grid; grid-template-columns: 2rem 1fr 7.5rem 3.5rem; align-items: center; gap: 0.5rem;
             padding: 0.42rem 0; margin: 0; font-size: 0.9375rem; color: var(--ink); }
.ranked .n { font-family: var(--mono); font-size: 0.8125rem; color: var(--faint); }
.ranked .v { font-family: var(--mono); font-size: 0.8125rem; color: var(--muted); text-align: right; }
.bar { height: 6px; border-radius: 999px; background: rgba(17, 28, 45, 0.07); overflow: hidden; }
.bar span { display: block; height: 100%; border-radius: inherit; background: var(--accent); }
.trial-head h3 { font-size: 1.125rem; font-weight: 600; color: var(--ink); margin: 0; padding: 0; letter-spacing: -0.01em; }
.st-key-trials [data-testid="stButtonGroup"] { display: flex; justify-content: flex-end; }
.stHtml .trials { list-style: none; margin: 0.5rem 0 0; padding: 0; }
.trials li { display: grid; grid-template-columns: 8.5rem 1fr auto; align-items: center; gap: 1.25rem;
             padding: 0.95rem 0; margin: 0; border-top: 1px solid var(--line); }
.trials a { font-family: var(--mono); font-size: 0.8125rem; color: var(--accent); text-decoration: none; }
.trials a:hover { text-decoration: underline; text-underline-offset: 0.2em; }
.trials a:focus-visible { outline: 2px solid var(--accent); outline-offset: 3px; border-radius: 3px; }
.trials .t { font-size: 0.9375rem; color: var(--ink); white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
@media (max-width: 760px) {
  [data-testid="stMainBlockContainer"] { padding-top: 2rem; }
  h1.title { font-size: 1.875rem; }
  .lead { font-size: 1rem; }
  .results { grid-template-columns: minmax(0, 1fr); }
  .results .glass { padding: 1.25rem 1.1rem; }
  .ranked li { grid-template-columns: 1.6rem minmax(0, 1fr) 4.5rem 3rem; }
  .st-key-inputs, .st-key-trials { padding: 1.25rem 1.1rem 1rem; }
  .trials li { grid-template-columns: 1fr auto; gap: 0.25rem 1rem; }
  .trials .t { grid-column: 1; white-space: normal; }
  .trials .pill { grid-row: 1 / span 2; grid-column: 2; }
}
</style>"""

st.set_page_config(page_title="Clinical trial keyword recommender", page_icon=":material/clinical_notes:")
st.html(STYLE)


@st.cache_resource
def load():
    return Recommender()


def found_panel(rec, found):
    head = f'<div class="panel-head"><h3>Found in your text</h3><span>{len(found)} found</span></div>'
    if not found:
        return head + ('<p class="empty">No MeSH keywords found. Try naming the condition or the treatment, '
                       'for example "type 2 diabetes" or "metformin".</p>')
    chips = "".join(f'<span class="chip{"" if ui in rec.index else " rare"}">{escape(rec.ext.names[ui])}</span>'
                    for ui in found)
    note = ('<p class="note">Dashed keywords appear in fewer than 0.5% of trials, so the model does not use them.</p>'
            if any(ui not in rec.index for ui in found) else "")
    return f'{head}<div class="chips">{chips}</div>{note}'


def keyword_panel(rec, found, keywords):
    head = '<div class="panel-head"><h3>Recommended keywords</h3><span>score</span></div>'
    if not keywords:
        return head + (f'<p class="empty">None of these keywords is among the {len(rec.vocab):,} the model knows, '
                       'so there is nothing to recommend from.</p>' if found else
                       '<p class="empty">Recommendations appear once a keyword is found.</p>')
    top = keywords[0][1]
    rows = "".join(f'<li><span class="n">{i}.</span><span>{escape(name)}</span>'
                   f'<span class="bar"><span style="width: {100 * score / top:.0f}%"></span></span>'
                   f'<span class="v">{score:.3f}</span></li>' for i, (name, score) in enumerate(keywords, 1))
    return f'{head}<ol class="ranked">{rows}</ol>'


def trial_rows(trials):
    return "".join(f'<li><a href="https://clinicaltrials.gov/study/{nct}" target="_blank" rel="noopener">{nct}</a>'
                   f'<span class="t" title="{escape(title)}">{escape(title)}</span>'
                   f'<span class="pill">{sim:.2f}</span></li>' for nct, title, sim in trials)


rec = load()
st.html('<h1 class="title">Keyword recommendation for clinical trial search</h1>'
        '<p class="lead">Describe a study in plain words. The app finds the MeSH keywords in it, recommends related '
        'keywords that similar trials usually mention, and lists the closest trials.</p>'
        f'<p class="meta">{len(rec.vocab):,} MeSH keywords &middot; {len(rec.ids):,} ClinicalTrials.gov trials '
        f'&middot; Model: {escape(rec.model_name)}</p>')

with st.container(key="inputs"):
    example = st.selectbox("Pick an example or write your own below", EXAMPLES)
    text = st.text_area("Describe the study you are looking for", value=example, height=110)
    with st.container(key="controls"):
        label, slider, value, box = st.columns([2.3, 2.6, 0.5, 2.6], vertical_alignment="center")
        label.html('<span class="inline-label">Number of keywords to recommend</span>')
        k = slider.slider("Number of keywords to recommend", 3, 20, 10, label_visibility="collapsed")
        value.html(f'<span class="pill">{k}</span>')
        generic = box.checkbox("Rank by full prediction",
                               help="Shows generic keywords such as 'Patients' and 'Therapeutics' first.")

if not text.strip():
    st.html('<div class="results" style="grid-template-columns: 1fr"><div class="glass"><p class="empty">'
            'Write a study description above, naming the condition, treatment or population.</p></div></div>')
    st.stop()

found, x, expanded, keywords = rec.recommend(text, k, generic)
st.html(f'<div class="results"><div class="glass">{found_panel(rec, found)}</div>'
        f'<div class="glass">{keyword_panel(rec, found, keywords)}</div></div>')

if x.any():
    with st.container(key="trials"):
        head, toggle = st.columns([1, 1.4], vertical_alignment="center")
        head.html('<div class="trial-head"><h3>Most similar trials</h3></div>')
        tab = toggle.segmented_control("Trials to show", TABS, default=TABS[0], label_visibility="collapsed")
        vec = x if tab == TABS[1] else expanded
        st.html(f'<ul class="trials">{trial_rows(rec.similar_trials(vec))}</ul>')

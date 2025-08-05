
import math
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(page_title="Vue pédagogique — Clean‑Spark FR", layout="wide")

st.title("Vue pédagogique — Clean‑Spark & Dispatch (France, Intraday)")
st.caption("Bougez les paramètres et observez : clean‑spark, décision ON/OFF, tailles de hedge, et PnL attendu.")

# Sidebar: plant parameters
st.sidebar.header("Paramètres centrale (CCGT)")
heat_rate = st.sidebar.slider("Heat-Rate (MWh_th / MWh_el)", min_value=1.4, max_value=2.4, value=1.75, step=0.01)
ef_el = st.sidebar.slider("Facteur d'émission (tCO₂ / MWh_el)", min_value=0.25, max_value=0.45, value=0.35, step=0.01)
vom = st.sidebar.slider("VOM (€/MWh)", min_value=0.0, max_value=10.0, value=3.0, step=0.1)
start_cost = st.sidebar.number_input("Coût de démarrage (k€)", min_value=0.0, value=45.0, step=1.0) * 1000.0
min_up = st.sidebar.slider("Min up (heures)", min_value=1, max_value=8, value=3, step=1)
capacity = st.sidebar.number_input("Puissance (MW)", min_value=50.0, max_value=500.0, value=400.0, step=10.0)

# Sidebar: market parameters
st.sidebar.header("Marché (prix)")
gas = st.sidebar.number_input("Gaz TTF within-day (€/MWh_th)", min_value=-50.0, max_value=200.0, value=30.0, step=1.0)
co2 = st.sidebar.number_input("CO₂ EUA spot (€/t)", min_value=0.0, max_value=200.0, value=70.0, step=1.0)

# Build a simple DA curve template (France typical shape) in €/MWh
hours = pd.date_range("2025-01-01", periods=24, freq="H")  # dummy day; we only use hour-of-day
h = hours.hour
# Piecewise base shape: night low, day medium, evening peak
power_da = np.where(h < 6, 65,
             np.where(h < 8, 75,
             np.where(h < 17, 85,
             np.where(h < 21, 115, 80)))).astype(float)

# User controls for DA level and ID shocks
st.sidebar.header("Scénario de prix électricité")
da_shift = st.sidebar.slider("Décalage global DA (€/MWh)", min_value=-40, max_value=40, value=0, step=1)
power_da = power_da + da_shift

peak_start = st.sidebar.slider("Début fenêtre 'peak' (h)", min_value=16, max_value=20, value=18, step=1)
peak_end = st.sidebar.slider("Fin fenêtre 'peak' (h)", min_value=19, max_value=23, value=20, step=1)
id_shock = st.sidebar.slider("Prime intraday durant la fenêtre (€/MWh)", min_value=-80, max_value=120, value=40, step=1)
id_noise = st.sidebar.slider("Bruit intraday hors fenêtre (écart-type €/MWh)", min_value=0, max_value=20, value=2, step=1)

rng = np.random.default_rng(42)  # deterministic for pedagogy
power_id = power_da.copy()
mask_peak = (h >= peak_start) & (h <= peak_end)
power_id[mask_peak] += id_shock
noise = rng.normal(0, id_noise, size=len(power_id))
power_id[~mask_peak] = power_id[~mask_peak] + noise[~mask_peak]

# Clean-spark DA / ID
csp_da = power_da - heat_rate * gas - ef_el * co2 - vom
csp_id = power_id - heat_rate * gas - ef_el * co2 - vom

df = pd.DataFrame({
    "Heure": h,
    "Power_DA_€/MWh": power_da,
    "Power_ID_€/MWh": power_id,
    "CSP_DA_€/MWh": csp_da,
    "CSP_ID_€/MWh": csp_id,
})
df.index = hours

# Simple dispatch logic (pedagogical): turn ON where CSP_ID>0, but require blocks of at least min_up hours
on = np.zeros(24, dtype=int)
i = 0
while i < 24:
    if csp_id[i] > 0:
        # find run length
        j = i
        while j < 24 and csp_id[j] > 0:
            j += 1
        run_len = j - i
        if run_len >= min_up:
            on[i:j] = 1
        i = j
    else:
        i += 1

# Check if the positive segments cover start cost (per start)
# If not, drop the shortest qualifying segment until the total margin >= start cost per start.
segments = []
i = 0
while i < 24:
    if on[i] == 1:
        j = i
        while j < 24 and on[j] == 1:
            j += 1
        segments.append((i, j))  # inclusive start, exclusive end
        i = j
    else:
        i += 1

# Evaluate PnL for each segment at 'capacity'
pnl_segments = []
for (a, b) in segments:
    hours_range = np.arange(a, b)
    margin = float(np.sum(csp_id[a:b]) * capacity)  # €/MWh * MW = €/h; sum over hours
    pnl_segments.append({"start": a, "end": b, "hours": b-a, "gross_margin_eur": margin})

# Apply start cost per segment; drop segments that don't cover start cost
kept = []
pnl_total = 0.0
for seg in pnl_segments:
    net = seg["gross_margin_eur"] - start_cost
    if net > 0:
        kept.append(seg | {"net_margin_eur": net})
        pnl_total += net
    else:
        # Remove this segment from ON schedule
        on[seg["start"]:seg["end"]] = 0

df["ON"] = on
df["MW"] = capacity * on
df["PNL_hourly_eur"] = (df["CSP_ID_€/MWh"] * df["MW"]).astype(float)

# Greeks / hedge
d_power = 1.0
d_gas = -heat_rate
d_co2 = -ef_el

# LAYOUT
col1, col2 = st.columns([2,1], gap="large")

with col1:
    st.subheader("Prix & Clean‑Spark (€/MWh)")
    import plotly.express as px
    long_prices = df[["Power_DA_€/MWh", "Power_ID_€/MWh"]].reset_index().melt(id_vars="index", var_name="Série", value_name="€/MWh")
    long_prices.rename(columns={"index": "Heure"}, inplace=True)
    fig1 = px.line(long_prices, x="Heure", y="€/MWh", color="Série")
    fig1.update_layout(height=340)
    st.plotly_chart(fig1, use_container_width=True)

    long_csp = df[["CSP_DA_€/MWh", "CSP_ID_€/MWh"]].reset_index().melt(id_vars="index", var_name="Série", value_name="€/MWh")
    long_csp.rename(columns={"index": "Heure"}, inplace=True)
    fig2 = px.line(long_csp, x="Heure", y="€/MWh", color="Série")
    # Shade ON hours
    for (a,b) in segments:
        if any((k>=a and k<b and on[k]==1) for k in range(a,b)):
            fig2.add_vrect(x0=long_csp['Heure'][a], x1=long_csp['Heure'][b-1], fillcolor="LightGreen", opacity=0.2, line_width=0)
    fig2.update_layout(height=340)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Planning & PnL (MW / €)")
    fig3 = px.bar(df.reset_index(), x="index", y="MW")
    fig3.update_layout(height=200, yaxis_title="MW", xaxis_title="Heure")
    st.plotly_chart(fig3, use_container_width=True)
    st.metric("PNL net estimé (€/jour)", f"{pnl_total:,.0f}".replace(",", " "))

with col2:
    st.subheader("Décision & Hedge")
    if len(kept) == 0:
        st.info("Aucun segment ne couvre le coût de démarrage avec les contraintes. Reste OFF (ou vise le marché DA).")
    else:
        for idx, seg in enumerate(kept, 1):
            st.success(f"Segment {idx}: h{seg['start']:02d}–h{seg['end']:02d} ({seg['hours']} h) — Marge nette ≈ {seg['net_margin_eur']:,.0f} €".replace(",", " "))

    st.markdown("**Deltas clean‑spark (par MWh_élec)**")
    st.write(f"dPower = +{d_power:.2f}, dGaz = {d_gas:.2f}, dCO₂ = {d_co2:.2f}")
    st.caption("Interprétation : acheter 1 MWh_élec s'accompagne d'une vente de HR MWh_th de gaz et EF tCO₂ pour rester neutre CSP.")

    st.subheader("Ticket indicatif (heure la plus attractive)")
    # Pick best hour by CSP_ID
    best_hour = int(np.nanargmax(csp_id))
    best_csp = float(csp_id[best_hour])
    if best_csp <= 0 or on[best_hour] == 0:
        st.write("Pas d'heure attractive selon le scénario actuel.")
    else:
        power_qty = capacity  # hedge per hour when ON
        gas_qty = -d_gas * power_qty
        co2_qty = -d_co2 * power_qty
        st.write(f"Heure: h{best_hour:02d}")
        st.write(f"Action: **Acheter** électricité intraday {power_qty:.0f} MWh ; **Vendre** gaz {gas_qty:.0f} MWh_th ; **Vendre** CO₂ {co2_qty:.0f} t.")
        st.write(f"Rationale: CSP_ID={best_csp:.1f} €/MWh, fenêtre peak={peak_start}–{peak_end}h, prime ID={id_shock:+.0f} €/MWh.")
        st.write("Stops/Targets: utiliser 1×/2× la volatilité intraday locale (non modélisée ici).")

with st.expander("Détails des calculs"):
    st.code(
        "CSP = Power - HR * Gas - EF * CO2 - VOM\n"
        "PNL segment = somme(CSP_ID_heures) * MW - StartCost\n"
        "Deltas = ( +1 Power, -HR Gas, -EF CO2 )"
    )

st.markdown("""
---
**Note pédagogique :** Cette appli montre l'effet de **HR/EF**, des **prix** (power/gaz/CO₂) et du **coût de démarrage** sur :
- la **marge clean‑spark** (DA vs Intraday),
- la **décision ON/OFF** sous contrainte de *min up*,
- les **tailles de hedge** (deltas),
- et la **PNL nette** après démarrage.
Vous pouvez reproduire des cas d'école : *matin froid sans vent* (prime ID forte), *journée ventée* (prime faible), etc.
""")

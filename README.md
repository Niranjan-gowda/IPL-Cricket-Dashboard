# IPL Player Matchup & Phase Strategy Dashboard

An interactive analytics dashboard built with Streamlit to analyse player matchups and phase-based strategies in the Indian Premier League (IPL), using 200,000+ ball-by-ball deliveries. It provides insights into batter vs bowler dynamics, dismissal patterns and phase-specific performance for coaches, analysts and fans.

## Features

**Batter View** — career stats, all-time IPL leaderboards, dismissal-type breakdowns, phase-wise strike rates (Powerplay, Middle Overs, Death) and performance against each opponent team.

**Bowler View** — economy, wicket-taking ability, dismissal types induced and phase-wise economy analysis, with head-to-head bowler comparison.

**Batter vs Bowler Matchup** — one-on-one matchup summaries; where no historical data exists, predicted strike rate, economy and runs-per-over are derived from each player's overall record.

**Clustering & Insights** — K-Means clustering (with PCA) groups batters into styles (Power Hitters, Anchors, Finishers, Consistent Performers) and bowlers by type and control profile, then recommends strategic player alternatives for a given matchup.

**Advanced Metrics** — dot-ball pressure, boundary percentage, wide/no-ball control rates, batting-position insights.

## Tech Stack

Python (pandas, NumPy, scikit-learn), Streamlit, Plotly.

## Data

Uses the standard IPL ball-by-ball deliveries dataset (`deliveries.csv`, 200,000+ rows). Upload your own CSV in the same format via the sidebar; the app validates required columns before running.

## Installation

```bash
git clone https://github.com/Niranjan-gowda/IPL-Cricket-Dashboard.git
cd IPL-Cricket-Dashboard
pip install -r requirements.txt
```

## Usage

```bash
streamlit run Decision_lab.py
```

Then open the local URL shown in the terminal (default http://localhost:8501) and upload the deliveries CSV via the sidebar.

## Applications

Coaches and analysts can refine match strategies with player insights; broadcasters can use data-backed visuals for commentary; fans can explore player rivalries and tactical trends.

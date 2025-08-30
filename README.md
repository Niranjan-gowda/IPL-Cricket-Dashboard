IPL Player Matchup & Phase Strategy Dashboard

This project is an interactive analytics dashboard built with Streamlit to analyze player matchups and phase-based strategies in the Indian Premier League (IPL). It provides insights into batter vs. bowler dynamics, dismissal patterns, and phase-specific performance, making it useful for coaches, analysts, and fans.

Features
Batter View – Analyze batting performance by position, match phase (Powerplay, Middle Overs, Death), and situation.
Bowler View – Explore bowling economy, wicket-taking ability, and efficiency across phases.
Batter vs Bowler Matchups – Compare head-to-head stats with predicted runs, dismissal probability, and boundary percentage.
Clustering & Insights – Groups players with similar performance patterns and suggests alternatives for strategy.
Advanced Metrics 
* Dot Ball Pressure Index
* Boundary Analysis
* Dismissal Type Heatmaps
* Batting Position Insights

Data
* Uses IPL ball-by-ball deliveries dataset.
* Data is cleaned and aggregated for matchup and phase analysis.
* Supports user-uploaded CSVs for additional analysis.

Tech Stack
* Python: pandas, numpy, scikit-learn
* Streamlit: interactive dashboard framework
* Plotly / Matplotlib: data visualizations

Installation
Clone the repository and install dependencies:
git clone https://github.com/Niranjan-gowda/IPL-Cricket-Dashboard.git
cd IPL-Cricket-Dashboard
pip install -r requirements.txt

Usage
Run the Streamlit app:
streamlit run Decision_lab.py
Then open the local URL shown in the terminal (default: http://localhost:8501) in your browser.

Applications
* Coaches and analysts – refine match strategies with player insights.
* Broadcasters – use data-backed visuals for commentary.
* Fans – explore player rivalries and tactical trends.
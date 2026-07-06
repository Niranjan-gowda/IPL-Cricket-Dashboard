# IPL Player Matchup & Phase Strategy Dashboard

Interactive Streamlit dashboard that analyses IPL ball-by-ball data to answer strategic questions: *Which bowler should face this batter at the death? Who handles the powerplay best?*

> 🎯 **Business framing:** This is a decision-support tool — the same matchup/segmentation logic applies to customer-vs-product analysis, campaign phase performance, or resource allocation.

<!-- TODO: add a screenshot or GIF of the dashboard here -->
<!-- TODO: add live demo link once deployed to Streamlit Community Cloud -->

## Key insights the dashboard surfaces

- **Batter vs Bowler matchups** — head-to-head stats with predicted runs, dismissal probability, and boundary %
- **Phase analysis** — performance split by Powerplay, Middle Overs, and Death
- **Player clustering** — scikit-learn groups players with similar profiles and suggests tactical alternatives
- **Advanced metrics** — Dot Ball Pressure Index, boundary analysis, dismissal-type heatmaps

## Tech stack

Python (pandas, NumPy, scikit-learn) · Streamlit · Plotly

## Quick start

```bash
git clone https://github.com/Niranjan-gowda/IPL-Cricket-Dashboard.git
cd IPL-Cricket-Dashboard
pip install -r requirements.txt
streamlit run Decision_lab.py
```

Open http://localhost:8501, then upload the included `deliveries.csv` (or any CSV in the IPL deliveries format) via the sidebar.

## Project files

```
├── Decision_lab.py           # Streamlit app
├── deliveries.csv            # ball-by-ball IPL dataset
├── Decision_lab_report.docx  # analysis report
└── requirements.txt
```

## What I'd improve next

- Add unit tests for metric calculations
- Cache aggregations for faster load on large datasets
- Deploy to Streamlit Community Cloud

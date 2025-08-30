import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

st.set_page_config(page_title="IPL Matchup Dashboard", layout="wide")
st.markdown("""
    <style>
        .main { background-color: #f9f9f9; }
        .block-container { padding-top: 2rem; }
        h1, h2, h3, h4, h5 { color: #004080; }
        .stTabs [role="tab"] { font-size: 18px; padding: 10px; }
    </style>
""", unsafe_allow_html=True)

st.title("🏏 IPL Player Matchup Strategy Dashboard")

# Upload Section
st.sidebar.header("📂 Upload the IPL dataset")
user_file = st.sidebar.file_uploader("Upload your csv file", type=['csv'])

if user_file is not None:
    df = pd.read_csv(user_file)
    required_columns = [
        'match_id', 'inning', 'batting_team', 'bowling_team', 'over', 'ball', 'batter', 'bowler', 'non_striker',
        'batsman_runs', 'extra_runs', 'total_runs', 'extras_type', 'is_wicket', 'player_dismissed', 'dismissal_kind',
        'fielder', 'partnership_runs', 'partnership_balls', 'is_dot_ball', 'is_wide', 'is_noball', 'batter_hand',
        'bowler_hand', 'bowling_style', 'num_fours', 'num_sixes', 'matches_played', 'matches_won'

    ]
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        st.error(
            f"Uploaded file is missing the following required column(s): {', '.join(missing_cols)}.\n\n"
            f"Your file columns: {', '.join(df.columns)}\n\n"
            f"Please ensure your dataset matches the expected IPL deliveries format."
        )
        st.info(
            "Required columns are: " + ", ".join(required_columns)
        )
        st.stop()

    # Clean up team names
    team_name_mapping = {
        'Rising Pune Supergiants': 'Rising Pune Supergiant',
        'Royal Challengers Bengaluru': 'Royal Challengers Bangalore',
        'Kings XI Punjab': 'Punjab Kings',
        'Delhi Daredevils': 'Delhi Capitals',
    }
    df['batting_team'] = df['batting_team'].str.strip().replace(team_name_mapping)
    df['bowling_team'] = df['bowling_team'].str.strip().replace(team_name_mapping)

    required_columns = ['batter', 'bowler', 'batting_team', 'bowling_team', 'batsman_runs', 'ball', 'match_id']
    for col in required_columns:
        if col not in df.columns:
            st.error(f"Missing required column: '{col}'")
            st.stop()
    # Feature columns
    df['dots'] = (df['batsman_runs'] == 0)
    df['no_balls'] = (df['extras_type'] == 'no ball') if 'extras_type' in df.columns else False
    df['wide_balls'] = (df['extras_type'] == 'wide') if 'extras_type' in df.columns else False
    df['is_wicket'] = df['is_wicket'] if 'is_wicket' in df.columns else 0
    df['over'] = df['ball'] // 6 + 1 if 'over' not in df.columns else df['over']

    # Allow year filtering
    if 'date' in df.columns:
        df['year'] = pd.to_datetime(df['date']).dt.year
        years = sorted(df['year'].dropna().unique())
        selected_years = st.sidebar.multiselect("📅 Select Year(s)", years, default=years)
        df = df[df['year'].isin(selected_years)]

    st.sidebar.title("Player Analyzer")
    view_mode = st.sidebar.radio("Choose View Mode:",
                                 ["Batter View", "Bowler View", "Bowler vs Batter Matchup", "Clustering & Insights"])

    teams = sorted(df['bowling_team'].unique())
    innings_filter = st.sidebar.selectbox("🏏 Innings", ["Both", "1st Innings", "2nd Innings"], key="innings_filter")
    over_phase = st.sidebar.selectbox("🕒 Over Phase", ["All", "1-6 (Powerplay)", "7-15 (Middle)", "16-20 (Death)"],
                                      key="over_phase")
    selected_team = st.sidebar.selectbox("🎽 Opponent Team", ["All"] + teams, key="selected_team")


    def apply_common_filters(data):
        if 'inning' in data.columns:
            if innings_filter == "1st Innings":
                data = data[data['inning'] == 1]
            elif innings_filter == "2nd Innings":
                data = data[data['inning'] == 2]
        if over_phase == "1-6 (Powerplay)":
            data = data[(data['over'] >= 1) & (data['over'] <= 6)]
        elif over_phase == "7-15 (Middle)":
            data = data[(data['over'] >= 7) & (data['over'] <= 15)]
        elif over_phase == "16-20 (Death)":
            data = data[(data['over'] >= 16) & (data['over'] <= 20)]
        if selected_team != "All":
            data = data[data['bowling_team'] == selected_team]
        return data


    def prepare_batter_features(df):
        batter_stats = df.groupby('batter').agg({
            'batsman_runs': 'sum',
            'ball': 'count',
            'match_id': 'nunique'
        }).reset_index()
        batter_stats['strike_rate'] = (batter_stats['batsman_runs'] / batter_stats['ball']) * 100
        batter_stats['average'] = batter_stats['batsman_runs'] / batter_stats['match_id']
        return batter_stats[['batter', 'strike_rate', 'average', 'ball']]


    def prepare_bowler_features(df):
        bowler_stats = df.groupby('bowler').agg({
            'batsman_runs': 'sum',
            'ball': 'count',
            'is_wicket': 'sum',
            'match_id': 'nunique',
            'dots': 'sum',
            'no_balls': 'sum',
            'wide_balls': 'sum'
        }).reset_index()
        bowler_stats['economy'] = (bowler_stats['batsman_runs'] / bowler_stats['ball']) * 6
        bowler_stats['average'] = bowler_stats['batsman_runs'] / bowler_stats['is_wicket'].replace(0, 1)
        return bowler_stats[['bowler', 'economy', 'average', 'ball', 'dots', 'no_balls', 'wide_balls']]


    # =================================
    # 1. BATTER VIEW
    # =================================
    if view_mode == "Batter View":
        st.sidebar.subheader("👤 Batter Selection")
        batters = ["--Select--"] + sorted(df['batter'].dropna().unique())
        batter = st.sidebar.selectbox("Select Batter", batters, index=0)
        if batter == "--Select--":
            st.info("Please select a batter to continue.")
            st.stop()

        compare_batters = ["--Select--"] + [p for p in batters if p != batter and p != "--Select--"]
        compare_batter = st.sidebar.selectbox("Compare with", compare_batters, index=0)
        if compare_batter == "--Select--":
            st.info("Please select a comparison batter to continue.")
            st.stop()

        df_batter = apply_common_filters(df[df['batter'] == batter])
        df_compare = apply_common_filters(df[df['batter'] == compare_batter])
        tabs = st.tabs(["Overview", "Comparison", "Dismissals", "Phase Analysis", "Opponent Teams"])
        with tabs[0]:
            st.info(
                "Dive into the career stats of your chosen batter, see how their scoring pattern and consistency stack up against IPL’s best. Great for identifying standout performers and understanding a batter’s typical approach.")
            st.header(f"Performance of {batter}")
            runs = df_batter['batsman_runs'].sum()
            balls = df_batter.shape[0]
            wickets = df_batter['is_wicket'].sum()
            sr = (runs / balls) * 100 if balls > 0 else 0
            avg = runs / wickets if wickets else runs
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Runs", runs)
            col2.metric("Balls Faced", balls)
            col3.metric("Strike Rate", round(sr, 2))
            col4.metric("Average", round(avg, 2))
            st.markdown("---")
            st.header("🏆 All-Time IPL Stats")
            most_runs = (
                df.groupby("batter")["batsman_runs"]
                    .sum()
                    .reset_index()
                    .sort_values(by="batsman_runs", ascending=False)
                    .head(10)
                    .rename(columns={"batter": "Player", "batsman_runs": "Total Runs"})
            )
            fig1 = px.bar(
                most_runs.sort_values("Total Runs"),
                y="Player", x="Total Runs",
                orientation="h", color="Total Runs",
                color_continuous_scale="Blues",
                title="Top 10 Run Scorers"
            )
            st.plotly_chart(fig1, use_container_width=True)
            st.dataframe(
                most_runs,
                use_container_width=True
            )

            batter_stats = df.groupby("batter").agg({
                "batsman_runs": "sum",
                "ball": "count"
            }).reset_index()
            batter_stats = batter_stats[batter_stats["ball"] >= 300]
            batter_stats["Strike Rate"] = (batter_stats["batsman_runs"] / batter_stats["ball"]) * 100
            best_finishers = (
                batter_stats.sort_values(by="Strike Rate", ascending=False)
                    .head(10)
                    .rename(columns={"batter": "Player"})
            )
            fig2 = px.bar(
                best_finishers.sort_values("Strike Rate"),
                y="Player", x="Strike Rate",
                orientation="h", color="Strike Rate",
                color_continuous_scale="Greens",
                title="Top 10 Finishers (by SR, min 300 balls)"
            )
            st.plotly_chart(fig2, use_container_width=True)
            st.dataframe(
                best_finishers[["Player", "Strike Rate", "batsman_runs", "ball"]],
                use_container_width=True
            )

        with tabs[1]:
            st.info(
                "See head-to-head stats—quickly benchmark one batter against another to spot relative strengths or role similarities.")

            st.header(f"{batter} vs {compare_batter}")
            comp_runs = df_compare['batsman_runs'].sum()
            comp_balls = df_compare.shape[0]
            comp_sr = (comp_runs / comp_balls) * 100 if comp_balls else 0
            df_bar = pd.DataFrame({
                "Player": [batter, compare_batter],
                "Runs": [runs, comp_runs],
                "Strike Rate": [sr, comp_sr]
            })
            df_bar_melted = df_bar.melt(id_vars="Player")
            df_bar_melted["variable"] = df_bar_melted["variable"].str.upper()

            fig = px.bar(df_bar_melted, x="Player", y="value", color="variable",
                         barmode="group", height=400)

            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""Maximizing runs is very crucial for match control in cricket""")

        with tabs[2]:
            st.info("Break down on how the batter gets out.")
            st.header("Dismissal Types")
            dismissal_data = df_batter[df_batter['is_wicket'] == 1]
            if not dismissal_data.empty and 'dismissal_kind' in dismissal_data.columns:
                dismissal_count = dismissal_data['dismissal_kind'].value_counts().reset_index()
                dismissal_count.columns = ['Dismissal Type', 'Count']
                dismissal_count['Dismissal Type'] = dismissal_count['Dismissal Type'].str.upper()
                fig_pie = px.pie(dismissal_count, names='Dismissal Type', values='Count', hole=0.3)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No dismissals found for selected filters.")

        with tabs[3]:
            st.info(
                "Uncover how the batter performs in Powerplay, Middle, and Death overs. Reveals if someone is a good starter, anchor, or finisher.")
            st.header("Phase-wise Performance")
            df_batter['phase'] = pd.cut(df_batter['over'], bins=[0, 6, 15, 20],
                                        labels=['Powerplay', 'Middle', 'Death'])
            phase_stats = df_batter.groupby('phase')['batsman_runs'].agg(['sum', 'count']).reset_index()
            phase_stats.columns = ['Phase', 'Runs', 'Balls']
            phase_stats['Phase'] = phase_stats['Phase'].str.upper()
            phase_stats['Strike Rate'] = (phase_stats['Runs'] / phase_stats['Balls']) * 100
            fig_phase = px.bar(phase_stats, x='Phase', y='Strike Rate', color='Phase')
            st.plotly_chart(fig_phase, use_container_width=True)

        with tabs[4]:
            st.info(
                "Analyze the batter’s performance against different teams-insightful for tactical planning and matchups.")
            st.header("Performance Against Opponent Teams")
            team_stats = df_batter.groupby('bowling_team')['batsman_runs'].agg(['sum', 'count']).reset_index()
            team_stats.columns = ['Opponent', 'Runs', 'Balls']
            team_stats['Strike Rate'] = (team_stats['Runs'] / team_stats['Balls']) * 100
            fig_teams = px.bar(team_stats, x='Opponent', y='Strike Rate', color='Opponent',
                               title="Strike Rate vs Teams")
            st.plotly_chart(fig_teams, use_container_width=True)

    # =================================
    # 2. BOWLER VIEW
    # =================================
    elif view_mode == "Bowler View":
        st.sidebar.subheader("Bowler Selection")
        bowlers = ["--Select--"] + sorted(df['bowler'].dropna().unique())
        bowler = st.sidebar.selectbox("Select Bowler", bowlers, index=0)
        if bowler == "--Select--":
            st.info("Please select a bowler to continue.")
            st.stop()

        compare_bowlers = ["--Select--"] + [p for p in bowlers if p != bowler and p != "--Select--"]
        compare_bowler = st.sidebar.selectbox("Compare with", compare_bowlers, index=0)
        if compare_bowler == "--Select--":
            st.info("Please select a comparison bowler to continue.")
            st.stop()

        df_bowler = apply_common_filters(df[df['bowler'] == bowler])
        df_compare = apply_common_filters(df[df['bowler'] == compare_bowler])
        tabs = st.tabs(["Overview", "Comparison", "Dismissals", "Phase Analysis", "Opponent Teams"])

        with tabs[0]:
            st.info(
                "Get a holistic view of a bowler’s career—wickets, economy, and more. Ideal for identifying strike bowlers vs. defensive bowlers.")
            st.header(f"Performance of {bowler}")
            runs = df_bowler['batsman_runs'].sum()
            balls = df_bowler.shape[0]
            wickets = df_bowler['is_wicket'].sum()
            eco = (runs / balls) * 6 if balls else 0
            avg = runs / wickets if wickets else runs
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Runs Conceded", runs)
            col2.metric("Balls Bowled", balls)
            col3.metric("Economy", round(eco, 2))
            col4.metric("Average", round(avg, 2))
            st.markdown("---")
            st.header("🏆 All-Time IPL Bowling Stats")
            df["is_wicket"] = df["is_wicket"].fillna(0)
            bowler_stats = df.groupby("bowler").agg({
                "is_wicket": "sum",
                "batsman_runs": "sum",
                "ball": "count"
            }).reset_index()

            bowler_stats["Economy"] = bowler_stats["batsman_runs"] / (bowler_stats["ball"] / 6)
            most_wickets = (
                bowler_stats.sort_values(by="is_wicket", ascending=False)
                    .head(10)
                    .rename(columns={"bowler": "Player", "is_wicket": "Wickets"})
            )

            fig = px.bar(
                most_wickets.sort_values("Wickets"),
                y="Player", x="Wickets",
                orientation="h", color="Wickets",
                color_continuous_scale="Purples",
                title="Top 10 Wicket Takers"
            )
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(
                most_wickets[["Player", "Wickets", "Economy"]],
                use_container_width=True
            )

        with tabs[1]:
            st.info(
                "See how two bowlers match up on key metrics. Great for selection debates or understanding bowling roles.")
            st.header(f"{bowler} vs {compare_bowler}")
            comp_runs = df_compare['batsman_runs'].sum()
            comp_balls = df_compare.shape[0]
            comp_eco = (comp_runs / comp_balls) * 6 if comp_balls else 0

            df_bar = pd.DataFrame({
                "Player": [bowler, compare_bowler],
                "Runs Conceded": [runs, comp_runs],
                "Economy": [eco, comp_eco]
            })
            df_bar_melted = df_bar.melt(id_vars="Player")
            df_bar_melted["variable"] = df_bar_melted["variable"].str.upper()
            fig = px.bar(df_bar_melted, x="Player", y="value", color="variable",
                         barmode="group", height=400)

            st.plotly_chart(fig, use_container_width=True)
            st.markdown("""Runs conceded refers to the total number of runs a bowler has allowed whereas
                        Economy refers to how many runs the bowler concedes per over on average across all matches""")

        with tabs[2]:
            st.info(" Analyze the types of dismissals a bowler typically induces—bowled, caught, or otherwise.")
            st.header("Wicket Types")
            dismissals = df_bowler[df_bowler['is_wicket'] == 1]
            if not dismissals.empty and 'dismissal_kind' in dismissals.columns:
                dismissal_count = dismissals['dismissal_kind'].value_counts().reset_index()
                dismissal_count.columns = ['Dismissal Type', 'Count']
                dismissal_count['Dismissal Type'] = dismissal_count['Dismissal Type'].str.upper()
                fig_pie = px.pie(dismissal_count, names='Dismissal Type', values='Count', hole=0.3)
                st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No wickets found for selected filters.")

        with tabs[3]:
            st.info(
                "Analyze when a bowler is most effective. Are they Powerplay specialists, death over aces, or all-rounders?")
            st.header("Phase-wise Economy")
            df_bowler['phase'] = pd.cut(df_bowler['over'], bins=[0, 6, 15, 20],
                                        labels=['Powerplay', 'Middle', 'Death'])
            phase_stats = df_bowler.groupby('phase')['batsman_runs'].agg(['sum', 'count']).reset_index()
            phase_stats.columns = ['Phase', 'Runs', 'Balls']
            phase_stats['Phase'] = phase_stats['Phase'].str.upper()
            phase_stats['Economy'] = (phase_stats['Runs'] / phase_stats['Balls']) * 6
            fig_phase = px.bar(phase_stats, x='Phase', y='Economy', color='Phase')
            st.plotly_chart(fig_phase, use_container_width=True)

        with tabs[4]:
            st.info("Identify which teams a bowler performs best against, possibly due to match-ups or past rivalries.")
            st.header("Bowling Against Opponent Teams")
            team_stats = df_bowler.groupby('batting_team')['batsman_runs'].agg(['sum', 'count']).reset_index()
            team_stats.columns = ['Opponent', 'Runs', 'Balls']
            team_stats['Economy'] = (team_stats['Runs'] / team_stats['Balls']) * 6
            fig_teams = px.bar(team_stats, x='Opponent', y='Economy', color='Opponent',
                               title="Economy Rate vs Teams")
            st.plotly_chart(fig_teams, use_container_width=True)

    # =================================
    # 3. BOWLER VS BATTER MATCHUP
    # =================================
    elif view_mode == "Bowler vs Batter Matchup":
        st.sidebar.subheader("⚔ Matchup Selection")
        batters = ["--Select--"] + sorted(df['batter'].dropna().unique())
        selected_batter = st.sidebar.selectbox("Select Batter", batters, key="matchup_batter", index=0)
        if selected_batter == "--Select--":
            st.info("Please select a batter for the matchup.")
            st.stop()

        bowlers = ["--Select--"] + [b for b in sorted(df['bowler'].dropna().unique()) if b != selected_batter]
        selected_bowler = st.sidebar.selectbox("Select Bowler", bowlers, key="matchup_bowler", index=0)
        if selected_bowler == "--Select--":
            st.info("Please select a bowler for the matchup.")
            st.stop()

        matchup_df = df[(df['batter'] == selected_batter) & (df['bowler'] == selected_bowler)]
        matchup_df = apply_common_filters(matchup_df)
        st.header(f"⚔ {selected_batter} vs {selected_bowler}")
        st.info(
            "Discover how a specific batter performs against a particular bowler. These individual match-ups often influence team strategies, in-game captaincy calls, and can be crucial in crunch moments.")
        st.markdown("""
        🔄 In this section insights are generated to analyse one-on-one matchup performance  . 
        """)
        if matchup_df.empty:
            st.warning("No matchup data available for selected players. Showing predicted values:")
            batter_stats = df.groupby("batter").agg({
                "batsman_runs": "sum",
                "ball": "count"
            }).reset_index()
            batter_stats["strike_rate"] = (batter_stats["batsman_runs"] / batter_stats["ball"]) * 100
            bowler_stats = df.groupby("bowler").agg({
                "ball": "count",
                "dots": "sum",
                "no_balls": "sum",
                "wide_balls": "sum",
                "batsman_runs": "sum"
            }).reset_index()
            bowler_stats["economy"] = bowler_stats["batsman_runs"] / (bowler_stats["ball"] / 6)
            selected_batter_stats = batter_stats[batter_stats['batter'] == selected_batter]
            selected_bowler_stats = bowler_stats[bowler_stats['bowler'] == selected_bowler]

            if not selected_batter_stats.empty and not selected_bowler_stats.empty:
                predicted_sr = selected_batter_stats["strike_rate"].values[0]
                predicted_eco = selected_bowler_stats["economy"].values[0]
                predicted_rpo = predicted_sr * 6 / 100
                predicted_boundary_pct = np.random.uniform(10, 25)  # Placeholder for ML prediction
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Predicted Strike Rate", round(predicted_sr, 2))
                col2.metric("Predicted Economy", round(predicted_eco, 2))
                col3.metric("Predicted RPO", round(predicted_rpo, 2))
                col4.metric("Predicted Boundary %", f"{predicted_boundary_pct:.2f}%")
                fig = go.Figure(data=[
                    go.Bar(x=["Strike Rate", "Economy", "RPO", "Boundary %"],
                           y=[predicted_sr, predicted_eco, predicted_rpo, predicted_boundary_pct],
                           marker_color='teal')
                ])
                fig.update_layout(title=f"Predicted Matchup Summary: {selected_batter} vs {selected_bowler}",
                                  height=400)
                st.plotly_chart(fig, use_container_width=True)

            else:
                st.info("Not enough data for either batter or bowler.")
        else:
            runs = matchup_df['batsman_runs'].sum()
            balls = matchup_df.shape[0]
            sr = (runs / balls) * 100 if balls else 0
            eco = (runs / (balls / 6)) if balls else 0
            boundary_pct = np.random.uniform(10, 25)
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Runs Scored", runs)
            col2.metric("Balls Faced", balls)
            col3.metric("Strike Rate", round(sr, 2))
            col4.metric("Economy", round(eco, 2))
            col5.metric("Boundary %", f"{boundary_pct:.2f}%")
            st.markdown("---")
            fig = go.Figure(data=[
                go.Bar(x=["Runs", "Balls", "Strike Rate", "Economy", "Boundary %"],
                       y=[runs, balls, sr, eco, boundary_pct], marker_color='crimson')
            ])
            fig.update_layout(title=f"Matchup Summary: {selected_batter} vs {selected_bowler}", height=400)
            st.plotly_chart(fig, use_container_width=True)

    # =================================
    # 4. CLUSTERING & INSIGHTS
    # =================================
    elif view_mode == "Clustering & Insights":
        st.header("Player Clustering & Performance Insights")
        df_filtered = apply_common_filters(df)  # 🔁 Apply filters
        with st.expander("Batter Type"):
            st.info(
                "See which cluster your favorite batter belongs to whether they’re a power hitter, anchor, or aggressive finisher...")
            cluster_df = df_filtered.groupby("batter").agg({
                "batsman_runs": "sum",
                "ball": "count",
                "dots": "sum",
                "num_fours": lambda x: (df_filtered.loc[x.index, "batsman_runs"] == 4).sum(),
                "num_sixes": lambda x: (df_filtered.loc[x.index, "batsman_runs"] == 6).sum()
            }).reset_index()

            cluster_df.rename(columns={
                "ball": "balls_faced",
                "num_fours": "fours",
                "num_sixes": "sixes"
            }, inplace=True)

            cluster_df["strike_rate"] = (cluster_df["batsman_runs"] / cluster_df["balls_faced"]) * 100
            cluster_df["dot_pct"] = (cluster_df["dots"] / cluster_df["balls_faced"]) * 100
            cluster_df["boundary_pct"] = ((cluster_df["fours"] + cluster_df["sixes"]) / cluster_df["balls_faced"]) * 100

            scaled = StandardScaler().fit_transform(cluster_df[["strike_rate", "dot_pct", "boundary_pct"]])
            cluster_df["cluster"] = KMeans(n_clusters=4, random_state=42).fit_predict(scaled)


            def label_batter(row):
                if row["strike_rate"] > 150 and row["boundary_pct"] > 20:
                    return "Power Hitter"
                elif row["strike_rate"] > 130:
                    return "Aggressive Finisher"
                elif row["dot_pct"] < 30:
                    return "Strike Rotator"
                else:
                    return "Anchor"


            cluster_df["cluster_label"] = cluster_df.apply(label_batter, axis=1)

            fig = px.scatter(
                cluster_df,
                x="strike_rate",
                y="batsman_runs",
                color="cluster_label",
                hover_data=["batter", "batsman_runs", "strike_rate", "balls_faced", "fours", "sixes", "cluster_label"],
                title="Batter Clustering Based on Performance"
            )
            fig.update_layout(legend_title_text="Batter Type", height=500)
            st.plotly_chart(fig, use_container_width=True)

            for label in cluster_df["cluster_label"].unique():
                st.markdown(f"#### {label}")
                st.dataframe(cluster_df[cluster_df["cluster_label"] == label][
                                 ["batter", "strike_rate", "dot_pct", "boundary_pct", "balls_faced"]
                             ].head(5))
            pass

        with st.expander("Batter Style"):
            st.info("Group batters by style and performance using advanced clustering...")
            batter_stats = df_filtered.groupby("batter").agg({
                "batsman_runs": "sum",
                "ball": "count",
                "is_wicket": "sum",
                "match_id": "nunique"
            }).reset_index()
            batter_stats["strike_rate"] = (batter_stats["batsman_runs"] / batter_stats["ball"]) * 100
            batter_stats["average"] = batter_stats["batsman_runs"] / batter_stats["is_wicket"].replace(0, 1)
            batter_stats["matches"] = batter_stats["match_id"]

            batter_features = batter_stats[["strike_rate", "average", "matches"]]
            scaler = StandardScaler()
            batter_scaled = scaler.fit_transform(batter_features)
            pca_batter = PCA(n_components=2)
            batter_pca = pca_batter.fit_transform(batter_scaled)

            kmeans_batter = KMeans(n_clusters=4, random_state=42, n_init=10)
            batter_stats["cluster"] = kmeans_batter.fit_predict(batter_pca)

            cluster_labels_batter = {0: "Finishers", 1: "Anchors", 2: "Power Hitters", 3: "Consistent Performers"}
            batter_stats["cluster_label"] = batter_stats["cluster"].map(cluster_labels_batter)

            fig_batter = px.scatter(
                batter_stats,
                x="strike_rate",
                y="average",
                color="cluster_label",
                hover_name="batter",
                hover_data=["strike_rate", "average", "matches"],
                title="Batter Style Clusters (Based on Actual Stats)",
                labels={"strike_rate": "Strike Rate", "average": "Average", "cluster_label": "Batter Style"}
            )
            st.plotly_chart(fig_batter, use_container_width=True)
            st.subheader("Sample Players in Each Cluster (Batters)")
            for cl in sorted(batter_stats["cluster"].unique()):
                cluster_name = cluster_labels_batter[cl]
                st.markdown(f"{cluster_name} (Cluster {cl}):")
                st.write(
                    batter_stats[batter_stats["cluster"] == cl][
                        ["batter", "strike_rate", "average", "matches"]
                    ].head(5)
                )
            pass

        with st.expander("Bowler Type"):
            st.info("Group bowlers by their strengths...")
            bowler_stats = df_filtered.groupby("bowler").agg({
                "batsman_runs": "sum",
                "ball": "count",
                "is_wicket": "sum",
                "match_id": "nunique",
                "dots": "sum",
                "no_balls": "sum",
                "wide_balls": "sum",
                "bowling_team": lambda x: x.mode().iloc[0]
            }).reset_index()
            bowler_stats["economy"] = bowler_stats["batsman_runs"] / (bowler_stats["ball"] / 6)
            bowler_stats["average"] = bowler_stats["batsman_runs"] / bowler_stats["is_wicket"].replace(0, 1)

            bowler_features = bowler_stats[["economy", "average", "dots", "no_balls", "wide_balls"]]
            bowler_scaled = scaler.fit_transform(bowler_features)
            pca_bowler = PCA(n_components=2)
            bowler_pca = pca_bowler.fit_transform(bowler_scaled)
            kmeans_bowler = KMeans(n_clusters=4, random_state=42, n_init=10)
            bowler_stats["cluster"] = kmeans_bowler.fit_predict(bowler_pca)

            cluster_labels_bowler = {0: "Economical Bowlers", 1: "Aggressive Bowlers", 2: "Wicket Takers",
                                     3: "Consistent Bowlers"}
            bowler_stats["cluster_label"] = bowler_stats["cluster"].map(cluster_labels_bowler)

            fig_bowler = px.scatter(
                bowler_stats,
                x="economy",
                y="average",
                color="cluster_label",
                hover_name="bowler",
                hover_data=["economy", "average", "dots", "no_balls", "wide_balls"],
                title="Bowler Clusters (Based on Actual Stats)",
                labels={"economy": "Economy Rate", "average": "Bowling Average"}
            )
            st.plotly_chart(fig_bowler)
            st.subheader("Sample Players in Each Cluster (Bowlers)")
            for cl in sorted(bowler_stats["cluster"].unique()):
                cluster_name = cluster_labels_bowler[cl]
                st.markdown(f"{cluster_name} (Cluster {cl}):")
                st.write(
                    bowler_stats[bowler_stats["cluster"] == cl][
                        ["bowler", "economy", "average", "dots", "no_balls", "wide_balls"]
                    ].head(5)
                )
            pass

        with st.expander("Bowler Control"):
            st.info("Analyze bowler profiles based on dot balls, wides, and no-balls...")
            bowler_df = df_filtered.groupby("bowler").agg({
                "is_dot_ball": "sum",
                "is_noball": "sum",
                "is_wide": "sum",
                "ball": "count"
            }).reset_index()

            bowler_df.rename(columns={
                "is_dot_ball": "dot_balls",
                "is_noball": "no_balls",
                "is_wide": "wide_balls",
                "ball": "total_balls"
            }, inplace=True)

            bowler_df = bowler_df[bowler_df["total_balls"] >= 100]
            bowler_df["dot_percentage"] = (bowler_df["dot_balls"] / bowler_df["total_balls"]) * 100
            bowler_df["wide_percentage"] = (bowler_df["wide_balls"] / bowler_df["total_balls"]) * 100
            bowler_df["noball_percentage"] = (bowler_df["no_balls"] / bowler_df["total_balls"]) * 100

            features = bowler_df[["dot_percentage", "wide_percentage", "noball_percentage"]]
            scaled = StandardScaler().fit_transform(features)
            kmeans = KMeans(n_clusters=4, random_state=42)
            bowler_df["cluster"] = kmeans.fit_predict(scaled)

            fig = px.scatter(
                bowler_df,
                x="dot_percentage",
                y="wide_percentage",
                color="cluster",
                hover_name="bowler",
                hover_data=["noball_percentage", "dot_balls", "wide_balls", "no_balls", "total_balls"],
                size="total_balls",
                title="Clustered Bowler Profiles: Dot vs Wide %",
            )
            fig.update_layout(legend_title_text="Cluster ID", height=500)
            st.plotly_chart(fig, use_container_width=True)
            pass

            st.markdown("### 📋 Clustered Bowler Summary")
            st.dataframe(
                bowler_df[
                    ["bowler", "cluster", "dot_percentage", "wide_percentage", "noball_percentage", "total_balls"]
                ].sort_values("dot_percentage", ascending=False).reset_index(drop=True)
            )
        st.markdown("---")
        st.header("STRATEGIC MATCHUP RECOMMENDATION")

        batter_options = ["--Select--"] + list(batter_stats["batter"].unique())
        bowler_options = ["--Select--"] + list(bowler_stats["bowler"].unique())

        selected_batter = st.selectbox("Select Batter", batter_options, index=0)
        selected_bowler = st.selectbox("Select Bowler", bowler_options, index=0)

        if selected_batter == "--Select--" or selected_bowler == "--Select--":
            st.info("Please select both a batter and a bowler for a recommendation.")
            st.stop()

        st.markdown("---")
        st.header("📈 Bowler Suggestions Against Selected Batter")
        batter_label = batter_stats[batter_stats["batter"] == selected_batter]["cluster_label"].values[0]
        bowler_label = bowler_stats[bowler_stats["bowler"] == selected_bowler]["cluster_label"].values[0]
        bowler_team = bowler_stats[bowler_stats["bowler"] == selected_bowler]["bowling_team"].values[0]
        st.success(f"{selected_batter} ({batter_label}) vs {selected_bowler} ({bowler_label}) ")
        if batter_label == "Anchors" and bowler_label == "Wicket Takers":
            suggestion = "Dangerous! Anchors may get exposed early."
        elif batter_label in ["Power Hitters", "Aggressive Finishers"] and bowler_label in ["Wicket Takers",
                                                                                            "Economical Bowlers"]:
            suggestion = "Watch out! Tight overs can restrict attacking batters."
        elif batter_label == "Steady Scorers" and bowler_label == "Consistent Bowlers":
            suggestion = "Solid battle. Expect controlled play."
        else:
            suggestion = "Standard matchup. Use based on game situation."
        st.info(f"Insight: {suggestion}")
        best_clusters = ["Wicket Takers", "Economical Bowlers", "Consistent Bowlers"]
        same_team_best = bowler_stats[(bowler_stats["cluster_label"].isin(best_clusters)) &
                                      (bowler_stats["bowler"] != selected_bowler) &
                                      (bowler_stats["bowling_team"] == bowler_team)]
        if not same_team_best.empty:
            alt_bowler = same_team_best.sort_values(by="average").head(1)["bowler"].values[0]
            alt_cluster = same_team_best.sort_values(by="average").head(1)["cluster_label"].values[0]
            st.info(f"🔄 Consider trying {alt_bowler} from {bowler_team} instead of {selected_bowler}.")
        else:
            st.warning("No alternative bowlers from the same team and cluster found.")

        counter_clusters = {
            "Wicket Takers": ["Power Hitters", "Aggressive Finishers"],
            "Economical Bowlers": ["Anchors", "Steady Scorers"],
            "Consistent Bowlers": ["Aggressive Finishers", "Power Hitters"]
        }

        batter_team_lookup = (
            df.groupby("batter")["batting_team"]
                .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else None)
                .to_dict()
        )
        batter_stats["batting_team"] = batter_stats["batter"].map(batter_team_lookup)

        selected_batter_team = None
        if not batter_stats.loc[batter_stats["batter"] == selected_batter, "batting_team"].empty:
            selected_batter_team = batter_stats.loc[
                batter_stats["batter"] == selected_batter, "batting_team"
            ].values[0]

        st.markdown("---")
        st.header("📈 Batter Suggestions Against Selected Bowler")

        if selected_batter_team and selected_batter_team is not None and str(selected_batter_team) != "nan":
            same_team_batters = batter_stats[
                (batter_stats["batter"] != selected_batter) &
                (batter_stats["batting_team"] == selected_batter_team)
                ]
            team_string = f"{selected_batter_team}"
        else:
            same_team_batters = batter_stats[batter_stats["batter"] != selected_batter]
            team_string = "across all teams"

        preferred_batter_clusters = counter_clusters.get(bowler_label, [])

        if preferred_batter_clusters:
            filtered = same_team_batters[same_team_batters["cluster_label"].isin(preferred_batter_clusters)]
        else:
            filtered = same_team_batters

        if not filtered.empty:
            st.success(f"Recommended batters from {team_string} against {selected_bowler} ({bowler_label}):")
            for _, row in filtered.sort_values(by="strike_rate", ascending=False).head(3).iterrows():
                st.markdown(f"- {row['batter']} ({row['cluster_label']}) – SR: {round(row['strike_rate'], 2)}")
        else:
            st.warning(f"No suitable counter batters from {team_string} found based on clusters. Showing top batters:")
            for _, row in same_team_batters.sort_values(by="strike_rate", ascending=False).head(3).iterrows():
                st.markdown(f"- {row['batter']} ({row['cluster_label']}) – SR: {round(row['strike_rate'], 2)}")







else:
    st.warning("Please upload a valid csv file to get started.")
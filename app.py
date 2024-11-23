import pandas as pd
from flask import Flask, render_template, request
import random
import joblib  # For loading the trained model

app = Flask(__name__)

# Load the CSV files
deliveries_df = pd.read_csv('deliveries.csv')
matches_df = pd.read_csv('matches.csv')

# Load the trained model
rf_model = joblib.load('cricket_match_predictor_model.pkl')


# Function to filter head-to-head matches between two teams
def get_head_to_head_matches(team1, team2):
    # Filter matches where either team1 or team2 is involved
    filtered_matches = matches_df[(
                                          (matches_df['team1'] == team1) & (matches_df['team2'] == team2)) |
                                  ((matches_df['team1'] == team2) & (matches_df['team2'] == team1))
                                  ]

    # Convert 'date' column to datetime using .loc
    filtered_matches.loc[:, 'date'] = pd.to_datetime(filtered_matches['date'])

    # Sort the filtered matches by date (most recent first)
    sorted_matches = filtered_matches.sort_values(by='date', ascending=False)
    return sorted_matches.head(5)  # Get the last 5 matches


def head_to_head_win_ratio(matches_df, team1, team2):
    # Filter matches where either team1 or team2 is involved
    head_to_head_matches = matches_df[
        ((matches_df['team1'] == team1) & (matches_df['team2'] == team2)) |
        ((matches_df['team1'] == team2) & (matches_df['team2'] == team1))
        ]

    # Count wins for team1 and team2
    team1_wins = head_to_head_matches[head_to_head_matches['winner'] == team1].shape[0]
    team2_wins = head_to_head_matches[head_to_head_matches['winner'] == team2].shape[0]

    # Calculate the win ratio (team1 wins / total matches)
    total_matches = team1_wins + team2_wins
    if total_matches > 0:
        win_ratio = team1_wins / total_matches
    else:
        win_ratio = 0  # No previous matches, or no winner recorded

    return win_ratio


# Function to predict match winner based on model and input data
def predict_winner(team1, team2, toss_winner):
    # Calculate head-to-head win ratio for the input teams
    win_ratio = head_to_head_win_ratio(matches_df, team1, team2)

    # Determine toss_winner_is_team1 (1 if toss winner is team1, else 0)
    toss_winner_is_team1 = 1 if toss_winner == team1 else 0

    # Prepare input features for the model
    input_features = pd.DataFrame({
        'toss_winner_is_team1': [toss_winner_is_team1],
        'team1_win_ratio': [win_ratio]
    })

    # Predict winner using the model
    prediction = rf_model.predict(input_features)
    predicted_winner = team1 if prediction[0] == 1 else team2
    return predicted_winner


# Route for the homepage
@app.route('/')
def index():
    teams = sorted(matches_df['team1'].unique())  # Get all unique teams
    return render_template('index.html', teams=teams)


# Route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    team1 = request.form['team1']
    team2 = request.form['team2']
    toss_winner = request.form['toss_winner']

    # Get the last 5 head-to-head matches
    last_matches = get_head_to_head_matches(team1, team2)

    # Predict the winner using the trained model
    winner = predict_winner(team1, team2, toss_winner)

    # Get the highest scoring batsman and highest wicket-taking bowler in head-to-head
    head_to_head_deliveries = deliveries_df[
        ((deliveries_df['batting_team'] == team1) | (deliveries_df['batting_team'] == team2)) &
        ((deliveries_df['bowling_team'] == team1) | (deliveries_df['bowling_team'] == team2))
        ]

    # Highest scoring batsman
    batsman_runs = head_to_head_deliveries.groupby('batter')['batsman_runs'].sum().reset_index()
    top_batsman = batsman_runs.loc[batsman_runs['batsman_runs'].idxmax()]

    # Bowler with the most wickets
    bowler_wickets = head_to_head_deliveries.groupby('bowler')['is_wicket'].sum().reset_index()
    top_bowler = bowler_wickets.loc[bowler_wickets['is_wicket'].idxmax()]

    # Render the prediction page with results
    return render_template('predict.html', winner=winner, last_matches=last_matches.to_dict('records'),
                           top_batsman=top_batsman, top_bowler=top_bowler, team1=team1, team2=team2)


# Start the Flask app
if __name__ == '__main__':
    app.run(debug=True)


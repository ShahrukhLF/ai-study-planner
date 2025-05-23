import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from flask import Flask, render_template, request, flash, redirect, url_for
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'

# Configuration
MAX_HOURS_PER_DAY = 8
MIN_HOURS_PER_DAY = 0.5

def load_resources():
    try:
        models = {
            'kmeans': joblib.load("models/kmeans_model.pkl"),
            'knn': joblib.load("models/knn_model.pkl"),
            'scaler': joblib.load("models/scaler.pkl"),
            'cluster_order': joblib.load("models/cluster_order.pkl")
        }
        study_data = pd.read_csv("data/study_data.csv")
        with open("models/accuracy.txt") as f:
            accuracy = f.read().strip()
        return models, study_data, accuracy
    except Exception as e:
        print(f"Error loading resources: {str(e)}")
        return None, None, "Not available"

models, study_data, accuracy = load_resources()

@app.route('/')
def index():
    accuracy = read_accuracy_from_file()  # Make sure this returns something like "Accuracy: 85%"
    return render_template('index.html', topics=TOPICS, MIN_HOURS_PER_DAY=MIN_HOURS_PER_DAY,
                           MAX_HOURS_PER_DAY=MAX_HOURS_PER_DAY, accuracy=accuracy)


@app.route("/generate_plan", methods=["POST"])
def generate_plan():
    if models is None or study_data is None or study_data.empty:
        flash("System not ready. Please try again later.", "danger")
        return redirect(url_for('index'))

    try:
        # Get and validate inputs
        selected_topics = request.form.getlist('topics')
        study_hours = min(MAX_HOURS_PER_DAY,
                          max(MIN_HOURS_PER_DAY, float(request.form['study_hours'])))
        deadline = datetime.strptime(request.form['deadline'], '%Y-%m-%d')
        today = datetime.now()
        available_days = (deadline - today).days + 1  # Include today

        if available_days <= 0:
            flash("Deadline must be in the future!", "danger")
            return redirect(url_for('index'))

        if not selected_topics:
            flash("Please select at least one subject!", "danger")
            return redirect(url_for('index'))

        selected_data = study_data[study_data['topic'].isin(selected_topics)].copy()

        X = selected_data[['difficulty', 'importance', 'time_required']]
        X_scaled = models['scaler'].transform(X)
        clusters = models['kmeans'].predict(X_scaled)
        selected_data['cluster'] = clusters

        # Predict days needed
        X_days = np.column_stack([clusters, np.full(len(clusters), study_hours)])
        days_needed = np.maximum(1, np.round(models['knn'].predict(X_days)))
        selected_data['days_needed'] = days_needed

        # Adjust if total exceeds available days
        total_days = int(days_needed.sum())
        if total_days > available_days:
            difficulty_scores = np.array([models['cluster_order'][c] for c in clusters])
            weights = difficulty_scores + 1
            weight_sum = np.sum(weights)
            proportional_days = np.floor((weights / weight_sum) * available_days).astype(int)
            proportional_days = np.maximum(1, proportional_days)

            # Redistribute remaining days
            remaining_days = available_days - np.sum(proportional_days)
            if remaining_days > 0:
                priority_order = np.argsort(-weights)
                for i in priority_order[:remaining_days]:
                    proportional_days[i] += 1

            selected_data['days_needed'] = proportional_days

        # Ensure each topic gets enough days to complete required hours
        selected_data['hours_per_day'] = np.minimum(
            study_hours,
            np.maximum(MIN_HOURS_PER_DAY,
                       np.round(selected_data['time_required'] / selected_data['days_needed'], 1))
        )

        selected_data['actual_hours'] = selected_data['hours_per_day'] * selected_data['days_needed']
        total_required_hours = selected_data['time_required'].sum()
        total_allocated_hours = selected_data['actual_hours'].sum()

        # If any subject's actual_hours < required, allocate more days if possible
        for idx, row in selected_data.iterrows():
            while row['actual_hours'] < row['time_required'] and selected_data['days_needed'].sum() < available_days:
                selected_data.at[idx, 'days_needed'] += 1
                row['days_needed'] += 1
                row['hours_per_day'] = min(study_hours, max(MIN_HOURS_PER_DAY,
                    round(row['time_required'] / row['days_needed'], 1)))
                row['actual_hours'] = row['hours_per_day'] * row['days_needed']
                selected_data.at[idx, 'hours_per_day'] = row['hours_per_day']
                selected_data.at[idx, 'actual_hours'] = row['actual_hours']

        # Final achievability check
        total_allocated_hours = selected_data['actual_hours'].sum()
        achieved_percent = min(100, int((total_allocated_hours / total_required_hours) * 100)) if total_required_hours > 0 else 0

        # Generate schedule
        schedule = []
        current_date = today
        for _, row in selected_data.iterrows():
            topic_days = int(row['days_needed'])
            schedule.append({
                'topic_name': row['topic'],
                'difficulty': ['Easy', 'Medium', 'Hard'][models['cluster_order'][row['cluster']]],
                'days_needed': topic_days,
                'hours_per_day': float(row['hours_per_day']),
                'start_date': current_date.strftime('%Y-%m-%d'),
                'end_date': (current_date + timedelta(days=topic_days - 1)).strftime('%Y-%m-%d')
            })
            current_date += timedelta(days=topic_days)

                        # Convert accuracy to percentage (robust version)
        try:
            import re
            match = re.search(r'R.?Score:\s*(-?\d+(?:\.\d+)?)', accuracy)
            if match:
                r2 = float(match.group(1))
                accuracy_percent = min(100, max(0, int((r2 + 1) * 50)))
            else:
                accuracy_percent = "N/A"
        except Exception as e:
            print(f"Accuracy parsing error: {e}")
            accuracy_percent = "N/A"



        return render_template("index.html",
                               topics=study_data['topic'].tolist(),
                               schedule=schedule,
                               accuracy=f"AI Accuracy: {accuracy_percent}%",
                               total_hours=round(total_required_hours, 1),
                               total_days=int(selected_data['days_needed'].sum()),
                               available_days=available_days,
                               completion_percentage=achieved_percent,
                               study_hours=study_hours,
                               MIN_HOURS_PER_DAY=MIN_HOURS_PER_DAY,
                               MAX_HOURS_PER_DAY=MAX_HOURS_PER_DAY)

    except Exception as e:
        flash(f"Error generating plan: {str(e)}", "danger")
        return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)

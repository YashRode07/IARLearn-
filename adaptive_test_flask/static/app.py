from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize

app = Flask(__name__)
app.secret_key = "adaptive_test_secret"

def logistic_1pl(theta, b):
    return 1 / (1 + np.exp(-1.7 * (theta - b)))

def negative_log_likelihood(theta, responses, difficulties):
    responses = np.array(responses)
    probabilities = 1 / (1 + np.exp(-(theta - np.array(difficulties))))
    probabilities = np.clip(probabilities, 1e-9, 1 - 1e-9)
    likelihood = responses * np.log(probabilities) + (1 - responses) * np.log(1 - probabilities)
    return -np.sum(likelihood)

def update_theta(responses, difficulties, initial_theta=0.0):
    result = minimize(negative_log_likelihood, initial_theta, args=(responses, difficulties), method='BFGS')
    new_theta = result.x[0] if result.success else initial_theta
    step_size = 0.1
    theta = min(initial_theta + step_size, new_theta) if new_theta > initial_theta else max(initial_theta - step_size, new_theta)
    return np.clip(theta, 0, 2)

def load_dataset(filename="questions.csv"):
    try:
        df = pd.read_csv(filename)
        return df
    except FileNotFoundError:
        return None

def get_latest_profile_number(test_taker):
    try:
        df = pd.read_csv("test_taker_profile.csv", names=["Test Taker", "Final Ability Estimate", "Weak Topics"])
        df = df[df["Test Taker"].str.startswith(test_taker)]
        if not df.empty:
            numbers = [int(name.split('-')[-1]) for name in df["Test Taker"] if '-' in name and name.split('-')[-1].isdigit()]
            return max(numbers) if numbers else 0
    except FileNotFoundError:
        return 0
    return 0

def select_question(theta, dataset, weak_topics, asked_questions):
    dataset = dataset[~dataset["Question ID"].isin(asked_questions)].copy()
    if weak_topics:
        dataset = dataset[dataset["Topic"].isin(weak_topics)]
    if not dataset.empty:
        dataset["Abs Diff"] = abs(dataset["Difficulty (b)"] - theta)
        best_qid = dataset.loc[dataset["Abs Diff"].idxmin(), "Question ID"]
        return dataset[dataset["Question ID"] == best_qid].iloc[0]
    return None

def save_test_results(test_taker, final_theta, weak_topics):
    weak_topics_str = ",".join(weak_topics) if weak_topics else "None"
    df = pd.DataFrame({"Test Taker": [test_taker], "Final Ability Estimate": [final_theta], "Weak Topics": [weak_topics_str]})
    df.to_csv("test_taker_profile.csv", mode='a', header=False, index=False)

def load_past_weak_topics(test_taker):
    latest_profile_num = get_latest_profile_number(test_taker)
    test_taker_latest = f"{test_taker}-{latest_profile_num}"
    try:
        df = pd.read_csv("test_taker_profile.csv", names=["Test Taker", "Final Ability Estimate", "Weak Topics"])
        df = df[df["Test Taker"] == test_taker_latest]
        if not df.empty:
            last_entry = df.iloc[-1]["Weak Topics"]
            return set(str(last_entry).split(",")) if pd.notna(last_entry) and last_entry != "None" else set()
    except FileNotFoundError:
        pass
    return set()

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        session['test_taker'] = request.form['name']
        session['theta'] = 0.0
        session['asked_questions'] = []
        dataset = load_dataset()
        session['weak_topics'] = list(load_past_weak_topics(session['test_taker']))
        return redirect(url_for('test'))
    return render_template('home.html')

@app.route('/test', methods=['GET', 'POST'])
def test():
    dataset = load_dataset()
    if request.method == 'POST':
        selected_answer = request.form['answer']
        question = session['current_question']
        correct_option = next((option for option in ["A", "B", "C", "D"] 
                      if str(question.get(f"Option {option}", "")).strip().upper() == str(question.get("Answer", "")).strip().upper()), None)
        is_correct = selected_answer == correct_option
        session['responses'].append(1 if is_correct else 0)
        session['difficulties'].append(question["Difficulty (b)"])
        if not is_correct:
            session['weak_topics'].append(question['Topic'])
        elif question['Topic'] in session['weak_topics']:
            session['weak_topics'].remove(question['Topic'])
        session['theta'] = update_theta(session['responses'], session['difficulties'], session['theta'])
        session['asked_questions'].append(question['Question ID'])
        if len(session['asked_questions']) >= 10:
            return redirect(url_for('profile'))
    question = select_question(session['theta'], dataset, session['weak_topics'], session['asked_questions'])
    if question is None:
        return redirect(url_for('profile'))
    session['current_question'] = question.to_dict()
    question_number = len(session['asked_questions']) + 1
    return render_template('test.html', question=question, question_number=question_number)

@app.route('/profile')
def profile():
    test_taker = session.get('test_taker', 'Unknown')
    save_test_results(test_taker, session['theta'], session['weak_topics'])
    return render_template('profile.html', name=test_taker, theta=session['theta'], weak_topics=session['weak_topics'])

if __name__ == "__main__":
    app.run(debug=True)

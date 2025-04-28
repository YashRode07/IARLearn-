from flask import Flask, render_template, request, redirect, url_for, session, flash
from dotenv import load_dotenv  # Add this with other imports
import numpy as np
import pandas as pd
import random
from scipy.optimize import minimize
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
import os
import re
import hashlib

USER_CSV = "users.csv"

app = Flask(__name__)
app.secret_key = "adaptive_test_secret"


# Initialize Gemini model after Flask app creation
load_dotenv()
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.3, max_tokens=500)


# Add RAG prompt template
system_prompt = (
    "You are an expert educational assistant. Generate personalized recommendations based on test profile. "
    "Follow these rules:\n"
    "1. Provide exactly 5 recommendations\n"
    "2. Each must include a YouTube video URL\n"
    "3. Focus on improving weak areas\n"
    "4. Avoid repeating previous suggestions\n\n"
    "Student Profile:\n"
    "Name: {profile_name}\n"
    "Ability Estimate Using IRT model: {ability_estimate}\n"
    "Weak Topics: {weak_topic}\n"
    "Weak Subtopics: {weak_sub_topic}\n"
    "Previous Recommendations: {previous_recommendations}"
)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

rag_chain = prompt | llm


def get_previous_recommendations(test_taker):
    try:
        current_topic = session.get('selected_topic', 'general')
        pattern = fr"^{re.escape(test_taker)}-(\d+)-{re.escape(current_topic)}"
        
        # CORRECTED: Read with header and use actual columns
        df = pd.read_csv("test_profiles.csv", header=0)
        
        # Extract and sort profiles properly
        df['Profile Number'] = df["Test Taker"].str.extract(pattern).astype(float)
        relevant_profiles = df.dropna(subset=['Profile Number']).sort_values('Profile Number', ascending=False)
        
        return relevant_profiles.iloc[0]['Previous Recommendations'] if not relevant_profiles.empty else "None"
    except Exception as e:
        print(f"Error getting previous recommendations: {str(e)}")
        return "None"

def generate_recommendations(profile_name, ability, weak_topic, weak_subtopic, previous_recs):
    try:
        response = rag_chain.invoke({
            "input": "Generate 5 personalized learning recommendations with YouTube video links",
            "profile_name": profile_name,
            "ability_estimate": ability,
            "weak_topic": weak_topic,
            "weak_sub_topic": weak_subtopic,
            "previous_recommendations": previous_recs
        })
        return response.content
    except Exception as e:
        print(f"Recommendation generation error: {str(e)}")
        return "None"



# Initialize Q-Table for Reinforcement Learning
Q_TABLE = {}

def update_q_table(state, action, reward, next_state, alpha=0.1, gamma=0.9):
    if state not in Q_TABLE:
        Q_TABLE[state] = {}
    if action not in Q_TABLE[state]:
        Q_TABLE[state][action] = 0
    max_future_q = max(Q_TABLE[next_state].values(), default=0) if next_state in Q_TABLE else 0
    Q_TABLE[state][action] += alpha * (reward + gamma * max_future_q - Q_TABLE[state][action])

def logistic_1pl(theta, b):
    return 1 / (1 + np.exp(-1.7 * (theta - b)))

def negative_log_likelihood(theta, responses, difficulties):
    responses = np.array(responses)
    # Add the 1.7 discrimination factor to match standard IRT practice
    probabilities = 1 / (1 + np.exp(-1.7 * (theta - np.array(difficulties))))  # Changed line
    probabilities = np.clip(probabilities, 1e-9, 1 - 1e-9)
    likelihood = responses * np.log(probabilities) + (1 - responses) * np.log(1 - probabilities)
    return -np.sum(likelihood)

def update_theta(responses, difficulties, initial_theta=0.0):
    result = minimize(negative_log_likelihood, initial_theta, 
                     args=(responses, difficulties), method='BFGS')
    # Remove step_size clamping and use optimized theta directly
    new_theta = result.x[0] if result.success else initial_theta  # Changed line
    # Use wider realistic theta range (-3 to 3 is common in psychometrics)
    return np.clip(new_theta, -3, 3)  # Changed line

def load_dataset(filename="questions.csv"):
    try:
        return pd.read_csv(filename)
    except FileNotFoundError:
        return None

def select_question(theta, dataset, weak_subtopics, asked_questions, subtopic_distribution=None):
    if dataset is None or dataset.empty:
        return None

    # Normalize for consistency
    dataset['Subtopic'] = dataset['Subtopic'].str.lower().str.strip()
    weak_subtopics = [sub.lower().strip() for sub in weak_subtopics]

    # Filter unanswered questions
    dataset = dataset[~dataset["Question ID"].isin(asked_questions)].copy()

    # Filter by weak subtopics
    if weak_subtopics:
        dataset = dataset[dataset["Subtopic"].isin(weak_subtopics)]
        dataset = dataset[~dataset["Question ID"].isin(asked_questions)]

        # Apply subtopic distribution control
        if subtopic_distribution:
            available_subtopics = [
                sub for sub in weak_subtopics 
                if subtopic_distribution.get(sub, 0) > 0 and not dataset[dataset["Subtopic"] == sub].empty
            ]
            if available_subtopics:
                chosen_sub = max(available_subtopics, key=lambda x: subtopic_distribution[x])
                dataset = dataset[dataset["Subtopic"] == chosen_sub]
                subtopic_distribution[chosen_sub] -= 1

    if dataset.empty:
        return None

    # Choose question with closest difficulty to current theta
    dataset["Abs Diff"] = abs(dataset["Difficulty (b)"] - theta)
    return dataset.nsmallest(1, "Abs Diff").iloc[0]

def save_test_results(test_taker, final_theta, weak_topics, weak_subtopics, wrong_answered):
    # Load dataset with error handling
    dataset = load_dataset()
    current_topic = session.get('selected_topic', 'general').replace('-', ' ')
    
    # Handle missing dataset case
    if dataset is None:
        filtered_weak_subtopics = []
        filtered_weak_topics = [t for t in weak_topics if t.lower() == current_topic.lower()]
    else:
        valid_subtopics = dataset[dataset['Topic'].str.lower() == current_topic.lower()]['Subtopic'].str.lower().values
        filtered_weak_subtopics = [st for st in weak_subtopics if st.lower() in valid_subtopics]
        filtered_weak_topics = [t for t in weak_topics if t.lower() == current_topic.lower()]


    # Add CSV initialization check
    if not os.path.exists("test_profiles.csv"):
        with open("test_profiles.csv", "w") as f:
            f.write("Test Taker,Ability Estimate,Wrong Answered Questions,Weak Sub Topic,Weak Topic,Previous Recommendations\n")

    # Load dataset for validation
    dataset = load_dataset()
    current_topic = session.get('selected_topic', 'general').replace('-', ' ')
    
    # Filter weak topics/subtopics based on current topic
    filtered_weak_topics = [t for t in weak_topics if t.lower() == current_topic.lower()]
    
    # Handle dataset availability
    if dataset is not None:
        valid_subtopics = dataset[
            dataset['Topic'].str.lower() == current_topic.lower()
        ]['Subtopic'].str.lower().values
        filtered_weak_subtopics = [st for st in weak_subtopics if st.lower() in valid_subtopics]
    else:
        filtered_weak_subtopics = []

    # Rest of the function remains the same
    topic = session.get('selected_topic', 'general')
    subtopic = session.get('selected_subtopic', '')
    
    profile_suffix = topic.lower().replace(" ", "-")
    if subtopic:
        profile_suffix += "-" + subtopic.lower().replace(" ", "-")
    
    # Get the next profile number for this topic
    profile_num = get_latest_profile_number(test_taker, topic)
    
    test_taker_profile = f"{test_taker}-{profile_num}-{profile_suffix}"

    # Get previous recommendations
    prev_recs = get_previous_recommendations(test_taker)
    
    # Generate new recommendations
    new_recommendations = generate_recommendations(
        test_taker_profile,
        round(final_theta, 2),
        ", ".join(filtered_weak_topics) if filtered_weak_topics else "None",
        ", ".join(filtered_weak_subtopics) if filtered_weak_subtopics else "None",
        prev_recs
    )

    # Format wrong answered questions with full details
    wrong_entries = []
    for qid, text, options, chosen in wrong_answered:
        formatted_options = [f"{opt}" for opt in options]
        options_str = " | ".join(formatted_options)
        clean_text = text.replace('\n', ' ').strip()
        wrong_entries.append(f"{qid}: {clean_text} [Options: {options_str}] [Chosen: {chosen}]")
    
    wrong_str = "||".join(wrong_entries) if wrong_entries else "No wrong answers"
    
    df = pd.DataFrame({
        "Test Taker": [test_taker_profile],
        "Ability Estimate": [round(final_theta, 2)],
        "Wrong Answered Questions": [wrong_str],
        "Weak Sub Topic": [",".join(filtered_weak_subtopics) if filtered_weak_subtopics else "None"],
        "Weak Topic": [",".join(filtered_weak_topics) if filtered_weak_topics else "None"],
        "Previous Recommendations": [new_recommendations]
    })
    
    # Save to CSV with proper escaping
    header = not os.path.exists("test_profiles.csv")
    df.to_csv("test_profiles.csv", mode='a', header=header, index=False, escapechar='\\')

def get_latest_profile_number(test_taker, topic):
    try:
        normalized_topic = topic.lower().replace(" ", "-")
        pattern = fr"^{re.escape(test_taker)}-(\d+)-{re.escape(normalized_topic)}(?:-|$)"
        
        # CORRECTED: Read with header
        df = pd.read_csv("test_profiles.csv", header=0)
        
        # Extract profile numbers safely with capture group
        matches = df["Test Taker"].str.extract(pattern)
        valid_numbers = matches[0].dropna().astype(int)
        
        return valid_numbers.max() + 1 if not valid_numbers.empty else 1
    except Exception as e:
        print(f"Error getting profile number: {str(e)}")
        return 1


def save_baseline_results(test_taker, ability, weak_topics, weak_subtopics, wrong_answered):
    wrong_entries = []
    for ans in wrong_answered:
        if not ans['correct']:
            q = ans['question']
            options = [f"{opt}: {q.get(f'Option {opt}', '')}" for opt in ["A", "B", "C", "D"]]
            wrong_entries.append(f"{q['Question ID']}: {q['Question']} [Options: {' | '.join(options)}] [Chosen: {ans['selected']}]")
    
    wrong_str = "||".join(wrong_entries) if wrong_entries else "No wrong answers"
    
    df = pd.DataFrame({
        "Test Taker": [test_taker],
        "Ability Estimate": [ability],
        "Wrong Answered Questions": [wrong_str],
        "Weak Sub Topic": [",".join(weak_subtopics) if weak_subtopics else "None"],
        "Weak Topic": [",".join(weak_topics) if weak_topics else "None"],
        "Previous Recommendations": ["Baseline Test"]
    })
    
    df.to_csv("test_profiles.csv", mode='a', header=False, index=False, escapechar='\\')



def load_past_weak_subtopics(test_taker, current_topic):
    try:
        dataset = load_dataset()
        current_topic = current_topic.lower().replace('-', ' ')
        valid_subs = dataset[dataset['Topic'].str.lower() == current_topic.lower()]['Subtopic'].unique()
        
        # Match profiles for current topic and get the latest one
        pattern = fr"^{re.escape(test_taker)}-(\d+)-{re.escape(current_topic.replace(' ', '-'))}"
        df = pd.read_csv("test_profiles.csv", names=[
            "Test Taker", "Ability Estimate", "Wrong Answered Questions",
            "Weak Sub Topic", "Weak Topic", "Previous Recommendations"
        ])
        
        # Extract profile numbers and find the latest
        df['Profile_Number'] = df["Test Taker"].str.extract(pattern).astype(float)
        relevant_profiles = df.dropna(subset=['Profile_Number']).sort_values("Profile_Number", ascending=False)
        
        if not relevant_profiles.empty:
            # Get ONLY the latest profile's weak subtopics
            latest_profile = relevant_profiles.iloc[0]
            if pd.notna(latest_profile['Weak Sub Topic']) and latest_profile['Weak Sub Topic'] != 'None':
                all_subs = latest_profile['Weak Sub Topic'].split(',')
                return [sub.strip().lower() for sub in all_subs if sub.strip().lower() in valid_subs]
        
        return []
        
    except Exception as e:
        print(f"Error loading weak subtopics: {str(e)}")
        return []

def load_past_weak_topics(test_taker, current_topic):
    try:
        df = pd.read_csv("test_profiles.csv", names=[
            "Test Taker", "Ability Estimate", "Wrong Answered Questions", 
            "Weak Sub Topic", "Weak Topic", "Previous Recommendations"
        ])
        
        # Only consider profiles for current topic
        pattern = fr"^{re.escape(test_taker)}-\d+-{re.escape(current_topic)}(-|$)"
        df = df[df["Test Taker"].str.contains(pattern, regex=True, na=False)]
        
        all_topics = set()
        for topics in df['Weak Topic']:
            if pd.notna(topics) and topics != "None":
                all_topics.update(topics.split(','))
        return all_topics
    except Exception as e:
        print(f"Error loading topics: {str(e)}")
    return set()




@app.route('/', methods=['GET', 'POST'])
def home():
    if 'test_taker' in session:
        return redirect(url_for('select_topic'))

    if request.method == 'POST':
        action = request.form.get('action')
        test_taker = request.form.get('test_taker', '').strip()
        password = request.form.get('password', '').strip()

        if action == 'register':
            # Registration logic
            if os.path.exists(USER_CSV):
                df = pd.read_csv(USER_CSV)
                if test_taker in df['test_taker'].values:
                    flash('Test taker name already exists!', 'error')
                    return redirect(url_for('home'))

            hashed_pw = hashlib.sha256(password.encode()).hexdigest()
            new_user = pd.DataFrame([[test_taker, hashed_pw]], 
                                   columns=["test_taker", "password"])
            new_user.to_csv(USER_CSV, mode='a', 
                           header=not os.path.exists(USER_CSV), 
                           index=False)

            # Initialize session after successful registration
            session['test_taker'] = test_taker
            session.update({
                'theta': 0.0,
                'asked_questions': [],
                'responses': [],
                'difficulties': [],
                'original_weak_subtopics': [],
                'new_weak_subtopics': [],
                'correctly_answered_subtopics': [],
                'original_weak_topics': [],
                'new_weak_topics': [],
                'correctly_answered_topics': [],
                'topic_distribution': {},
                'wrong_answered': [],
                'weak_subtopics': []
            })
            return redirect(url_for('select_topic'))

        elif action == 'login':
            # Login logic
            if not os.path.exists(USER_CSV):
                flash('No users registered yet!', 'error')
                return redirect(url_for('home'))

            df = pd.read_csv(USER_CSV)
            hashed_pw = hashlib.sha256(password.encode()).hexdigest()
            user = df[(df['test_taker'] == test_taker) & 
                    (df['password'] == hashed_pw)]

            if not user.empty:
                # Initialize session with existing test data
                session['test_taker'] = test_taker
                # Add code here to load previous session data if needed
                return redirect(url_for('select_topic'))

            flash('Invalid credentials!', 'error')

        return redirect(url_for('home'))

    return render_template('home.html')


@app.route('/logout')
def logout():
    session.pop('test_taker', None)
    return redirect(url_for('home'))


# Add new route for topic selection
@app.route('/select_topic', methods=['GET', 'POST'])
def select_topic():
    if 'test_taker' not in session:  # Add this check
        return redirect(url_for('home'))
    # Rest of the code
    
    if request.method == 'POST':
        # Normalize topic format for consistency
        raw_topic = request.form.get('topic')
        normalized_topic = raw_topic.lower().replace(" ", "-")
        
        session['selected_topic'] = normalized_topic
        session['selected_subtopic'] = request.form.get('subtopic', '').lower().replace(" ", "-")
        return redirect(url_for('test'))
    
    # Rest of the GET handling remains the same
    dataset = load_dataset()
    if dataset is None or dataset.empty:
        return redirect(url_for('home'))
    
    topics = dataset['Topic'].unique().tolist()
    subtopic_map = dataset.groupby('Topic')['Subtopic'].unique().apply(list).to_dict()
    
    return render_template('select_topic.html', 
                         topics=topics,
                         subtopic_map=subtopic_map)


@app.route('/dashboard', methods=['GET', 'POST'])
def dashboard():
    if 'test_taker' not in session:
        return redirect(url_for('home'))
    
    test_taker = session['test_taker']
    dataset = load_dataset()
    
    if dataset is None:
        return redirect(url_for('home'))
    
    # Normalize topics for dropdown and processing
    dataset['Norm_Topic'] = dataset['Topic'].str.lower().str.replace(' ', '-')
    topics = dataset['Topic'].unique().tolist()  # Display names with spaces
    
    selected_topic = None
    charts_data = {
        'ability': {'labels': [], 'data': []},
        'subtopics': {'labels': [], 'data': []}
    }

    if request.method == 'POST':
        # Get display topic and convert to normalized format
        display_topic = request.form.get('topic', '')
        selected_topic = display_topic.lower().replace(' ', '-')
        session['selected_topic'] = selected_topic
    else:
        selected_topic = session.get('selected_topic', '')

    try:
        df = pd.read_csv("test_profiles.csv", header=0)
        user_profiles = df[df["Test Taker"].str.startswith(test_taker)]
        
        if selected_topic:
            # Match profiles using normalized topic format
            pattern = fr"{re.escape(test_taker)}-\d+-{re.escape(selected_topic)}"
            relevant_profiles = user_profiles[
                user_profiles["Test Taker"].str.contains(pattern, regex=True)
            ].copy()
            
            # Extract profile numbers correctly
            relevant_profiles['Profile Number'] = relevant_profiles["Test Taker"].str.extract(r'-(\d+)-').astype(int)
            relevant_profiles = relevant_profiles.sort_values('Profile Number')
            
            # Prepare chart data
            ability_labels = relevant_profiles['Profile Number'].astype(str).tolist()
            ability_data = relevant_profiles['Ability Estimate'].tolist()
            
            # Get valid subtopics from dataset
            valid_subtopics = dataset[dataset['Norm_Topic'] == selected_topic]['Subtopic'].str.lower().str.replace(' ', '-').unique()
            
            # Collect weak subtopics
            weak_subtopics = []
            for _, row in relevant_profiles.iterrows():
                if pd.notna(row['Weak Sub Topic']):
                    subs = [s.strip().lower().replace(' ', '-') 
                           for s in str(row['Weak Sub Topic']).split(',')]
                    weak_subtopics.extend([s for s in subs if s in valid_subtopics])
            
            # Prepare subtopic chart data
            subtopic_counts = pd.Series(weak_subtopics).value_counts()
            
            charts_data = {
                'ability': {
                    'labels': ability_labels,
                    'data': ability_data
                },
                'subtopics': {
                    'labels': subtopic_counts.index.tolist(),
                    'data': subtopic_counts.values.tolist()
                }
            }
    
    except Exception as e:
        print(f"Dashboard error: {str(e)}")
    
    return render_template('dashboard.html',
                          topics=topics,
                          selected_topic=display_topic if request.method == 'POST' else selected_topic.replace('-', ' '),
                          charts=charts_data)

def get_previous_theta(test_taker, topic):
    try:
        df = pd.read_csv("test_profiles.csv", names=[
            "Test Taker", "Ability Estimate", "Wrong Answered Questions",
            "Weak Sub Topic", "Weak Topic", "Previous Recommendations"
        ])
        topic = topic.lower().replace(" ", "-")
        pattern = fr"^{re.escape(test_taker)}-(\d+)-{re.escape(topic)}(?:-|$)"
        df['Profile Number'] = df["Test Taker"].str.extract(pattern).astype(float)
        relevant = df.dropna(subset=['Profile Number']).sort_values("Profile Number", ascending=False)
        if not relevant.empty:
            return float(relevant.iloc[0]["Ability Estimate"])
    except Exception as e:
        print(f"Error fetching previous theta: {e}")
    return 0.0



@app.route('/new_test')
def new_test():
    # Preserve current topic and subtopic before clearing session
    temp_topic = session.get('selected_topic', 'general')
    temp_subtopic = session.get('selected_subtopic', '')

    # Clear all session except test taker
    session_keys = [key for key in session.keys() if key not in ['test_taker']]
    for key in session_keys:
        session.pop(key, None)

    # Load previous theta estimate for continuity
    theta_start = get_previous_theta(session['test_taker'], temp_topic)

    # Reinitialize all session state
    session.update({
        'theta': theta_start,
        'responses': [],
        'difficulties': [],
        'asked_questions': [],
        'original_weak_subtopics': [],
        'new_weak_subtopics': [],
        'correctly_answered_subtopics': [],
        'original_weak_topics': [],
        'new_weak_topics': [],
        'correctly_answered_topics': [],
        'topic_distribution': {},
        'wrong_answered': [],
        'weak_subtopics': [],
        'selected_topic': temp_topic,
        'selected_subtopic': temp_subtopic
    })

    # Load historical weak subtopics and topics
    test_taker = session['test_taker']
    past_weak_subtopics = load_past_weak_subtopics(test_taker, temp_topic)
    past_weak_topics = load_past_weak_topics(test_taker, temp_topic)

    # Normalize and store
    session['original_weak_subtopics'] = [s.lower().strip() for s in set(past_weak_subtopics)]
    session['original_weak_topics'] = [s.lower().strip() for s in set(past_weak_topics)]

    return redirect(url_for('select_topic'))

@app.route('/test', methods=['GET', 'POST'])
def test():
    session.modified = True
    # Add debug prints
    print(f"Current weak subtopics: {session.get('original_weak_subtopics', [])}")
    print(f"Current weak topics: {session.get('original_weak_topics', [])}")

    dataset = load_dataset()
    dataset['Subtopic'] = dataset['Subtopic'].str.lower().str.strip()

    if dataset is None or dataset.empty:
        return redirect(url_for('home'))
        
    # Apply topic filter with normalization
    selected_topic = session.get('selected_topic')
    if selected_topic:
        # Normalize dataset topics for comparison
        dataset['Norm_Topic'] = dataset['Topic'].str.lower().str.replace(' ', '-')
        dataset = dataset[dataset['Norm_Topic'] == selected_topic]

        selected_subtopic = session.get('selected_subtopic')
        weak_subs = session.get('original_weak_subtopics', [])

        # Only restrict dataset to a manually selected subtopic if adaptive mode isn't active
        if selected_subtopic and (not weak_subs or len(weak_subs) == 0):
            dataset['Norm_Subtopic'] = dataset['Subtopic'].str.lower().str.replace(' ', '-')
            dataset = dataset[dataset['Norm_Subtopic'] == selected_subtopic]

    
    # Initialize topic distribution on first question
    if len(session.get('asked_questions', [])) == 0 and 'original_weak_subtopics' in session and session['original_weak_subtopics']:
        available_counts = {}
        for sub in session['original_weak_subtopics']:
            count = len(dataset[dataset["Subtopic"] == sub])
            available_counts[sub] = count
        
        total_questions = 10
        num_subs = len(session['original_weak_subtopics'])
        base = total_questions // num_subs
        remainder = total_questions % num_subs
        
        session['subtopic_distribution'] = {}
        for i, sub in enumerate(session['original_weak_subtopics']):
            allocation = base + (1 if i < remainder else 0)
            session['subtopic_distribution'][sub] = min(allocation, available_counts[sub])
        
        total_allocated = sum(session['subtopic_distribution'].values())
        if total_allocated < 10:
            remaining = 10 - total_allocated
            subs = [s for s in session['original_weak_subtopics'] if available_counts.get(s, 0) > 0]
            while remaining > 0 and len(subs) > 0:
                for subs in subs:
                    if remaining <= 0:
                        break
                    session['subtopic_distribution'][sub] += 1
                    remaining -= 1

    if 'wrong_answered' not in session:
        session['wrong_answered'] = []

    if 'weak_subtopics' not in session:
        session['weak_subtopics'] = []


    if request.method == 'POST':
        if 'current_question' not in session:
            return redirect(url_for('home'))
        
        selected_answer = request.form['answer']
        question = session['current_question']
        
        # Check correctness
        correct_option = next(
            (option for option in ["A", "B", "C", "D"] 
             if str(question.get(f"Option {option}", "")).strip().upper() == str(question.get("Answer", "")).strip().upper()),
            None
        )
        is_correct = selected_answer == correct_option
        reward = 1 if is_correct else -1
        
        # Track responses
        session['responses'].append(1 if is_correct else 0)
        session['difficulties'].append(question["Difficulty (b)"])
        update_q_table(session['theta'], question["Question ID"], reward, session['theta'] + (0.1 if is_correct else -0.1))

        if not is_correct:
            # Ensure proper option formatting
            options = [
                f"{opt}: {question.get(f'Option {opt}', '').strip()}"
                for opt in ["A", "B", "C", "D"]
            ]
            # Clean question text
            question_text = question.get('Question', '').replace('\n', ' ').strip()
            session['wrong_answered'].append((
                question['Question ID'],
                question_text,
                options,
                selected_answer
            ))
            
            # Track subtopics
            # Track both topic and subtopic
            current_topic = question.get('Topic')
            current_subtopic = question.get('Subtopic')

            # Validate topic-subtopic mapping
            current_topic = session.get('selected_topic', '').replace('-', ' ')
            valid_subtopics = dataset[dataset['Topic'].str.lower() == current_topic.lower()]['Subtopic'].unique()


            if current_topic:
                if current_topic not in session['new_weak_topics']:
                    session['new_weak_topics'].append(current_topic)

            if current_subtopic:
                if current_subtopic not in session['weak_subtopics']:
                    session['weak_subtopics'].append(current_subtopic)
                if current_subtopic not in session['new_weak_subtopics']:
                    session['new_weak_subtopics'].append(current_subtopic)
        else:
            current_subtopic = question.get('Subtopic')
            current_topic = question.get('Topic')
            if current_topic and current_topic not in session['correctly_answered_topics']:
                session['correctly_answered_topics'].append(current_topic)
            if current_subtopic and current_subtopic not in session['correctly_answered_subtopics']:
                session['correctly_answered_subtopics'].append(current_subtopic)

        # Update topic tracking
        if is_correct:
            if question['Subtopic'] in session['original_weak_subtopics'] and question['Subtopic'] not in session['correctly_answered_subtopics']:
                session['correctly_answered_subtopics'].append(question['Subtopic'])
        else:
            if question['Subtopic'] not in session['new_weak_subtopics']:
                session['new_weak_subtopics'].append(question['Subtopic'])
        
        # Update theta and track questions
        session['theta'] = update_theta(session['responses'], session['difficulties'], session['theta'])
        session['asked_questions'].append(question['Question ID'])
        
        if len(session['asked_questions']) >= 10:
            return redirect(url_for('profile'))
    
    # Select next question
    try:
        session['original_weak_subtopics'] = [
            sub.lower().strip() for sub in session.get('original_weak_subtopics', [])]

        question = select_question(
            session['theta'], dataset,
            session['original_weak_subtopics'],
            session['asked_questions'],
            session.get('subtopic_distribution')
        )
    
        # If no questions found in weak areas, fall back to general questions
        if question is None or question.empty:
            dataset = dataset[~dataset["Question ID"].isin(session['asked_questions'])]
            question = select_question(session['theta'], dataset,session['original_weak_subtopics'],session['asked_questions'],session.get('subtopic_distribution'))
    
        if question.empty:
            # If truly no questions left, redirect to profile
            return redirect(url_for('profile'))
    
        session['current_question'] = question.to_dict()
        return render_template('test.html', 
                            question=session['current_question'], 
                            question_number=len(session['asked_questions']) + 1)

    except Exception as e:
        print(f"Critical error: {str(e)}")
        return redirect(url_for('profile'))


@app.route('/baseline_test')
def baseline_test():
    session_keys = ['baseline_question_ids', 'baseline_answers', 'current_question_index']
    for key in session_keys:
        session.pop(key, None)
    
    dataset = load_dataset()
    if dataset is None:
        flash('Question bank unavailable')
        return redirect(url_for('select_topic'))
    
    # Select 30 random questions
    if len(dataset) < 10:
        flash('Not enough questions in the database')
        return redirect(url_for('select_topic'))
    
    random_questions = dataset.sample(n=10)
    baseline_question_ids = random_questions['Question ID'].tolist()
    random.shuffle(baseline_question_ids)
    
    session['baseline_question_ids'] = baseline_question_ids
    session['baseline_answers'] = []
    session['current_question_index'] = 0
    
    return redirect(url_for('baseline_question'))

@app.route('/baseline_question', methods=['GET', 'POST'])
def baseline_question():
    if 'baseline_question_ids' not in session:
        return redirect(url_for('select_topic'))
    
    dataset = load_dataset()
    question_id = session['baseline_question_ids'][session['current_question_index']]
    question = dataset[dataset['Question ID'] == question_id].iloc[0].to_dict()

    if request.method == 'POST':
        selected_answer = request.form['answer']
        
        # Store only essential answer data
        session['baseline_answers'].append({
            'question_id': question_id,
            'selected': selected_answer,
            'correct': selected_answer == question['Answer'].strip().upper()
        })
        
        session['current_question_index'] += 1
        
        if session['current_question_index'] >= len(session['baseline_question_ids']):
            return redirect(url_for('baseline_results'))
    
    return render_template('baseline_test.html', 
                         question=question,
                         question_number=session['current_question_index'] + 1)

@app.route('/baseline_results')
def baseline_results():
    if 'baseline_answers' not in session:
        flash('Test session expired or invalid', 'error')
        return redirect(url_for('select_topic'))
    
    try:
        dataset = load_dataset()
        if dataset is None or dataset.empty:
            flash('Question data unavailable', 'error')
            return redirect(url_for('select_topic'))

        # Calculate score with division guard
        total_questions = len(session['baseline_answers'])
        if total_questions == 0:
            flash('No test answers recorded', 'error')
            return redirect(url_for('select_topic'))

        correct_answers = sum(1 for ans in session['baseline_answers'] if ans.get('correct', False))
        scaled_score = round((correct_answers / total_questions) * 3 - 3, 2) if total_questions > 0 else -3.0

        # Collect weak areas with validation
        wrong_subtopics = set()
        wrong_topics = set()
        
        for ans in session['baseline_answers']:
            if not ans.get('correct', True):
                try:
                    question_row = dataset[dataset['Question ID'] == ans.get('question_id')]
                    if not question_row.empty:
                        question = question_row.iloc[0]
                        wrong_subtopics.add(question.get('Subtopic', 'Unknown Subtopic'))
                        wrong_topics.add(question.get('Topic', 'Unknown Topic'))
                except Exception as e:
                    print(f"Error processing question {ans.get('question_id')}: {str(e)}")
                    continue

        # Save results with fallback values
        test_taker = f"{session['test_taker']}-baseline"
        save_baseline_results(
            test_taker,
            scaled_score,
            list(wrong_topics) or ['General Knowledge Gap'],
            list(wrong_subtopics) or ['General Subtopic Gap'],
            session['baseline_answers']
        )

        # Generate recommendations with empty checks
        prev_recs = get_previous_recommendations(test_taker)
        recommendations = generate_recommendations(
            test_taker,
            scaled_score,
            ", ".join(wrong_topics) if wrong_topics else "General Knowledge Gaps",
            ", ".join(wrong_subtopics) if wrong_subtopics else "General Subtopic Gaps",
            prev_recs
        )

        return render_template('baseline_profile.html',
                            name=session['test_taker'],
                            theta=scaled_score,
                            weak_topics=wrong_topics,
                            recommendations=recommendations.split('\n') if recommendations else [])

    except Exception as e:
        print(f"Critical error in baseline results: {str(e)}")
        flash('An error occurred processing your test results', 'error')
        return redirect(url_for('select_topic'))

@app.route('/profile')
def profile():
    # Calculate final weak topics and subtopics separately
    original_weak_topics = set(session.get('original_weak_topics', []))
    original_weak_subtopics = set(session.get('original_weak_subtopics', []))
    
    # Calculate final weak topics
    correct_answered_topics = set(session.get('correctly_answered_topics', []))
    new_weak_topics = set(session.get('new_weak_topics', []))
    final_weak_topics = list((original_weak_topics - correct_answered_topics).union(new_weak_topics))
    
    # Calculate final weak subtopics
    correct_answered_subtopics = set(session.get('correctly_answered_subtopics', []))
    new_weak_subtopics = set(session.get('new_weak_subtopics', []))
    final_weak_subtopics = list((original_weak_subtopics - correct_answered_subtopics).union(new_weak_subtopics))

    # Save results with proper separation
    save_test_results(
        session['test_taker'],
        session['theta'],
        final_weak_topics,  # Now passing topics here
        final_weak_subtopics,  # And subtopics here
        session.get('wrong_answered', [])
    )
    
    # Get latest recommendations with proper column names
    column_names = [
        "Test Taker", 
        "Ability Estimate", 
        "Wrong Answered Questions",
        "Weak Sub Topic", 
        "Weak Topic", 
        "Previous Recommendations"
    ]
    
    try:
        df = pd.read_csv("test_profiles.csv", names=column_names)
        latest = df[df["Test Taker"].str.startswith(session['test_taker'])].iloc[-1]
        recommendations = latest["Previous Recommendations"].split('\n') if pd.notna(latest["Previous Recommendations"]) else []
    except Exception as e:
        print(f"Error loading profile: {str(e)}")
        recommendations = []

    return render_template('profile.html', 
        name=session['test_taker'], 
        theta=round(session['theta'], 2), 
        weak_topics=final_weak_topics,
        recommendations=recommendations
    )

if __name__ == "__main__":
    app.run(debug=True)
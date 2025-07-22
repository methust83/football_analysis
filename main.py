
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import warnings
import io

warnings.filterwarnings('ignore')

# Set page configuration - this is like setting up the overall look and feel of your application
st.set_page_config(
    page_title="⚽ Football Prediction System",
    page_icon="⚽",
    layout="wide",  # Use full width of the browser
    initial_sidebar_state="expanded"  # Start with sidebar open
)


class StreamlitFootballPredictor:
    """
    Interactive Football Match Prediction System using Streamlit

    This class wraps our prediction logic with Streamlit's interactive components,
    creating a user-friendly web interface for football match analysis and betting predictions.
    """

    def __init__(self):
        # Initialize session state variables - these persist across user interactions
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'models_trained' not in st.session_state:
            st.session_state.models_trained = False
        if 'predictor_data' not in st.session_state:
            st.session_state.predictor_data = {}

    def load_data_interface(self):
        """
        Create the data loading interface with file upload and data preview
        This is like creating a reception area where users first enter your application
        """
        st.header("📁 Data Loading & Exploration")

        st.markdown("""
        Welcome to the Football Match Prediction System! This application uses machine learning 
        to analyze historical match data and predict future outcomes for betting analysis.

        **To get started:** Upload your CSV files containing historical match data. The system 
        expects files with columns for teams, scores, shots, fouls, and other match statistics.
        """)

        # File upload component - allows multiple files for different seasons
        uploaded_files = st.file_uploader(
            "Upload CSV files with match data",
            type=['csv'],
            accept_multiple_files=True,
            help="You can upload multiple season files. Each file should contain match statistics."
        )

        if uploaded_files:
            return self.process_uploaded_data(uploaded_files)

        # Show sample data format if no files uploaded
        if not st.session_state.data_loaded:
            st.info("👆 Upload your match data CSV files to begin analysis")
            self.show_expected_format()

        return None

    def process_uploaded_data(self, uploaded_files):
        """
        Process uploaded CSV files and combine them into a unified dataset
        This function handles the technical details of data loading while providing user feedback
        """
        try:
            with st.spinner("Loading and processing your match data..."):
                all_data = []

                # Process each uploaded file
                for i, uploaded_file in enumerate(uploaded_files):
                    # Read CSV with error handling for different formats
                    df = pd.read_csv(uploaded_file)
                    df['Season'] = f'Season_{i + 1}'
                    all_data.append(df)

                    # Show progress to user
                    st.success(f"✓ Loaded {len(df)} matches from {uploaded_file.name}")

                # Combine all seasons into single dataset
                combined_df = pd.concat(all_data, ignore_index=True)

                # Handle date parsing with multiple format attempts
                combined_df = self.parse_dates_flexible(combined_df)

                # Store in session state for persistence across page interactions
                st.session_state.df = combined_df
                st.session_state.data_loaded = True
                st.session_state.predictor_data['raw_data'] = combined_df

                # Display data summary
                self.show_data_summary(combined_df)

                return combined_df

        except Exception as e:
            st.error(f"Error processing files: {str(e)}")
            st.info("Please check that your CSV files have the expected format and columns.")
            return None

    def parse_dates_flexible(self, df):
        """
        Flexibly parse date columns to handle different formats
        This demonstrates robust data handling - essential for real-world applications
        """
        try:
            # Try multiple date formats, starting with most specific
            df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y')
        except ValueError:
            try:
                df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y')
            except ValueError:
                # Let pandas infer format as last resort
                df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, infer_datetime_format=True)

        return df.sort_values('Date').reset_index(drop=True)

    def show_expected_format(self):
        """
        Display expected data format to help users prepare their files correctly
        This is like providing clear instructions before users attempt a task
        """
        st.subheader("Expected Data Format")

        sample_data = {
            'Date': ['11/08/23', '12/08/23', '13/08/23'],
            'HomeTeam': ['Arsenal', 'Chelsea', 'Liverpool'],
            'AwayTeam': ['Brighton', 'West Ham', 'Bournemouth'],
            'FTHG': [2, 1, 3],  # Full Time Home Goals
            'FTAG': [1, 0, 1],  # Full Time Away Goals
            'FTR': ['H', 'H', 'H'],  # Full Time Result (H/D/A)
            'HS': [15, 12, 18],  # Home Shots
            'AS': [8, 6, 10]  # Away Shots
        }

        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df)

        st.markdown("""
        **Required Columns:**
        - `Date`: Match date in DD/MM/YY or DD/MM/YYYY format
        - `HomeTeam`, `AwayTeam`: Team names
        - `FTHG`, `FTAG`: Full-time goals for home and away teams
        - `FTR`: Full-time result (H=Home win, D=Draw, A=Away win)
        - `HS`, `AS`: Shots for home and away teams
        - Additional columns like `HST` (shots on target), `HF` (fouls), `HC` (corners) will enhance predictions
        """)

    def show_data_summary(self, df):
        """
        Display comprehensive summary of loaded data
        This gives users immediate feedback about their data quality and coverage
        """
        st.subheader("📊 Data Summary")

        # Create columns for organized layout
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Matches", len(df))

        with col2:
            date_range = f"{df['Date'].min().strftime('%b %Y')} - {df['Date'].max().strftime('%b %Y')}"
            st.metric("Date Range", date_range)

        with col3:
            unique_teams = len(set(df['HomeTeam'].tolist() + df['AwayTeam'].tolist()))
            st.metric("Unique Teams", unique_teams)

        with col4:
            avg_goals = (df['FTHG'] + df['FTAG']).mean()
            st.metric("Avg Goals/Match", f"{avg_goals:.1f}")

        # Show outcome distribution using Plotly for interactivity
        st.subheader("Match Outcomes Distribution")

        outcome_counts = df['FTR'].value_counts()
        outcome_labels = {'H': 'Home Wins', 'A': 'Away Wins', 'D': 'Draws'}

        fig = px.pie(
            values=outcome_counts.values,
            names=[outcome_labels.get(x, x) for x in outcome_counts.index],
            title="Distribution of Match Results"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Data quality check
        missing_data = df.isnull().sum()
        if missing_data.sum() > 0:
            st.warning("⚠️ Some missing data detected:")
            st.write(missing_data[missing_data > 0])
        else:
            st.success("✅ No missing data detected - excellent data quality!")

    def feature_engineering_interface(self):
        """
        Interactive interface for feature engineering and data preparation
        This is where raw match data transforms into predictive features
        """
        if not st.session_state.data_loaded:
            st.warning("Please load data first using the sidebar.")
            return

        st.header("🔧 Feature Engineering")

        st.markdown("""
        Feature engineering transforms raw match statistics into predictive features. 
        Think of this as a chef preparing ingredients - we take basic components (goals, shots, fouls) 
        and create sophisticated measures (team form, efficiency ratios, attacking strength) that 
        better predict match outcomes.
        """)

        # Allow user to configure feature engineering parameters
        col1, col2 = st.columns(2)

        with col1:
            rolling_window = st.slider(
                "Recent Form Window (matches)",
                min_value=3,
                max_value=10,
                value=5,
                help="Number of recent matches to consider for team form calculations"
            )

        with col2:
            include_referee = st.checkbox(
                "Include Referee Effects",
                value=True,
                help="Whether to include referee influence in predictions"
            )

        if st.button("🚀 Generate Features"):
            with st.spinner("Engineering features from your match data..."):
                df = st.session_state.df.copy()

                # Perform feature engineering
                enhanced_df = self.create_enhanced_features(df, rolling_window, include_referee)

                # Store enhanced data
                st.session_state.enhanced_df = enhanced_df
                st.session_state.features_ready = True

                st.success("✅ Feature engineering completed!")

                # Show feature summary
                self.show_feature_summary(enhanced_df)

    def create_enhanced_features(self, df, rolling_window, include_referee):
        """
        Create sophisticated features for machine learning models
        This function embodies the analytical expertise that transforms data into insights
        """
        # Calculate team form and statistics
        df = self.calculate_team_form(df, rolling_window)

        # Create derived statistical features
        df['TotalGoals'] = df['FTHG'] + df['FTAG']
        df['GoalDifference'] = df['FTHG'] - df['FTAG']

        # Shot efficiency - a key indicator of team quality
        df['HomeShotEfficiency'] = np.where(df['HS'] > 0, df['FTHG'] / df['HS'], 0)
        df['AwayShotEfficiency'] = np.where(df['AS'] > 0, df['FTAG'] / df['AS'], 0)

        # Shots on target ratio - indicates shot quality
        if 'HST' in df.columns and 'AST' in df.columns:
            df['HomeSOTRatio'] = np.where(df['HS'] > 0, df['HST'] / df['HS'], 0)
            df['AwaySOTRatio'] = np.where(df['AS'] > 0, df['AST'] / df['AS'], 0)
        else:
            # Create dummy columns if not available
            df['HomeSOTRatio'] = 0.4  # League average approximation
            df['AwaySOTRatio'] = 0.4

        # Disciplinary features if available
        if all(col in df.columns for col in ['HY', 'AY', 'HR', 'AR']):
            df['HomeCards'] = df['HY'] + (df['HR'] * 2)
            df['AwayCards'] = df['AY'] + (df['AR'] * 2)
        else:
            df['HomeCards'] = 2.0  # Average cards per match
            df['AwayCards'] = 2.0

        # Time-based features
        df['Month'] = df['Date'].dt.month
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        df['IsWeekend'] = (df['DayOfWeek'] >= 5).astype(int)

        # Encode categorical variables
        le_home = LabelEncoder()
        le_away = LabelEncoder()

        df['HomeTeam_encoded'] = le_home.fit_transform(df['HomeTeam'])
        df['AwayTeam_encoded'] = le_away.fit_transform(df['AwayTeam'])

        if include_referee and 'Referee' in df.columns:
            le_ref = LabelEncoder()
            df['Referee_encoded'] = le_ref.fit_transform(df['Referee'].fillna('Unknown'))
        else:
            df['Referee_encoded'] = 0

        # Store encoders for later use in predictions
        st.session_state.label_encoders = {
            'home': le_home,
            'away': le_away,
            'referee': le_ref if include_referee else None
        }

        return df

    def calculate_team_form(self, df, window):
        """
        Calculate rolling team form metrics
        This captures the dynamic nature of team performance over time
        """
        # Initialize form tracking lists
        home_form_points = []
        away_form_points = []
        home_goals_avg = []
        away_goals_avg = []
        home_performance = []
        away_performance = []

        # Calculate form for each match
        for idx, row in df.iterrows():
            home_team = row['HomeTeam']
            away_team = row['AwayTeam']

            # Get recent form for both teams
            home_recent = self.get_recent_form(df, home_team, idx, window)
            away_recent = self.get_recent_form(df, away_team, idx, window)

            home_form_points.append(home_recent['points'])
            away_form_points.append(away_recent['points'])
            home_goals_avg.append(home_recent['goals_avg'])
            away_goals_avg.append(away_recent['goals_avg'])
            home_performance.append(home_recent['performance_score'])
            away_performance.append(away_recent['performance_score'])

        # Add calculated features to dataframe
        df['HomeFormPoints'] = home_form_points
        df['AwayFormPoints'] = away_form_points
        df['HomeGoalsAvg'] = home_goals_avg
        df['AwayGoalsAvg'] = away_goals_avg
        df['HomePerformance'] = home_performance
        df['AwayPerformance'] = away_performance

        return df

    def get_recent_form(self, df, team, current_idx, window):
        """Calculate recent form metrics for a specific team"""
        # Get previous matches for this team
        prev_matches = df.iloc[:current_idx]
        team_matches = prev_matches[
            (prev_matches['HomeTeam'] == team) | (prev_matches['AwayTeam'] == team)
            ].tail(window)

        if len(team_matches) == 0:
            return {'points': 0, 'goals_avg': 1.0, 'performance_score': 50}

        points = 0
        goals_for = 0
        goals_against = 0

        for _, match in team_matches.iterrows():
            if match['HomeTeam'] == team:
                gf, ga = match['FTHG'], match['FTAG']
                if match['FTR'] == 'H':
                    points += 3
                elif match['FTR'] == 'D':
                    points += 1
            else:
                gf, ga = match['FTAG'], match['FTHG']
                if match['FTR'] == 'A':
                    points += 3
                elif match['FTR'] == 'D':
                    points += 1

            goals_for += gf
            goals_against += ga

        avg_goals = goals_for / len(team_matches) if len(team_matches) > 0 else 1.0

        # Performance score combining multiple factors
        avg_points = points / len(team_matches) if len(team_matches) > 0 else 0
        performance_score = (avg_points * 20) + (avg_goals * 10) - (goals_against / len(team_matches) * 5)
        performance_score = max(0, min(100, performance_score))

        return {
            'points': points,
            'goals_avg': avg_goals,
            'performance_score': performance_score
        }

    def show_feature_summary(self, df):
        """Display summary of engineered features"""
        st.subheader("Generated Features Summary")

        # Feature categories
        form_features = ['HomeFormPoints', 'AwayFormPoints', 'HomeGoalsAvg', 'AwayGoalsAvg', 'HomePerformance',
                         'AwayPerformance']
        efficiency_features = ['HomeShotEfficiency', 'AwayShotEfficiency', 'HomeSOTRatio', 'AwaySOTRatio']
        contextual_features = ['Month', 'DayOfWeek', 'IsWeekend', 'HomeCards', 'AwayCards']

        col1, col2, col3 = st.columns(3)

        with col1:
            st.write("**📈 Team Form Features:**")
            for feature in form_features:
                if feature in df.columns:
                    avg_val = df[feature].mean()
                    st.write(f"- {feature}: {avg_val:.2f}")

        with col2:
            st.write("**🎯 Efficiency Features:**")
            for feature in efficiency_features:
                if feature in df.columns:
                    avg_val = df[feature].mean()
                    st.write(f"- {feature}: {avg_val:.3f}")

        with col3:
            st.write("**🕒 Contextual Features:**")
            for feature in contextual_features:
                if feature in df.columns:
                    if feature in ['Month', 'DayOfWeek']:
                        st.write(f"- {feature}: Available")
                    else:
                        avg_val = df[feature].mean()
                        st.write(f"- {feature}: {avg_val:.2f}")

    def model_training_interface(self):
        """
        Interactive interface for training machine learning models
        This is where statistical analysis meets user-friendly interaction
        """
        if not hasattr(st.session_state, 'features_ready'):
            st.warning("Please complete feature engineering first.")
            return

        st.header("🤖 Model Training & Evaluation")

        st.markdown("""
        Now we train machine learning models to learn patterns from your historical match data. 
        Think of this as teaching three different experts to analyze football matches, 
        each with their own analytical approach and strengths.
        """)

        # Model configuration options
        col1, col2 = st.columns(2)

        with col1:
            test_size = st.slider(
                "Test Data Size (%)",
                min_value=10,
                max_value=40,
                value=20,
                help="Percentage of data reserved for testing model performance"
            )

        with col2:
            models_to_train = st.multiselect(
                "Select Models to Train",
                ['Random Forest', 'Gradient Boosting', 'Logistic Regression'],
                default=['Random Forest', 'Gradient Boosting', 'Logistic Regression'],
                help="Choose which machine learning algorithms to train and compare"
            )

        if st.button("🚂 Train Models"):
            if not models_to_train:
                st.error("Please select at least one model to train.")
                return

            with st.spinner("Training your prediction models..."):
                results = self.train_models(test_size / 100, models_to_train)

                if results:
                    st.session_state.model_results = results
                    st.session_state.models_trained = True
                    st.success("✅ Models trained successfully!")

                    # Display results
                    self.display_model_results(results)

    def train_models(self, test_size, selected_models):
        """Train selected machine learning models"""
        try:
            df = st.session_state.enhanced_df

            # Define feature set
            feature_columns = [
                'HomeFormPoints', 'AwayFormPoints', 'HomeGoalsAvg', 'AwayGoalsAvg',
                'HomePerformance', 'AwayPerformance', 'HS', 'AS',
                'HomeShotEfficiency', 'AwayShotEfficiency', 'HomeSOTRatio', 'AwaySOTRatio',
                'HomeCards', 'AwayCards', 'Month', 'DayOfWeek', 'IsWeekend',
                'HomeTeam_encoded', 'AwayTeam_encoded', 'Referee_encoded'
            ]

            # Filter to available features
            available_features = [f for f in feature_columns if f in df.columns]

            X = df[available_features].fillna(0)
            y = df['FTR']

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )

            # Scale features for logistic regression
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # Define available models
            model_dict = {
                'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
                'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
                'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
            }

            results = {}
            trained_models = {}

            # Train selected models
            for model_name in selected_models:
                if model_name in model_dict:
                    model = model_dict[model_name]

                    # Use scaled data for logistic regression, original for tree-based models
                    if model_name == 'Logistic Regression':
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5)

                    accuracy = accuracy_score(y_test, y_pred)

                    results[model_name] = {
                        'accuracy': accuracy,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std(),
                        'predictions': y_pred,
                        'actual': y_test.values
                    }

                    trained_models[model_name] = model

            # Store trained models and other necessary data
            st.session_state.trained_models = trained_models
            st.session_state.scaler = scaler
            st.session_state.feature_columns = available_features

            # Select best model
            if results:
                best_model_name = max(results.keys(), key=lambda x: results[x]['cv_mean'])
                st.session_state.best_model_name = best_model_name
                st.session_state.best_model = trained_models[best_model_name]

            return results

        except Exception as e:
            st.error(f"Error training models: {str(e)}")
            return None

    def display_model_results(self, results):
        """Display comprehensive model training results"""
        st.subheader("🎯 Model Performance Results")

        # Performance comparison chart
        model_names = list(results.keys())
        accuracies = [results[name]['cv_mean'] for name in model_names]

        fig = go.Figure(data=[
            go.Bar(x=model_names, y=accuracies,
                   text=[f'{acc:.3f}' for acc in accuracies],
                   textposition='auto')
        ])
        fig.update_layout(
            title="Model Accuracy Comparison (Cross-Validation)",
            yaxis_title="Accuracy",
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)

        # Detailed metrics table
        metrics_data = []
        for name, result in results.items():
            metrics_data.append({
                'Model': name,
                'Test Accuracy': f"{result['accuracy']:.3f}",
                'CV Mean': f"{result['cv_mean']:.3f}",
                'CV Std': f"{result['cv_std']:.3f}",
                'Status': '🥇 Best' if name == st.session_state.best_model_name else '✅ Trained'
            })

        metrics_df = pd.DataFrame(metrics_data)
        st.dataframe(metrics_df, use_container_width=True)

        st.success(f"🏆 Best performing model: **{st.session_state.best_model_name}**")

    def prediction_interface(self):
        """
        Interactive interface for making match predictions
        This is where all the analytical work pays off with practical predictions
        """
        if not st.session_state.models_trained:
            st.warning("Please train models first.")
            return

        st.header("🔮 Match Prediction")

        st.markdown("""
        Use your trained models to predict upcoming match outcomes and calculate betting odds. 
        Simply select the teams and let the AI analyze their current form, historical performance, 
        and statistical patterns to generate predictions.
        """)

        # Get available teams from the data
        df = st.session_state.enhanced_df
        teams = sorted(list(set(df['HomeTeam'].tolist() + df['AwayTeam'].tolist())))

        col1, col2, col3 = st.columns(3)

        with col1:
            home_team = st.selectbox("🏠 Home Team", teams)

        with col2:
            away_teams_available = [t for t in teams if t != home_team]
            away_team = st.selectbox("✈️ Away Team", away_teams_available)

        with col3:
            prediction_confidence = st.slider(
                "Minimum Confidence",
                min_value=0.3,
                max_value=0.9,
                value=0.6,
                help="Only show predictions above this confidence level"
            )

        if st.button("🎯 Predict Match"):
            with st.spinner("Analyzing teams and generating prediction..."):
                prediction_result = self.predict_match(home_team, away_team)

                if prediction_result:
                    self.display_prediction_results(prediction_result, home_team, away_team, prediction_confidence)

    def predict_match(self, home_team, away_team):
        """Generate prediction for a specific match"""
        try:
            # Get recent form for both teams
            df = st.session_state.enhanced_df

            home_recent = self.get_team_recent_stats(df, home_team, 5)
            away_recent = self.get_team_recent_stats(df, away_team, 5)

            # Create feature vector
            feature_data = {
                'HomeFormPoints': home_recent['form_points'],
                'AwayFormPoints': away_recent['form_points'],
                'HomeGoalsAvg': home_recent['goals_avg'],
                'AwayGoalsAvg': away_recent['goals_avg'],
                'HomePerformance': home_recent['performance'],
                'AwayPerformance': away_recent['performance'],
                'HS': home_recent['avg_shots'],
                'AS': away_recent['avg_shots'],
                'HomeShotEfficiency': home_recent['shot_efficiency'],
                'AwayShotEfficiency': away_recent['shot_efficiency'],
                'HomeSOTRatio': home_recent['sot_ratio'],
                'AwaySOTRatio': away_recent['sot_ratio'],
                'HomeCards': home_recent['avg_cards'],
                'AwayCards': away_recent['avg_cards'],
                'Month': pd.Timestamp.now().month,
                'DayOfWeek': pd.Timestamp.now().dayofweek,
                'IsWeekend': 1 if pd.Timestamp.now().dayofweek >= 5 else 0,
                'HomeTeam_encoded': self.encode_team(home_team, 'home'),
                'AwayTeam_encoded': self.encode_team(away_team, 'away'),
                'Referee_encoded': 0  # Default referee encoding
            }

            # Create feature array
            feature_columns = st.session_state.feature_columns
            feature_array = np.array([[feature_data.get(f, 0) for f in feature_columns]])

            # Make prediction with best model
            best_model = st.session_state.best_model
            best_model_name = st.session_state.best_model_name

            if best_model_name == 'Logistic Regression':
                feature_array = st.session_state.scaler.transform(feature_array)

            prediction = best_model.predict(feature_array)[0]
            probabilities = best_model.predict_proba(feature_array)[0]

            # Map probabilities to outcomes
            classes = best_model.classes_
            prob_dict = dict(zip(classes, probabilities))

            # Calculate betting odds
            odds = self.calculate_betting_odds(prob_dict)

            return {
                'prediction': prediction,
                'probabilities': prob_dict,
                'odds': odds,
                'confidence': max(probabilities),
                'model_used': best_model_name,
                'team_stats': {'home': home_recent, 'away': away_recent}
            }

        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
            return None

    def get_team_recent_stats(self, df, team, window=5):
        """Get recent statistics for a team"""
        recent_matches = df[
            (df['HomeTeam'] == team) | (df['AwayTeam'] == team)
            ].tail(window)

        if len(recent_matches) == 0:
            return {
                'form_points': 0, 'goals_avg': 1.0, 'performance': 50,
                'avg_shots': 10, 'shot_efficiency': 0.1, 'sot_ratio': 0.4,
                'avg_cards': 2
            }

        points = 0
        goals_for = 0
        shots = 0
        cards = 0

        for _, match in recent_matches.iterrows():
            is_home = match['HomeTeam'] == team

            if is_home:
                gf = match['FTHG']
                shots += match['HS']
                cards += match.get('HomeCards', 2)

                if match['FTR'] == 'H':
                    points += 3
                elif match['FTR'] == 'D':
                    points += 1
            else:
                gf = match['FTAG']
                shots += match['AS']
                cards += match.get('AwayCards', 2)

                if match['FTR'] == 'A':
                    points += 3
                elif match['FTR'] == 'D':
                    points += 1

            goals_for += gf

        n_matches = len(recent_matches)

        return {
            'form_points': points,
            'goals_avg': goals_for / n_matches,
            'performance': (points / n_matches) * 33.33,
            'avg_shots': shots / n_matches,
            'shot_efficiency': (goals_for / shots) if shots > 0 else 0,
            'sot_ratio': 0.4,  # Approximation if not available
            'avg_cards': cards / n_matches
        }

    def encode_team(self, team_name, team_type):
        """Encode team name using stored label encoder"""
        try:
            encoder = st.session_state.label_encoders[team_type]
            if team_name in encoder.classes_:
                return encoder.transform([team_name])[0]
        except:
            pass
        return 0  # Default for unknown teams

    def calculate_betting_odds(self, probabilities):
        """Convert probabilities to betting odds with bookmaker margin"""
        odds = {}
        margin = 0.05  # 5% bookmaker margin

        for outcome, prob in probabilities.items():
            if prob > 0:
                fair_odds = 1 / prob
                bookmaker_odds = fair_odds * (1 + margin)
                odds[outcome] = round(bookmaker_odds, 2)
            else:
                odds[outcome] = 999.99

        return odds

    def display_prediction_results(self, result, home_team, away_team, min_confidence):
        """Display comprehensive prediction results"""
        st.subheader(f"🎯 Prediction: {home_team} vs {away_team}")

        confidence = result['confidence']

        # Show confidence warning if below threshold
        if confidence < min_confidence:
            st.warning(f"⚠️ Prediction confidence ({confidence:.3f}) is below your threshold ({min_confidence:.3f})")

        # Main prediction display
        outcome_names = {'H': 'Home Win', 'D': 'Draw', 'A': 'Away Win'}
        predicted_outcome = outcome_names.get(result['prediction'], result['prediction'])

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("🏆 Predicted Outcome", predicted_outcome)

        with col2:
            st.metric("🎯 Confidence", f"{confidence:.1%}")

        with col3:
            st.metric("🤖 Model Used", result['model_used'])

        # Probability breakdown
        st.subheader("📊 Outcome Probabilities")

        prob_data = []
        for outcome, prob in result['probabilities'].items():
            outcome_name = outcome_names.get(outcome, outcome)
            prob_data.append({'Outcome': outcome_name, 'Probability': prob})

        prob_df = pd.DataFrame(prob_data)

        # Create probability visualization
        fig = px.bar(prob_df, x='Outcome', y='Probability',
                     title="Predicted Outcome Probabilities")
        fig.update_layout(yaxis_title="Probability", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # Betting odds
        st.subheader("💰 Recommended Betting Odds")

        odds_data = []
        for outcome, odd in result['odds'].items():
            outcome_name = outcome_names.get(outcome, outcome)
            prob = result['probabilities'][outcome]

            # Calculate implied probability from odds
            implied_prob = 1 / odd if odd > 0 else 0
            value = prob - implied_prob  # Positive value indicates good bet

            odds_data.append({
                'Outcome': outcome_name,
                'Fair Odds': odd,
                'Probability': f"{prob:.1%}",
                'Value': f"{value:.3f}" if value > 0 else f"{value:.3f}"
            })

        odds_df = pd.DataFrame(odds_data)
        st.dataframe(odds_df, use_container_width=True)

        # Team comparison
        st.subheader("⚖️ Team Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.write(f"**🏠 {home_team} (Recent Form)**")
            home_stats = result['team_stats']['home']
            st.write(f"• Form Points: {home_stats['form_points']}")
            st.write(f"• Goals per Game: {home_stats['goals_avg']:.1f}")
            st.write(f"• Performance Score: {home_stats['performance']:.0f}/100")
            st.write(f"• Shot Efficiency: {home_stats['shot_efficiency']:.2f}")

        with col2:
            st.write(f"**✈️ {away_team} (Recent Form)**")
            away_stats = result['team_stats']['away']
            st.write(f"• Form Points: {away_stats['form_points']}")
            st.write(f"• Goals per Game: {away_stats['goals_avg']:.1f}")
            st.write(f"• Performance Score: {away_stats['performance']:.0f}/100")
            st.write(f"• Shot Efficiency: {away_stats['shot_efficiency']:.2f}")

    def betting_analysis_interface(self):
        """
        Interactive betting analysis dashboard
        This transforms predictions into actionable betting strategies
        """
        if not st.session_state.models_trained:
            st.warning("Please train models first.")
            return

        st.header("💰 Betting Analysis Dashboard")

        st.markdown("""
        Analyze betting opportunities by comparing model predictions with simulated bookmaker odds. 
        This helps identify value bets where the model's assessment differs significantly from market prices.
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            min_confidence = st.slider("Min Confidence", 0.4, 0.9, 0.6)

        with col2:
            min_odds = st.slider("Min Odds", 1.2, 5.0, 1.8)

        with col3:
            bet_amount = st.number_input("Bet Amount ($)", 5, 100, 10)

        if st.button("🔍 Analyze Betting Opportunities"):
            with st.spinner("Analyzing betting opportunities..."):
                opportunities = self.analyze_betting_opportunities(min_confidence, min_odds, bet_amount)
                self.display_betting_analysis(opportunities)

    def analyze_betting_opportunities(self, min_confidence, min_odds, bet_amount):
        """Analyze historical betting opportunities"""
        # This is a simplified simulation for demonstration
        # In practice, you would run this analysis on test data

        results = st.session_state.model_results
        opportunities = []
        total_profit = 0
        total_bets = 0

        # Simulate betting analysis on model results
        for model_name, result in results.items():
            if model_name == st.session_state.best_model_name:
                predictions = result['predictions']
                actual = result['actual']

                for pred, act in zip(predictions, actual):
                    # Simulate confidence and odds
                    confidence = 0.75 if pred == act else 0.45
                    simulated_odds = np.random.uniform(1.5, 3.0)

                    if confidence >= min_confidence and simulated_odds >= min_odds:
                        total_bets += 1

                        if pred == act:  # Winning bet
                            profit = bet_amount * (simulated_odds - 1)
                            total_profit += profit
                            result_status = 'WIN'
                        else:  # Losing bet
                            total_profit -= bet_amount
                            profit = -bet_amount
                            result_status = 'LOSS'

                        opportunities.append({
                            'prediction': pred,
                            'actual': act,
                            'confidence': confidence,
                            'odds': simulated_odds,
                            'bet_amount': bet_amount,
                            'profit': profit,
                            'result': result_status
                        })

        return {
            'opportunities': opportunities,
            'total_profit': total_profit,
            'total_bets': total_bets,
            'win_rate': len([o for o in opportunities if o['result'] == 'WIN']) / max(total_bets, 1),
            'roi': (total_profit / (total_bets * bet_amount)) * 100 if total_bets > 0 else 0
        }

    def display_betting_analysis(self, analysis):
        """Display comprehensive betting analysis results"""
        opportunities = analysis['opportunities']

        if not opportunities:
            st.info("No betting opportunities found with current criteria. Try adjusting the filters.")
            return

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Bets", analysis['total_bets'])

        with col2:
            st.metric("Win Rate", f"{analysis['win_rate']:.1%}")

        with col3:
            st.metric("Total P&L", f"${analysis['total_profit']:.2f}")

        with col4:
            st.metric("ROI", f"{analysis['roi']:.1f}%")

        # Profit/loss chart
        cumulative_profit = []
        running_total = 0

        for opp in opportunities:
            running_total += opp['profit']
            cumulative_profit.append(running_total)

        if cumulative_profit:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=cumulative_profit,
                mode='lines+markers',
                name='Cumulative Profit',
                line=dict(color='green' if analysis['total_profit'] > 0 else 'red')
            ))
            fig.update_layout(
                title="Cumulative Profit/Loss Over Time",
                yaxis_title="Profit ($)",
                xaxis_title="Bet Number"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Opportunities table
        if len(opportunities) > 0:
            st.subheader("📋 Betting Opportunities")

            opp_df = pd.DataFrame(opportunities)

            # Format for display
            display_df = opp_df.copy()
            display_df['confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
            display_df['odds'] = display_df['odds'].apply(lambda x: f"{x:.2f}")
            display_df['profit'] = display_df['profit'].apply(lambda x: f"${x:.2f}")

            st.dataframe(display_df.head(20), use_container_width=True)


def main():
    """
    Main application function that orchestrates the entire Streamlit interface
    This is the conductor of your analytical orchestra
    """

    # Initialize the application
    predictor = StreamlitFootballPredictor()

    # Create sidebar navigation
    st.sidebar.title("⚽ Navigation")

    # Navigation options
    pages = {
        "📁 Data Loading": "data",
        "🔧 Feature Engineering": "features",
        "🤖 Model Training": "training",
        "🔮 Predictions": "predictions",
        "💰 Betting Analysis": "betting"
    }

    selected_page = st.sidebar.radio("Choose Analysis Step:", list(pages.keys()))
    current_page = pages[selected_page]

    # Add some helpful information in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    **📋 Quick Guide:**
    1. Upload your match data CSV files
    2. Configure feature engineering  
    3. Train prediction models
    4. Generate match predictions
    5. Analyze betting opportunities
    """)

    # Main title
    st.title("⚽ Football Match Prediction System")
    st.markdown("*Advanced machine learning for football betting analysis*")

    # Route to appropriate page
    if current_page == "data":
        predictor.load_data_interface()

    elif current_page == "features":
        predictor.feature_engineering_interface()

    elif current_page == "training":
        predictor.model_training_interface()

    elif current_page == "predictions":
        predictor.prediction_interface()

    elif current_page == "betting":
        predictor.betting_analysis_interface()

    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("*Built with Streamlit & Scikit-learn*")


if __name__ == "__main__":
    main()
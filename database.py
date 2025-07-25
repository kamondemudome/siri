# database.py
import sqlite3
import bcrypt
import os
import pandas as pd
from datetime import datetime
from constants import FEATURES  # Import FEATURES to validate user_data

# Database file path
DB_PATH = "D:/Myproject/diabetes_app.db"

def init_db():
    """Initialize the SQLite database and create tables."""
    try:
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()

        # Create users table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                email TEXT UNIQUE NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Create prediction_history table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS prediction_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                HighBP INTEGER,
                HighChol INTEGER,
                BMI REAL,
                GenHlth INTEGER,
                Smoker INTEGER,
                PhysActivity INTEGER,
                Fruits INTEGER,
                Veggies INTEGER,
                Age INTEGER,
                Income INTEGER,
                Probability REAL,
                Prediction TEXT,
                Timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        # Create volunteer_applications table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS volunteer_applications (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER,
                name TEXT NOT NULL,
                email TEXT NOT NULL,
                message TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
            )
        """)

        conn.commit()
    except sqlite3.Error as e:
        print(f"Database initialization error: {str(e)}")
        raise
    finally:
        conn.close()

def register_user(username, password, email):
    """Register a new user with hashed password."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
        cursor.execute(
            "INSERT INTO users (username, password, email) VALUES (?, ?, ?)",
            (username, hashed_password, email)
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Username or email already exists
    except sqlite3.Error as e:
        print(f"Database error during registration: {str(e)}")
        return False
    finally:
        conn.close()

def authenticate_user(username, password=None):
    """Authenticate a user or retrieve user_id by username."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT id, password FROM users WHERE username = ?", (username,))
        user = cursor.fetchone()
        if not user:
            return None
        if password is None:
            return user[0]  # Return user_id if no password is provided
        if bcrypt.checkpw(password.encode('utf-8'), user[1]):
            return user[0]  # Return user_id if password matches
        return None
    except sqlite3.Error as e:
        print(f"Database error during authentication: {str(e)}")
        return None
    finally:
        conn.close()

def save_prediction(user_id, user_data, prob, prediction):
    """Save prediction data for a user."""
    try:
        # Validate user_data
        if not all(feature in user_data for feature in FEATURES):
            raise ValueError(f"Missing required features in user_data: {FEATURES}")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO prediction_history (
                user_id, HighBP, HighChol, BMI, GenHlth, Smoker, PhysActivity, Fruits, Veggies, Age, Income, Probability, Prediction
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            user_id,
            user_data['HighBP'], user_data['HighChol'], user_data['BMI'], user_data['GenHlth'],
            user_data['Smoker'], user_data['PhysActivity'], user_data['Fruits'], user_data['Veggies'],
            user_data['Age'], user_data['Income'], prob, prediction
        ))
        conn.commit()
    except (sqlite3.Error, ValueError) as e:
        print(f"Error saving prediction: {str(e)}")
        raise
    finally:
        conn.close()

def get_user_predictions(user_id):
    """Retrieve prediction history for a user."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM prediction_history WHERE user_id = ?", (user_id,))
        columns = [desc[0] for desc in cursor.description]
        df = pd.read_sql_query(
            "SELECT * FROM prediction_history WHERE user_id = ?",
            conn,
            params=(user_id,),
            parse_dates=['Timestamp']
        )
        return df
    except sqlite3.Error as e:
        print(f"Error retrieving predictions: {str(e)}")
        return pd.DataFrame()
    finally:
        conn.close()

def save_volunteer_application(user_id, name, email, message):
    """Save a volunteer application for a user."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO volunteer_applications (user_id, name, email, message, timestamp)
            VALUES (?, ?, ?, ?, ?)
            """,
            (user_id, name, email, message, datetime.now())
        )
        conn.commit()
    except sqlite3.Error as e:
        raise Exception(f"Database error: {str(e)}")
    finally:
        conn.close()

def debug_view_db():
    """Debug function to view database contents."""
    try:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users")
        print("Users:", cursor.fetchall())
        cursor.execute("SELECT * FROM prediction_history")
        print("Predictions:", cursor.fetchall())
        cursor.execute("SELECT * FROM volunteer_applications")
        print("Volunteer Applications:", cursor.fetchall())
    except sqlite3.Error as e:
        print(f"Debug error: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    init_db()
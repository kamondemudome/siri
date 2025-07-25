import sqlite3
from database import init_db, register_user, DB_PATH

# Initialize database
init_db()

# Function to register admin user
def create_admin_user():
    try:
        # Register admin user with username "admin" and password "Admin123!"
        username = "admin"
        password = "Admin123!"
        email = "admin@diabetesapp.com"  # Placeholder email
        success = register_user(username, password, email)
        if success:
            print(f"Admin user '{username}' created successfully.")
        else:
            print(f"Failed to create admin user. Username '{username}' or email may already exist.")
    except sqlite3.Error as e:
        print(f"Error creating admin user: {str(e)}")

if __name__ == "__main__":
    create_admin_user()
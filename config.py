import os

# Database configuration
basedir = os.path.abspath(os.path.dirname(__file__))
SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', f'sqlite:///{os.path.join(basedir, "app.db")}')
SQLALCHEMY_TRACK_MODIFICATIONS = False

# Secret key
SECRET_KEY = os.getenv('SECRET_KEY', 'your-secret-key-here')

# Flask-Mail configuration
MAIL_SERVER = os.getenv('MAIL_SERVER', 'smtp.example.com')
MAIL_PORT = int(os.getenv('MAIL_PORT', 587))
MAIL_USE_TLS = os.getenv('MAIL_USE_TLS', 'true').lower() in ['true', 'on', '1']
MAIL_USERNAME = os.getenv('MAIL_USERNAME')
MAIL_PASSWORD = os.getenv('MAIL_PASSWORD')
MAIL_DEFAULT_SENDER = os.getenv('MAIL_DEFAULT_SENDER', 'noreply@example.com')

# NLP configuration
SKIP_NLP_INIT = os.getenv('SKIP_NLP_INIT', 'false').lower() in ['true', 'on', '1']
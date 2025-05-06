from flask import Flask, render_template, request, redirect, url_for, flash, abort
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from models.database import db, init_db
from models.user import User
from models.complaint import Complaint
import os
from datetime import datetime
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pandas as pd
import string
import nltk
import smtplib
from email.mime.text import MIMEText
from twilio.rest import Client

# Import training data from separate file
from training_data import get_training_data, get_department_info

app = Flask(__name__)
app.config.from_pyfile('config.py')

# Initialize database
init_db(app)

# Flask-Login setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

class ComplaintClassifier:
    def __init__(self):
        if app.config.get('SKIP_NLP_INIT', False):
            self.vectorizer = None
            self.classifier = None
            return
            
        try:
            # Download NLTK resources
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            
            # Initialize ML components
            self.vectorizer = TfidfVectorizer(
                max_features=10000,
                ngram_range=(1, 2),
                stop_words='english'
            )
            self.classifier = LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                solver='saga',
                penalty='elasticnet',
                l1_ratio=0.5
            )
            self.lemmatizer = WordNetLemmatizer()
            self.stop_words = set(stopwords.words('english'))
            
            # Load training data
            self.train_data = get_training_data()
            self.train_model()
        except Exception as e:
            app.logger.error(f"Error initializing NLP: {e}")
            self.vectorizer = None
            self.classifier = None
    
    def preprocess_text(self, text):
        """Clean and preprocess text for classification"""
        # Convert to lowercase
        text = text.lower()
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Tokenize
        tokens = word_tokenize(text)
        # Remove stopwords and lemmatize
        tokens = [self.lemmatizer.lemmatize(token) 
                 for token in tokens 
                 if token not in self.stop_words and len(token) > 2]
        return ' '.join(tokens)
    
    def train_model(self):
        """Train the classification model"""
        if not hasattr(self, 'train_data') or self.train_data.empty:
            app.logger.error("No training data available")
            return
            
        try:
            # Preprocess all training texts
            self.train_data['processed_text'] = self.train_data['text'].apply(self.preprocess_text)
            
            # Vectorize text
            X = self.vectorizer.fit_transform(self.train_data['processed_text'])
            y = self.train_data['category']
            
            # Train model
            self.classifier.fit(X, y)
            
            # Log training accuracy
            train_acc = self.classifier.score(X, y)
            app.logger.info(f"Model trained successfully. Training accuracy: {train_acc:.2f}")
            
        except Exception as e:
            app.logger.error(f"Training failed: {e}")
            raise
    
    def predict_category(self, text):
        """Predict the category for a new complaint"""
        if not self.vectorizer or not self.classifier:
            app.logger.error("Classifier not initialized properly")
            return 'General Complaint'
            
        try:
            processed_text = self.preprocess_text(text)
            if not processed_text.strip():
                return 'General Complaint'
                
            # Vectorize and predict
            X = self.vectorizer.transform([processed_text])
            
            # Get prediction with confidence
            probs = self.classifier.predict_proba(X)[0]
            max_prob = max(probs)
            predicted_idx = probs.argmax()
            predicted_category = self.classifier.classes_[predicted_idx]
            
            # Only accept confident predictions
            if max_prob < 0.6:
                app.logger.warning(f"Low confidence prediction: {predicted_category} ({max_prob:.2f})")
                return 'General Complaint'
                
            app.logger.info(f"Predicted category: {predicted_category} (confidence: {max_prob:.2f})")
            return predicted_category
            
        except Exception as e:
            app.logger.error(f"Prediction failed: {e}")
            return 'General Complaint'

# Initialize classifier
classifier = ComplaintClassifier()

def send_department_notification(complaint):
    """Send notifications to the concerned department"""
    # Email notification
    msg = MIMEText(f"""
    New Complaint Received:
    ID: {complaint.id}
    Category: {complaint.category}
    Title: {complaint.title}
    Description: {complaint.description[:200]}...
    
    Please login to the system to take action.
    """)
    
    msg['Subject'] = f'[Action Required] New {complaint.category} Complaint (ID:{complaint.id})'
    msg['From'] = app.config['MAIL_DEFAULT_SENDER']
    msg['To'] = complaint.department_email
    
    try:
        with smtplib.SMTP(app.config['SMTP_SERVER'], app.config['SMTP_PORT']) as server:
            server.starttls()
            server.login(app.config['SMTP_USERNAME'], app.config['SMTP_PASSWORD'])
            server.send_message(msg)
            app.logger.info(f"Email sent to {complaint.department_email}")
    except Exception as e:
        app.logger.error(f"Failed to send email: {e}")
    
    # SMS notification
    if app.config.get('TWILIO_ENABLED', False):
        try:
            client = Client(app.config['TWILIO_SID'], app.config['TWILIO_TOKEN'])
            message = client.messages.create(
                body=f"New {complaint.category} complaint (ID:{complaint.id}) assigned to your department",
                from_=app.config['TWILIO_NUMBER'],
                to=complaint.department_phone
            )
            app.logger.info(f"SMS sent to {complaint.department_phone}")
        except Exception as e:
            app.logger.error(f"Failed to send SMS: {e}")

# ======================
# Authentication Routes
# ======================

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists', 'error')
            return redirect(url_for('register'))
        
        if User.query.filter_by(email=email).first():
            flash('Email already exists', 'error')
            return redirect(url_for('register'))
        
        new_user = User(
            username=username,
            email=email,
            password_hash=generate_password_hash(password)
        )
        
        db.session.add(new_user)
        db.session.commit()
        
        flash('Registration successful. Please login.', 'success')
        return redirect(url_for('login'))
    
    return render_template('auth/register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        remember = True if request.form.get('remember') else False
        
        user = User.query.filter_by(username=username).first()
        
        if not user or not check_password_hash(user.password_hash, password):
            flash('Invalid username or password', 'error')
            return redirect(url_for('login'))
        
        login_user(user, remember=remember)
        
        if user.is_admin:
            return redirect(url_for('admin_dashboard'))
        return redirect(url_for('dashboard'))
    
    return render_template('auth/login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

# ======================
# User Complaint Routes
# ======================

@app.route('/dashboard')
@login_required
def dashboard():
    user_complaints = Complaint.query.filter_by(user_id=current_user.id)\
        .order_by(Complaint.created_at.desc())\
        .all()
    return render_template('dashboard.html', complaints=user_complaints)

@app.route('/file-complaint', methods=['GET', 'POST'])
@login_required
def file_complaint():
    if request.method == 'POST':
        title = request.form['title']
        description = request.form['description']
        is_anonymous = 'anonymous' in request.form
        
        # Classify complaint
        try:
            category = classifier.predict_category(f"{title} {description}")
            dept_info = get_department_info(category)
        except Exception as e:
            app.logger.error(f"Classification failed: {e}")
            category = 'General Complaint'
            dept_info = get_department_info(category)
        
        # Create complaint record
        complaint = Complaint(
            user_id=current_user.id,
            title=title,
            description=description,
            category=category,
            department=dept_info['department'],
            department_email=dept_info['email'],
            department_phone=dept_info['phone'],
            is_anonymous=is_anonymous,
            status='Submitted',
            forwarded_at=datetime.utcnow()
        )
        
        db.session.add(complaint)
        db.session.commit()
        
        # Send notifications
        send_department_notification(complaint)
        
        flash(f'Complaint submitted to {dept_info["department"]}', 'success')
        return redirect(url_for('dashboard'))
    
    return render_template('complaint.html')

@app.route('/complaint-status/<int:complaint_id>')
@login_required
def complaint_status(complaint_id):
    complaint = Complaint.query.get_or_404(complaint_id)
    
    # Check authorization
    if complaint.user_id != current_user.id and not current_user.is_admin:
        abort(403)
    
    return render_template('status.html', complaint=complaint)

# ======================
# Admin Routes
# ======================

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    if not current_user.is_admin:
        abort(403)
    
    # Get all complaints with user info
    complaints = db.session.query(
        Complaint,
        User.username
    ).outerjoin(
        User, Complaint.user_id == User.id
    ).order_by(
        Complaint.created_at.desc()
    ).all()
    
    # Get department statistics
    departments = db.session.query(
        Complaint.department,
        db.func.count(Complaint.id).label('count')
    ).group_by(
        Complaint.department
    ).all()
    
    return render_template('admin/dashboard.html', 
                         complaints=complaints,
                         departments=departments)

@app.route('/admin/department/<department_name>')
@login_required
def department_view(department_name):
    if not current_user.is_admin:
        abort(403)
    
    complaints = Complaint.query.filter_by(department=department_name)\
        .order_by(Complaint.created_at.desc())\
        .all()
    
    status_counts = db.session.query(
        Complaint.status,
        db.func.count(Complaint.id).label('count')
    ).filter_by(
        department=department_name
    ).group_by(
        Complaint.status
    ).all()
    
    return render_template('admin/department.html',
                         department=department_name,
                         complaints=complaints,
                         status_counts=status_counts)

@app.route('/admin/update-status/<int:complaint_id>', methods=['POST'])
@login_required
def update_status(complaint_id):
    if not current_user.is_admin:
        abort(403)
    
    complaint = Complaint.query.get_or_404(complaint_id)
    new_status = request.form.get('status')
    notes = request.form.get('notes', '')
    
    valid_statuses = ['Submitted', 'In Progress', 'Resolved', 'Rejected']
    if new_status not in valid_statuses:
        flash('Invalid status', 'error')
    else:
        complaint.status = new_status
        complaint.department_notes = notes
        complaint.updated_at = datetime.utcnow()
        db.session.commit()
        flash('Status updated successfully', 'success')
    
    return redirect(request.referrer or url_for('admin_dashboard'))

@app.route('/admin/delete-complaint/<int:complaint_id>', methods=['POST'])
@login_required
def delete_complaint(complaint_id):
    if not current_user.is_admin:
        abort(403)
    
    complaint = Complaint.query.get_or_404(complaint_id)
    db.session.delete(complaint)
    db.session.commit()
    flash('Complaint deleted successfully', 'success')
    return redirect(url_for('admin_dashboard'))

if __name__ == '__main__':
    app.run(debug=True)
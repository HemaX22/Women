from app import app, db
from models.user import User

def create_admin():
    with app.app_context():
        # Check if admin already exists
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            admin = User(username='admin', email='admin@example.com', is_admin=True)
            admin.set_password('admin123')
            db.session.add(admin)
            db.session.commit()
            print("Admin user created successfully!")
        else:
            print("Admin user already exists")

if __name__ == '__main__':
    # Skip NLP initialization during admin creation
    app.config['SKIP_NLP_INIT'] = True
    create_admin()
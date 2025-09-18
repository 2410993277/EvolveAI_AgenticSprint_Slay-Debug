# 📊 CFO AI Assistant

An AI-powered financial assistant built with **Django, Python, and Machine Learning** to help CFOs and finance teams analyze reports, track risks, and generate insights.  

---

## 🚀 Features
- **Report Uploads**: Upload PDF/CSV financial reports for automated analysis  
- **AI-Powered Insights**: Extract patterns, risks, and recommendations from financial data  
- **Financial Health Tracking**: Classify reports as **High Risk**, **Moderate/OK**, or **Low/Healthy**  
- **Organized Dashboard**: Sidebar navigation with interconnected pages  
- **User Accounts**: Authentication system with user profiles and emergency contact forms  

---

## 🛠️ Tech Stack
- **Backend**: Django (Python)  
- **Frontend**: HTML, CSS, Bootstrap, JavaScript  
- **Database**: SQLite / PostgreSQL  
- **AI/ML**: Pandas, Scikit-learn (for data processing and risk classification)  
- **Deployment**: GitHub + (optional: Docker/Heroku/AWS if planned)  

---

## Project Structure
Agentic_Sprint/
│── Ai_Assistant/ # Main Django app
│ ├── media/ # Uploaded reports (ignored in git)
│ ├── templates/ # HTML templates
│ ├── static/ # CSS, JS, images
│ └── ...
│
│── env/ # Virtual environment (ignored in git)
│── manage.py # Django project entry point
│── requirements.txt # Dependencies
│── README.md # Project documentation


## Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/2410993277/CFO-AI-ASSISTANT.git
   cd CFO-AI-ASSISTANT

2. Create a virtual environment
python -m venv env
# Activate it:
env\Scripts\activate       # On Windows
source env/bin/activate    # On Linux/Mac

3. Install dependencies
pip install -r requirements.txt

4. Run migrations
python manage.py migrate
   
6. Start the development server
python manage.py runserver

8. Open in browser:

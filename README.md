# 🩺 MEDS AI

![MEDS AI Logo](frontend/src/assets/medsai-logo2-white.png)


**MEDS AI** is a smart, AI-powered web platform for predicting diseases based on symptoms entered by the user. It recommends relevant doctors and home remedies, making healthcare advice more accessible, especially in regions with limited resources.

---

## 🚀 Live Application

- **Link**: [https://medsai.netlify.app/](https://medsai.netlify.app/)


---

## 🧰 Tech Stack

### Frontend:
- **React.js**
- **Tailwind CSS**
- **Axios** for API calls
- **Hosted on**: Netlify

### Backend:
- **Python 3**
- **Flask**
- **Flask-CORS**
- **Flask-PyMongo**

### Database:
- **MongoDB Atlas**

### Machine Learning:
- Trained on a comprehensive symptom–disease dataset with 132 symptoms.
- Implemented multiple machine learning models:
  - **Support Vector Machine (SVM)**
  - **Naive Bayes (NB)**
  - **Random Forest (RF)**
  - **XGBoost** (Extreme Gradient Boosting)
- Final predictions are made using a **Voting Classifier**, which combines the predictions of all models to improve overall accuracy and robustness.


---

## 🛠️ How to Run Locally

### 📦 Backend Setup (Flask + Python + MongoDB)

1. **Clone the repo & navigate into backend folder**:

```bash
git clone <repository>
cd backend
```

2. **Create virtual environment**:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:

```bash
pip install -r requirements.txt
```

4. **Set environment variables**:
Example `.env`:

```
MONGO_URI= mongodb url
FLASK_ENV= dev
SECRET_KEY= ***
```

5. **Run the Flask server**:

```bash
flask run
```

Server starts at: `http://localhost:5000`

### 💻 Frontend Setup (React)

1. **Navigate to frontend folder**:

```bash
cd ../frontend
```

2. **Install dependencies**:

```bash
npm install
```

3. **Run the development server**:

```bash
npm run dev
```


4. **Make sure** `.env` file contains the API base URL:

```env
VITE_BASE_URL=http://127.0.0.1:5000

```



## 👨‍💻 Project Contributors
* **Adhith K L**
* **Ann Geo**
* **Tony K Seby**


**Guide**: *Ms. Iris Jose*, Assistant Professor *Christ College of Engineering, Irinjalakuda*

© 2025 – MEDS AI. Built with 💙 for healthcare innovation.

# 🚀 AI Resume & Job Matcher

A full-stack AI application that helps job seekers optimize their resumes. It fetches real-time job listings, compares them against a user's uploaded PDF resume using Natural Language Processing (NLP), and provides a compatibility score along with a detailed skill gap analysis.

![Python](https://img.shields.io/badge/Python-3.10-blue)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-green)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)
![PostgreSQL](https://img.shields.io/badge/Database-PostgreSQL-blue)
![Docker](https://img.shields.io/badge/Deployment-Docker-informational)

---

## 🌟 Features

* **Real-Time Job Search**: Fetches live job postings using the **TheirStack API**.
* **AI-Powered Matching**: Uses **TF-IDF Vectorization** and **Cosine Similarity** to mathematically calculate how well a resume fits a job description.
* **Skill Gap Analysis**: Automatically extracts technical skills and identifies which ones are missing from the resume.
* **Secure Authentication**: User registration and login system powered by **JWT (JSON Web Tokens)** and **Bcrypt** hashing.
* **Match History Dashboard**: Persists all scan results in a cloud **PostgreSQL** database so users can track their applications.
* **PDF Parsing**: Extracts text from PDF resumes with high accuracy.

---

## 🏗️ Architecture

The application follows a decoupled **Client-Server architecture**:

1. **Frontend**: Built with **Streamlit** for a responsive, interactive UI. It communicates with the backend via REST API calls.
2. **Backend**: Built with **FastAPI**. It handles business logic, PDF processing, AI calculations, and database interactions.
3. **Database**: **PostgreSQL** (hosted on Neon.tech) stores user profiles and match history.
4. **AI Engine**: Uses `scikit-learn` for vectorizing text and calculating similarity scores.

---

## 🛠️ Tech Stack

* **Language:** Python 3.10+
* **Backend:** FastAPI, Uvicorn, SQLAlchemy, Pydantic
* **Frontend:** Streamlit, Pandas
* **Database:** PostgreSQL (Neon.tech for Prod, Docker for Local)
* **AI/NLP:** Scikit-learn, PyPDF, NLTK (optional)
* **Security:** Python-Jose (JWT), Passlib (Bcrypt)
* **Infrastructure:** Docker, Docker Compose, Render.com (Cloud Hosting)

---

## 🚀 Live Demo

* **Frontend App:** https://resume-jobmatcher.streamlit.app/
* **Backend API Docs:** https://resume-job-matcher-g7ja.onrender.com/docs

---

## ⚙️ Local Installation & Setup

Follow these steps to run the project on your local machine.

### Prerequisites

* Docker & Docker Compose
* Python 3.10+ (if running without Docker)
* Git

### 1. Clone the Repository

```bash
git clone https://github.com/PrathmeshnPawar/resume-job-matcher.git
cd resume-job-matcher


### 2\. Configure Environment Variables

Create a `.env` file in the root directory:

```ini
# Database (Use local or cloud URL)
DATABASE_URL=postgresql://matcher_admin:dbpassword123@db:5432/resumematcher

# API Key for Job Search (Get free key from TheirStack)
THEIRSTACK_API_KEY=your_api_key_here

# Security
SECRET_KEY=supersecretkey
```

### 3\. Run with Docker (Recommended)

This will spin up the Backend and a local PostgreSQL database automatically.

```bash
docker-compose up --build
```

*The Backend will be available at `http://localhost:8002`.*

### 4\. Run the Frontend

Open a new terminal window:

```bash
# Create virtual environment (optional but recommended)
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run Streamlit
streamlit run frontend.py
```

-----

## 🧠 How the AI Works

The core matching logic is based on **Vector Space Modeling**:

1.  **Preprocessing:** The app cleans the text, normalizing terms (e.g., converting "C\#" to "csharp", ".NET" to "dotnet") to ensure accurate matching.
2.  **Vectorization (TF-IDF):** It converts the Job Description and Resume into numerical vectors. Rare, important words (like "Kubernetes") are given higher weight than common words (like "the").
3.  **Cosine Similarity:** The app calculates the cosine of the angle between the two vectors.
      * **0° (Score 1.0):** Perfect Match.
      * **90° (Score 0.0):** No relevance.
4.  **Scaling:** The raw mathematical score is scaled to a human-readable percentage (0-100%).

-----

## 📸 Screenshots

### 1\. Job Search Dashboard

*(Add a screenshot of your Search Tab here)*<img width="1546" height="671" alt="Screenshot From 2025-11-23 20-06-51" src="https://github.com/user-attachments/assets/7a965a2c-0c7f-49db-af53-025e5ccd14c5" />


### 2\. AI Match Results

*(Add a screenshot of your Analysis Result here)*<img width="1546" height="859" alt="Screenshot From 2025-11-23 20-09-11" src="https://github.com/user-attachments/assets/8e2d73e6-1c1e-417f-b4ff-29e3f2b31f88" />


-----

## 🤝 Contributing

Contributions are welcome\! Please open an issue or submit a pull request.

## 📄 License

This project is licensed under the MIT License.

```
```

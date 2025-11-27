import os
import re
import requests
import hashlib
import httpx
import logging
import time
import json
from datetime import datetime, timedelta
from typing import List, Optional, Union

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Text, create_engine
from sqlalchemy import text as sa_text
from sqlalchemy.orm import Session, relationship, sessionmaker, declarative_base
from passlib.context import CryptContext
from jose import JWTError, jwt
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# 1. Load Config
load_dotenv()
DATABASE_URL = os.getenv("DATABASE_URL")
# Job Search Config
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")
RAPIDAPI_HOST = os.getenv("RAPIDAPI_HOST", "jsearch.p.rapidapi.com")
# AI Config
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
# Security Config
SECRET_KEY = "YOUR_SUPER_SECRET_KEY_HERE"  # Change this in production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

# 2. Database Setup
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 3. Models
class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    hashed_password = Column(String)

class Match(Base):
    __tablename__ = "matches"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    job_title = Column(String)
    company = Column(String)
    match_score = Column(Float)
    match_date = Column(DateTime, default=datetime.utcnow)
    ai_feedback = Column(Text)

# Create Tables
Base.metadata.create_all(bind=engine)
# Ensure ai_feedback column exists (for deployments without migrations)
try:
    with engine.begin() as conn:
        conn.execute(sa_text("ALTER TABLE matches ADD COLUMN IF NOT EXISTS ai_feedback TEXT"))
except Exception:
    logger.exception("Failed to ensure ai_feedback column exists; continuing")

# 4. Auth Logic
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def verify_password(plain_password, hashed_password):
    if len(plain_password) > 72:
        plain_password = hashlib.sha256(plain_password.encode('utf-8')).hexdigest()
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    if len(password) > 72:
        password = hashlib.sha256(password.encode('utf-8')).hexdigest()
    return pwd_context.hash(password)

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception
    return user

# 5. Pydantic Schemas
class UserCreate(BaseModel):
    email: str
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str

class JobSearchRequest(BaseModel):
    title: str
    location: str = ""
    page: int = 0
    employment_type: Optional[str] = None
    remote_type: Optional[str] = None

class JobResult(BaseModel):
    id: str
    title: str
    company: str
    location: str
    description: str
    skills: List[str]

class MatchResponse(BaseModel):
    match_percentage: float
    missing_skills: List[str]
    matched_skills: List[str]
    ai_feedback: str

# 6. App & Routes
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- AUTH ROUTES ---
@app.post("/register", response_model=Token)
def register(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    hashed_password = get_password_hash(user.password)
    new_user = User(email=user.email, hashed_password=hashed_password)
    db.add(new_user)
    db.commit()
    
    access_token = create_access_token(data={"sub": new_user.email})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=400, detail="Incorrect email or password")
    
    access_token = create_access_token(data={"sub": user.email})
    return {"access_token": access_token, "token_type": "bearer"}

# --- JOB SEARCH ENGINE ---
class JobSearchEngine:
    def __init__(self):
        self.host = RAPIDAPI_HOST
        self.key = RAPIDAPI_KEY
        self.base = f"https://{self.host}"

    def _clean_html(self, html: str) -> str:
        if not html: return ""
        text = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", text).strip()

    def search(self, title: str, location: str, page: int = 0, employment_type: str = None, remote_type: str = None) -> List[JobResult]:
        if not self.key:
            raise RuntimeError("RAPIDAPI_KEY must be set in environment")

        url = f"{self.base}/search"
        params = {
            "query": title or "",
            "location": location or "",
            "page": max(1, page + 1),
            "num_pages": 1,
            "limit": 5,
        }

        # Add filters
        if employment_type and employment_type != "Any":
            params["query"] = (params["query"] + " " + employment_type).strip()
        if remote_type and remote_type != "Any":
            params["query"] = (params["query"] + " " + remote_type).strip()

        headers = {
            "X-RapidAPI-Key": self.key,
            "X-RapidAPI-Host": self.host,
        }

        # RETRY LOGIC
        max_retries = 3
        data = None
        backoff = 1
        for attempt in range(1, max_retries + 1):
            try:
                # Single request with retry/backoff on failures or rate-limits
                resp = requests.get(url, params=params, headers=headers, timeout=30)
                if resp.status_code == 200:
                    data = resp.json()
                    break

                # Respect rate-limit/backoff for retryable statuses
                if resp.status_code in (429, 503) and attempt < max_retries:
                    logger.warning("Provider returned %s, retrying attempt %s", resp.status_code, attempt)
                    time.sleep(backoff)
                    backoff *= 2
                    continue

                # Non-retryable error from provider
                logger.error("RapidAPI jobs returned %s: %s", resp.status_code, resp.text)
                raise HTTPException(status_code=502, detail=f"Jobs provider error: {resp.status_code} - {resp.text}")

            except requests.exceptions.RequestException as e:
                logger.warning("Request attempt %s failed: %s", attempt, e)
                if attempt == max_retries:
                    logger.exception("All RapidAPI attempts failed")
                    raise HTTPException(status_code=502, detail=f"Jobs provider request failed: {e}")
                time.sleep(backoff)
                backoff *= 2

            except Exception as e:
                logger.error("Attempt %s failed: %s", attempt, e)
                if attempt == max_retries:
                    logger.exception("All RapidAPI attempts failed (unexpected error)")
                    raise HTTPException(status_code=502, detail=f"Jobs provider unexpected error: {e}")
                time.sleep(backoff)
                backoff *= 2
        
        if not data:
            # Return empty list instead of crashing if API fails after retries
            logger.error("Job search failed after retries")
            return []

        results = []
        for item in data.get("data", []):
            jid = item.get("job_id") or item.get("job_uuid") or item.get("id") or item.get("job_link")
            title_text = item.get("job_title") or item.get("title") or "No Title"
            company = item.get("employer_name") or item.get("company_name") or item.get("job_employer") or "Unknown"
            
            city = item.get("job_city") or item.get("location") or "Remote"
            country = item.get("job_country") or ""
            loc = (city + (", " + country if country else "")).strip()
            
            description = self._clean_html(item.get("job_description") or item.get("description") or "")
            job_id = str(jid or f"rapid_{hash(title_text+company+loc)}")
            
            results.append(JobResult(
                id=job_id, 
                title=title_text, 
                company=company, 
                location=loc or "Remote", 
                description=description, 
                skills=[]
            ))

        return results

search_engine = JobSearchEngine()

@app.post("/search-jobs")
def search_jobs(request: JobSearchRequest):
    return search_engine.search(request.title, request.location, request.page, request.employment_type, request.remote_type)

# --- GEMINI AI ENGINE ---
def call_gemini(prompt_text: str, job_desc: str = None) -> Union[str, dict]:
    """
    Calls Gemini 2.0 Flash.
    - If job_desc is provided, returns TEXT critique (for Match Resume).
    - If job_desc is None, returns JSON object (for Review Resume).
    """
    if not GEMINI_API_KEY:
        return "AI Critique Unavailable (Missing API Key)" if job_desc else {}
    
    model = "gemini-2.0-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={GEMINI_API_KEY}"
    headers = {"Content-Type": "application/json"}
    
    # Mode 1: Match Critique (Text Output)
    if job_desc:
        instruction = f"""
        You are an expert Technical Recruiter. 
        Compare this Resume to the Job Description.
        
        RESUME:
        {prompt_text[:4000]} 
        
        JOB DESCRIPTION:
        {job_desc[:4000]}
        
        Provide a helpful critique in markdown format:
        1. **Strengths:** Mention 2 things the candidate matches well.
        2. **Missing Keywords:** List 3 specific technical skills or tools missing from the resume that are in the JD.
        3. **Recommendation:** One sentence on how to improve.
        """
        response_type = "text/plain"
        
    # Mode 2: General Review (JSON Output)
    else:
        instruction = f"""
        You are an expert resume reviewer. 
        Analyze the following resume and provide a structured critique.
        
        RESUME:
        {prompt_text[:4000]} 
        
        Output a valid JSON object with keys: "score" (0-100), "criticisms" (list of strings), "suggestions" (list of strings).
        Do NOT use Markdown blocks.
        """
        response_type = "application/json"

    payload = {
        "contents": [{"parts": [{"text": instruction}]}],
        "generationConfig": {
            "temperature": 0.3,
            "maxOutputTokens": 1000,
            "responseMimeType": response_type
        }
    }

    try:
        # Increased timeout to 60s for AI
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            result_text = data["candidates"][0]["content"]["parts"][0]["text"]
            
            if job_desc:
                return result_text # Return Text
            else:
                # Return JSON object
                clean_json = re.sub(r"```json|```", "", result_text).strip()
                return json.loads(clean_json)
            
    except Exception as e:
        logger.error(f"Gemini Error: {e}")
        return "AI Analysis unavailable." if job_desc else {}

# --- MATCHING ENGINE ---
def calculate_match(resume_text, jd):
    def clean(text):
        text = text.lower()
        text = text.replace("c#", "csharp").replace(".net", "dotnet").replace("c++", "cpp")
        return re.sub(r'[^a-z0-9\s]', ' ', text)

    clean_res = clean(resume_text)
    clean_jd = clean(jd)
    if not clean_res.strip() or not clean_jd.strip(): return 0.0
    
    try:
        tfidf = TfidfVectorizer(stop_words='english').fit_transform([clean_res, clean_jd])
        score = cosine_similarity(tfidf[0:1], tfidf[1:2])[0][0]
        final_score = float(score * 4.0)
        final_score = round(min(final_score, 1.0) * 100, 2)
        return final_score
    except:
        return 0.0

@app.post("/match-resume", response_model=MatchResponse)
async def match_resume(
    file: UploadFile = File(...), 
    job_description: str = Form(...),
    job_skills: str = Form(""), 
    current_user: User = Depends(get_current_user), 
    db: Session = Depends(get_db)
):
    try:
        reader = PdfReader(file.file)
        text = "".join([page.extract_text() or "" for page in reader.pages])
    except Exception:
        text = ""

    score = calculate_match(text, job_description)
    
    # Auto-detect skills
    COMMON_TECH_STACK = ["Python", "Java", "Spring", "React", "AWS", "Docker", "Kubernetes", "SQL", "NoSQL", "Linux", "Git", "CI/CD", "Azure", "C#", ".NET", "JavaScript", "HTML", "CSS"]
    
    if job_skills:
        jd_skills_list = [s.strip() for s in job_skills.split(",")]
    else:
        jd_skills_list = [tech for tech in COMMON_TECH_STACK if tech.lower() in job_description.lower()]

    missing = [s for s in jd_skills_list if s.lower() not in text.lower()]
    matched = [s for s in jd_skills_list if s.lower() in text.lower()]

    # Call Gemini (Text Mode)
    ai_critique = call_gemini(text, job_desc=job_description)
    
    # Ensure ai_critique is a string (in case of error returning dict)
    if not isinstance(ai_critique, str):
        ai_critique = "AI Analysis unavailable."

    db.add(Match(
        user_id=current_user.id, 
        job_title=job_description[:50] + "...", 
        company="Unknown", 
        match_score=score,
        ai_feedback=ai_critique 
    ))
    db.commit()

    return MatchResponse(
        match_percentage=score, 
        missing_skills=missing, 
        matched_skills=matched,
        ai_feedback=ai_critique
    )

@app.post("/review-resume")
async def review_resume(
    file: UploadFile = File(...),
    job_description: str = Form(None),
):
    """
    Endpoint for generic resume review (JSON output).
    """
    try:
        # FIX: Read the file content into memory first
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
            
        # Pass the bytes to PdfReader using io.BytesIO
        reader = PdfReader(io.BytesIO(content))
        resume_text = "".join([page.extract_text() or "" for page in reader.pages])
        
        if len(resume_text.strip()) < 50:
             raise HTTPException(status_code=400, detail="PDF is image-based or empty. Please upload a text PDF.")
             
    except Exception as e:
        logger.exception(f"PDF Parse Error: {e}")
        raise HTTPException(status_code=400, detail=f"Unable to read PDF: {e}")

    prompt_text = resume_text
    if job_description:
        prompt_text = f"Job Description:\n{job_description}\n\nResume:\n{resume_text}"

    # Call Gemini (JSON Mode)
    result = call_gemini(prompt_text, job_desc=None)
    
    if not result:
        raise HTTPException(status_code=500, detail="AI Analysis failed")
        
    return result

@app.get("/history")
def get_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(Match).filter(Match.user_id == current_user.id).order_by(Match.match_date.desc()).all()

# --- HEALTH CHECK (Fixes Render Shutdown) ---
@app.api_route("/", methods=["GET", "HEAD"])
def root():
    return {"message": "Resume Matcher API is Online!"}

@app.get("/healthz")
def health_check():
    return {"status": "ok"}
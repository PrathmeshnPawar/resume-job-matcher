import os
import re
import requests
import hashlib
import httpx
import logging
import json
from datetime import datetime, timedelta
from typing import List, Optional

from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, Form, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from sqlalchemy import Column, Integer, String, Float, ForeignKey, DateTime, Text, create_engine
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
THEIRSTACK_API_KEY = os.getenv("THEIRSTACK_API_KEY")
SECRET_KEY = "YOUR_SUPER_SECRET_KEY_HERE" # Change this in production!
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

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

# Create Tables
Base.metadata.create_all(bind=engine)

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
    employment_type: Optional[str] = None # New Filter: Full-time, Part-time
    remote_type: Optional[str] = None     # New Filter: Remote, Hybrid

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

# --- JOB SEARCH ENGINE (Adzuna) ---
ADZUNA_APP_ID = os.getenv("ADZUNA_APP_ID")
ADZUNA_APP_KEY = os.getenv("ADZUNA_APP_KEY")
# ADZUNA_COUNTRY = os.getenv("ADZUNA_COUNTRY", "us")


class JobSearchEngine:
    def __init__(self):
        self.base = "https://api.adzuna.com/v1/api/jobs"
        self.app_id = ADZUNA_APP_ID
        self.app_key = ADZUNA_APP_KEY
        # self.country = ADZUNA_COUNTRY

    def _clean_html(self, html: str) -> str:
        if not html:
            return ""
        text = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", text).strip()

    def search(self, title: str, location: str, page: int = 0, employment_type: str = None, remote_type: str = None) -> List[JobResult]:
        if not self.app_id or not self.app_key:
            # Fail early with a helpful error
            raise RuntimeError("ADZUNA_APP_ID and ADZUNA_APP_KEY must be set in environment")

        try:
            # Adzuna uses 1-indexed pages
            page_num = max(1, page + 1)
            url = f"{self.base}/{self.country}/search/{page_num}"
            params = {
                "app_id": self.app_id,
                "app_key": self.app_key,
                "what": title or "",
                "where": location or "",
                "results_per_page": 5,
            }

            if employment_type and employment_type != "Any":
                params["what"] = (params["what"] + " " + employment_type).strip()
            if remote_type and remote_type != "Any":
                params["what"] = (params["what"] + " " + remote_type).strip()

            headers = {"User-Agent": "resume-matcher/1.0"}
            resp = requests.get(url, params=params, headers=headers, timeout=15)
            resp.raise_for_status()
            data = resp.json()

            results = []
            for item in data.get("results", []):
                title = item.get("title") or item.get("label") or "No Title"
                company = (item.get("company") or {}).get("display_name") or "Unknown"
                loc = (item.get("location") or {}).get("display_name") or item.get("location", "") or "Remote"
                description = self._clean_html(item.get("description") or "")
                job_id = str(item.get("id") or item.get("redirect_url") or f"adz_{hash(title+company+loc)}")
                skills = []

                results.append(JobResult(id=job_id, title=title, company=company, location=loc, description=description, skills=skills))

            return results
        except Exception as e:
            print(f"Adzuna API error: {e}")
            return []


search_engine = JobSearchEngine()

# GEMINI / Generative Language API key
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MOCK = os.getenv("GEMINI_MOCK", "false").lower() in ("1", "true", "yes")

logger = logging.getLogger("app")
logging.basicConfig(level=logging.INFO)


def call_gemini(prompt_text: str, model: str = "chat-bison-001", max_tokens: int = 800):
    """Call the Generative Language REST API (Gemini/chat-bison style).
    Returns a parsed JSON object when possible, otherwise returns dict with 'raw'.
    """
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")

    url = f"https://generativelanguage.googleapis.com/v1beta2/models/{model}:generate?key={GEMINI_API_KEY}"

    system_prompt = (
        "You are an expert resume reviewer. Return ONLY a JSON object with keys: "
        '"score" (0-100), "criticisms" (list of short points), "suggestions" (list of short actionable items), '
        '"improvements" (optional list of suggested rewrites). Do not include extra text.'
    )

    full_prompt = system_prompt + "\n\nResume Text:\n" + prompt_text

    payload = {
        "prompt": {"text": full_prompt},
        "temperature": 0.2,
        "max_output_tokens": max_tokens,
    }

    try:
        with httpx.Client(timeout=60.0) as client:
            r = client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
    except httpx.HTTPStatusError as e:
        # include response content for debugging
        content = None
        try:
            content = e.response.text
        except Exception:
            content = str(e)
        logger.error("Gemini API returned HTTP error: %s", content)
        raise RuntimeError(f"Gemini API HTTP error: {e.response.status_code} - {content}")
    except Exception as e:
        logger.exception("Error calling Gemini API: %s", e)
        raise RuntimeError(f"Error calling Gemini API: {e}")

    generated = ""
    # Common shapes: 'candidates' or 'output'
    if isinstance(data, dict) and "candidates" in data and len(data["candidates"]) > 0:
        generated = data["candidates"][0].get("content", "")
    elif isinstance(data, dict) and "output" in data and isinstance(data["output"], list):
        pieces = []
        for item in data["output"]:
            if isinstance(item, dict) and "content" in item:
                for c in item["content"]:
                    if c.get("type") == "output_text":
                        pieces.append(c.get("text", ""))
        generated = "\n".join(pieces)
    else:
        # fallback: stringify
        try:
            generated = data.get("candidates", [{}])[0].get("content", str(data))
        except Exception:
            generated = str(data)

    # Try to parse JSON out of the generated text
    try:
        m = re.search(r"\{[\s\S]*\}", generated)
        json_text = m.group(0) if m else generated
        return json.loads(json_text)
    except Exception:
        logger.warning("Failed to parse JSON from Gemini output; returning raw text")
        return {"raw": generated}

@app.post("/search-jobs")
def search_jobs(request: JobSearchRequest):
    return search_engine.search(request.title, request.location, request.page, request.employment_type, request.remote_type)

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
    reader = PdfReader(file.file)
    text = "".join([page.extract_text() or "" for page in reader.pages])
    
    score = calculate_match(text, job_description)
    
    # Skills comparison removed per request — return empty lists for matched/missing skills
    matched = []
    missing = []

    db.add(Match(
        user_id=current_user.id, 
        job_title=job_description[:50] + "...", 
        company="Unknown", 
        match_score=score
    ))
    db.commit()

    return MatchResponse(match_percentage=score, missing_skills=missing, matched_skills=matched)


@app.post("/review-resume")
async def review_resume(
    file: UploadFile = File(...),
    job_description: str = Form(None),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Extract resume text and call Gemini to produce a structured critique and suggestions."""
    try:
        reader = PdfReader(file.file)
        resume_text = "".join([page.extract_text() or "" for page in reader.pages])
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Unable to read PDF: {e}")

    prompt_text = resume_text
    if job_description:
        prompt_text = f"Job Description:\n{job_description}\n\nResume:\n{resume_text}"

    # If GEMINI_API_KEY is not set, either return a mock response (if enabled)
    # or return an informative error so the frontend can surface it.
    if not GEMINI_API_KEY:
        if GEMINI_MOCK:
            logger.info("GEMINI_API_KEY not set - returning mock AI review (GEMINI_MOCK=true)")
            mock = {
                "score": 75,
                "criticisms": ["Summary is short", "Add quantifiable achievements"],
                "suggestions": ["Expand summary to 3-4 lines", "Add metrics to projects"],
                "improvements": [{"section": "Summary", "rewrite": "Experienced Software Engineer with 5+ years..."}]
            }
            return mock
        else:
            logger.error("GEMINI_API_KEY not set and GEMINI_MOCK is false")
            raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured on server. Set GEMINI_API_KEY or enable GEMINI_MOCK for testing.")

    try:
        result = call_gemini(prompt_text)
    except Exception as e:
        logger.exception("call_gemini failed: %s", e)
        raise HTTPException(status_code=500, detail=f"AI call failed: {e}")

    return result

@app.get("/history")
def get_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(Match).filter(Match.user_id == current_user.id).order_by(Match.match_date.desc()).all()

@app.get("/")
def root():
    return {"message": "Resume Matcher API is Online and Connected to Neon DB!"}
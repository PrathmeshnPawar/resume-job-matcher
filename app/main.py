import os
import re
import requests
import hashlib
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

# --- JOB SEARCH ENGINE ---
class JobSearchEngine:
    def __init__(self):
        self.base_url = "https://api.theirstack.com/v1/jobs/search"
        self.api_key = THEIRSTACK_API_KEY

    def search(self, title: str, location: str, page: int = 0, employment_type: str = None, remote_type: str = None) -> List[JobResult]:
        try:
            headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
            
            payload = {
                "job_title_or": [title],
                "posted_at_max_age_days": 45,
                "limit": 5,
                "page": page, 
                "include_total_results": False
            }

            # 1. Location & Work Style Logic
            locations = []
            if location and location.strip().lower() not in ["", "anywhere", "global", "remote"]:
                locations.append(location)
            
            if remote_type and remote_type != "Any":
                if remote_type.lower() == "remote":
                    locations.append("remote")
                elif remote_type.lower() == "hybrid":
                    locations.append("hybrid")
            
            if locations:
                payload["job_location_pattern_or"] = locations

            # 2. Employment Type Logic (Using description matching)
            if employment_type and employment_type != "Any":
                patterns = []
                et = employment_type.lower()
                if "full" in et: patterns = ["full-time", "full time"]
                elif "part" in et: patterns = ["part-time", "part time"]
                elif "contract" in et: patterns = ["contract", "contractor"]
                elif "intern" in et: patterns = ["intern", "internship"]
                elif "freelance" in et: patterns = ["freelance"]
                
                if patterns:
                    payload["job_description_pattern_or"] = patterns

            response = requests.post(self.base_url, json=payload, headers=headers, timeout=20)
            response.raise_for_status()
            data = response.json()
            
            clean_jobs = []
            for item in data.get('data', []):
                loc = item.get("job_location") or "Remote"
                clean_jobs.append(JobResult(
                    id=str(item.get("id")), 
                    title=item.get("job_title") or "No Title",
                    company=item.get("company_object", {}).get("name") or "Unknown",
                    location=loc,
                    description=item.get("description") or "No description", 
                    skills=item.get("technologies", [])
                ))
            return clean_jobs
        except Exception as e:
            print(f"API Error: {e} -> Returning Mock Data")
            return [JobResult(id="mock", title="Mock Job", company="Mock Corp", location="Remote", description="Mock Desc", skills=["Python"])]

search_engine = JobSearchEngine()

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
    
    # Normalize incoming job_skills. Accept comma-separated strings, JSON lists,
    # or empty values. Try to extract readable skill names if elements are dict-like.
    jd_skills_list = []
    if job_skills:
        # If looks like a JSON array, try to parse
        try:
            import json
            parsed = json.loads(job_skills)
            if isinstance(parsed, list):
                # Extract string values or object 'name' fields
                for el in parsed:
                    if isinstance(el, str):
                        jd_skills_list.append(el.strip())
                    elif isinstance(el, dict):
                        name = el.get('name') or el.get('title') or el.get('tech')
                        if name: jd_skills_list.append(str(name).strip())
        except Exception:
            # Fallback: comma-separated
            jd_skills_list = [s.strip() for s in job_skills.split(",") if s.strip()]

    # If the upstream job API didn't provide structured skills, infer them
    # from the job description using a curated alias -> canonical mapping.
    if not jd_skills_list:
        jd_lower = (job_description or "").lower()
        skill_aliases = {
            'python': 'Python', 'java': 'Java', 'c++': 'C++', 'c#': 'C#', 'csharp': 'C#', 'cpp': 'C++',
            'javascript': 'JavaScript', 'js': 'JavaScript', 'typescript': 'TypeScript', 'node.js': 'Node.js', 'node': 'Node.js',
            'react': 'React', 'react.js': 'React', 'angular': 'Angular', 'vue': 'Vue',
            'django': 'Django', 'flask': 'Flask', 'spring': 'Spring',
            'sql': 'SQL', 'postgresql': 'PostgreSQL', 'postgres': 'PostgreSQL', 'mysql': 'MySQL', 'nosql': 'NoSQL',
            'mongodb': 'MongoDB', 'redis': 'Redis', 'elasticsearch': 'Elasticsearch',
            'aws': 'AWS', 'azure': 'Azure', 'gcp': 'GCP', 'google cloud': 'GCP',
            'docker': 'Docker', 'kubernetes': 'Kubernetes', 'k8s': 'Kubernetes', 'helm': 'Helm',
            'git': 'Git', 'linux': 'Linux', 'bash': 'Bash', 'shell': 'Shell',
            'html': 'HTML', 'css': 'CSS', 'sass': 'Sass',
            'pandas': 'Pandas', 'numpy': 'NumPy', 'scikit-learn': 'Scikit-Learn', 'sklearn': 'Scikit-Learn',
            'tensorflow': 'TensorFlow', 'pytorch': 'PyTorch', 'keras': 'Keras',
            'spark': 'Spark', 'hadoop': 'Hadoop', 'scala': 'Scala', 'golang': 'Go', 'go': 'Go', 'rust': 'Rust',
            'graphql': 'GraphQL', 'rest': 'REST', 'restful': 'REST', 'api': 'API',
            'microservice': 'Microservices', 'microservices': 'Microservices', 'ci/cd': 'CI/CD', 'jenkins': 'Jenkins',
            'terraform': 'Terraform', 'ansible': 'Ansible', 'vagrant': 'Vagrant',
            'redis': 'Redis', 'kafka': 'Kafka'
        }

        inferred = set()
        for alias, canonical in skill_aliases.items():
            # Match using word boundaries, allow dots and dashes in aliases
            pattern = r"\b" + re.escape(alias) + r"\b"
            if re.search(pattern, jd_lower):
                inferred.add(canonical)

        # Also look for common multi-word skills explicitly
        multi_word = ['machine learning', 'data science', 'deep learning', 'natural language processing']
        for phrase in multi_word:
            if phrase in jd_lower:
                inferred.add(phrase.title())

        jd_skills_list = list(inferred)

    # Use word-boundary matching against the resume text to determine matched/missing skills
    missing = []
    matched = []
    text_lower = text.lower()
    for s in jd_skills_list:
        if not s:
            continue
        s_lower = s.lower()
        pattern = r"\b" + re.escape(s_lower) + r"\b"
        if re.search(pattern, text_lower):
            matched.append(s)
        else:
            missing.append(s)

    # Debug log for inferred/normalized skills (helps debugging in dev)
    try:
        print(f"[match-resume] jd_skills_list={jd_skills_list} matched={matched} missing={missing}")
    except Exception:
        pass

    db.add(Match(
        user_id=current_user.id, 
        job_title=job_description[:50] + "...", 
        company="Unknown", 
        match_score=score
    ))
    db.commit()

    return MatchResponse(match_percentage=score, missing_skills=missing, matched_skills=matched)

@app.get("/history")
def get_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(Match).filter(Match.user_id == current_user.id).order_by(Match.match_date.desc()).all()

@app.get("/")
def root():
    return {"message": "Resume Matcher API is Online and Connected to Neon DB!"}
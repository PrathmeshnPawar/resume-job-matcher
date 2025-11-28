import streamlit as st
import streamlit.components.v1 as components
import httpx
import pandas as pd
import os

# SMART CONFIG
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8002") 
st.set_page_config(page_title="JobFit", page_icon="🚀", layout="wide")

# --- CUSTOM CSS FOR LANDING PAGE ---
def load_custom_css():
    st.markdown("""
        <style>
            /* Import Fonts */
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Playfair+Display:wght@700&display=swap');
            
            /* Variable Definitions */
            :root {
                --bg-gradient: radial-gradient(at 0% 0%, rgba(76, 29, 149, 0.4) 0px, transparent 50%),
                               radial-gradient(at 90% 10%, rgba(59, 130, 246, 0.3) 0px, transparent 50%),
                               radial-gradient(at 50% 50%, rgba(30, 27, 75, 0.8) 0px, transparent 50%),
                               radial-gradient(at 0% 100%, rgba(76, 29, 149, 0.5) 0px, transparent 50%),
                               radial-gradient(at 100% 100%, rgba(59, 130, 246, 0.4) 0px, transparent 50%);
            }

            /* Apply Abstract Background only if not logged in */
            [data-testid="stAppViewContainer"] {
                background-color: #0f0c29;
                background-image: var(--bg-gradient);
                background-attachment: fixed;
                background-size: cover;
                color: white;
            }
            
            /* Typography Overrides */
            h1, h2, h3 {
                font-family: 'Playfair Display', serif !important;
            }
            
            p, div, button {
                font-family: 'Inter', sans-serif !important;
            }

            /* Custom Button Styling to match the reference */
            div.stButton > button:first-child {
                background-color: #4f46e5; /* Indigo-600 */
                color: white;
                border-radius: 8px;
                padding: 0.75rem 2rem;
                border: none;
                font-weight: 600;
                box-shadow: 0 10px 15px -3px rgba(79, 70, 229, 0.3);
                transition: all 0.3s ease;
            }
            
            div.stButton > button:first-child:hover {
                background-color: #4338ca; /* Indigo-700 */
                transform: translateY(-2px);
            }

            /* Glassmorphism Card Effect */
            .glass-card {
                background: rgba(255, 255, 255, 0.03);
                backdrop-filter: blur(10px);
                border: 1px solid rgba(255, 255, 255, 0.05);
                border-radius: 16px;
                padding: 20px;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.25);
            }
        </style>
    """, unsafe_allow_html=True)

# --- SESSION STATE ---
if 'token' not in st.session_state: st.session_state.token = None
if 'user_email' not in st.session_state: st.session_state.user_email = None
if 'page' not in st.session_state: st.session_state.page = 0
if 'show_auth' not in st.session_state: st.session_state.show_auth = False

# Apply CSS only
load_custom_css()

# --- AUTH FUNCTIONS ---
def login(email, password):
    try:
        with httpx.Client(trust_env=False) as client:
            res = client.post(f"{BACKEND_URL}/login", data={"username": email, "password": password})
            if res.status_code == 200:
                data = res.json()
                st.session_state.token = data['access_token']
                st.session_state.user_email = email
                st.session_state.show_auth = False # Close dialog
                st.rerun()
            else: st.error("Invalid email or password")
    except Exception as e: st.error(f"Connection Error: {e}")

def register(email, password):
    try:
        with httpx.Client(trust_env=False) as client:
            res = client.post(f"{BACKEND_URL}/register", json={"email": email, "password": password})
            if res.status_code == 200: st.success("Account created! Please log in.")
            else: st.error("Registration failed.")
    except Exception as e: st.error(f"Connection Error: {e}")

def logout():
    st.session_state.token = None
    st.session_state.user_email = None
    st.rerun()

# --- HEADER & AUTH BUTTON ---
# Only show standard header if logged in OR if auth modal is open
if st.session_state.token or st.session_state.show_auth:
    c1, c2 = st.columns([8, 1])
    with c1:
        st.title("🚀 JobFit AI")
    with c2:
        if st.session_state.token:
            if st.button("Logout"):
                logout()

# --- AUTH DIALOG (MODAL) ---
if st.session_state.show_auth and not st.session_state.token:
    with st.container():
        st.markdown("<br>", unsafe_allow_html=True)
        col_spacer, col_login, col_spacer2 = st.columns([1, 2, 1])
        with col_login:
            with st.container(border=True):
                st.subheader("Welcome Back 👋")
                tab_login, tab_signup = st.tabs(["Login", "Create Account"])
                
                with tab_login:
                    email = st.text_input("Email", key="login_email")
                    password = st.text_input("Password", type="password", key="login_pass")
                    if st.button("Sign In", type="primary", use_container_width=True):
                        login(email, password)
                
                with tab_signup:
                    new_email = st.text_input("New Email", key="signup_email")
                    new_pass = st.text_input("New Password", type="password", key="signup_pass")
                    if st.button("Register", type="primary", use_container_width=True):
                        register(new_email, new_pass)
                
                if st.button("Close", use_container_width=True):
                    st.session_state.show_auth = False
                    st.rerun()
        st.stop() 

# --- LANDING PAGE (If not logged in) ---
if not st.session_state.token:
    # Using Streamlit layout to replicate the design
    # Empty spacer for top padding
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    hero_col1, hero_col2 = st.columns([1, 1])
    
    with hero_col1:
        st.markdown("""
<div style="margin-top: 40px;">
    <h1 style="font-size: 4.5rem; line-height: 1.1; margin-bottom: 20px;">
        Resume Matching <br> 
        <span style="background: linear-gradient(to right, #60a5fa, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">Reimagined</span>
    </h1>
    <p style="font-size: 1.1rem; color: #cbd5e1; margin-bottom: 30px; max-width: 500px; line-height: 1.6;">
        Stop guessing. Our AI analyzes your resume against real job descriptions to give you a precise match score and actionable feedback. Land your dream job with data, not luck.
    </p>
</div>
""", unsafe_allow_html=True)
        
        if st.button("Get Started", type="primary"):
            st.session_state.show_auth = True
            st.rerun()
            
        st.markdown("""
<div style="display: flex; gap: 40px; margin-top: 60px; color: #94a3b8; font-size: 0.8rem; font-weight: 600; letter-spacing: 1px;">
    <div>
        <span style="display: block; color: white; font-size: 1.5rem; font-weight: 700;">95%</span>
        ACCURACY
    </div>
        <div>
        <span style="display: block; color: white; font-size: 1.5rem; font-weight: 700;">10k+</span>
        MATCHES
    </div>
        <div>
        <span style="display: block; color: white; font-size: 1.5rem; font-weight: 700;">24/7</span>
        AI COACH
    </div>
</div>
""", unsafe_allow_html=True)

    with hero_col2:
        html = """
        <!-- Glow Effect -->
        <div style="position: relative; width: 100%; height: 420px;">
            <div style="position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); width: 300px; height: 300px; background: #4f46e5; filter: blur(100px); opacity: 0.4; border-radius: 50%;"></div>

            <div class="glass-card" style="position: absolute; top: 20px; right: 20px; width: 280px;">
                <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 12px;">
                    <div style="width: 44px; height: 44px; background: #22c55e; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: bold; color: white;">85</div>
                    <div>
                        <div style="font-weight: 700; font-size: 0.95rem;">Match Score</div>
                        <div style="font-size: 0.75rem; color: #4ade80;">High Compatibility</div>
                    </div>
                </div>
                <div style="height: 8px; background: rgba(255,255,255,0.06); border-radius: 4px; width: 100%; margin-bottom: 8px;"></div>
                <div style="height: 8px; background: linear-gradient(90deg,#4ade80 75%, rgba(255,255,255,0.06) 0%); border-radius: 4px; width: 75%;"></div>
            </div>

            <div class="glass-card" style="position: absolute; bottom: 20px; left: 12px; width: 240px;">
                <div style="font-size: 0.72rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px;">Missing Skills</div>
                <div style="display: flex; gap: 8px; flex-wrap: wrap;">
                    <span style="padding: 6px 10px; background: rgba(239,68,68,0.12); color: #fca5a5; border-radius: 6px; font-size: 0.78rem; border: 1px solid rgba(239,68,68,0.18);">Docker</span>
                    <span style="padding: 6px 10px; background: rgba(239,68,68,0.12); color: #fca5a5; border-radius: 6px; font-size: 0.78rem; border: 1px solid rgba(239,68,68,0.18);">Kubernetes</span>
                </div>
            </div>
        </div>
        <style>
            .glass-card { background: rgba(255,255,255,0.03); backdrop-filter: blur(10px); border: 1px solid rgba(255,255,255,0.05); border-radius: 16px; padding: 16px; box-shadow: 0 25px 50px -12px rgba(0,0,0,0.25); }
        </style>
        """
        components.html(html, height=420)


    st.stop()

# --- MAIN DASHBOARD (Only visible after login) ---
with st.sidebar:
    st.write(f"👤 **{st.session_state.user_email}**")
    if st.button("Logout", use_container_width=True):
        logout()
    st.divider()
    st.markdown("### 🧭 Navigation")

tab1, tab2 = st.tabs(["🔍 Find Jobs", "📜 Match History"])

# --- TAB 1: SEARCH & MATCH ---
with tab1:
    col_search, col_results = st.columns([1, 2])
    
    def perform_search():
        with st.spinner(f"Searching Page {st.session_state.page + 1}..."):
            try:
                payload = {
                    "title": st.session_state.search_title, 
                    "location": st.session_state.search_location,
                    "page": st.session_state.page,
                    "employment_type": st.session_state.search_employment if st.session_state.search_employment != "Any" else None,
                    "remote_type": st.session_state.search_remote if st.session_state.search_remote != "Any" else None
                }
                with httpx.Client(trust_env=False, timeout=60.0) as client:
                    res = client.post(f"{BACKEND_URL}/search-jobs", json=payload, timeout=60.0)
                    if res.status_code == 200:
                        st.session_state.jobs = res.json()
                    else:
                        try: body = res.json()
                        except: body = res.text
                        st.error(f"Search failed: {res.status_code} - {body}")
            except Exception as e: st.error(f"Error: {e}")

    with col_search:
        st.subheader("Filters")
        if 'search_title' not in st.session_state: st.session_state.search_title = "Python Developer"
        if 'search_location' not in st.session_state: st.session_state.search_location = ""
        if 'search_employment' not in st.session_state: st.session_state.search_employment = "Any"
        if 'search_remote' not in st.session_state: st.session_state.search_remote = "Any"

        st.session_state.search_title = st.text_input("Job Title", st.session_state.search_title)
        st.session_state.search_location = st.text_input("City / Country", st.session_state.search_location)
        
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.search_employment = st.selectbox("Type", ["Any", "Full-time", "Part-time", "Contract", "Internship", "Freelance"])
        with c2:
            st.session_state.search_remote = st.selectbox("Work Style", ["Any", "Remote", "Hybrid", "On-site"])

        if st.button("Find Jobs", type="primary", use_container_width=True):
            st.session_state.page = 0
            perform_search()

    with col_results:
        if 'jobs' in st.session_state:
            st.subheader(f"Results (Page {st.session_state.page + 1})")
            for job in st.session_state.jobs:
                with st.expander(f"💼 {job.get('title','Untitled')} @ {job.get('company','Unknown')}"):
                    st.markdown(f"**Location:** {job.get('location','N/A')}")
                    short_desc = (job.get('description') or "").strip()
                    if short_desc:
                        preview = short_desc[:300] + ("..." if len(short_desc) > 300 else "")
                        st.write(preview)
                    else:
                        st.info("No job description available.")

                    st.markdown("---")
                    st.subheader("Resume")
                    file_key = f"file_{job.get('id')}"
                    uploaded = st.file_uploader("Drag and drop or browse to upload (PDF)", type=["pdf"], key=file_key)

                    st.markdown("---")
                    c_an, c_rev = st.columns(2)
                    
                    analyze_key = f"btn_{job.get('id')}"
                    review_key = f"review_{job.get('id')}"
                    
                    with c_an:
                        if st.button("⚡ Analyze Match", key=analyze_key, use_container_width=True):
                            if not uploaded:
                                st.warning("Upload resume first.")
                            else:
                                with st.spinner("Analyzing..."):
                                    headers = {"Authorization": f"Bearer {st.session_state.token}"}
                                    files = {"file": (uploaded.name, uploaded, "application/pdf")}
                                    data = {"job_description": job.get('description') or "", "job_skills": ",".join(job.get('skills', []))}
                                    try:
                                        with httpx.Client(trust_env=False, timeout=90.0) as client:
                                            res = client.post(f"{BACKEND_URL}/match-resume", data=data, files=files, headers=headers, timeout=90.0)
                                        if res.status_code == 200:
                                            result = res.json()
                                            st.session_state[f"match_{job.get('id')}"] = result
                                        else: st.error("Analysis failed.")
                                    except Exception as e: st.error(f"Error: {e}")

                    with c_rev:
                        if st.button("🤖 AI Career Coach", key=review_key, use_container_width=True):
                            if not uploaded:
                                st.warning("Upload resume first.")
                            else:
                                with st.spinner("Asking Gemini..."):
                                    headers = {"Authorization": f"Bearer {st.session_state.token}"}
                                    files = {"file": (uploaded.name, uploaded, "application/pdf")}
                                    data = {"job_description": job.get('description') or ""}
                                    try:
                                        with httpx.Client(trust_env=False, timeout=180.0) as client:
                                            res = client.post(f"{BACKEND_URL}/review-resume", data=data, files=files, headers=headers, timeout=180.0)
                                        if res.status_code == 200:
                                            st.session_state[f"ai_review_{job.get('id')}"] = res.json()
                                        else: st.error(f"AI review failed: {res.status_code}")
                                    except Exception as e: st.error(f"AI error: {e}")

                    # Render Match Results
                    match_key = f"match_{job.get('id')}"
                    if match_key in st.session_state:
                        res = st.session_state[match_key]
                        sc = res.get('match_percentage', 0)
                        st.divider()
                        if sc >= 75: st.success(f"### 🚀 Match Score: {sc}%")
                        elif sc >= 50: st.warning(f"### ⚠️ Match Score: {sc}%")
                        else: st.error(f"### ❌ Match Score: {sc}%")
                        
                        c1, c2 = st.columns(2)
                        with c1: 
                            st.success("✅ **Skills You Have**")
                            if res.get('matched_skills'):
                                for s in res['matched_skills']: st.write(f"- {s}")
                            else: st.caption("None detected")
                        with c2: 
                            st.error("❌ **Skills Missing**")
                            if res.get('missing_skills'):
                                for s in res['missing_skills']: st.write(f"- {s}")
                            else: st.caption("None detected")

                    # Render AI Review Results
                    ai_key = f"ai_review_{job.get('id')}"
                    if ai_key in st.session_state:
                        review = st.session_state[ai_key]
                        st.divider()
                        st.subheader("🤖 AI Critique")
                        if isinstance(review, dict):
                            if review.get('score'): st.metric("AI Resume Score", f"{review['score']}/100")
                            if review.get('criticisms'):
                                st.write("#### ⚠️ Areas for Improvement")
                                for c in review['criticisms']: st.error(f"• {c}")
                            if review.get('suggestions'):
                                st.write("#### 💡 Actionable Suggestions")
                                for s in review['suggestions']: st.info(f"• {s}")
                        else:
                            st.write(review)

            st.divider()
            c1, c2, c3 = st.columns([1, 2, 1])
            if c1.button("⬅️ Prev") and st.session_state.page > 0:
                st.session_state.page -= 1
                perform_search()
                st.rerun()
            if c3.button("Next ➡️") and len(st.session_state.jobs) == 5:
                st.session_state.page += 1
                perform_search()
                st.rerun()

# --- TAB 2: HISTORY ---
with tab2:
    if st.button("Refresh History"):
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        try:
            with httpx.Client(trust_env=False) as client:
                res = client.get(f"{BACKEND_URL}/history", headers=headers)
                if res.status_code == 200:
                    df = pd.DataFrame(res.json())
                    if not df.empty: st.dataframe(df[['job_title', 'match_score', 'match_date']], use_container_width=True)
                    else: st.info("No history yet.")
        except: st.error("Error fetching history")
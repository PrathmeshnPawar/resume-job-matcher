import streamlit as st
import httpx
import pandas as pd
import os

# SMART CONFIG
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8002") 
st.set_page_config(page_title="Resume Matcher Pro", page_icon="💼", layout="wide")

# --- SESSION STATE ---
if 'token' not in st.session_state: st.session_state.token = None
if 'user_email' not in st.session_state: st.session_state.user_email = None
if 'page' not in st.session_state: st.session_state.page = 0

# --- AUTH ---
def login(email, password):
    try:
        with httpx.Client(trust_env=False) as client:
            res = client.post(f"{BACKEND_URL}/login", data={"username": email, "password": password})
            if res.status_code == 200:
                data = res.json()
                st.session_state.token = data['access_token']
                st.session_state.user_email = email
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
    st.rerun()

# --- LOGIN SCREEN ---
if not st.session_state.token:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔒 Login")
        tab1, tab2 = st.tabs(["Login", "Register"])
        with tab1:
            e = st.text_input("Email")
            p = st.text_input("Password", type="password")
            if st.button("Sign In"): login(e, p)
        with tab2:
            ne = st.text_input("New Email")
            np = st.text_input("New Password", type="password")
            if st.button("Register"): register(ne, np)
    st.stop()

# --- MAIN DASHBOARD ---
with st.sidebar:
    st.write(f"👤 **{st.session_state.user_email}**")
    if st.button("Logout"): logout()
    st.divider()

st.title("🚀 Resume Matcher Pro")
tab1, tab2 = st.tabs(["🔍 Find Jobs", "📜 History"])

with tab1:
    col_search, col_results = st.columns([1, 2])
    
    # Search Function
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
                with httpx.Client(trust_env=False) as client:
                    res = client.post(f"{BACKEND_URL}/search-jobs", json=payload)
                    if res.status_code == 200:
                        st.session_state.jobs = res.json()
                    else:
                        st.error("Search failed.")
            except Exception as e: st.error(f"Error: {e}")

    with col_search:
        st.subheader("Filters")
        if 'search_title' not in st.session_state: st.session_state.search_title = "Python Developer"
        if 'search_location' not in st.session_state: st.session_state.search_location = ""
        if 'search_employment' not in st.session_state: st.session_state.search_employment = "Any"
        if 'search_remote' not in st.session_state: st.session_state.search_remote = "Any"

        st.session_state.search_title = st.text_input("Job Title", st.session_state.search_title)
        st.session_state.search_location = st.text_input("City / Country", st.session_state.search_location)
        
        # NEW: Filter Columns
        c1, c2 = st.columns(2)
        with c1:
            st.session_state.search_employment = st.selectbox("Type", ["Any", "Full-time", "Part-time", "Contract", "Internship", "Freelance"])
        with c2:
            st.session_state.search_remote = st.selectbox("Work Style", ["Any", "Remote", "Hybrid", "On-site"])

        if st.button("Find Jobs", type="primary"):
            st.session_state.page = 0
            perform_search()

    with col_results:
        if 'jobs' in st.session_state:
            st.subheader(f"Results (Page {st.session_state.page + 1})")
            for job in st.session_state.jobs:
                with st.expander(f"💼 {job.get('title','Untitled')} @ {job.get('company','Unknown')}"):
                    # Vertical layout with clear sections: Job info -> Resume uploader -> Analyze -> Job description
                    st.markdown(f"**Location:** {job.get('location','N/A')}")
                    short_desc = (job.get('description') or "").strip()
                    if short_desc:
                        preview = short_desc[:300] + ("..." if len(short_desc) > 300 else "")
                        st.write(preview)
                    else:
                        st.info("No job description available.")

                    # Section: Resume upload
                    st.markdown("---")
                    st.subheader("Resume")
                    file_key = f"file_{job.get('id')}"
                    uploaded = st.file_uploader("Drag and drop or browse to upload (PDF)", type=["pdf"], key=file_key)

                    # Section: Analyze button
                    st.markdown("---")
                    analyze_key = f"btn_{job.get('id')}"
                    if st.button("Analyze", key=analyze_key):
                        if not uploaded:
                            st.warning("Please upload a resume PDF before analyzing.")
                        else:
                            with st.spinner("Analyzing..."):
                                headers = {"Authorization": f"Bearer {st.session_state.token}"}
                                files = {"file": (uploaded.name, uploaded, "application/pdf")}
                                job_description = job.get('description') or ""
                                data = {"job_description": job_description, "job_skills": ",".join(job.get('skills', []))}
                                try:
                                    with httpx.Client(trust_env=False) as client:
                                        res = client.post(f"{BACKEND_URL}/match-resume", data=data, files=files, headers=headers, timeout=30)
                                    if res.status_code == 200:
                                        result = res.json()
                                        score = result.get('match_percentage', 0)
                                        st.divider()
                                        if score >= 75:
                                            st.success(f"### 🚀 Match: {score}%")
                                        elif score >= 50:
                                            st.warning(f"### ⚠️ Match: {score}%")
                                        else:
                                            st.error(f"### ❌ Match: {score}%")

                                        st.divider()
                                        # Skills display removed per user request.
                                        with st.expander("📄 Full Job Description"):
                                            st.markdown(job.get('description') or "No description provided.")
                                    else:
                                        st.error("Analysis failed.")
                                except Exception as e:
                                    st.error(f"Error: {e}")

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
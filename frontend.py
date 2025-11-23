import streamlit as st
import httpx
import pandas as pd
import os

# CONFIG: Ensure this matches the port your Backend is running on!
BACKEND_URL = os.getenv("BACKEND_URL", "http://127.0.0.1:8002")
st.set_page_config(page_title="Resume Matcher Pro", page_icon="💼", layout="wide")

# --- SESSION STATE INIT ---
if 'token' not in st.session_state:
    st.session_state.token = None
if 'user_email' not in st.session_state:
    st.session_state.user_email = None

# --- AUTH FUNCTIONS ---
def login(email, password):
    try:
        # trust_env=False bypasses Nginx/Proxies
        with httpx.Client(trust_env=False) as client:
            res = client.post(f"{BACKEND_URL}/login", data={"username": email, "password": password})
            if res.status_code == 200:
                data = res.json()
                st.session_state.token = data['access_token']
                st.session_state.user_email = email
                st.rerun()
            else:
                st.error("Invalid email or password")
    except Exception as e:
        st.error(f"Connection Error: {e}")

def register(email, password):
    try:
        with httpx.Client(trust_env=False) as client:
            res = client.post(f"{BACKEND_URL}/register", json={"email": email, "password": password})
            if res.status_code == 200:
                st.success("Account created! Please log in.")
            else:
                st.error("Registration failed (Email might exist).")
    except Exception as e:
        st.error(f"Connection Error: {e}")

def logout():
    st.session_state.token = None
    st.session_state.user_email = None
    st.rerun()

# --- UI: LOGIN SCREEN ---
if not st.session_state.token:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("🔒 Login to Resume Matcher")
        tab_login, tab_signup = st.tabs(["Login", "Create Account"])
        
        with tab_login:
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            if st.button("Sign In"):
                login(email, password)
        
        with tab_signup:
            new_email = st.text_input("New Email")
            new_pass = st.text_input("New Password", type="password")
            if st.button("Register"):
                register(new_email, new_pass)
    st.stop() # Stop here if not logged in

# --- UI: MAIN DASHBOARD ---
# Sidebar
with st.sidebar:
    st.write(f"👤 **{st.session_state.user_email}**")
    if st.button("Logout"):
        logout()
    st.divider()
    st.markdown("### 🧭 Navigation")
    
st.title("🚀 Resume Matcher Pro")

tab1, tab2 = st.tabs(["🔍 Find Jobs", "📜 History"])

# TAB 1: SEARCH
with tab1:
    col_search, col_results = st.columns([1, 2])
    with col_search:
        st.subheader("Search Parameters")
        job_title = st.text_input("Job Title", "Python Developer")
        location = st.text_input("Location", "Remote")
        
        if st.button("Find Jobs", type="primary"):
            with st.spinner("Searching..."):
                with httpx.Client(trust_env=False) as client:
                    res = client.post(f"{BACKEND_URL}/search-jobs", json={"title": job_title, "location": location})
                    if res.status_code == 200:
                        st.session_state.jobs = res.json()
                    else:
                        st.error("Search failed.")

    with col_results:
        if 'jobs' in st.session_state:
            st.subheader("Results")
            for job in st.session_state.jobs:
                with st.expander(f"💼 {job['title']} @ {job['company']}"):
                    st.markdown(f"**Location:** {job['location']}")
                    # Show partial description initially
                    st.write(job['description'][:200] + "...")
                    
                    uploaded_file = st.file_uploader("Upload Resume", type=["pdf"], key=job['id'])
                    
                    if uploaded_file and st.button(f"Analyze Match", key=f"btn_{job['id']}"):
                        with st.spinner("Analyzing..."):
                            headers = {"Authorization": f"Bearer {st.session_state.token}"}
                            files = {"file": (uploaded_file.name, uploaded_file, "application/pdf")}
                            data = {"job_description": job['description'], "job_skills": ",".join(job['skills'])}
                            
                            try:
                                with httpx.Client(trust_env=False) as client:
                                    res = client.post(
                                        f"{BACKEND_URL}/match-resume", 
                                        data=data, 
                                        files=files, 
                                        headers=headers, 
                                        timeout=30
                                    )
                                
                                if res.status_code == 200:
                                    result = res.json()
                                    score = result['match_percentage']
                                    
                                    # --- 1. VISUAL SCORECARD ---
                                    st.divider()
                                    if score >= 75:
                                        st.success(f"### 🚀 Match Score: {score}%")
                                    elif score >= 50:
                                        st.warning(f"### ⚠️ Match Score: {score}%")
                                    else:
                                        st.error(f"### ❌ Match Score: {score}%")
                                    
                                    # --- 2. SKILLS COMPARISON (Side-by-Side) ---
                                    st.subheader("Skill Gap Analysis")
                                    c1, c2 = st.columns(2)
                                    
                                    with c1:
                                        st.success("✅ **Skills You Have**")
                                        if result['matched_skills']:
                                            for s in result['matched_skills']:
                                                st.write(f"- {s}")
                                        else:
                                            st.write("*(No specific skills matched)*")

                                    with c2:
                                        st.error("❌ **Skills Missing**")
                                        if result['missing_skills']:
                                            for s in result['missing_skills']:
                                                st.write(f"- {s}")
                                        else:
                                            st.write("*(No missing skills detected)*")
                                    
                                    # --- 3. FULL DESCRIPTION (Expandable) ---
                                    st.divider()
                                    with st.expander("📄 Click to Read Full Job Description"):
                                        st.markdown(job['description'])
                                        
                                else:
                                    st.error(f"Analysis failed. Backend said: {res.text}")
                            except Exception as e:
                                st.error(f"Error: {e}")

# TAB 2: HISTORY
with tab2:
    if st.button("Refresh History"):
        headers = {"Authorization": f"Bearer {st.session_state.token}"}
        try:
            with httpx.Client(trust_env=False) as client:
                res = client.get(f"{BACKEND_URL}/history", headers=headers)
                if res.status_code == 200:
                    df = pd.DataFrame(res.json())
                    if not df.empty:
                        # Clean up columns for display
                        display_df = df[['job_title', 'match_score', 'match_date']].copy()
                        display_df.columns = ['Job Title', 'Score (%)', 'Date']
                        st.dataframe(display_df, use_container_width=True)
                    else:
                        st.info("No history yet.")
                else:
                    st.error("Could not fetch history.")
        except Exception as e:
             st.error(f"Connection Error: {e}")

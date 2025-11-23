# app/job_search.py
import requests
import os
from dotenv import load_dotenv

load_dotenv()

class TheirStackEngine:
    def __init__(self):
        self.api_key = os.getenv("THEIRSTACK_API_KEY")
        self.base_url = "https://api.theirstack.com/v1/jobs/search"

    def search_jobs(self, job_title: str, location: str = "remote", limit: int = 5):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # TheirStack uses a rich filter payload
        payload = {
            "job_title_or": [job_title],
            "job_location_pattern_or": [location],
            "posted_at_max_age_days": 30, # Only fresh jobs
            "limit": limit,
            "include_total_results": False
        }

        try:
            response = requests.post(self.base_url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            # Transform TheirStack's response to a simple format for your Frontend
            jobs = []
            for item in data.get('data', []):
                jobs.append({
                    "id": item.get("id"),
                    "title": item.get("job_title"),
                    "company": item.get("company_object", {}).get("name"),
                    "location": item.get("job_location"),
                    "description": item.get("description"), # The full text for NLP
                    "skills_detected": item.get("technologies", []) # TheirStack's superpower!
                })
            return jobs

        except Exception as e:
            print(f"TheirStack API Error: {e}")
            return []

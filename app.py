import streamlit as st
import requests
import json
import os
import re
import textwrap
import sqlite3
from transformers import pipeline
from pathlib import Path

# --- Page Configuration (do this first) ---
st.set_page_config(page_title="🩺 Medical Safety Assistant", layout="wide", initial_sidebar_state="collapsed")

# --- NEW: Function to download large files from Hugging Face Hub ---
def download_file_from_hf(repo_id, filename, dest_path="."):
    # ... (this function stays exactly the same) ...
    local_path = Path(dest_path) / filename
    if local_path.exists():
        return str(local_path)
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    st.info(f"Downloading {filename} from Hugging Face Hub... (this happens once)")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(local_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        st.success(f"✅ Downloaded {filename}.")
        return str(local_path)
    except Exception as e:
        st.error(f"Failed to download {filename}. Error: {e}")
        return None

# --- NEW: Define your data repository and download files ---

DATA_REPO_ID = "FassikaF/medical-safety-app-data" 
DB_FILENAME = "ddi_database.db"
NDC_FILENAME = "drug_names.txt"

db_path = download_file_from_hf(DATA_REPO_ID, DB_FILENAME)
ndc_path = download_file_from_hf(DATA_REPO_ID, NDC_FILENAME)


# --- Model & Data Loading (with Caching) ---
# Use Streamlit's caching to load models only once
@st.cache_resource
def load_ner_model():
    try:
        # No need for st.info here, the download function already does it
        ner_pipeline = pipeline(
            "ner",
            model="d4data/biomedical-ner-all",
            aggregation_strategy="simple"
        )
        st.success("✅ NER model loaded.")
        return ner_pipeline
    except Exception as e:
        st.error(f"Fatal: Could not load NER model. Error: {e}")
        return None

@st.cache_resource
def load_drug_names_from_txt(filepath):
    if not filepath or not os.path.exists(filepath):
        st.warning("Drug name file not available.")
        return set()
    
    ndc_set = set()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                ndc_set.add(line.strip())
        st.success(f"✅ Loaded {len(ndc_set)} unique drug names.")
        return ndc_set
    except Exception as e:
        st.error(f"Failed to load or parse drug name file: {e}")
        return set()

# Load the resources using the downloaded paths
ner_pipeline = load_ner_model()
ndc_drug_names = load_drug_names_from_txt(ndc_path) # Pass the path here

# --- Helper Functions (Update the database query) ---
def query_ddi_database(drug1: str, drug2: str):
    # This function now uses the downloaded db_path
    if not db_path or not os.path.exists(db_path):
        st.error("Database file not available for query.")
        return None
        
    conn = sqlite3.connect(db_path)
    # ... (rest of the function is the same)
    cursor = conn.cursor()
    query = "SELECT level FROM ddi_interactions WHERE (LOWER(drug1) = ? AND LOWER(drug2) = ?) OR (LOWER(drug1) = ? AND LOWER(drug2) = ?)"
    cursor.execute(query, (drug1.lower(), drug2.lower(), drug2.lower(), drug1.lower()))
    result = cursor.fetchone()
    conn.close()
    return {"level": result[0]} if result else None


# ... (The rest of your app.py file remains exactly the same) ...


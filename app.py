import streamlit as st
import requests
import json
import os
import re
import sqlite3
from transformers import pipeline
from pathlib import Path

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(
    page_title="🩺 Medical Safety Assistant",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- File Downloading Logic ---
def download_file_from_hf(repo_id: str, filename: str, dest_path: str = "."):
    """
    Downloads a file from a Hugging Face Hub dataset repository if it doesn't exist locally.
    """
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

# --- Data and Model Loading ---
DATA_REPO_ID = "FassikaF/medical-safety-app-data" 
DB_FILENAME = "ddi_database.db"
NDC_FILENAME = "drug_names.txt"

db_path = download_file_from_hf(DATA_REPO_ID, DB_FILENAME)
ndc_path = download_file_from_hf(DATA_REPO_ID, NDC_FILENAME)

@st.cache_resource
def load_ner_model():
    """Loads the lightweight NER model, cached for performance."""
    try:
        st.info("Loading NER model...")
        ner_pipeline = pipeline(
            "ner",
            model="dslim/bert-base-NER",
            aggregation_strategy="simple"
        )
        st.success("✅ NER model loaded.")
        return ner_pipeline
    except Exception as e:
        st.error(f"Fatal: Could not load NER model. Error: {e}")
        return None

@st.cache_resource
def load_drug_names_from_txt(filepath: str):
    """Loads drug names from a simple text file, cached for performance."""
    if not filepath or not os.path.exists(filepath):
        st.warning("Drug name file not available.")
        return set()
    
    drug_set = set()
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                drug_set.add(line.strip())
        st.success(f"✅ Loaded {len(drug_set)} unique drug names.")
        return drug_set
    except Exception as e:
        st.error(f"Failed to load or parse drug name file: {e}")
        return set()

ner_pipeline = load_ner_model()
ndc_drug_names = load_drug_names_from_txt(ndc_path)

# --- Core Logic Functions ---
def extract_terms(text: str):
    """Extracts medical terms using NER and a dictionary lookup."""
    if not text.strip():
        return []
    
    found_terms = set()
    lower_text = text.strip().lower()

    # Layer 1: Dictionary Lookup (Primary Method)
    for drug in ndc_drug_names:
        if re.search(r'\b' + re.escape(drug) + r'\b', lower_text):
            found_terms.add(drug)

    # Layer 2: NER Model (Secondary/Backup Method)
    if ner_pipeline:
        try:
            entities = ner_pipeline(text)
            for entity in entities:
                found_terms.add(entity['word'].strip().lower().replace("##", ""))
        except Exception:
            # Silently fail if NER has issues, as dictionary is primary
            pass
    
    return list(found_terms)

def query_ddi_database(drug1: str, drug2: str):
    """Queries the local SQLite database for drug interactions."""
    if not db_path or not os.path.exists(db_path):
        st.error("Database file not available for query.")
        return None
        
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    query = "SELECT level FROM ddi_interactions WHERE (LOWER(drug1) = ? AND LOWER(drug2) = ?) OR (LOWER(drug1) = ? AND LOWER(drug2) = ?)"
    cursor.execute(query, (drug1.lower(), drug2.lower(), drug2.lower(), drug1.lower()))
    result = cursor.fetchone()
    conn.close()
    return {"level": result[0]} if result else None

def get_llm_details_from_openrouter(drug1: str, drug2: str, level: str):
    """Calls the OpenRouter API to get a detailed analysis of the interaction."""
    api_key = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key:
        st.error("OpenRouter API key not found. Please set it in your Streamlit secrets.")
        return "Analysis unavailable: API key is missing."

    prompt = f"""
    You are a medical safety assistant. Provide a concise, factual explanation of the drug-drug interaction between {drug1} and {drug2}, which has a {level} interaction level.
    Include:
    - Mechanism: The pharmacokinetic or pharmacodynamic basis.
    - Side Effects: Specific adverse effects.
    - Management: Recommendations for monitoring or mitigation.
    - Confidence estimate (%) based on established evidence.
    Do not include disclaimers or closing remarks. If no reliable data, state 'Insufficient data for detailed analysis.' Keep it under 150 words.
    """
    
    try:
        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "kin/kimi-k2-free",
                "messages": [
                    {"role": "system", "content": "You are a helpful medical safety assistant."},
                    {"role": "user", "content": prompt}
                ]
            }
        )
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        st.error(f"API Error: {e}")
        return "Error retrieving detailed analysis from the API."

# --- Streamlit User Interface ---
st.title("🧠 Medical Safety Assistant")
st.markdown("Check for potential interactions. *This tool is for informational purposes only.*")

st.subheader("Enter Your Information")
col1, col2 = st.columns(2)
with col1:
    current_input = st.text_area("💊 Current Profile", height=150, placeholder="e.g., Fleroxacin")
with col2:
    new_input = st.text_area("➕ New Addition", height=150, placeholder="e.g., Fenoprofen")

if st.button("🔎 Analyze for Safety", use_container_width=True):
    if not current_input.strip() or not new_input.strip():
        st.warning("Please fill in both fields.")
    else:
        with st.spinner("Analyzing..."):
            current_terms = extract_terms(current_input)
            new_terms = extract_terms(new_input)

            st.markdown("---")
            st.subheader("🔬 Analysis Results")
            
            st.markdown(f"**Detected Terms (Current):** `{', '.join(current_terms) if current_terms else 'None'}`")
            st.markdown(f"**Detected Terms (New):** `{', '.join(new_terms) if new_terms else 'None'}`")
            st.markdown("---")

            if current_terms and new_terms:
                interaction_found = False
                for drug1 in current_terms:
                    for drug2 in new_terms:
                        ddi = query_ddi_database(drug1, drug2)
                        if ddi:
                            interaction_found = True
                            level = ddi['level'].lower()
                            st.markdown(f"#### Interaction Found: **{drug1.title()} & {drug2.title()}**")
                            st.markdown(f"**Risk Level:** `{level.upper()}`")
                            details = get_llm_details_from_openrouter(drug1, drug2, level)
                            st.markdown(details)
                
                if not interaction_found:
                    st.info("No specific interaction found in the database. A general analysis will now be performed using the first detected terms.")
                    details = get_llm_details_from_openrouter(current_terms[0], new_terms[0], "unknown")
                    st.markdown(details)
            else:
                st.error("Could not detect enough medical terms to perform an analysis.")
else:
    st.info("Enter your information and click the 'Analyze' button to see results.")

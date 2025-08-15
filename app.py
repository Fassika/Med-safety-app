import streamlit as st
import requests
import json
import os
import re
import sqlite3
from transformers import pipeline
from pathlib import Path
from thefuzz import process as fuzzy_process # Import the fuzzy matching library

# --- Page Configuration (Must be the first Streamlit command) ---
st.set_page_config(page_title="🩺 Medical Safety Assistant", layout="wide", initial_sidebar_state="collapsed")

# --- File Downloading Logic ---
def download_file_from_hf(repo_id: str, filename: str, dest_path: str = "."):
    local_path = Path(dest_path) / filename
    if local_path.exists():
        return str(local_path)
    url = f"https://huggingface.co/datasets/{repo_id}/resolve/main/{filename}"
    st.info(f"Downloading {filename}...")
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
DRUG_MAP_FILENAME = "drug_map.json"

db_path = download_file_from_hf(DATA_REPO_ID, DB_FILENAME)
ndc_path = download_file_from_hf(DATA_REPO_ID, NDC_FILENAME)
drug_map_path = download_file_from_hf(DATA_REPO_ID, DRUG_MAP_FILENAME)

@st.cache_resource
def load_ner_model():
    try:
        st.info("Loading Biomedical NER model...")
        ner_pipeline = pipeline("ner", model="d4data/biomedical-ner-all", aggregation_strategy="simple")
        st.success("✅ Biomedical NER model loaded.")
        return ner_pipeline
    except Exception as e:
        st.error(f"Fatal: Could not load NER model. Error: {e}")
        return None

@st.cache_resource
def load_all_drug_data(map_filepath: str, list_filepath: str):
    drug_map, drug_list = {}, []
    # Load the high-precision map
    if map_filepath and os.path.exists(map_filepath):
        try:
            with open(map_filepath, "r", encoding="utf-8") as f:
                drug_map = json.load(f)
            st.success(f"✅ Loaded drug map with {len(drug_map)} variations.")
        except Exception as e:
            st.error(f"Failed to load drug map file: {e}")
    
    # Load the full list for fuzzy matching
    if list_filepath and os.path.exists(list_filepath):
        try:
            with open(list_filepath, "r", encoding="utf-8") as f:
                drug_list = [line.strip() for line in f if line.strip()]
            st.success(f"✅ Loaded drug list with {len(drug_list)} names for fuzzy matching.")
        except Exception as e:
            st.error(f"Failed to load drug list file: {e}")
            
    return drug_map, drug_list

ner_pipeline = load_ner_model()
drug_map, drug_list = load_all_drug_data(drug_map_path, ndc_path)

# --- Core Logic Functions ---
def extract_terms(text: str):
    """
    Extracts medical terms using a robust three-layer defense system:
    1. High-precision canonical name map.
    2. High-recall biomedical NER model.
    3. Fuzzy string matching safety net.
    """
    if not text.strip():
        return []
    
    found_canonical_terms = set()
    
    # Pre-process text: lowercase and split into potential terms
    potential_terms = set(text.lower().replace(',', ' ').replace('.', ' ').split())

    # --- Layer 1: High-Precision Canonical Map ---
    words_to_check_further = set()
    for term in potential_terms:
        if term in drug_map:
            found_canonical_terms.add(drug_map[term])
        else:
            words_to_check_further.add(term)

    # --- Layer 2: High-Recall Biomedical NER ---
    # Run NER on the original, unprocessed text for best results
    if ner_pipeline:
        try:
            entities = ner_pipeline(text)
            for entity in entities:
                term = entity['word'].strip().lower().replace("##", "")
                # Check if the found term is a key in our map to get the canonical name
                if term in drug_map:
                    found_canonical_terms.add(drug_map[term])
                else: # Otherwise, add the term as is
                    found_canonical_terms.add(term)
        except Exception as e:
            st.warning(f"NER extraction had an issue: {e}")

    # --- Layer 3: Fuzzy Matching Safety Net ---
    if drug_list:
        # Check remaining words that weren't found in the map
        for term in words_to_check_further:
            # Find the best match from our full drug list
            # We use a high threshold (e.g., 90) to avoid incorrect matches
            best_match = fuzzy_process.extractOne(term, drug_list, score_cutoff=90)
            if best_match:
                # best_match is a tuple: ('matched_word', score)
                found_canonical_terms.add(best_match[0])
                
    return list(found_canonical_terms)

def query_ddi_database(drug1: str, drug2: str):
    # This function is unchanged
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
    # This function is unchanged
    api_key = st.secrets.get("OPENROUTER_API_KEY")
    if not api_key:
        st.error("OpenRouter API key not found.")
        return "Analysis unavailable: API key is missing."
    your_app_url = "https://fassikaf-med-safety-app-app-axpxqg.streamlit.app/"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json", "HTTP-Referer": your_app_url, "X-Title": "Medical Safety Assistant"}
    prompt = f"""
    You are a medical safety assistant providing a concise, factual explanation of the drug-drug interaction between {drug1} and {drug2}, which has a known {level} interaction level.
    Structure your response with these four sections:
    - Mechanism: The pharmacokinetic or pharmacodynamic basis.
    - Side Effects: Specific adverse effects of the interaction.
    - Management: Recommendations for monitoring or mitigation.
    - Confidence Level: A qualitative assessment (e.g., High, Moderate, Low) based on established evidence.
    Do not include any other disclaimers or closing remarks. If no reliable data exists, state 'Insufficient data for detailed analysis.'
    Do not leave any sentence unfinished.
    """
    json_payload = {"model": "nousresearch/nous-hermes-2-mixtral-8x7b-dpo", "max_tokens": 300, "messages": [{"role": "system", "content": "You are a helpful medical safety assistant."}, {"role": "user", "content": prompt}]}
    try:
        response = requests.post("https://openrouter.ai/api/v1/chat/completions", headers=headers, json=json_payload)
        response.raise_for_status() 
        result = response.json()
        return result['choices'][0]['message']['content'].strip()
    except requests.exceptions.RequestException as e:
        error_details = "No additional details in response."
        if e.response is not None:
            try:
                error_details = json.dumps(e.response.json(), indent=2)
            except json.JSONDecodeError:
                error_details = e.response.text
        st.error(f"API Error: {e}\n\nServer Response:\n```\n{error_details}\n```")
        return "Error retrieving detailed analysis from the API."

# --- Streamlit User Interface (unchanged) ---
st.title("🧠 Medical Safety Assistant")
st.markdown("Check for potential interactions. *This tool is for informational purposes only.*")
st.subheader("Enter Your Information")
col1, col2 = st.columns(2)
with col1:
    current_input = st.text_area("💊 Current Profile", height=150, placeholder="e.g., Warfarin, Coumadin")
with col2:
    new_input = st.text_area("➕ New Addition", height=150, placeholder="e.g., asprn, Tylenol")
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


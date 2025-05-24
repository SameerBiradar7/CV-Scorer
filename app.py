from flask import Flask, request, render_template, redirect, url_for
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging
import spacy
import re
from collections import Counter
import io # Needed for handling in-memory files

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    logging.error("spaCy model 'en_core_web_sm' not found. Please run: python -m spacy download en_core_web_sm")
    raise SystemExit("spaCy model 'en_core_web_sm' not found. Please download it using 'python -m spacy download en_core_web_sm'.")

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads' # Still used for clarity, though files processed in memory
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'pdf'}

# --- New: Common Skills List (Extend this list as needed for your domain) ---
COMMON_SKILLS_LIST = [
    'python', 'java', 'javascript', 'sql', 'mysql', 'mongodb', 'react', 'angular', 'vue',
    'flask', 'django', 'node.js', 'express.js', 'html', 'css', 'api', 'rest', 'git', 'github',
    'docker', 'kubernetes', 'aws', 'azure', 'google cloud', 'gcp', 'linux', 'unix',
    'agile', 'scrum', 'project management', 'data analysis', 'machine learning',
    'deep learning', 'nlp', 'communication', 'teamwork', 'leadership', 'problem-solving',
    'critical thinking', 'software development', 'web development', 'mobile development',
    'full-stack', 'backend', 'frontend', 'database management', 'cloud computing',
    'cybersecurity', 'devops', 'testing', 'qa', 'automation', 'ui/ux', 'design',
    'algorithms', 'data structures', 'object-oriented programming', 'oop', 'microservices'
]
COMMON_SKILLS_LIST = [skill.lower() for skill in COMMON_SKILLS_LIST]

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_file_obj):
    """
    Extracts text from a PDF file object (BytesIO).
    """
    try:
        pdf_file_obj.seek(0) # Go to the beginning of the file-like object
        reader = PyPDF2.PdfReader(pdf_file_obj)
        text = ''
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        logging.info(f"Successfully extracted text (first 200 chars): {text[:200]}...")
        return text
    except PyPDF2.errors.PdfReadError as e:
        logging.error(f"PyPDF2 Read Error: {str(e)} - The PDF might be corrupted or encrypted.")
        return ''
    except Exception as e:
        logging.error(f"Error extracting text from PDF: {str(e)}")
        return ''

def extract_personal_details(text):
    """
    Extracts name, email, and mobile number using regex and spaCy.
    """
    details = {
        'name': None,
        'email': None,
        'mobile_number': None
    }

    # Email extraction
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if email_match:
        details['email'] = email_match.group(0)

    # Mobile number extraction (basic for 10 digits, adjust regex for specific formats like +91)
    # This regex looks for 10 digits, optionally starting with +country_code (e.g., +91)
    mobile_match = re.search(r'(?:\+?\d{1,3}[-.\s]?)?(\d{10})\b', text)
    if mobile_match:
        details['mobile_number'] = mobile_match.group(1) # Extract just the 10 digits

    # Name extraction (more complex, relies on spaCy PERSON entity or initial capitalization)
    doc = nlp(text.split('\n')[0].strip()) # Focus on the first line for name
    for ent in doc.ents:
        if ent.label_ == 'PERSON' and len(ent.text.split()) >= 2: # Look for multi-word names
            details['name'] = ent.text
            break
    
    if not details['name']: # Fallback if spaCy didn't find it in the first line
        # Try to find a capitalized phrase at the beginning of the document
        initial_lines = text.split('\n')[:5] # Check first few lines
        for line in initial_lines:
            match = re.match(r'^[A-Z][a-z]+(?: [A-Z][a-z]+)+$', line.strip()) # Simple title-case pattern
            if match and len(match.group(0).split()) >= 2:
                details['name'] = match.group(0)
                break

    return details

def extract_skills_and_keywords(text):
    """
    Extracts key skills and relevant keywords using spaCy and a common skills list.
    """
    if not text:
        return []
    
    doc = nlp(text.lower())
    
    # Method 1: Extract skills from a predefined list (prioritized)
    found_skills_from_list = set()
    for skill in COMMON_SKILLS_LIST:
        if skill in text.lower():
            found_skills_from_list.add(skill)

    # Method 2: Extract general relevant keywords using spaCy's POS and NER
    general_keywords = []
    for token in doc:
        # Include nouns, verbs, and adjectives that are not stop words and are alphanumeric
        if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and not token.is_stop and token.is_alpha and len(token.lemma_) > 2:
            general_keywords.append(token.lemma_)
    
    # Include named entities like organizations, products, skills (if spaCy identifies them)
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT', 'SKILL', 'GPE', 'LOC', 'NORP', 'FAC']:
            general_keywords.append(ent.text.lower())
    
    # Combine and prioritize: start with found_skills_from_list, then add unique general keywords
    combined_keywords = list(found_skills_from_list)
    for kw in general_keywords:
        if kw not in combined_keywords and kw not in ['experience', 'management', 'development', 'system', 'inc', 'corp', 'llc']:
            combined_keywords.append(kw)
    
    # Get frequency to pick most common, but also ensure diversity
    keyword_freq = Counter(combined_keywords)
    # Return top 50 unique keywords/skills
    filtered_keywords = [kw for kw, freq in keyword_freq.most_common(50) if len(kw) > 2]
    
    logging.debug(f"Extracted skills and keywords: {filtered_keywords}")
    return filtered_keywords

def compute_score(resume_extracted_data, job_desc_text=''):
    """
    Computes similarity score based on the mode (resume only or resume + JD).
    """
    resume_raw_text = resume_extracted_data.get('raw_text', '')
    resume_skills_keywords = resume_extracted_data.get('skills_keywords', []) # Now named skills_keywords
    
    if not resume_raw_text:
        return 0.0, [], [], "No text extracted from resume for scoring."

    if not job_desc_text:
        # "Resume Only" Scoring: Based on count of extracted skills/keywords
        MAX_RELEVANT_SKILLS_FOR_SCORE = 20 # Target for 100% score based on skills/keywords
        
        score = (min(len(resume_skills_keywords), MAX_RELEVANT_SKILLS_FOR_SCORE) / MAX_RELEVANT_SKILLS_FOR_SCORE) * 100
        
        # Display top 20 skills/keywords from the resume itself
        matches_for_display = resume_skills_keywords[:20] 
        
        missing_keywords = [] # No missing keywords in 'resume only' mode
        
        logging.info(f"Resume-only score (based on intrinsic skills/keywords): {score:.2f}% (Found: {len(resume_skills_keywords)})")
        return round(score, 2), matches_for_display, missing_keywords, None
    
    else:
        # "Resume + Job Description" Scoring: Cosine similarity + keyword matching
        job_keywords = set(extract_skills_and_keywords(job_desc_text))
        
        # Determine matched and missing keywords based on job description vs resume's skills/keywords
        matches = [keyword for keyword in job_keywords if keyword in resume_skills_keywords]
        missing_keywords = [keyword for keyword in job_keywords if keyword not in resume_skills_keywords]

        documents = [resume_raw_text, job_desc_text]
        vectorizer = TfidfVectorizer(stop_words='english', min_df=1) 
        try:
            vectors = vectorizer.fit_transform(documents)
            if vectors.shape[0] < 2:
                logging.warning("Not enough vectors to compute similarity. One or both documents might be too short.")
                return 0.0, matches, missing_keywords, "Not enough textual content for similarity comparison."
            
            similarity = cosine_similarity(vectors[0:1], vectors[1:2])[0][0]
            score = similarity * 100
            logging.info(f"Resume + JD score calculated: {score:.2f}% (Matches: {len(matches)}/{len(job_keywords)})")
            return round(score, 2), matches, missing_keywords, None
        except ValueError as e:
            logging.error(f"Error during TF-IDF vectorization or cosine similarity: {e}")
            return 0.0, matches, missing_keywords, "An error occurred during text similarity calculation. Ensure both documents have sufficient text."

def get_ats_tips():
    return [
        "Tailor your resume to the job description by including relevant keywords.",
        "Use standard section headings like 'Work Experience', 'Education', and 'Skills'.",
        "Include exact matches for keywords (e.g., 'Agile methodology' instead of just 'Agile').",
        "Use a simple, text-based PDF format without headers, footers, or tables.",
        "Quantify achievements (e.g., 'Increased sales by 20%') and use action verbs like 'Developed' or 'Led'.",
        "Avoid keyword stuffing—use keywords naturally in context.",
        "Include relevant certifications and education (e.g., 'AWS Certified Solutions Architect').",
        "Ensure your resume is a text-based PDF, not a scanned image, for better parsing."
    ]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', 
                           resume_score=None, 
                           jd_score=None, 
                           error=None, 
                           matches=[], 
                           missing_keywords=[], 
                           ats_tips=get_ats_tips(),
                           person_name=None, # Added for display
                           person_email=None, # Added for display
                           person_mobile=None) # Added for display

@app.route('/score_resume_only', methods=['POST'])
def score_resume_only():
    resume_score = None
    error = None
    matches = []
    missing_keywords = []
    person_name = None
    person_email = None
    person_mobile = None
    
    if 'resume' not in request.files:
        error = 'No resume file part in the request. Please select a file.'
        return render_template('index.html', resume_score=None, jd_score=None, error=error, matches=[], missing_keywords=[], ats_tips=get_ats_tips(),
                               person_name=person_name, person_email=person_email, person_mobile=person_mobile)
    
    resume_file = request.files['resume']
    
    if resume_file.filename == '':
        error = 'No resume file selected. Please choose a PDF.'
        return render_template('index.html', resume_score=None, jd_score=None, error=error, matches=[], missing_keywords=[], ats_tips=get_ats_tips(),
                               person_name=person_name, person_email=person_email, person_mobile=person_mobile)

    if resume_file and allowed_file(resume_file.filename):
        # Read the file into BytesIO to process in memory
        resume_bytes_io = io.BytesIO(resume_file.read())
        
        try:
            resume_text = extract_text_from_pdf(resume_bytes_io)
            
            if resume_text:
                # Extract personal details and skills/keywords
                personal_details = extract_personal_details(resume_text)
                person_name = personal_details.get('name')
                person_email = personal_details.get('email')
                person_mobile = personal_details.get('mobile_number')

                skills_keywords = extract_skills_and_keywords(resume_text)

                # Prepare data for compute_score
                resume_extracted_data = {
                    'raw_text': resume_text,
                    'skills_keywords': skills_keywords,
                    'name': person_name,
                    'email': person_email,
                    'mobile_number': person_mobile
                }

                resume_score, matches, missing_keywords, score_error = compute_score(resume_extracted_data)
                if score_error:
                    error = score_error
            else:
                error = 'Could not extract text from resume. Ensure it is a text-based PDF, not a scanned image, or the PDF is corrupted.'
        except Exception as e:
            logging.error(f"Error processing resume-only file: {e}")
            error = f"An unexpected error occurred during file processing: {e}"
    else:
        error = 'Invalid file type. Please upload a **PDF** file for your resume.'

    return render_template('index.html', 
                           resume_score=resume_score, 
                           jd_score=None, 
                           error=error, 
                           matches=matches, 
                           missing_keywords=missing_keywords, 
                           ats_tips=get_ats_tips(),
                           person_name=person_name, 
                           person_email=person_email, 
                           person_mobile=person_mobile)

@app.route('/score_resume_jd', methods=['POST'])
def score_resume_jd():
    jd_score = None
    error = None
    matches = []
    missing_keywords = []
    person_name = None
    person_email = None
    person_mobile = None

    if 'resume_jd' not in request.files:
        error = 'No resume file part in the request for JD matching.'
        return render_template('index.html', resume_score=None, jd_score=None, error=error, matches=[], missing_keywords=[], ats_tips=get_ats_tips(),
                               person_name=person_name, person_email=person_email, person_mobile=person_mobile)
    
    resume_file_jd = request.files['resume_jd']
    job_desc = request.form.get('job_desc', '').strip()

    if resume_file_jd.filename == '':
        error = 'No resume file selected for JD matching. Please choose a PDF.'
        return render_template('index.html', resume_score=None, jd_score=None, error=error, matches=[], missing_keywords=[], ats_tips=get_ats_tips(),
                               person_name=person_name, person_email=person_email, person_mobile=person_mobile)
    
    if not job_desc:
        error = 'Job Description cannot be empty for JD matching. Please paste the job description text.'
        return render_template('index.html', resume_score=None, jd_score=None, error=error, matches=[], missing_keywords=[], ats_tips=get_ats_tips(),
                               person_name=person_name, person_email=person_email, person_mobile=person_mobile)

    if resume_file_jd and allowed_file(resume_file_jd.filename):
        # Read the file into BytesIO to process in memory
        resume_bytes_io_jd = io.BytesIO(resume_file_jd.read())

        try:
            resume_text = extract_text_from_pdf(resume_bytes_io_jd)
            
            if resume_text:
                # Extract personal details and skills/keywords
                personal_details = extract_personal_details(resume_text)
                person_name = personal_details.get('name')
                person_email = personal_details.get('email')
                person_mobile = personal_details.get('mobile_number')

                skills_keywords = extract_skills_and_keywords(resume_text)

                # Prepare data for compute_score
                resume_extracted_data = {
                    'raw_text': resume_text,
                    'skills_keywords': skills_keywords,
                    'name': person_name,
                    'email': person_email,
                    'mobile_number': person_mobile
                }

                jd_score, matches, missing_keywords, score_error = compute_score(resume_extracted_data, job_desc)
                if score_error:
                    error = score_error
            else:
                error = 'Could not extract text from resume for JD matching. Ensure it is a text-based PDF, not a scanned image, or the PDF is corrupted.'
        except Exception as e:
            logging.error(f"Error processing resume+JD file: {e}")
            error = f"An unexpected error occurred during file processing: {e}"
    else:
        error = 'Invalid file type. Please upload a **PDF** file for your resume (for JD matching).'

    return render_template('index.html', 
                           resume_score=None, 
                           jd_score=jd_score, 
                           error=error, 
                           matches=matches, 
                           missing_keywords=missing_keywords, 
                           ats_tips=get_ats_tips(),
                           person_name=person_name, 
                           person_email=person_email, 
                           person_mobile=person_mobile)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0',port=5000)
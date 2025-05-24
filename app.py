from flask import Flask, request, render_template, redirect, url_for
import PyPDF2
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import logging
import spacy
import re
from collections import Counter
import io

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

try:
    # --- THIS IS THE CRITICAL CHANGE FOR VERCEL SIZE LIMIT ---
    # Loading the much smaller 'en_core_web_nan' model
    nlp = spacy.load("en_core_web_nan") 
except OSError:
    logging.error("spaCy model 'en_core_web_nan' not found. Please run: python -m spacy download en_core_web_nan")
    raise SystemExit("spaCy model 'en_core_web_nan' not found. Please download it using 'python -m spacy download en_core_web_nan'.")

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

ALLOWED_EXTENSIONS = {'pdf'}

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
    try:
        pdf_file_obj.seek(0)
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
    details = {
        'name': None,
        'email': None,
        'mobile_number': None
    }

    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    if email_match:
        details['email'] = email_match.group(0)

    mobile_match = re.search(r'(?:\+?\d{1,3}[-.\s]?)?(\d{10})\b', text)
    if mobile_match:
        details['mobile_number'] = mobile_match.group(1)

    doc = nlp(text.split('\n')[0].strip())
    for ent in doc.ents:
        if ent.label_ == 'PERSON' and len(ent.text.split()) >= 2:
            details['name'] = ent.text
            break
    
    if not details['name']:
        initial_lines = text.split('\n')[:5]
        for line in initial_lines:
            match = re.match(r'^[A-Z][a-z]+(?: [A-Z][a-z]+)+$', line.strip())
            if match and len(match.group(0).split()) >= 2:
                details['name'] = match.group(0)
                break

    return details

def extract_skills_and_keywords(text):
    if not text:
        return []
    
    doc = nlp(text.lower())
    
    found_skills_from_list = set()
    for skill in COMMON_SKILLS_LIST:
        if skill in text.lower():
            found_skills_from_list.add(skill)

    general_keywords = []
    for token in doc:
        if token.pos_ in ['NOUN', 'VERB', 'ADJ'] and not token.is_stop and token.is_alpha and len(token.lemma_) > 2:
            general_keywords.append(token.lemma_)
    
    for ent in doc.ents:
        if ent.label_ in ['ORG', 'PRODUCT', 'SKILL', 'GPE', 'LOC', 'NORP', 'FAC']:
            general_keywords.append(ent.text.lower())
    
    combined_keywords = list(found_skills_from_list)
    for kw in general_keywords:
        if kw not in combined_keywords and kw not in ['experience', 'management', 'development', 'system', 'inc', 'corp', 'llc']:
            combined_keywords.append(kw)
    
    keyword_freq = Counter(combined_keywords)
    filtered_keywords = [kw for kw, freq in keyword_freq.most_common(50) if len(kw) > 2]
    
    logging.debug(f"Extracted skills and keywords: {filtered_keywords}")
    return filtered_keywords

def compute_score(resume_extracted_data, job_desc_text=''):
    resume_raw_text = resume_extracted_data.get('raw_text', '')
    resume_skills_keywords = resume_extracted_data.get('skills_keywords', [])
    
    if not resume_raw_text:
        return 0.0, [], [], "No text extracted from resume for scoring."

    if not job_desc_text:
        MAX_RELEVANT_SKILLS_FOR_SCORE = 20 
        
        score = (min(len(resume_skills_keywords), MAX_RELEVANT_SKILLS_FOR_SCORE) / MAX_RELEVANT_SKILLS_FOR_SCORE) * 100
        
        matches_for_display = resume_skills_keywords[:20] 
        
        missing_keywords = [] 
        
        logging.info(f"Resume-only score (based on intrinsic skills/keywords): {score:.2f}% (Found: {len(resume_skills_keywords)})")
        return round(score, 2), matches_for_display, missing_keywords, None
    
    else:
        job_keywords = set(extract_skills_and_keywords(job_desc_text))
        
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
                           person_name=None, 
                           person_email=None, 
                           person_mobile=None)

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
        resume_bytes_io = io.BytesIO(resume_file.read())
        
        try:
            resume_text = extract_text_from_pdf(resume_bytes_io)
            
            if resume_text:
                personal_details = extract_personal_details(resume_text)
                person_name = personal_details.get('name')
                person_email = personal_details.get('email')
                person_mobile = personal_details.get('mobile_number')

                skills_keywords = extract_skills_and_keywords(resume_text)

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
        resume_bytes_io_jd = io.BytesIO(resume_file_jd.read())

        try:
            resume_text = extract_text_from_pdf(resume_bytes_io_jd)
            
            if resume_text:
                personal_details = extract_personal_details(resume_text)
                person_name = personal_details.get('name')
                person_email = personal_details.get('email')
                person_mobile = personal_details.get('mobile_number')

                skills_keywords = extract_skills_and_keywords(resume_text)

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
    app.run(debug=False, host='0.0.0.0',port=5000)
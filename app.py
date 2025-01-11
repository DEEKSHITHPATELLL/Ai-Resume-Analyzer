import os
import io
import re
import json
import logging
import tempfile
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
import PyPDF2
import docx2txt
from docx import Document
from dotenv import load_dotenv
import google.generativeai as genai
import numpy as np
import pandas as pd
import shap

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Configure Google Generative AI
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

ALLOWED_EXTENSIONS = {'pdf', 'docx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(file_content):
    """Extract text from PDF using multiple methods including OCR"""
    try:
        logger.info(f"Starting PDF extraction, content size: {len(file_content)} bytes")
        
        # Try extracting with PyPDF2 first
        try:
            reader = PyPDF2.PdfReader(io.BytesIO(file_content))
            logger.info(f"PDF has {len(reader.pages)} pages")
            
            if reader.is_encrypted:
                logger.error("PDF is password protected")
                return None
                
            text_parts = []
            for i, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text and page_text.strip():
                        text_parts.append(page_text.strip())
                        logger.info(f"Page {i+1}: Extracted {len(page_text)} characters")
                    else:
                        logger.warning(f"Page {i+1}: No text extracted, might be scanned")
                except Exception as e:
                    logger.error(f"Error on page {i+1}: {str(e)}")
                    continue
            
            # If we got meaningful text, return it
            if text_parts:
                text = '\n'.join(text_parts)
                if len(text.strip()) > 50:
                    logger.info(f"Successfully extracted {len(text)} characters with PyPDF2")
                    return text
                    
            logger.info("PyPDF2 extraction yielded no meaningful text, trying OCR...")
            
        except Exception as e:
            logger.error(f"PyPDF2 extraction failed: {str(e)}")
            
        # Try OCR as fallback
        try:
            import pytesseract
            import fitz  # PyMuPDF
            from PIL import Image
            import os
            
            # Check for Tesseract installation
            tesseract_path = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
            if not os.path.exists(tesseract_path):
                logger.error(f"Tesseract not found at {tesseract_path}")
                logger.error("Please install from: https://github.com/UB-Mannheim/tesseract/wiki")
                return None
                
            # Set Tesseract path
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            
            # Try OCR using PyMuPDF
            try:
                # Load PDF
                pdf_document = fitz.open(stream=file_content, filetype="pdf")
                
                # Get first page
                page = pdf_document[0]
                
                # Convert to image
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # 2x zoom for better quality
                
                # Convert to PIL Image
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                
                # Convert to grayscale for better OCR
                img = img.convert('L')
                
                # Perform OCR
                text = pytesseract.image_to_string(img)
                
                if text and len(text.strip()) > 50:
                    logger.info(f"Successfully extracted {len(text)} characters with OCR")
                    return text.strip()
                    
                logger.warning("OCR yielded no meaningful text")
                
            except Exception as e:
                logger.error(f"Error during OCR: {str(e)}")
            
            logger.error("OCR extraction failed")
            return None
            
        except ImportError as e:
            logger.error(f"OCR dependencies not installed: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"OCR extraction failed: {str(e)}")
            return None
            
    except Exception as e:
        logger.error(f"PDF extraction failed: {str(e)}")
        return None

def extract_text_from_docx(file_content):
    """Extract text from DOCX using multiple methods"""
    try:
        logger.info(f"Starting DOCX extraction, content size: {len(file_content)} bytes")

        # Create temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.docx') as temp_file:
            temp_file.write(file_content)
            temp_file.flush()
            temp_path = temp_file.name
            
        try:
            # Try docx2txt first
            try:
                text = docx2txt.process(temp_path)
                if text and len(text.strip()) > 50:
                    logger.info(f"Successfully extracted {len(text)} characters with docx2txt")
                    return text.strip()
                else:
                    logger.warning("docx2txt extraction yielded no meaningful text")
            except Exception as e:
                logger.error(f"docx2txt extraction failed: {str(e)}")
            
            # Try python-docx as fallback
            doc = Document(temp_path)
            text_parts = []
            
            # Extract from paragraphs
            for para in doc.paragraphs:
                if para.text and para.text.strip():
                    text_parts.append(para.text.strip())
                    
            # Extract from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        if cell.text and cell.text.strip():
                            text_parts.append(cell.text.strip())
            
            if text_parts:
                text = '\n'.join(text_parts)
                if len(text.strip()) > 50:
                    logger.info(f"Successfully extracted {len(text)} characters with python-docx")
                    return text
                    
            logger.error("No meaningful text extracted from DOCX")
            return None
            
        finally:
            try:
                os.unlink(temp_path)
            except Exception as e:
                logger.error(f"Error deleting temp file: {str(e)}")
                
    except Exception as e:
        logger.error(f"DOCX extraction failed: {str(e)}")
        return None

def analyze_resume_with_ai(text):
    """Analyze resume text using Google's Generative AI"""
    try:
        model = genai.GenerativeModel('gemini-pro')
        
        # First, extract skills and basic info
        skills_prompt = f"""Extract skills, experience, and education from this resume text. Format your response EXACTLY as JSON:
{{
    "skills": ["list", "of", "skills"],
    "experience": "summary of experience",
    "education": "summary of education"
}}

Resume text:
{text}"""

        skills_response = model.generate_content(skills_prompt)
        
        try:
            extracted_info = json.loads(skills_response.text)
        except:
            logger.error("Failed to parse skills response")
            extracted_info = {
                "skills": [],
                "experience": "",
                "education": ""
            }

        # Then, get career guidance and job opportunities
        guidance_prompt = f"""Based on these skills and experience, provide career guidance and job opportunities. Format your response EXACTLY as JSON:
{{
    "career_paths": [
        {{
            "role": "Software Engineer",
            "description": "Role description",
            "growth_potential": "High - explanation",
            "salary_range": "$XX,XXX - $YY,XXX"
        }}
    ],
    "recommendations": [
        "Specific recommendation 1",
        "Specific recommendation 2"
    ],
    "job_opportunities": [
        {{
            "title": "Job Title",
            "description": "Brief job description",
            "required_skills": ["skill1", "skill2"],
            "matching_skills": ["skill1", "skill2"],
            "missing_skills": ["skill3"],
            "typical_salary": "$XX,XXX - $YY,XXX",
            "demand_level": "High/Medium/Low",
            "companies": ["Company1", "Company2"]
        }}
    ],
    "market_insights": {{
        "industry_demand": "Description of current market demand",
        "emerging_skills": ["skill1", "skill2"],
        "salary_trends": "Description of salary trends",
        "top_locations": ["Location1", "Location2"]
    }}
}}

Skills and Experience:
{json.dumps(extracted_info, indent=2)}"""

        guidance_response = model.generate_content(guidance_prompt)
        
        try:
            career_guidance = json.loads(guidance_response.text)
        except:
            logger.error("Failed to parse career guidance")
            career_guidance = {
                "career_paths": [],
                "recommendations": [],
                "job_opportunities": [],
                "market_insights": {
                    "industry_demand": "",
                    "emerging_skills": [],
                    "salary_trends": "",
                    "top_locations": []
                }
            }

        # Calculate feature importance
        skills = extracted_info.get("skills", [])
        
        # Categorize skills
        technical_skills = [s for s in skills if any(tech in s.lower() for tech in [
            "python", "java", "javascript", "html", "css", "sql", "react", "angular",
            "node", "aws", "azure", "docker", "kubernetes", "git", "api", "programming",
            "development", "software", "web", "database", "cloud", "ai", "ml"
        ])]
        
        soft_skills = [s for s in skills if any(soft in s.lower() for soft in [
            "communication", "leadership", "management", "team", "project", "agile",
            "scrum", "problem solving", "analytical", "organization", "planning"
        ])]
        
        domain_skills = [s for s in skills if s not in technical_skills and s not in soft_skills]

        # Calculate scores
        tech_score = min(len(technical_skills) * 0.2, 1.0)
        soft_score = min(len(soft_skills) * 0.2, 1.0)
        domain_score = min(len(domain_skills) * 0.2, 1.0)
        experience_score = 0.5 if extracted_info.get("experience") else 0.0
        education_score = 0.5 if extracted_info.get("education") else 0.0

        feature_importance = {
            "Technical Skills": tech_score,
            "Soft Skills": soft_score,
            "Domain Knowledge": domain_score,
            "Experience": experience_score,
            "Education": education_score
        }

        # Calculate overall score
        score = sum(feature_importance.values()) / len(feature_importance)

        # Add skill match percentages to job opportunities
        if "job_opportunities" in career_guidance:
            for job in career_guidance["job_opportunities"]:
                required_skills = set(s.lower() for s in job.get("required_skills", []))
                if required_skills:
                    user_skills = set(s.lower() for s in skills)
                    matching = required_skills.intersection(user_skills)
                    job["match_percentage"] = round(len(matching) / len(required_skills) * 100)
                else:
                    job["match_percentage"] = 0

        # Prepare final analysis
        analysis = {
            "extracted_info": extracted_info,
            "career_guidance": career_guidance,
            "feature_importance": feature_importance,
            "score": score
        }

        logger.info(f"Extracted {len(skills)} skills")
        logger.info(f"Technical skills: {len(technical_skills)}")
        logger.info(f"Soft skills: {len(soft_skills)}")
        logger.info(f"Domain skills: {len(domain_skills)}")
        
        return analysis
            
    except Exception as e:
        logger.error(f"Error in AI analysis: {str(e)}")
        return create_default_analysis()

def create_default_analysis():
    """Create a default analysis when AI fails"""
    return {
        "extracted_info": {
            "skills": [],
            "experience": "Could not extract experience information",
            "education": "Could not extract education information"
        },
        "career_guidance": {
            "career_paths": [
                {
                    "role": "Career Analysis Unavailable",
                    "description": "We encountered an error analyzing your resume. Please try again later.",
                    "growth_potential": "Unknown",
                    "salary_range": "Unknown"
                }
            ],
            "recommendations": [
                "Please try uploading your resume again",
                "Make sure your resume is in PDF or DOCX format",
                "Ensure your resume is not password protected"
            ],
            "job_opportunities": [],
            "market_insights": {
                "industry_demand": "Information not available",
                "emerging_skills": [],
                "salary_trends": "Information not available",
                "top_locations": []
            }
        },
        "feature_importance": {
            "Technical Skills": 0.0,
            "Soft Skills": 0.0,
            "Domain Knowledge": 0.0,
            "Experience": 0.0,
            "Education": 0.0
        },
        "score": 0.0
    }

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        if 'resume' not in request.files:
            logger.error("No resume file in request")
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['resume']
        if not file or file.filename == '':
            logger.error("Empty filename")
            return jsonify({'error': 'No file selected'}), 400

        filename = secure_filename(file.filename)
        if not allowed_file(filename):
            logger.error(f"Invalid file type: {filename}")
            return jsonify({'error': 'Invalid file type. Please upload PDF or DOCX only.'}), 400

        # Read file content
        file_content = file.read()
        if not file_content:
            logger.error("Empty file content")
            return jsonify({'error': 'Empty file uploaded'}), 400
            
        logger.info(f"Processing file: {filename} ({len(file_content)} bytes)")

        # Extract text based on file type
        file_extension = filename.rsplit('.', 1)[1].lower()
        if file_extension == 'pdf':
            text = extract_text_from_pdf(file_content)
        else:
            text = extract_text_from_docx(file_content)

        if not text:
            logger.error("Text extraction failed")
            return jsonify({
                'error': 'Could not extract text from file. Please ensure:\n' +
                        '1. The file is not corrupted\n' +
                        '2. The file is not password protected\n' +
                        '3. The file contains actual text (not scanned images)\n' +
                        '4. The file is in the correct format (PDF or DOCX)',
                'debug_info': {
                    'filename': filename,
                    'file_size': len(file_content),
                    'file_type': file_extension
                }
            }), 400

        # Log first few characters for debugging
        preview = text[:200].replace('\n', ' ')
        logger.info(f"Extracted text preview: {preview}...")

        # Analyze with AI
        analysis = analyze_resume_with_ai(text)
        if not analysis:
            return jsonify({'error': 'Failed to analyze resume content'}), 500

        # Prepare response
        response = {
            'score': float(analysis['score']),
            'feature_importance': analysis['feature_importance'],
            'extracted_info': analysis['extracted_info'],
            'career_guidance': analysis['career_guidance']
        }

        logger.info("Analysis completed successfully")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error during analysis: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        # Check dependencies before starting server
        try:
            import pytesseract
            if not os.path.exists("C:\\Program Files\\Tesseract-OCR\\tesseract.exe"):
                logger.warning("Tesseract OCR not found - OCR functionality will be disabled")
            else:
                logger.info("Tesseract OCR found")
                
            # Set Tesseract path
            pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
        except ImportError:
            logger.warning("pytesseract not installed - OCR functionality will be disabled")
            
        # Initialize Gemini
        try:
            genai.configure(api_key=os.getenv('GOOGLE_API_KEY'))
            logger.info("Gemini API initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {str(e)}")
            raise
            
        # Try different ports if default is busy
        ports = [5000, 8000, 8080, 3000]
        for port in ports:
            try:
                logger.info(f"Attempting to start server on port {port}...")
                app.run(host='127.0.0.1', port=port, debug=True, use_reloader=False)
                break
            except OSError as e:
                if port == ports[-1]:  # Last port attempt
                    logger.error(f"Could not bind to any port. Last error: {str(e)}")
                    raise
                else:
                    logger.warning(f"Port {port} is busy, trying next port...")
                    continue
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise

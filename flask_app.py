from flask import Flask, render_template, request, jsonify, send_file, session
import numpy as np
import speech_recognition as sr
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import requests
from together import Together
import json
import re
from dotenv import load_dotenv
import os
import tempfile
from datetime import datetime
import threading
from gtts import gTTS
import uuid
import base64
import io

load_dotenv()

app = Flask(__name__)
app.secret_key = os.urandom(24)

# Initialize Together client
your_api_key = os.getenv("TOGETHER_API_KEY")
client = Together(api_key=your_api_key)

def prompt_llm(prompt, show_cost=False):
    """Function to send prompt to an LLM via the Together API."""
    model = "meta-llama/Meta-Llama-3-8B-Instruct-Lite"
    tokens = len(prompt.split())

    if show_cost:
        print(f"\nNumber of tokens: {tokens}")
        cost = (0.1 / 1_000_000) * tokens
        print(f"Estimated cost for {model}: ${cost:.10f}\n")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
        )
        content = response.choices[0].message.content
        
        # Safety check for empty responses
        if not content or len(content.strip()) < 10:
            print(f"Warning: LLM returned empty or very short response: '{content}'")
            return "The LLM response was too short or empty. Please try again with more detailed input."
        return content
    except Exception as e:
        print(f"Error calling LLM API: {str(e)}")
        return "An error occurred while generating content. Please check your API key and try again."

class Analyzer:
    def __init__(self):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def parse_job_info(self, job_description, company_values):
        """Extracts key insights and fills relevant fields."""
        prompt = f"""
        SYSTEM: You are an expert career coach and interviewer with over 30 years of experience in the tech industry. Your task is to thoroughly analyze the job description and company values to extract and classify all relevant information.

        INSTRUCTIONS:
        1. First, identify the company name and position title.
        2. Then analyze the entire text for any skills, requirements, or values, looking for both explicit and implicit mentions.
        3. Classify all found information into these categories:

        Company Name:
        - Extract the company name from the text
        - If not explicitly stated, write "Company name not specified"

        Position Title:
        - Extract the job title/position from the text
        - It contains no additional adjectives or adverbs, only the job title
        - If not explicitly stated, write "Position title not specified"
        - Example: From "NetNation is seeking Junior to Mid-Range UX/UI Software Developers"
          Extract only: "UX/UI Software Developer"
          (Remove level indicators like "Junior to Mid-Range")

        Company Values:
        - Look for mentions of culture, principles, mission, values
        - Include both explicit values and implied ones from the company's description
        - Examples: integrity, innovation, customer focus, diversity, sustainability

        Technical Skills:
        - All technical requirements, tools, languages, platforms
        - Both required and preferred technical qualifications
        - Include domain-specific technical knowledge
        - Examples: programming languages, frameworks, methodologies, systems

        Soft Skills:
        - All interpersonal and professional skills
        - Leadership and management capabilities
        - Personal qualities and attributes
        - Examples: communication, leadership, problem-solving, teamwork

        Job Duties:
        - Primary responsibilities and expectations
        - Day-to-day tasks and long-term objectives
        - Project responsibilities and deliverables
        - Team and organizational contributions

        FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:
        **Company Name:**
        [company name]

        **Position Title:**
        [position title]
        
        **Key Company Values:**
        - [value 1]
        - [value 2]
        
        **Essential Technical Skills:**
        - [skill 1]
        - [skill 2]
        
        **Necessary Soft Skills:**
        - [skill 1]
        - [skill 2]
        
        **Summary of Key Job Duties:**
        - [duty 1]
        - [duty 2]

        Keep each bullet point concise (under 10 words).

        JOB DESCRIPTION: {job_description}
        COMPANY VALUES: {company_values}
        """
        response = prompt_llm(prompt)
        
        # Add new regex patterns for company name and position title
        company_name_match = re.search(r"\*\*Company Name:\*\*(.*?)\*\*Position Title:\*\*", response, re.DOTALL)
        position_title_match = re.search(r"\*\*Position Title:\*\*(.*?)\*\*Key Company Values:\*\*", response, re.DOTALL)
        company_values_match = re.search(r"\*\*Key Company Values:\*\*(.*?)\*\*Essential Technical Skills:\*\*", response, re.DOTALL)
        tech_skills_match = re.search(r"\*\*Essential Technical Skills:\*\*(.*?)\*\*Necessary Soft Skills:\*\*", response, re.DOTALL)
        soft_skills_match = re.search(r"\*\*Necessary Soft Skills:\*\*(.*?)\*\*Summary of Key Job Duties:\*\*", response, re.DOTALL)
        job_duties_match = re.search(r"\*\*Summary of Key Job Duties:\*\*(.*)", response, re.DOTALL)

        company_name = company_name_match.group(1).strip() if company_name_match else "Company name not specified"
        position_title = position_title_match.group(1).strip() if position_title_match else "Position title not specified"
        company_values = company_values_match.group(1).strip() if company_values_match else "Not found"
        tech_skills = tech_skills_match.group(1).strip() if tech_skills_match else "Not found"
        soft_skills = soft_skills_match.group(1).strip() if soft_skills_match else "Not found"
        job_duties = job_duties_match.group(1).strip() if job_duties_match else "Not found"

        parsed_info = {
            "company_name": company_name,
            "position_title": position_title,
            "company_values": company_values,
            "tech_skills": tech_skills,
            "soft_skills": soft_skills,
            "job_duties": job_duties
        }
        return parsed_info

class Drafter:
    def generate_answer(self, question, company_info, job_description, resume, voice_answer):
        """Drafts a model answer based on user inputs."""
        prompt = f"""
        SYSTEM: You are a professional interview coach and writer with over 30 years of experience in the tech industry. Draft a strong, structured answer to get this user hired by a top tech company, based on the following inputs:
        
        INSTRUCTIONS:
        - Ensure clarity and logical flow.
        - Incorporate company values where relevant.
        - Highlight technical and soft skills from the job description.
        - Use Amazon Leadership Principles to guide the answer.
        - Use user's voice answer, experience and skills in the resume to answer the question.
        - Use the situation, task, action, and result (STAR) method to structure the answer.
        - Improve conciseness while maintaining completeness.
        - Maintain a confident and positive tone.
        - Keep the answer in 90 seconds to 2 minutes long.
        - If possible, use the same language as the user's voice answer.
        - If there is no information, just output "Not found".
        
        QUESTION: {question}
        COMPANY INFO: {company_info}
        JOB DESCRIPTION: {job_description}
        USER RESUME: {resume}
        USER VOICE ANSWER: {voice_answer}
        """
        return prompt_llm(prompt)

class Evaluator:
    def evaluate_answer(self, voice_answer, job_description, company_values):
        """Evaluates the user's voice answer based on clarity, relevance, and confidence."""
        prompt = f"""
        SYSTEM: You are an experienced interviewer in the tech industry for over 30 years. Also you are an expert evaluator for interview responses. Assess the answer based on the following criteria:
        
        INSTRUCTIONS:
        - Clarity: Is the response structured and easy to understand?
        - Relevance: Does it address the job's required skills and reflect company values?
        - Confidence: Does the tone convey certainty and professionalism?
        - Consider that the user could be nervous, so don't be too strict.
        - Consider that the user is not a native English speaker, so don't be too strict.
        - Provide constructive feedback and a score out of 10 for each category.
        - Always give some positive feedback at the begining, then give some feedback on what to improve.
        - Use a friendly and professional tone and encourage the user to do better.
        - Keep the feedback concise and to the point.
        - Keep the feedback in 150 words to 250 words.

        USER VOICE ANSWER: {voice_answer}
        JOB DESCRIPTION: {job_description}
        COMPANY VALUES: {company_values}
        """
        response = prompt_llm(prompt)
        
        clarity_match = re.search(r"Clarity: (\d+)/10", response)
        relevance_match = re.search(r"Relevance: (\d+)/10", response)
        confidence_match = re.search(r"Confidence: (\d+)/10", response)
        
        clarity_score = int(clarity_match.group(1)) if clarity_match else 0
        relevance_score = int(relevance_match.group(1)) if relevance_match else 0
        confidence_score = int(confidence_match.group(1)) if confidence_match else 0
        feedback = response
        
        scores = {
            "clarity": clarity_score,
            "relevance": relevance_score,
            "confidence": confidence_score
        }
        
        return scores, feedback

class FollowUpQuestioner:
    def generate_follow_up_questions(self, job_description, resume, question, answer):
        """Generates insightful follow-up questions based on the user's answer."""
        prompt = f"""
        SYSTEM: You are an expert interviewer with 30 years of experience hiring for top tech companies. Your task is to generate 2-3 thoughtful follow-up questions based on a candidate's interview answer.

        INSTRUCTIONS:
        - Analyze the candidate's answer for areas that could be explored further
        - Look for opportunities to dive deeper into their experience, skills, or thought process
        - Consider the job requirements and the candidate's resume when crafting questions
        - Generate questions that allow the candidate to elaborate on strengths
        - Include questions that help address potential weaknesses in a constructive way
        - Each question should be specific and related to the candidate's response
        - Keep questions concise and direct
        - Format the output with a numbered list (1., 2., 3.)

        JOB DESCRIPTION:
        {job_description}

        RESUME:
        {resume}

        INTERVIEW QUESTION:
        {question}

        CANDIDATE'S ANSWER:
        {answer}

        FOLLOW-UP QUESTIONS (generate exactly 2-3):
        """
        
        response = prompt_llm(prompt)
        
        # Extract the questions as a list
        questions = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and (line.startswith('1.') or line.startswith('2.') or line.startswith('3.')):
                questions.append(line)
        
        # If no questions were extracted, return the full response
        if not questions:
            # Split by newlines and filter out empty lines
            questions = [line.strip() for line in response.strip().split('\n') if line.strip()]
            # Limit to 3 questions
            questions = questions[:3]
        
        return questions

class InterviewAgentManager:
    def __init__(self):
        self.analyzer = Analyzer()
        self.drafter = Drafter()
        self.evaluator = Evaluator()
        self.follow_up_questioner = FollowUpQuestioner()
    
    def process_interview(self, job_description, company_values, question, company_info, resume, voice_answer):
        """Manages the full process from analysis to evaluation."""
        parsed_info = self.analyzer.parse_job_info(job_description, company_values)
        model_answer = self.drafter.generate_answer(question, company_info, job_description, resume, voice_answer)
        evaluation = self.evaluator.evaluate_answer(voice_answer, job_description, company_values)
        
        return {
            "parsed_info": parsed_info,
            "model_answer": model_answer,
            "evaluation": evaluation
        }

interview_manager = InterviewAgentManager()

def get_question_hints():
    """Return a dictionary of questions and their hints"""
    return {
        "Tell me about yourself": 
            "Focus on your professional background, key achievements, and why you're a good fit for this role.",
        "What's your greatest strength?": 
            "Choose a strength relevant to the job. Provide specific examples that demonstrate this strength.",
        "Why do you want this job?": 
            "Connect your skills and career goals to the role and company. Show you've done your research.",
        "Where do you see yourself in 5 years?": 
            "Discuss your career goals and how they align with the company's growth trajectory.",
        "Why do you want to work at our company?": 
            "Demonstrate your knowledge of the company's values, culture, and mission.",
        "Tell me about your most relevant experience for this role": 
            "Focus on experience that directly relates to the job requirements. Use the STAR method.",
        "Describe a time you led/motivated others. How were you able to?":
            "Describe a time when you led a team finishing a challenging task, tailor your approach to the people involved, and were positive and persuasive",
        "What's your greatest weakness?":
            "Choose a weakness that is not a deal breaker for the job, and show how you are working to improve it.",
        "What's your biggest accomplishment?":
            "Focus on a significant achievement that showcases your skills and dedication.",
        "What's your biggest failure?":
            "Share a failure that taught you a valuable lesson, emphasizing what you learned and how you overcame it.",
        "How do you motivate team members?":
            "Focus on your ability to inspire and guide others, using your own experiences as examples.",
        "Tell me about a time you worked in a team. How did you contribute?":
            "Describe your specific role and responsibilities within the team. Highlight a successful outcome that resulted from your teamwork.",
        "Can you describe a time when you faced a conflict in a team setting? How did you handle the situation, and what was the outcome?":
            "Highlight the positive outcome and lessons learned. Empathy, communication, negotiation, emotional intelligence",
        "Can you describe a situation where you had to work with a difficult colleague or client?":
            "Describe the specific challenges faced in the situation. Highlight the positive outcome from the experience.",
        "Describe a time you experienced a major change at work. How did you adapt?":
            "Pick an example where you were impacted by a big change and adapted efficiently; extra credit if you got others to do the same",
        "Can you share an example of a time when you used creativity to solve a challenging problem? What approach did you take, and what was the result?":
            "Analyze, creativity, optimization. Create your own opportunities.",
        "Can you share an example of a time when you took initiative? What was the situation, and what impact did your actions have?":
            "Creativity, proactive, positive changes, impact",
        "How do you tackle challenges? Name a difficult challenge you faced while working on a project, how you overcame it, and what you learned.":
            "Perseverance, resilience, resourcefulness, problem-solving",
        "How do you handle ambiguity or uncertainty in your work?":
            "Emphasize the strategies used to approach uncertain situations. Ownership. Adaptability"
    }

def generate_sample_questions(job_desc, company_info, resume):
    """Generate categorized interview questions based on all inputs"""
    # Organize questions by category
    categorized_questions = {
        "Introduction": [
            "Tell me about yourself",
            "Tell me about your most relevant experience for this role"
        ],
        "Strengths & Weaknesses": [
            "What's your greatest strength?",
            "What's your greatest weakness?",
            "What's your biggest accomplishment?", 
            "What's your biggest failure?"
        ],
        "Career Goals": [
            "Why do you want this job?",
            "Where do you see yourself in 5 years?",
            "Why do you want to work at our company?"
        ],
        "Teamwork & Collaboration": [
            "Tell me about a time you worked in a team. How did you contribute?",
            "Can you describe a time when you faced a conflict in a team setting? How did you handle the situation, and what was the outcome?",
            "Can you describe a situation where you had to work with a difficult colleague or client?"
        ],
        "Leadership & Initiative": [
            "Describe a time you led/motivated others. How were you able to?",
            "How do you motivate team members?",
            "Can you share an example of a time when you took initiative? What was the situation, and what impact did your actions have?"
        ],
        "Problem Solving & Adaptability": [
            "How do you tackle challenges? Name a difficult challenge you faced while working on a project, how you overcame it, and what you learned.",
            "Describe a time you experienced a major change at work. How did you adapt?",
            "Can you share an example of a time when you used creativity to solve a challenging problem? What approach did you take, and what was the result?",
            "How do you handle ambiguity or uncertainty in your work?"
        ]
    }
    
    # Add company-specific question if company info is provided
    if company_info.strip() and "Why do you want to work at our company?" not in categorized_questions["Career Goals"]:
        categorized_questions["Career Goals"].append("Why do you want to work at our company?")
    
    # Add experience-related question if resume is provided
    if resume.strip() and "Tell me about your most relevant experience for this role" not in categorized_questions["Introduction"]:
        categorized_questions["Introduction"].append("Tell me about your most relevant experience for this role")
    
    return categorized_questions

def speech_to_text(audio_data):
    """Convert speech to text using SpeechRecognition"""
    recognizer = sr.Recognizer()
    
    try:
        # Save base64 audio data to a temporary file
        audio_bytes = base64.b64decode(audio_data.split(',')[1])
        
        # Save as webm file first
        with tempfile.NamedTemporaryFile(delete=False, suffix='.webm') as temp_audio_file:
            temp_audio_file.write(audio_bytes)
            temp_webm_path = temp_audio_file.name
        
        # Convert to WAV using FFmpeg if available, otherwise use a direct approach
        try:
            import subprocess
            wav_path = temp_webm_path.replace('.webm', '.wav')
            subprocess.call(['ffmpeg', '-i', temp_webm_path, '-ar', '16000', '-ac', '1', wav_path])
            os.unlink(temp_webm_path)  # Delete the webm file
            
            with sr.AudioFile(wav_path) as source:
                audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            os.unlink(wav_path)  # Delete temp WAV file
            return text
        except (ImportError, FileNotFoundError):
            # If FFmpeg is not available, try direct approach with the webm file
            # Note: This might not work perfectly but worth trying
            with sr.AudioFile(temp_webm_path) as source:
                audio = recognizer.record(source)
            text = recognizer.recognize_google(audio)
            os.unlink(temp_webm_path)  # Delete temp file
            return text
    except Exception as e:
        # If an error occurs, try to delete any temporary files
        try:
            if 'temp_webm_path' in locals():
                os.unlink(temp_webm_path)
            if 'wav_path' in locals():
                os.unlink(wav_path)
        except:
            pass
        
        # Use a different approach as fallback - send directly to Google's API
        try:
            audio_bytes = base64.b64decode(audio_data.split(',')[1])
            audio_data_obj = sr.AudioData(audio_bytes, 16000, 2)  # Using default values
            text = recognizer.recognize_google(audio_data_obj)
            return text
        except Exception as inner_e:
            return f"Speech recognition failed: {str(e)}. Second attempt: {str(inner_e)}"

def get_voice_options():
    return {
        "US English": {"lang": "en", "tld": "com"},
        "UK English": {"lang": "en", "tld": "co.uk"},
        "Australian English": {"lang": "en", "tld": "com.au"},
        "Indian English": {"lang": "en", "tld": "co.in"},
        "French": {"lang": "fr", "tld": "fr"},
        "German": {"lang": "de", "tld": "de"},
        "Spanish": {"lang": "es", "tld": "es"},
        "Italian": {"lang": "it", "tld": "it"},
        "Japanese": {"lang": "ja", "tld": "co.jp"},
        "Korean": {"lang": "ko", "tld": "co.kr"}
    }

def text_to_speech(text, voice_option="US English"):
    """Convert text to speech and return as base64"""
    voice_options = get_voice_options()
    selected_voice = voice_options.get(voice_option, {"lang": "en", "tld": "com"})
    
    try:
        # Create a temporary file for the audio
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            temp_filename = fp.name
        
        # Generate the speech audio file with the selected voice
        tts = gTTS(text=text, lang=selected_voice["lang"], tld=selected_voice["tld"])
        tts.save(temp_filename)
        
        # Read the file and convert to base64
        with open(temp_filename, 'rb') as audio_file:
            audio_data = base64.b64encode(audio_file.read()).decode('utf-8')
        
        # Clean up the temporary file
        os.unlink(temp_filename)
        
        return f"data:audio/mp3;base64,{audio_data}"
    except Exception as e:
        print(f"TTS Error: {e}")
        return None

def save_to_html(job_desc, company_info, resume, company_name, position_title, company_values, tech_skills, soft_skills, job_duties, selected_question, answer_text, feedback, model_answer, follow_up_questions=None):
    """Generate HTML content for download."""
    # Get current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"interview_summary_{current_time}.html"
    
    html_content = f"""
    <html>
    <head>
        <title>Interview Preparation Summary</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
            
            body {{
                font-family: 'Inter', sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f5f5f0;
                color: #1E3932;
            }}
            
            .container {{
                max-width: 850px;
                margin: 0 auto;
                padding: 20px;
            }}
            
            header {{
                background: linear-gradient(to right, #006241, #1E3932);
                color: white;
                padding: 30px 0;
                margin-bottom: 30px;
            }}
            
            .header-content {{
                text-align: center;
                padding: 0 20px;
            }}
            
            h1 {{
                margin: 0;
                font-size: 28px;
                font-weight: 700;
            }}
            
            .company-info {{
                margin-top: 10px;
                font-size: 16px;
            }}
            
            .timestamp {{
                margin-top: 5px;
                font-size: 14px;
                opacity: 0.8;
            }}
            
            .section-title {{
                color: #006241;
                font-size: 22px;
                margin-top: 40px;
                margin-bottom: 15px;
                padding-bottom: 10px;
                border-bottom: 2px solid #D4E9E2;
            }}
            
            .info-item {{
                margin-bottom: 12px;
                line-height: 1.6;
            }}
            
            .info-section {{
                background-color: white;
                border-radius: 8px;
                padding: 20px;
                margin-bottom: 30px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                border: 1px solid #D4E9E2;
            }}
            
            .question {{
                font-weight: 600;
                margin-bottom: 10px;
                color: #006241;
            }}
            
            .answer {{
                background-color: #f5f5f0;
                border-left: 4px solid #006241;
                padding: 15px;
                margin-bottom: 20px;
                border-radius: 0 8px 8px 0;
            }}
            
            .feedback {{
                background-color: #f5f5f0;
                border: 1px solid #D4E9E2;
                padding: 15px;
                margin-top: 20px;
                border-radius: 8px;
            }}
            
            .score-section {{
                display: flex;
                justify-content: space-between;
                margin: 20px 0;
                flex-wrap: wrap;
            }}
            
            .score-item {{
                flex: 1;
                min-width: 150px;
                background: white;
                padding: 15px;
                border-radius: 8px;
                margin-right: 15px;
                margin-bottom: 15px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                border: 1px solid #D4E9E2;
            }}
            
            .score-item:last-child {{
                margin-right: 0;
            }}
            
            .score-title {{
                font-weight: 600;
                margin-bottom: 8px;
                color: #006241;
            }}
            
            .stars {{
                color: #006241;
                font-size: 18px;
            }}
            
            footer {{
                text-align: center;
                margin-top: 50px;
                padding: 20px 0;
                color: #666;
                font-size: 14px;
                border-top: 1px solid #D4E9E2;
            }}
            
            ul {{
                padding-left: 20px;
            }}
            
            li {{
                margin-bottom: 8px;
            }}
        </style>
    </head>
    <body>
        <header>
            <div class="header-content">
                <h1>Interview Preparation Summary</h1>
                <div class="company-info">
                    <strong>Company:</strong> {company_name if company_name else 'Not specified'} |
                    <strong>Position:</strong> {position_title if position_title else 'Not specified'}
                </div>
                <div class="timestamp">Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</div>
            </div>
        </header>
        <div class="container">
            <!-- Parsed Information -->
            <div class="info-section">
                <h2 class="section-title">Job Analysis</h2>
                <ul>
                    <li class="info-item"><strong>Company Values:</strong> {company_values}</li>
                    <li class="info-item"><strong>Tech Skills:</strong> {tech_skills}</li>
                    <li class="info-item"><strong>Soft Skills:</strong> {soft_skills}</li>
                    <li class="info-item"><strong>Job Duties:</strong> {job_duties}</li>
                </ul>
            </div>
            
            <!-- Question and Answer -->
            <div class="info-section">
                <h2 class="section-title">Interview Question</h2>
                <div class="question">
                    {selected_question}
                </div>
                
                <h3 style="margin-top: 25px; color: #006241;">Your Answer</h3>
                <div class="answer">
                    {answer_text}
                </div>
                
                <h3 style="margin-top: 25px; color: #006241;">Model Answer</h3>
                <div class="answer">
                    {model_answer}
                </div>
            </div>
            
            <!-- Follow-up Questions -->
            {'<div class="info-section"><h2 class="section-title">Potential Follow-up Questions</h2><div class="follow-up-questions">' + ''.join([f'<div class="follow-up-question"><p>{q}</p></div>' for q in (follow_up_questions or [])]) + '</div></div>' if follow_up_questions else ''}
            
            <!-- Feedback -->
            <div class="info-section">
                <h2 class="section-title">Performance Analysis</h2>
                <div class="score-section">
                    <div class="score-item">
                        <div class="score-title">Clarity</div>
                        <div class="stars">{"★" * int(feedback.split('Clarity:')[1].split('/')[0].strip() if 'Clarity:' in feedback else 5) + "☆" * (10 - int(feedback.split('Clarity:')[1].split('/')[0].strip() if 'Clarity:' in feedback else 5))}</div>
                    </div>
                    <div class="score-item">
                        <div class="score-title">Relevance</div>
                        <div class="stars">{"★" * int(feedback.split('Relevance:')[1].split('/')[0].strip() if 'Relevance:' in feedback else 5) + "☆" * (10 - int(feedback.split('Relevance:')[1].split('/')[0].strip() if 'Relevance:' in feedback else 5))}</div>
                    </div>
                    <div class="score-item">
                        <div class="score-title">Confidence</div>
                        <div class="stars">{"★" * int(feedback.split('Confidence:')[1].split('/')[0].strip() if 'Confidence:' in feedback else 5) + "☆" * (10 - int(feedback.split('Confidence:')[1].split('/')[0].strip() if 'Confidence:' in feedback else 5))}</div>
                    </div>
                </div>
                
                <div class="feedback">
                    <h3 style="margin-top: 0; margin-bottom: 15px; color: #006241;">Detailed Feedback</h3>
                    {feedback}
                </div>
            </div>
            
            <footer>
                <p>Husky Interview Prep &copy; 2025</p>
                <p>Generated on {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>
            </footer>
        </div>
    </body>
    </html>
    """
    
    # Create a temporary file to store the HTML content
    with tempfile.NamedTemporaryFile(delete=False, suffix='.html') as tmp_file:
        tmp_file.write(html_content.encode('utf-8'))
        tmp_file_path = tmp_file.name
    
    return tmp_file_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze-info', methods=['POST'])
def analyze_info_endpoint():
    data = request.get_json()
    job_desc = data.get('job_desc', '')
    company_info = data.get('company_info', '')
    
    parsed_info = interview_manager.analyzer.parse_job_info(job_desc, company_info)
    
    # Store in session for later use
    session['job_desc'] = job_desc
    session['company_info'] = company_info
    session['parsed_info'] = parsed_info
    
    return jsonify(parsed_info)

@app.route('/generate-questions', methods=['POST'])
def generate_questions_endpoint():
    data = request.get_json()
    job_desc = data.get('job_desc', '')
    company_info = data.get('company_info', '')
    resume = data.get('resume', '')
    
    # Store in session for later use
    session['job_desc'] = job_desc
    session['company_info'] = company_info
    session['resume'] = resume
    
    categorized_questions = generate_sample_questions(job_desc, company_info, resume)
    question_hints = get_question_hints()
    
    return jsonify({
        'questions': categorized_questions,
        'hints': question_hints
    })

@app.route('/speech-to-text', methods=['POST'])
def speech_to_text_endpoint():
    data = request.get_json()
    audio_data = data.get('audio', '')
    
    if not audio_data:
        return jsonify({'error': 'No audio data provided'}), 400
    
    text = speech_to_text(audio_data)
    return jsonify({'text': text})

@app.route('/analyze-answer', methods=['POST'])
def analyze_answer_endpoint():
    data = request.get_json()
    voice_answer = data.get('answer_text', '')
    job_desc = data.get('job_desc', session.get('job_desc', ''))
    company_values = data.get('company_values', '')
    
    # Input validation
    if not voice_answer:
        default_feedback = "No answer provided to analyze. Please record or type your answer."
        return jsonify({
            'scores': {'clarity': 0, 'relevance': 0, 'confidence': 0},
            'feedback': default_feedback,
            'formatted_output': default_feedback
        })
    
    try:
        scores, feedback = interview_manager.evaluator.evaluate_answer(voice_answer, job_desc, company_values)
        
        # Ensure feedback is not empty
        if not feedback or len(feedback.strip()) < 10:
            feedback = """I couldn't properly evaluate your answer. Here are some general tips:
            
- Structure your response with a clear beginning, middle, and end
- Relate your experience directly to the job requirements
- Use specific examples from your past experience
- Show confidence in your tone and delivery
            
Try recording again with these tips in mind."""
            scores = {'clarity': 5, 'relevance': 5, 'confidence': 5}
        
        # Create formatted output
        stars = lambda score: "⭐" * score + "☆" * (10 - score)
        combined_output = f"""SCORES:
Clarity: {stars(scores['clarity'])}
Relevance: {stars(scores['relevance'])}
Confidence: {stars(scores['confidence'])}

FEEDBACK:
{feedback}"""
        
        return jsonify({
            'scores': scores,
            'feedback': feedback,
            'formatted_output': combined_output
        })
    except Exception as e:
        print(f"Error in analyze_answer_endpoint: {str(e)}")
        default_feedback = "I'm having trouble analyzing your answer right now. This might be due to a connection issue or server load. Please try again in a moment."
        scores = {'clarity': 5, 'relevance': 5, 'confidence': 5}
        stars = lambda score: "⭐" * score + "☆" * (10 - score)
        combined_output = f"""SCORES:
Clarity: {stars(scores['clarity'])}
Relevance: {stars(scores['relevance'])}
Confidence: {stars(scores['confidence'])}

FEEDBACK:
{default_feedback}"""
        
        return jsonify({
            'scores': scores,
            'feedback': default_feedback,
            'formatted_output': combined_output
        })

@app.route('/generate-model-answer', methods=['POST'])
def generate_model_answer_endpoint():
    data = request.get_json()
    question = data.get('question', '')
    company_info = data.get('company_info', session.get('company_info', ''))
    job_desc = data.get('job_desc', session.get('job_desc', ''))
    resume = data.get('resume', session.get('resume', ''))
    voice_answer = data.get('answer_text', '')
    
    # Input validation
    if not question:
        return jsonify({
            'model_answer': 'No question provided. Please select a question first.'
        })
    
    try:
        model_answer = interview_manager.drafter.generate_answer(question, company_info, job_desc, resume, voice_answer)
        
        # Ensure model answer is not empty
        if not model_answer or len(model_answer.strip()) < 10:
            model_answer = f"""I couldn't generate a complete sample answer for this question: "{question}"
            
Here's a general structure you can follow:

1. Begin with a brief introduction relevant to the question
2. Use the STAR method for behavioral questions:
   - Situation: Describe the context
   - Task: Explain your responsibility
   - Action: Detail the steps you took
   - Result: Share the outcome and what you learned

3. Connect your answer to the specific job requirements
4. Keep your answer concise (about 1-2 minutes when spoken)
5. Practice your delivery to sound natural and confident"""
        
        return jsonify({'model_answer': model_answer})
    except Exception as e:
        print(f"Error in generate_model_answer_endpoint: {str(e)}")
        default_answer = f"""I'm having trouble generating a sample answer for the question: "{question}"

Here are some general tips for this type of question:
- Use the STAR method: Situation, Task, Action, Result
- Relate your answer to the job you're applying for
- Be specific and use concrete examples
- Keep your answer concise and to the point
- Practice your delivery to sound confident and prepared"""
        
        return jsonify({'model_answer': default_answer})

@app.route('/text-to-speech', methods=['POST'])
def text_to_speech_endpoint():
    data = request.get_json()
    text = data.get('text', '')
    voice_option = data.get('voice_option', 'US English')
    
    audio_base64 = text_to_speech(text, voice_option)
    
    return jsonify({'audio': audio_base64})

@app.route('/generate-follow-up-questions', methods=['POST'])
def generate_follow_up_questions_endpoint():
    data = request.get_json()
    question = data.get('question', '')
    job_desc = data.get('job_desc', session.get('job_desc', ''))
    resume = data.get('resume', session.get('resume', ''))
    answer_text = data.get('answer_text', '')
    
    if not question or not answer_text:
        return jsonify({
            'follow_up_questions': ['Please provide both a question and your answer to generate follow-up questions.']
        })
    
    try:
        follow_up_questions = interview_manager.follow_up_questioner.generate_follow_up_questions(
            job_desc, resume, question, answer_text
        )
        
        return jsonify({'follow_up_questions': follow_up_questions})
    except Exception as e:
        print(f"Error generating follow-up questions: {str(e)}")
        default_questions = [
            "Could you elaborate more on your experience in this area?",
            "How would you apply these skills in our company context?",
            "Can you provide a specific example of how you've handled similar situations?"
        ]
        return jsonify({'follow_up_questions': default_questions})

@app.route('/save-to-html', methods=['POST'])
def save_to_html_endpoint():
    data = request.get_json()
    job_desc = data.get('job_desc', session.get('job_desc', ''))
    company_info = data.get('company_info', session.get('company_info', ''))
    resume = data.get('resume', session.get('resume', ''))
    company_values = data.get('company_values', '')
    tech_skills = data.get('tech_skills', '')
    soft_skills = data.get('soft_skills', '')
    job_duties = data.get('job_duties', '')
    selected_question = data.get('selected_question', '')
    answer_text = data.get('answer_text', '')
    feedback = data.get('feedback', '')
    model_answer = data.get('model_answer', '')
    follow_up_questions = data.get('follow_up_questions', [])
    # Retrieve company name and position title from parsed info if available
    parsed_info = session.get('parsed_info', {})
    company_name = data.get('company_name', parsed_info.get('company_name', ''))
    position_title = data.get('position_title', parsed_info.get('position_title', ''))
    
    html_file_path = save_to_html(
        job_desc, company_info, resume, company_name, position_title, company_values, tech_skills, 
        soft_skills, job_duties, selected_question, answer_text, feedback, model_answer, follow_up_questions
    )
    
    # Generate a unique ID for this file for the frontend to request it
    file_id = str(uuid.uuid4())
    session[f'html_file_{file_id}'] = html_file_path
    
    return jsonify({'file_id': file_id})

@app.route('/download-html/<file_id>', methods=['GET'])
def download_html(file_id):
    file_path = session.get(f'html_file_{file_id}')
    if not file_path:
        return "File not found", 404
    
    # Set the name for the downloaded file
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    download_name = f"interview_summary_{current_time}.html"
    
    return send_file(file_path, as_attachment=True, download_name=download_name)

if __name__ == "__main__":
    # Create templates folder if it doesn't exist
    if not os.path.exists('templates'):
        os.makedirs('templates')
    
    # Create static folder if it doesn't exist
    if not os.path.exists('static'):
        os.makedirs('static')
    
    # Copy interviewer.png to static folder if it exists
    if os.path.exists('interviewer.png') and not os.path.exists('static/interviewer.png'):
        import shutil
        shutil.copy('interviewer.png', 'static/interviewer.png')
    
    # Create basic index.html template if it doesn't exist
    if not os.path.exists('templates/index.html'):
        with open('templates/index.html', 'w') as f:
            f.write('''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Husky Interview Prep</title>
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Alpine.js -->
    <script defer src="https://cdn.jsdelivr.net/npm/alpinejs@3.x.x/dist/cdn.min.js"></script>
    <!-- Font Awesome -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
        
        :root {
            --primary-color: #006241;    /* Starbucks green */
            --primary-light: #D4E9E2;    /* Light green */
            --primary-dark: #004C33;     /* Dark green */
            --accent-color: #CBA258;     /* Gold accent */
            --text-dark: #1E3932;        /* Dark green text */
            --text-light: #f9f9f9;       /* Light text */
            --spacing-base: 4px;         /* Base spacing unit */
            --transition-base: 0.2s ease-in-out;
        }
        
        body {
            font-family: 'Inter', sans-serif;
            color: var(--text-dark);
            line-height: 1.5;
            font-size: 16px;
            background-color: #f8f9fa;
        }
        
        /* Enhanced Backgrounds */
        .bg-primary { background-color: var(--primary-color); }
        .bg-primary-light { background-color: var(--primary-light); }
        .bg-primary-dark { background-color: var(--primary-dark); }
        
        /* Improved Buttons */
        .btn-primary {
            background-color: var(--primary-color);
            color: white;
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 500;
            cursor: pointer;
            transition: all var(--transition-base);
            border: none;
            outline: none;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
        }
        
        .btn-primary:hover, .btn-primary:focus {
            background-color: var(--primary-dark);
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            transform: translateY(-1px);
        }
        
        .btn-primary:disabled {
            opacity: 0.7;
            cursor: not-allowed;
            transform: none;
        }
        
        .btn-secondary {
            background-color: white;
            color: var(--primary-color);
            border: 1px solid var(--primary-color);
            border-radius: 8px;
            padding: 12px 24px;
            font-weight: 500;
            cursor: pointer;
            transition: all var(--transition-base);
            display: inline-flex;
            align-items: center;
            justify-content: center;
        }
        
        .btn-secondary:hover, .btn-secondary:focus {
            background-color: var(--primary-light);
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }
        
        /* Card Improvements */
        .card {
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
            transition: all var(--transition-base);
            margin-bottom: 24px;
            overflow: hidden;
        }
        
        .card-header {
            padding: 20px 24px;
            border-bottom: 1px solid #f0f0f0;
        }
        
        .card-body {
            padding: 24px;
        }
        
        /* Form Elements */
        .form-control {
            width: 100%;
            padding: 12px 16px;
            border-radius: 8px;
            border: 1px solid #e0e0e0;
            transition: all var(--transition-base);
            font-size: 16px;
        }
        
        .form-control:focus {
            border-color: var(--primary-color);
            box-shadow: 0 0 0 3px rgba(0, 98, 65, 0.1);
            outline: none;
        }
        
        .form-label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: var(--text-dark);
        }
        
        /* Status indicators */
        .spinner {
            display: inline-block;
            width: 20px;
            height: 20px;
            margin-right: 12px;
            border: 2px solid rgba(255, 255, 255, 0.3);
            border-radius: 50%;
            border-top-color: #fff;
            animation: spin 0.8s linear infinite;
        }
        
        @keyframes spin {
            to { transform: rotate(360deg); }
        }
        
        .recording-pulse {
            animation: pulse 1.5s infinite;
        }
        
        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0.7); }
            70% { box-shadow: 0 0 0 10px rgba(220, 38, 38, 0); }
            100% { box-shadow: 0 0 0 0 rgba(220, 38, 38, 0); }
        }
        
        /* Visual Tab System */
        .tab-active {
            color: var(--primary-color);
            border-bottom: 2px solid var(--primary-color);
            font-weight: 600;
        }
        
        /* Question Cards */
        .question-card {
            transition: all var(--transition-base);
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            border: 1px solid #eaeaea;
            background-color: #f9f9f9;
            cursor: pointer;
        }
        
        .question-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 8px 16px -4px rgba(0, 0, 0, 0.05);
            border-color: var(--primary-light);
        }
    </style>
</head>
<body class="text-gray-800">
    <div x-data="app()" x-init="init" class="min-h-screen flex flex-col">
        <!-- Hero Header -->
        <header class="bg-primary relative py-16 overflow-hidden">
            <div class="absolute inset-0 z-0">
                <div class="absolute inset-0 bg-gradient-to-r from-primary-dark to-primary opacity-90"></div>
            </div>
            <div class="container mx-auto px-4 sm:px-6 lg:px-8 relative z-10">
                <div class="text-center">
                    <div class="inline-block px-4 py-2 rounded-full bg-white/10 backdrop-blur-lg border border-white/20 mb-6">
                        <span class="inline-block w-2 h-2 rounded-full bg-white mr-2"></span>
                        <span class="text-white text-sm font-medium">AI-Powered Interview Prep</span>
                    </div>
                    <h1 class="text-4xl md:text-5xl font-bold text-white mb-4">Husky Interview Prep</h1>
                    <p class="text-xl text-white/80 max-w-2xl mx-auto">Master your interview with confidence. AI-powered preparation for job seekers.</p>
                </div>
            </div>
        </header>
        
        <main class="flex-grow py-12 px-4 sm:px-6 lg:px-8">
            <div class="container mx-auto max-w-5xl">
                <!-- Step 1: Enter Information -->
                <section class="card">
                    <div class="card-header">
                        <h2 class="text-2xl font-bold flex items-center">
                            <span class="inline-flex items-center justify-center w-8 h-8 rounded-full bg-primary-light text-primary mr-3">1</span>
                            Enter Your Information
                        </h2>
                    </div>
                    <div class="card-body">
                        <div class="grid grid-cols-1 gap-6">
                            <div>
                                <label class="form-label">Job Description</label>
                                <textarea 
                                    x-model="jobDesc" 
                                    class="form-control h-32" 
                                    placeholder="Paste the job description here..."
                                ></textarea>
                            </div>
                            <div>
                                <label class="form-label">Company Information</label>
                                <textarea 
                                    x-model="companyInfo" 
                                    class="form-control h-32" 
                                    placeholder="Enter information about the company..."
                                ></textarea>
                            </div>
                            <div>
                                <label class="form-label">Your Resume</label>
                                <textarea 
                                    x-model="resume" 
                                    class="form-control h-32" 
                                    placeholder="Paste your resume or relevant experience here..."
                                ></textarea>
                            </div>
                        </div>
                        
                        <div class="mt-8 flex justify-center">
                            <button 
                                @click="analyzeInfo()" 
                                :disabled="isAnalyzing" 
                                class="btn-primary"
                            >
                                <span class="spinner" x-show="isAnalyzing"></span>
                                <i class="fas fa-search mr-2" x-show="!isAnalyzing"></i>
                                <span x-text="isAnalyzing ? 'Analyzing...' : 'Analyze Information'"></span>
                            </button>
                        </div>
                
                        <template x-if="parsedInfo.company_values">
                            <div class="mt-8 space-y-6">
                                <!-- Company and position info -->
                                <div class="grid md:grid-cols-2 gap-6">
                                    <div class="bg-primary-light p-5 rounded-lg">
                                        <h3 class="text-lg font-semibold mb-2">Company Name</h3>
                                        <p x-text="parsedInfo.company_name"></p>
                                    </div>
                                    <div class="bg-primary-light p-5 rounded-lg">
                                        <h3 class="text-lg font-semibold mb-2">Position Title</h3>
                                        <p x-text="parsedInfo.position_title"></p>
                                    </div>
                                </div>
                                
                                <!-- Other job info -->
                                <div class="grid md:grid-cols-2 gap-6">
                                    <div class="bg-primary-light p-5 rounded-lg">
                                        <h3 class="text-lg font-semibold mb-2">Company Values</h3>
                                        <ul class="list-disc pl-5 space-y-1">
                                            <template x-for="(line, index) in parsedInfo.company_values.split('- ').filter(item => item.trim().length > 0)" :key="index">
                                                <li x-text="line.trim()"></li>
                                            </template>
                                        </ul>
                                    </div>
                                    <div class="bg-primary-light p-5 rounded-lg">
                                        <h3 class="text-lg font-semibold mb-2">Job Duties</h3>
                                        <ul class="list-disc pl-5 space-y-1">
                                            <template x-for="(line, index) in parsedInfo.job_duties.split('- ').filter(item => item.trim().length > 0)" :key="index">
                                                <li x-text="line.trim()"></li>
                                            </template>
                                        </ul>
                                    </div>
                                    <div class="bg-primary-light p-5 rounded-lg">
                                        <h3 class="text-lg font-semibold mb-2">Tech Skills</h3>
                                        <ul class="list-disc pl-5 space-y-1">
                                            <template x-for="(line, index) in parsedInfo.tech_skills.split('- ').filter(item => item.trim().length > 0)" :key="index">
                                                <li x-text="line.trim()"></li>
                                            </template>
                                        </ul>
                                    </div>
                                    <div class="bg-primary-light p-5 rounded-lg">
                                        <h3 class="text-lg font-semibold mb-2">Soft Skills</h3>
                                        <ul class="list-disc pl-5 space-y-1">
                                            <template x-for="(line, index) in parsedInfo.soft_skills.split('- ').filter(item => item.trim().length > 0)" :key="index">
                                                <li x-text="line.trim()"></li>
                                            </template>
                                        </ul>
                                    </div>
                                </div>
                                
                                <div class="flex justify-center">
                                    <button 
                                        @click="generateQuestions()" 
                                        :disabled="isGeneratingQuestions" 
                                        class="btn-primary"
                                    >
                                        <span class="spinner" x-show="isGeneratingQuestions"></span>
                                        <i class="fas fa-list-ul mr-2" x-show="!isGeneratingQuestions"></i>
                                        <span x-text="isGeneratingQuestions ? 'Generating...' : 'Generate Interview Questions'"></span>
                                    </button>
                                </div>
                            </div>
                        </template>
                    </div>
                </section>
            
                <!-- Step 2: Practice Questions -->
                <section x-show="questionsGenerated" class="card">
                    <div class="card-header">
                        <h2 class="text-2xl font-bold flex items-center">
                            <span class="inline-flex items-center justify-center w-8 h-8 rounded-full bg-primary-light text-primary mr-3">2</span>
                            Practice Questions
                        </h2>
                    </div>
                    <div class="card-body">
                        <!-- Question Category Tabs -->
                        <div class="mb-6">
                            <div class="flex flex-wrap space-x-2 border-b border-gray-200 pb-2">
                                <template x-for="(category, index) in Object.keys(questions)" :key="index">
                                    <button 
                                        @click="activeCategory = category" 
                                        :class="{'tab-active': activeCategory === category}"
                                        class="px-4 py-2 rounded-t-lg font-medium transition-all duration-200"
                                        x-text="category"
                                    ></button>
                                </template>
                            </div>
                        </div>
                
                        <!-- Questions for Selected Category -->
                        <div>
                            <template x-for="(categoryQuestions, category) in questions" :key="category">
                                <div x-show="activeCategory === category">
                                    <template x-for="(question, qIndex) in categoryQuestions" :key="qIndex">
                                        <div 
                                            @click="selectQuestion(question)"
                                            class="question-card"
                                            :class="{'border-primary bg-primary-light': selectedQuestion === question}"
                                        >
                                            <p class="text-gray-800" x-text="question"></p>
                                        </div>
                                    </template>
                                </div>
                            </template>
                        </div>
                    </div>
                </section>
            
                <!-- Step 3: Record Answer -->
                <section x-show="selectedQuestion" class="card">
                    <div class="card-header">
                        <h2 class="text-2xl font-bold flex items-center">
                            <span class="inline-flex items-center justify-center w-8 h-8 rounded-full bg-primary-light text-primary mr-3">3</span>
                            Record Your Answer
                        </h2>
                    </div>
                    <div class="card-body">
                        <!-- Company info bar -->
                        <div class="bg-primary-light p-4 rounded-lg mb-6">
                            <div class="flex flex-col sm:flex-row sm:items-center sm:justify-start sm:space-x-8">
                                <div class="flex items-center mb-2 sm:mb-0">
                                    <span class="text-sm font-medium text-primary-dark">Company:</span>
                                    <span class="ml-2 font-semibold text-primary" x-text="parsedInfo.company_name || 'Not specified'"></span>
                                </div>
                                <div class="flex items-center">
                                    <span class="text-sm font-medium text-primary-dark">Position:</span>
                                    <span class="ml-2 font-semibold text-primary" x-text="parsedInfo.position_title || 'Not specified'"></span>
                                </div>
                            </div>
                        </div>
                        
                        <!-- New grid layout for interview question and recording section -->
                        <div class="grid md:grid-cols-12 gap-6 mb-6">
                            <!-- Interviewer Column -->
                            <div class="md:col-span-4">
                                <label class="form-label">Interviewer</label>
                                <div class="bg-primary-light rounded-lg p-5 text-center h-full flex flex-col">
                                    <div class="rounded-lg overflow-hidden mb-4 mx-auto w-32 h-32 flex items-center justify-center bg-white">
                                        <img src="/static/interviewer.png" alt="Interviewer" class="max-w-full max-h-full">
                                    </div>
                                    
                                    <div class="mb-4 flex-grow">
                                        <label class="form-label text-center">Voice Accent</label>
                                        <select 
                                            x-model="voiceOption" 
                                            class="form-control text-sm"
                                        >
                                            <option value="US English">US English</option>
                                            <option value="UK English">UK English</option>
                                            <option value="Australian English">Australian English</option>
                                            <option value="Indian English">Indian English</option>
                                            <option value="French">French</option>
                                            <option value="German">German</option>
                                            <option value="Spanish">Spanish</option>
                                        </select>
                                    </div>
                                    
                                    <button 
                                        @click="readQuestionAloud()" 
                                        :disabled="isReadingAloud" 
                                        class="btn-secondary w-full"
                                    >
                                        <span class="spinner" x-show="isReadingAloud"></span>
                                        <i class="fas fa-volume-up mr-2" x-show="!isReadingAloud"></i>
                                        <span x-text="isReadingAloud ? 'Reading...' : 'Read Question Aloud'"></span>
                                    </button>
                                    
                                    <div x-show="audioPlaying" class="mt-4">
                                        <audio x-ref="audioPlayer" controls class="w-full">
                                            Your browser does not support the audio element.
                                        </audio>
                                    </div>
                                </div>
                            </div>
                
                            <!-- Question Column -->
                            <div class="md:col-span-8">
                                <label class="form-label">Current Question</label>
                                <div class="p-5 bg-primary-light rounded-lg mb-4">
                                    <p class="text-gray-800 font-medium text-lg" x-text="selectedQuestion"></p>
                                </div>
                                
                                <div class="p-5 bg-primary-light rounded-lg">
                                    <h3 class="font-semibold text-primary-dark mb-3">How to Answer</h3>
                                    <p class="leading-relaxed" x-text="questionHints[selectedQuestion] || 'No specific hint available for this question.'"></p>
                                </div>
                            </div>
                        </div>
                
                        <div class="grid md:grid-cols-12 gap-6">
                            <!-- Recording Button Column -->
                            <div class="md:col-span-4">
                                <label class="form-label">Record Your Answer</label>
                                <div class="flex flex-col items-center justify-center h-full p-6 bg-primary-light rounded-lg">
                                    <button @click="toggleRecording()" class="mb-4 relative">
                                        <div :class="{'recording-pulse': isRecording}" class="w-20 h-20 rounded-full bg-white border-2 border-gray-300 flex items-center justify-center">
                                            <i :class="isRecording ? 'fa-stop text-red-500' : 'fa-microphone text-gray-700'" class="fas text-2xl"></i>
                                        </div>
                                    </button>
                                    <p x-text="recordingStatus" class="text-sm font-medium text-primary-dark text-center"></p>
                                </div>
                            </div>
                
                            <!-- Transcription Textarea Column -->
                            <div class="md:col-span-8">
                                <label class="form-label">Your Answer (Transcribed)</label>
                                <textarea 
                                    x-model="answerText" 
                                    class="form-control h-48" 
                                    placeholder="Your transcribed answer will appear here..."
                                ></textarea>
                            </div>
                        </div>
                
                        <div class="mt-8 flex justify-center">
                            <button 
                                @click="analyzeAnswer()" 
                                :disabled="isAnalyzingAnswer" 
                                class="btn-primary"
                            >
                                <span class="spinner" x-show="isAnalyzingAnswer"></span>
                                <i class="fas fa-chart-line mr-2" x-show="!isAnalyzingAnswer"></i>
                                <span x-text="isAnalyzingAnswer ? 'Analyzing...' : 'Analyze Answer'"></span>
                            </button>
                        </div>
                    </div>
                </section>
            
                <!-- Step 4: Review Analysis -->
                <section x-show="feedbackText" class="card">
                    <div class="card-header">
                        <h2 class="text-2xl font-bold flex items-center">
                            <span class="inline-flex items-center justify-center w-8 h-8 rounded-full bg-primary-light text-primary mr-3">4</span>
                            Review Analysis
                        </h2>
                    </div>
                    <div class="card-body">
                        <div x-show="scores" class="grid md:grid-cols-3 gap-6 mb-6">
                            <div class="bg-primary-light p-5 rounded-lg">
                                <h3 class="font-semibold text-primary-dark mb-2 text-center">Clarity</h3>
                                <div class="text-2xl text-center mb-2">
                                    <template x-for="i in 10" :key="i">
                                        <span 
                                            :class="i <= scores.clarity ? 'text-primary' : 'text-gray-300'"
                                            x-text="i <= scores.clarity ? '★' : '☆'"
                                        ></span>
                                    </template>
                                </div>
                                <p class="text-sm text-center font-medium">Score: <span x-text="scores.clarity"></span>/10</p>
                            </div>
                            <div class="bg-primary-light p-5 rounded-lg">
                                <h3 class="font-semibold text-primary-dark mb-2 text-center">Relevance</h3>
                                <div class="text-2xl text-center mb-2">
                                    <template x-for="i in 10" :key="i">
                                        <span 
                                            :class="i <= scores.relevance ? 'text-primary' : 'text-gray-300'"
                                            x-text="i <= scores.relevance ? '★' : '☆'"
                                        ></span>
                                    </template>
                                </div>
                                <p class="text-sm text-center font-medium">Score: <span x-text="scores.relevance"></span>/10</p>
                            </div>
                            <div class="bg-primary-light p-5 rounded-lg">
                                <h3 class="font-semibold text-primary-dark mb-2 text-center">Confidence</h3>
                                <div class="text-2xl text-center mb-2">
                                    <template x-for="i in 10" :key="i">
                                        <span 
                                            :class="i <= scores.confidence ? 'text-primary' : 'text-gray-300'"
                                            x-text="i <= scores.confidence ? '★' : '☆'"
                                        ></span>
                                    </template>
                                </div>
                                <p class="text-sm text-center font-medium">Score: <span x-text="scores.confidence"></span>/10</p>
                            </div>
                        </div>
                        
                        <div class="bg-primary-light p-6 rounded-lg">
                            <h3 class="font-semibold text-primary-dark mb-4 text-lg">Detailed Feedback</h3>
                            <div class="prose max-w-none text-gray-700 whitespace-pre-line leading-relaxed" 
                                x-text="feedbackText">
                            </div>
                        </div>
                    </div>
                </section>
            
                <!-- Step 5: Get Model Answer -->
                <section x-show="feedbackText" class="card">
                    <div class="card-header">
                        <h2 class="text-2xl font-bold flex items-center">
                            <span class="inline-flex items-center justify-center w-8 h-8 rounded-full bg-primary-light text-primary mr-3">5</span>
                            Get Model Answer
                        </h2>
                    </div>
                    <div class="card-body">
                        <div class="flex justify-center mb-6">
                            <button 
                                @click="generateModelAnswer()" 
                                :disabled="isGeneratingModel" 
                                class="btn-primary"
                            >
                                <span class="spinner" x-show="isGeneratingModel"></span>
                                <i class="fas fa-magic mr-2" x-show="!isGeneratingModel"></i>
                                <span x-text="isGeneratingModel ? 'Generating...' : 'Generate Model Answer'"></span>
                            </button>
                        </div>
                        
                        <template x-if="modelAnswer">
                            <div class="bg-primary-light p-6 rounded-lg">
                                <h3 class="font-semibold text-primary-dark mb-4 text-lg">Sample Professional Answer</h3>
                                <div class="prose max-w-none text-gray-700 whitespace-pre-line leading-relaxed" 
                                    x-text="modelAnswer">
                                </div>
                            </div>
                        </template>
                    </div>
                </section>
            
                <!-- Step 5.5: Follow-up Questions -->
                <section x-show="feedbackText && modelAnswer" class="card">
                    <div class="card-header">
                        <h2 class="text-2xl font-bold flex items-center">
                            <span class="inline-flex items-center justify-center w-8 h-8 rounded-full bg-primary-light text-primary mr-3">5.5</span>
                            Potential Follow-up Questions
                        </h2>
                    </div>
                    <div class="card-body">
                        <div class="flex justify-center mb-6">
                            <button 
                                @click="generateFollowUpQuestions()" 
                                :disabled="isGeneratingFollowUp" 
                                class="btn-primary"
                            >
                                <span class="spinner" x-show="isGeneratingFollowUp"></span>
                                <i class="fas fa-question-circle mr-2" x-show="!isGeneratingFollowUp"></i>
                                <span x-text="isGeneratingFollowUp ? 'Generating...' : 'Generate Follow-up Questions'"></span>
                            </button>
                        </div>
                        
                        <template x-if="followUpQuestions && followUpQuestions.length > 0">
                            <div class="bg-primary-light p-6 rounded-lg">
                                <h3 class="font-semibold text-primary-dark mb-4 text-lg">Questions the Interviewer Might Ask Next</h3>
                                <ul class="space-y-4 list-disc pl-6">
                                    <template x-for="(question, index) in followUpQuestions" :key="index">
                                        <li class="prose leading-relaxed" x-text="question"></li>
                                    </template>
                                </ul>
                            </div>
                        </template>
                    </div>
                </section>
            
                <!-- Step 6: Save Work -->
                <section x-show="feedbackText && modelAnswer" class="card">
                    <div class="card-header">
                        <h2 class="text-2xl font-bold flex items-center">
                            <span class="inline-flex items-center justify-center w-8 h-8 rounded-full bg-primary-light text-primary mr-3">6</span>
                            Save Your Work
                        </h2>
                    </div>
                    <div class="card-body">
                        <div class="flex justify-center">
                            <button 
                                @click="saveToHTML()" 
                                :disabled="isSaving" 
                                class="btn-primary"
                            >
                                <span class="spinner" x-show="isSaving"></span>
                                <i class="fas fa-download mr-2" x-show="!isSaving"></i>
                                <span x-text="isSaving ? 'Saving...' : 'Save as HTML'"></span>
                            </button>
                        </div>
                        
                        <template x-if="downloadLink">
                            <div class="mt-6 p-5 bg-primary-light rounded-lg text-center">
                                <p class="text-primary-dark mb-4 font-medium">Your file is ready to download!</p>
                                <a 
                                    :href="downloadLink" 
                                    class="btn-primary inline-flex" 
                                    download
                                >
                                    <i class="fas fa-cloud-download-alt mr-2"></i> Download File
                                </a>
                            </div>
                        </template>
                    </div>
                </section>
            
                <!-- Step 7: Continue or Start New -->
                <section x-show="feedbackText && modelAnswer" class="card">
                    <div class="card-header">
                        <h2 class="text-2xl font-bold flex items-center">
                            <span class="inline-flex items-center justify-center w-8 h-8 rounded-full bg-primary-light text-primary mr-3">7</span>
                            Continue Your Practice
                        </h2>
                    </div>
                    <div class="card-body">
                        <div class="grid md:grid-cols-2 gap-6">
                            <div class="bg-primary-light p-6 rounded-lg text-center">
                                <h3 class="font-semibold text-primary-dark mb-4 text-lg">Practice Another Question</h3>
                                <p class="text-gray-600 mb-6">Continue practicing with the same job information and try another interview question.</p>
                                <button 
                                    @click="practiceAnotherQuestion()" 
                                    class="btn-primary"
                                >
                                    <i class="fas fa-redo mr-2"></i> Try Another Question
                                </button>
                            </div>
                            
                            <div class="bg-primary-light p-6 rounded-lg text-center">
                                <h3 class="font-semibold text-primary-dark mb-4 text-lg">Start Fresh</h3>
                                <p class="text-gray-600 mb-6">Clear all data and start over with a different company or position.</p>
                                <button 
                                    @click="startOver()" 
                                    class="btn-secondary"
                                >
                                    <i class="fas fa-sync mr-2"></i> Start New Practice
                                </button>
                            </div>
                        </div>
                    </div>
                </section>
            </div>
        </main>
        
        <footer class="bg-primary py-8 text-center mt-12">
            <div class="container mx-auto px-4">
                <p class="text-white/90">Husky Interview Prep © 2025. All rights reserved.</p>
                <p class="text-white/70 text-sm mt-2">Powered by Cathy and AI and created to help you succeed.</p>
            </div>
        </footer>
    </div>

    <script>
        function app() {
            return {
                jobDesc: '',
                companyInfo: '',
                resume: '',
                parsedInfo: {},
                
                // Questions
                questions: {},
                questionHints: {},
                activeCategory: 'Introduction',
                selectedQuestion: '',
                questionsGenerated: false,
                
                // Loading states
                isAnalyzing: false,
                isGeneratingQuestions: false,
                isReadingAloud: false,
                isTranscribing: false,
                isAnalyzingAnswer: false,
                isGeneratingModel: false,
                isSaving: false,
                
                // Recording
                isRecording: false,
                recorder: null,
                audioChunks: [],
                recordingStatus: 'Click to start recording',
                answerText: '',
                
                // Analysis
                scores: null,
                feedbackText: '',
                modelAnswer: '',
                
                // Follow-up questions
                followUpQuestions: [],
                isGeneratingFollowUp: false,
                
                // Audio playback
                voiceOption: 'US English',
                audioSrc: '',
                audioPlaying: false,
                
                // Download
                downloadLink: '',
                
                async analyzeInfo() {
                    if (!this.jobDesc || !this.companyInfo) {
                        alert('Please enter job description and company information.');
                        return;
                    }
                    
                    this.isAnalyzing = true;  // Start loading indicator
                    
                    try {
                        const response = await fetch('/analyze-info', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                job_desc: this.jobDesc,
                                company_info: this.companyInfo
                            }),
                        });
                        
                        this.parsedInfo = await response.json();
                    } catch (error) {
                        console.error('Error analyzing information:', error);
                        alert('Error analyzing information. Please try again.');
                    } finally {
                        this.isAnalyzing = false;  // Stop loading indicator
                    }
                },
                
                async generateQuestions() {
                    if (!this.jobDesc) {
                        alert('Please enter at least the job description.');
                        return;
                    }
                    
                    this.isGeneratingQuestions = true;  // Start loading indicator
                    
                    try {
                        const response = await fetch('/generate-questions', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                job_desc: this.jobDesc,
                                company_info: this.companyInfo,
                                resume: this.resume
                            }),
                        });
                        
                        const data = await response.json();
                        this.questions = data.questions;
                        this.questionHints = data.hints;
                        this.questionsGenerated = true;
                        
                        // Set default active category to the first one
                        if (Object.keys(this.questions).length > 0) {
                            this.activeCategory = Object.keys(this.questions)[0];
                        }
                    } catch (error) {
                        console.error('Error generating questions:', error);
                        alert('Error generating questions. Please try again.');
                    } finally {
                        this.isGeneratingQuestions = false;  // Stop loading indicator
                    }
                },
                
                selectQuestion(question) {
                    this.selectedQuestion = question;
                    // Reset related data
                    this.answerText = '';
                    this.scores = null;
                    this.feedbackText = '';
                    this.modelAnswer = '';
                    this.audioSrc = '';
                    this.audioPlaying = false;
                },
                
                async readQuestionAloud() {
                    if (!this.selectedQuestion) return;
                    
                    this.isReadingAloud = true;  // Start loading indicator
                    
                    try {
                        const response = await fetch('/text-to-speech', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                text: this.selectedQuestion,
                                voice_option: this.voiceOption
                            }),
                        });
                        
                        const data = await response.json();
                        if (data.audio) {
                            this.audioSrc = data.audio;
                            this.audioPlaying = true;
                            
                            // Give DOM time to update before accessing the audio element
                            setTimeout(() => {
                                const audio = this.$refs.audioPlayer;
                                if (audio) {
                                    audio.src = data.audio;
                                    audio.load();
                                    audio.play().catch(e => console.error('Error playing audio:', e));
                                }
                            }, 100);
                        }
                    } catch (error) {
                        console.error('Error reading question aloud:', error);
                        alert('Error reading question aloud. Please try again.');
                    } finally {
                        this.isReadingAloud = false;  // Stop loading indicator
                    }
                },
                
                toggleRecording() {
                    if (this.isRecording) {
                        this.stopRecording();
                    } else {
                        this.startRecording();
                    }
                },
                
                async startRecording() {
                    try {
                        // Set audio constraints for better compatibility
                        const stream = await navigator.mediaDevices.getUserMedia({
                            audio: {
                                channelCount: 1,
                                sampleRate: 16000,
                                sampleSize: 16,
                                echoCancellation: true,
                                noiseSuppression: true
                            }
                        });
                        
                        this.recorder = new MediaRecorder(stream, {
                            mimeType: 'audio/webm' // More widely supported in browsers
                        });
                        this.audioChunks = [];
                        
                        this.recorder.ondataavailable = (e) => {
                            if (e.data.size > 0) {
                                this.audioChunks.push(e.data);
                            }
                        };
                        
                        this.recorder.onstop = async () => {
                            const audioBlob = new Blob(this.audioChunks, { type: 'audio/webm' });
                            const reader = new FileReader();
                            reader.readAsDataURL(audioBlob);
                            
                            reader.onload = async () => {
                                const base64Audio = reader.result;
                                
                                try {
                                    const response = await fetch('/speech-to-text', {
                                        method: 'POST',
                                        headers: {
                                            'Content-Type': 'application/json',
                                        },
                                        body: JSON.stringify({ audio: base64Audio }),
                                    });
                                    
                                    const data = await response.json();
                                    this.answerText = data.text;
                                    this.recordingStatus = 'Recording transcribed';
                                } catch (error) {
                                    console.error('Error transcribing audio:', error);
                                    this.recordingStatus = 'Error transcribing audio';
                                } finally {
                                    this.isTranscribing = false;  // Stop transcribing indicator
                                }
                            };
                        };
                        
                        this.recorder.start();
                        this.isRecording = true;
                        this.recordingStatus = 'Recording... Click to stop';
                    } catch (error) {
                        console.error('Error starting recording:', error);
                        this.recordingStatus = 'Error accessing microphone';
                    }
                },
                
                stopRecording() {
                    if (this.recorder && this.recorder.state !== 'inactive') {
                        this.recorder.stop();
                        this.isRecording = false;
                        this.isTranscribing = true;  // Start transcribing indicator
                        this.recordingStatus = 'Processing...';
                        
                        // Stop all audio tracks
                        this.recorder.stream.getTracks().forEach(track => track.stop());
                    }
                },
                
                async analyzeAnswer() {
                    if (!this.answerText) {
                        alert('Please record or enter an answer to analyze.');
                        return;
                    }
                    
                    this.isAnalyzingAnswer = true;  // Start loading indicator
                    
                    try {
                        const response = await fetch('/analyze-answer', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                answer_text: this.answerText,
                                job_desc: this.jobDesc,
                                company_values: this.parsedInfo.company_values
                            }),
                        });
                        
                        const data = await response.json();
                        // Ensure scores are properly initialized with numeric values
                        this.scores = {
                            clarity: parseInt(data.scores.clarity) || 0,
                            relevance: parseInt(data.scores.relevance) || 0,
                            confidence: parseInt(data.scores.confidence) || 0
                        };
                        this.feedbackText = data.feedback;
                    } catch (error) {
                        console.error('Error analyzing answer:', error);
                        alert('Error analyzing answer. Please try again.');
                    } finally {
                        this.isAnalyzingAnswer = false;  // Stop loading indicator
                    }
                },
                
                async generateModelAnswer() {
                    if (!this.selectedQuestion) {
                        alert('Please select a question first.');
                        return;
                    }
                    
                    this.isGeneratingModel = true;  // Start loading indicator
                    
                    try {
                        const response = await fetch('/generate-model-answer', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                question: this.selectedQuestion,
                                company_info: this.companyInfo,
                                job_desc: this.jobDesc,
                                resume: this.resume,
                                answer_text: this.answerText
                            }),
                        });
                        
                        const data = await response.json();
                        this.modelAnswer = data.model_answer;
                    } catch (error) {
                        console.error('Error generating model answer:', error);
                        alert('Error generating model answer. Please try again.');
                    } finally {
                        this.isGeneratingModel = false;  // Stop loading indicator
                    }
                },
                
                async saveToHTML() {
                    this.isSaving = true;
                    
                    try {
                        const response = await fetch('/save-to-html', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                job_desc: this.jobDesc,
                                company_info: this.companyInfo,
                                resume: this.resume,
                                company_name: this.parsedInfo.company_name,
                                position_title: this.parsedInfo.position_title,
                                company_values: this.parsedInfo.company_values,
                                tech_skills: this.parsedInfo.tech_skills,
                                soft_skills: this.parsedInfo.soft_skills,
                                job_duties: this.parsedInfo.job_duties,
                                selected_question: this.selectedQuestion,
                                answer_text: this.answerText,
                                feedback: this.feedbackText,
                                model_answer: this.modelAnswer,
                                follow_up_questions: this.followUpQuestions
                            }),
                        });
                        
                        const data = await response.json();
                        this.downloadLink = `/download-html/${data.file_id}`;
                    } catch (error) {
                        console.error('Error saving to HTML:', error);
                        alert('Error saving to HTML. Please try again.');
                    } finally {
                        this.isSaving = false;
                    }
                },
                
                practiceAnotherQuestion() {
                    // Reset answer-related data but keep job info
                    this.selectedQuestion = '';
                    this.answerText = '';
                    this.scores = null;
                    this.feedbackText = '';
                    this.modelAnswer = '';
                    this.followUpQuestions = [];
                    this.audioSrc = '';
                    this.audioPlaying = false;
                    this.downloadLink = '';
                    
                    // Scroll to the questions section
                    document.querySelector('section:nth-of-type(2)').scrollIntoView({ behavior: 'smooth' });
                },
                
                startOver() {
                    // Reset everything
                    this.jobDesc = '';
                    this.companyInfo = '';
                    this.resume = '';
                    this.parsedInfo = {};
                    this.questions = {};
                    this.questionHints = {};
                    this.activeCategory = 'Introduction';
                    this.selectedQuestion = '';
                    this.questionsGenerated = false;
                    this.answerText = '';
                    this.scores = null;
                    this.feedbackText = '';
                    this.modelAnswer = '';
                    this.followUpQuestions = [];
                    this.audioSrc = '';
                    this.audioPlaying = false;
                    this.downloadLink = '';
                    
                    // Scroll to top
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                },
                
                async generateFollowUpQuestions() {
                    if (!this.selectedQuestion) {
                        alert('Please select a question first.');
                        return;
                    }
                    
                    this.isGeneratingFollowUp = true;  // Start loading indicator
                    
                    try {
                        const response = await fetch('/generate-follow-up-questions', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json',
                            },
                            body: JSON.stringify({
                                question: this.selectedQuestion,
                                job_desc: this.jobDesc,
                                resume: this.resume,
                                answer_text: this.answerText
                            }),
                        });
                        
                        const data = await response.json();
                        this.followUpQuestions = data.follow_up_questions;
                    } catch (error) {
                        console.error('Error generating follow-up questions:', error);
                        alert('Error generating follow-up questions. Please try again.');
                    } finally {
                        this.isGeneratingFollowUp = false;  // Stop loading indicator
                    }
                }
            };
        }
    </script>
</body>
</html>
''')
    
    app.run(debug=True, host='0.0.0.0', port=5002)
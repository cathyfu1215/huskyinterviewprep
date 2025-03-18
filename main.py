import gradio as gr
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

load_dotenv()

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

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content

class Analyzer:
    def __init__(self):
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def parse_job_info(self, job_description, company_values):
        """Extracts key insights and fills relevant fields."""
        prompt = f"""
        SYSTEM: You are an expert job description analyzer. Extract relevant details from the provided job description and company values.
        
        INSTRUCTIONS:
        - Identify key company values.
        - Extract essential technical skills required.
        - Extract necessary soft skills.
        - Summarize key job duties.
        
        JOB DESCRIPTION: {job_description}
        COMPANY VALUES: {company_values}
        """
        response = prompt_llm(prompt)

        # Print the response for debugging
        # print("LLM Response:", response)
        
        # Use regular expressions to extract the relevant sections
        company_values_match = re.search(r"\*\*Key Company Values:\*\*(.*?)\*\*Essential Technical Skills:\*\*", response, re.DOTALL)
        tech_skills_match = re.search(r"\*\*Essential Technical Skills:\*\*(.*?)\*\*Necessary Soft Skills:\*\*", response, re.DOTALL)
        soft_skills_match = re.search(r"\*\*Necessary Soft Skills:\*\*(.*?)\*\*Summary of Key Job Duties:\*\*", response, re.DOTALL)
        job_duties_match = re.search(r"\*\*Summary of Key Job Duties:\*\*(.*)", response, re.DOTALL)

        company_values = company_values_match.group(1).strip() if company_values_match else "Not found"
        tech_skills = tech_skills_match.group(1).strip() if tech_skills_match else "Not found"
        soft_skills = soft_skills_match.group(1).strip() if soft_skills_match else "Not found"
        job_duties = job_duties_match.group(1).strip() if job_duties_match else "Not found"

        parsed_info = {
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
        SYSTEM: You are a professional interview coach. Draft a strong, structured answer based on the following inputs:
        
        INSTRUCTIONS:
        - Ensure clarity and logical flow.
        - Incorporate company values where relevant.
        - Highlight technical and soft skills from the job description.
        - Improve conciseness while maintaining completeness.
        - Maintain a confident and positive tone.
        
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
        SYSTEM: You are an expert evaluator for interview responses. Assess the answer based on the following criteria:
        
        INSTRUCTIONS:
        - Clarity: Is the response structured and easy to understand?
        - Relevance: Does it address the job's required skills and reflect company values?
        - Confidence: Does the tone convey certainty and professionalism?
        - Provide constructive feedback and a score out of 10 for each category.
        
        USER VOICE ANSWER: {voice_answer}
        JOB DESCRIPTION: {job_description}
        COMPANY VALUES: {company_values}
        """
        response = prompt_llm(prompt)
        # print("LLM Response:", response)

        
        # Extract scores and feedback from the response
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

class InterviewAgentManager:
    def __init__(self):
        self.analyzer = Analyzer()
        self.drafter = Drafter()
        self.evaluator = Evaluator()
    
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

def analyze_info(job_desc, company_info):
    parsed_info = interview_manager.analyzer.parse_job_info(job_desc, company_info)
    return parsed_info

def generate_model_answer(question, company_info, job_desc, resume, voice_answer):
    """Generate a model answer based on the question and all available context"""
    return interview_manager.drafter.generate_answer(question, company_info, job_desc, resume, voice_answer)

def evaluate_answer(voice_answer, job_desc, company_values):
    return interview_manager.evaluator.evaluate_answer(voice_answer, job_desc, company_values)

# The integration with Gradio should call these functions appropriately

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
            "Focus on experience that directly relates to the job requirements. Use the STAR method."
    }

def generate_sample_questions(job_desc, company_info, resume):
    """Simulate generating interview questions based on all inputs"""
    # Basic questions
    questions = [
        "Tell me about yourself",
        "What's your greatest strength?",
        "Why do you want this job?",
        "Where do you see yourself in 5 years?"
    ]
    
    # Add company-specific question if company info is provided
    if company_info.strip():
        questions.append(f"Why do you want to work at our company?")
    
    # Add experience-related question if resume is provided
    if resume.strip():
        questions.append("Tell me about your most relevant experience for this role")
    
    return questions

def speech_to_text(audio_path):
    """Convert speech to text"""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except:
        return "Speech recognition failed. Please try again."

def analyze_answer(answer):
    """Simple mock analysis of the answer"""
    # Simulate scoring
    scores = {
        "clarity": np.random.uniform(0.6, 1.0),
        "confidence": np.random.uniform(0.6, 1.0),
        "relevance": np.random.uniform(0.6, 1.0)
    }
    
    feedback = "Sample feedback: Try to be more specific and provide concrete examples."
    return scores, feedback

def analyze_information(job_desc, company_info):
    """Analyze the job description and company information to extract values, tech skills, soft skills, and job duties"""
    parsed_info = interview_manager.analyzer.parse_job_info(job_desc, company_info)
    
    # Assuming the parsed_info is a dictionary with keys: 'company_values', 'tech_skills', 'soft_skills', 'job_duties'
    company_values = parsed_info.get('company_values', 'Not found')
    tech_skills = parsed_info.get('tech_skills', 'Not found')
    soft_skills = parsed_info.get('soft_skills', 'Not found')
    job_duties = parsed_info.get('job_duties', 'Not found')
    
    return company_values, tech_skills, soft_skills, job_duties

# Global variables to store persistent inputs
persistent_job_desc = ""
persistent_company_info = ""
persistent_resume = ""
persistent_company_values = ""
persistent_tech_skills = ""
persistent_soft_skills = ""
persistent_job_duties = ""

def save_to_html(job_desc, company_info, resume, company_values, tech_skills, soft_skills, job_duties, selected_question, answer_text, feedback, model_answer):
    """Generate HTML content for download."""
    # Get current date and time
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    file_name = f"interview_summary_{current_time}.html"

    html_content = f"""
    <html>
    <head>
        <title>Interview Prep Summary</title>
        <style>
            body {{
                font-family: 'Arial', sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f4f4f9;
                color: #333;
            }}
            h1, h2 {{
                color: #4f46e5;
            }}
            h1 {{
                text-align: center;
                margin-bottom: 40px;
            }}
            h2 {{
                margin-top: 30px;
                border-bottom: 2px solid #ddd;
                padding-bottom: 10px;
            }}
            p, ul {{
                line-height: 1.6;
            }}
            ul {{
                list-style-type: none;
                padding: 0;
            }}
            li {{
                background: #fff;
                margin: 5px 0;
                padding: 10px;
                border-radius: 5px;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            }}
        </style>
    </head>
    <body>
        <h1>Interview Prep Summary</h1>
        <h2>Job Description</h2>
        <p>{job_desc}</p>
        <h2>Company Information</h2>
        <p>{company_info}</p>
        <h2>Your Resume</h2>
        <p>{resume}</p>
        <h2>Parsed Information</h2>
        <ul>
            <li><strong>Company Values:</strong> {company_values}</li>
            <li><strong>Tech Skills:</strong> {tech_skills}</li>
            <li><strong>Soft Skills:</strong> {soft_skills}</li>
            <li><strong>Job Duties:</strong> {job_duties}</li>
        </ul>
        <h2>Selected Question</h2>
        <p>{selected_question}</p>
        <h2>Your Answer (Transcribed)</h2>
        <p>{answer_text}</p>
        <h2>Feedback</h2>
        <p>{feedback}</p>
        <h2>Model Answer</h2>
        <p>{model_answer}</p>
    </body>
    </html>
    """
    # Use a temporary file to store the HTML content for download
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
        tmp_file.write(html_content.encode('utf-8'))
        tmp_file_path = tmp_file.name
    return tmp_file_path

def create_demo():
    question_hints = get_question_hints()
    
    with gr.Blocks(css="""
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        #header-container {
            margin: -40px -20px 20px -20px;
            padding: 40px 20px;
            background: linear-gradient(to right, #9333ea, #4f46e5);
            position: relative;
            overflow: hidden;
            font-family: 'Inter', sans-serif;
        }
        
        /* Animated gradient orbs */
        #header-container::before {
            content: '';
            position: absolute;
            top: -50%;
            right: -20%;
            width: 300px;
            height: 300px;
            background: #a855f7;
            border-radius: 50%;
            filter: blur(80px);
            opacity: 0.3;
            animation: pulse 3s infinite;
        }
        
        #header-container::after {
            content: '';
            position: absolute;
            bottom: -50%;
            left: -20%;
            width: 300px;
            height: 300px;
            background: #6366f1;
            border-radius: 50%;
            filter: blur(80px);
            opacity: 0.3;
            animation: pulse 3s infinite 1.5s;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 0.3; }
            50% { opacity: 0.4; }
        }
        
        @keyframes float {
            0%, 100% { transform: translateY(0); }
            50% { transform: translateY(-10px); }
        }
        
        #header-badge {
            display: inline-block;
            padding: 8px 16px;
            background: rgba(255, 255, 255, 0.1);
            backdrop-filter: blur(8px);
            border-radius: 9999px;
            border: 1px solid rgba(255, 255, 255, 0.2);
            margin-bottom: 20px;
            animation: float 3s ease-in-out infinite;
        }
        
        #header-badge-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #4ade80;
            border-radius: 50%;
            margin-right: 8px;
            animation: pulse 2s infinite;
        }
        
        #header-title {
            font-size: 48px;
            font-weight: bold;
            color: white;
            text-align: center;
            margin: 0;
            padding: 20px 0;
            position: relative;
            z-index: 1;
        }
        
        #header-subtitle {
            text-align: center;
            color: rgba(255, 255, 255, 0.9);
            font-size: 20px;
            margin: 0;
            padding-bottom: 20px;
            position: relative;
            z-index: 1;
        }

        .container {
            background: white;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }

        .section-title {
            color: #4f46e5;
            font-family: 'Inter', sans-serif;
            font-weight: 600;
            margin-bottom: 16px;
        }
    """) as demo:
        # Stylish Header
        with gr.Column(elem_id="header-container"):
            gr.HTML(
                """
                <div id="header-badge">
                    <span id="header-badge-dot"></span>
                    <span style="color: white; font-size: 14px;">AI-Powered Interview Prep</span>
                </div>
                <h1 id="header-title">Husky Interview Prep</h1>
                <p id="header-subtitle">Master Your Interview with Confidence</p>
                """
            )
        
        # Main Content
        with gr.Column(elem_classes="container"):
            gr.Markdown(
                """### Step 1: Enter Your Information""",
                elem_classes="section-title"
            )
            job_desc = gr.Textbox(
                label="Job Description",
                placeholder="Paste the job description here...",
                lines=3
            )
            company_info = gr.Textbox(
                label="Company Information",
                placeholder="Enter information about the company...",
                lines=2
            )
            resume = gr.Textbox(
                label="Your Resume",
                placeholder="Paste your resume or relevant experience here...",
                lines=3
            )
            analyze_info_btn = gr.Button("Analyze Information", variant="primary")
            company_values = gr.Textbox(
                label="Company Values",
                placeholder="Waiting for parsing the information...",
                lines=2
            )
            tech_skills = gr.Textbox(
                label="Tech Skills",
                placeholder="Waiting for parsing the information...",
                lines=2
            )
            soft_skills = gr.Textbox(
                label="Soft Skills",
                placeholder="Waiting for parsing the information...",
                lines=2
            )
            job_duties = gr.Textbox(
                label="Job Duties",
                placeholder="Waiting for parsing the information...",
                lines=2
            )
            generate_btn = gr.Button("Generate Interview Questions", variant="primary")

        with gr.Column(elem_classes="container"):
            gr.Markdown(
                """### Step 2: Practice Questions""",
                elem_classes="section-title"
            )
            questions = gr.Radio(
                choices=[],
                label="Interview Questions",
                info="Select a question to practice"
            )

        with gr.Column(elem_classes="container"):
            gr.Markdown(
                """### Step 3: Record Your Answer""",
                elem_classes="section-title"
            )
            selected_question = gr.Textbox(
                label="Current Question",
                interactive=False
            )
            question_hint = gr.Textbox(
                label="How to Answer",
                interactive=False,
                lines=2
            )
            with gr.Row():
                audio_input = gr.Audio(
                    sources=["microphone"],
                    type="filepath",
                    label="Record Your Answer"
                )
                answer_text = gr.Textbox(
                    label="Your Answer (Transcribed)",
                    lines=3
                )
            analyze_btn = gr.Button("Analyze Answer", variant="primary")

        with gr.Column(elem_classes="container"):
            gr.Markdown(
                """### Step 4: Review Analysis""",
                elem_classes="section-title"
            )
            feedback = gr.Textbox(
                label="Analysis Results",
                lines=6
            )

        # Add new section for model answer
        with gr.Column(elem_classes="container"):
            gr.Markdown(
                """### Step 5: Get Model Answer""",
                elem_classes="section-title"
            )
            generate_model_btn = gr.Button("Generate Model Answer", variant="primary")
            model_answer = gr.Textbox(
                label="Model Answer",
                lines=4,
                interactive=False
            )

        # Add new section for saving to HTML
        with gr.Column(elem_classes="container"):
            gr.Markdown(
                """### Step 6: Save Your Work""",
                elem_classes="section-title"
            )
            save_btn = gr.Button("Save to HTML", variant="primary")
            download_link = gr.File(label="Download HTML File")

        # Event handlers
        def update_questions(job_desc, company_info, resume):
            global persistent_job_desc, persistent_company_info, persistent_resume
            persistent_job_desc = job_desc
            persistent_company_info = company_info
            persistent_resume = resume
            questions = generate_sample_questions(job_desc, company_info, resume)
            return gr.Radio(choices=questions)

        def update_selected_question(question):
            if question:
                hint = question_hints.get(question, "No hint available for this question.")
                return question, hint
            return "", ""

        def process_answer(audio, answer, job_desc, company_values):
            if audio is not None:
                answer = speech_to_text(audio)
            scores, feedback_text = interview_manager.evaluator.evaluate_answer(answer, job_desc, company_values)
            stars = lambda score: "⭐" * score + "☆" * (10 - score)
            combined_output = f"""SCORES:
Clarity: {stars(scores['clarity'])}
Relevance: {stars(scores['relevance'])}
Confidence: {stars(scores['confidence'])}

FEEDBACK:
{feedback_text}"""
            return combined_output

        def get_model_answer(question, company_info, job_desc, resume, answer_text):
            return generate_model_answer(question, company_info, job_desc, resume, answer_text)

        def analyze_info(job_desc, company_info):
            parsed_info = interview_manager.analyzer.parse_job_info(job_desc, company_info)
            return parsed_info['company_values'], parsed_info['tech_skills'], parsed_info['soft_skills'], parsed_info['job_duties']

        # Event bindings
      
        generate_btn.click(
            update_questions,
            inputs=[job_desc, company_info, resume],
            outputs=[questions]
        )

        questions.change(
            update_selected_question,
            inputs=[questions],
            outputs=[selected_question, question_hint]
        )

        audio_input.change(
            speech_to_text,
            inputs=[audio_input],
            outputs=[answer_text]
        )

        analyze_btn.click(
            process_answer,
            inputs=[audio_input, answer_text, job_desc, company_values],
            outputs=[feedback]
        )

        # Add new event binding for model answer
        generate_model_btn.click(
            get_model_answer,
            inputs=[selected_question, company_info, job_desc, resume, answer_text],
            outputs=[model_answer]
        )

        # Add new event binding for analyze information
        analyze_info_btn.click(
            analyze_info,
            inputs=[job_desc, company_info],
            outputs=[company_values, tech_skills, soft_skills, job_duties]
        )

        # Add new event binding for saving to HTML
        save_btn.click(
            save_to_html,
            inputs=[job_desc, company_info, resume, company_values, tech_skills, soft_skills, job_duties, selected_question, answer_text, feedback, model_answer],
            outputs=[download_link]
        )

    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)

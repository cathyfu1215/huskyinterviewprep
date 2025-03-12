import gradio as gr
import numpy as np
import pandas as pd
import openai
import speech_recognition as sr
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from fpdf import FPDF
import sqlite3
from datetime import datetime

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Use it in OpenAI
import openai
openai.api_key = openai_api_key

print("OpenAI API Key loaded successfully!")

def generate_questions(resume, cover_letter, job_description, company_info):
    # Simulate generating questions (replace with actual OpenAI call)
    questions = [
        "Tell me about yourself",
        "Why are you interested in this position?",
        "What's your greatest strength?",
        "Describe a challenging situation at work",
        "Why do you want to work for our company?",
        "Where do you see yourself in 5 years?",
        "Tell me about a project you're proud of",
        "How do you handle conflict?",
        "What's your leadership style?",
        "What questions do you have for us?"
    ]
    return questions

def speech_to_text(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except:
        return "Speech recognition failed. Please try again."

def analyze_answer(question, answer, resume, job_description, company_info):
    # Simulate analysis (replace with actual AI analysis)
    scores = {
        "clarity": np.random.uniform(0.6, 1.0),
        "confidence": np.random.uniform(0.6, 1.0),
        "situation": np.random.uniform(0.6, 1.0),
        "task": np.random.uniform(0.6, 1.0),
        "action": np.random.uniform(0.6, 1.0),
        "result": np.random.uniform(0.6, 1.0)
    }
    
    recommendations = "Focus on providing more specific examples and quantifiable results."
    model_answer = f"Model answer for: {question}\nThis is a sample model answer based on the provided resume and job description."
    
    return scores, recommendations, model_answer

def create_radar_chart(scores):
    categories = list(scores.keys())
    values = list(scores.values())
    values.append(values[0])
    categories.append(categories[0])
    
    fig = go.Figure(data=[
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself'
        )
    ])
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1]
            )
        ),
        showlegend=False
    )
    return fig

def save_to_database(data):
    conn = sqlite3.connect('interviews.db')
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS interviews
        (id INTEGER PRIMARY KEY AUTOINCREMENT,
         timestamp TEXT,
         resume TEXT,
         job_description TEXT,
         company_info TEXT,
         qa_pairs TEXT)
    ''')
    
    cursor.execute('''
        INSERT INTO interviews (timestamp, resume, job_description, company_info, qa_pairs)
        VALUES (?, ?, ?, ?, ?)
    ''', (datetime.now().isoformat(), data['resume'], data['job_description'],
          data['company_info'], str(data['qa_pairs'])))
    
    conn.commit()
    conn.close()

def create_pdf(data):
    pdf = FPDF()
    pdf.add_page()
    
    # Title
    pdf.set_font("Arial", "B", 16)
    pdf.cell(200, 10, txt="Interview Preparation Report", ln=True, align='C')
    pdf.ln(10)
    
    # Content
    pdf.set_font("Arial", size=12)
    
    # Interview Q&A Section
    pdf.set_font("Arial", "B", 14)
    pdf.cell(200, 10, txt="Interview Questions and Answers", ln=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", size=12)
    for qa in data['qa_pairs']:
        # Question
        pdf.set_font("Arial", "B", 12)
        pdf.multi_cell(0, 10, txt=f"Q: {qa['question']}")
        
        # Answer
        pdf.set_font("Arial", size=12)
        pdf.multi_cell(0, 10, txt=f"A: {qa['answer']}")
        pdf.ln(5)
    
    pdf_path = "interview_report.pdf"
    pdf.output(pdf_path)
    return pdf_path

def create_interface():
    # Shared state
    state = gr.State({
        "questions": [],
        "selected_questions": [],
        "current_question_index": 0,
        "answers": {},
        "analysis": {}
    })
    
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
    """) as app:
        # Header
        with gr.Column(elem_id="header-container"):
            gr.HTML(
                """
                <div id="header-badge">
                    <span id="header-badge-dot"></span>
                    <span style="color: white; font-size: 14px;">AI-Powered Interview Prep Training</span>
                </div>
                <h1 id="header-title">Husky Interview Prep</h1>
                <p id="header-subtitle">Master Your Interview with Confidence</p>
                """
            )
        
        with gr.Tabs() as tabs:
            with gr.Tab("Stage 1: Input Information", id="input"):
                resume = gr.Textbox(label="Resume", lines=5)
                cover_letter = gr.Textbox(label="Cover Letter", lines=5)
                job_description = gr.Textbox(label="Job Description", lines=5)
                company_info = gr.Textbox(label="Company Information", lines=3)
                generate_btn = gr.Button("Generate Questions")
                question_checkboxes = gr.CheckboxGroup(
                    choices=[], 
                    label="Select Questions",
                    value=[]  # This will be populated with all questions selected by default
                )
                next_btn_1 = gr.Button("Proceed to Interview")

            with gr.Tab("Stage 2: Interview Questions", id="interview"):
                current_question = gr.Textbox(label="Current Question", interactive=False)
                answer_hints = gr.Textbox(label="Hints for Answering", interactive=False)
                with gr.Row():  # Group the recording components together
                    audio_input = gr.Audio(sources=["microphone"], type="filepath")
                    transcribed_text = gr.Textbox(label="Your Answer")
                with gr.Row():  # Group the buttons together
                    save_answer_btn = gr.Button("Save Answer", variant="primary")
                    skip_btn = gr.Button("Skip Question", variant="secondary")
                    next_question_btn = gr.Button("Next Question", interactive=False)

            # Stage 3: Analysis
            with gr.Tab("Stage 3: Analysis"):
                question_selector = gr.Slider(minimum=0, maximum=1, step=1, label="Question Number")
                analysis_question = gr.Textbox(label="Question")
                user_answer = gr.Textbox(label="Your Answer")
                radar_plot = gr.Plot(label="Analysis Radar")
                recommendations = gr.Textbox(label="Recommendations")
                model_answer = gr.Textbox(label="Model Answer")

            # Stage 4: Export
            with gr.Tab("Stage 4: Export"):
                export_pdf_btn = gr.Button("Export to PDF")
                download_btn = gr.File(label="Download Report")

        # Event handlers
        def proceed_to_interview(selected_questions, state_dict):
            if not selected_questions:
                gr.Warning("Please select at least one question")
                return {
                    "questions": [],
                    "selected_questions": [],
                    "current_question_index": 0,
                    "answers": {},
                    "analysis": {}
                }, "", "", "", True, True, False
            
            # Update state
            new_state = {
                "questions": state_dict.get("questions", []),
                "selected_questions": selected_questions,
                "current_question_index": 0,
                "answers": {},
                "analysis": {}
            }
            
            # Get first question
            current_q = selected_questions[0]
            hint = f"Hint for question: {current_q}"
            
            return (
                new_state,          # updated state
                current_q,          # current question
                hint,               # hint text
                "",                 # clear answer textbox
                True,               # save button interactive
                True,               # skip button interactive
                False               # next button interactive
            )

        next_btn_1.click(
            proceed_to_interview,
            inputs=[
                question_checkboxes,
                state
            ],
            outputs=[
                state,
                current_question,
                answer_hints,
                transcribed_text,
                save_answer_btn,
                skip_btn,
                next_question_btn
            ]
        )

        # Separate function for tab navigation
        def switch_to_interview():
            return gr.Tabs(selected=1)

        # Add another click handler just for tab switching
        next_btn_1.click(
            switch_to_interview,
            outputs=[tabs]
        )

        # Event handler for generating questions
        def update_questions(resume, cover_letter, job_description, company_info):
            questions = generate_questions(resume, cover_letter, job_description, company_info)
            return gr.CheckboxGroup(choices=questions, value=questions)

        generate_btn.click(
            update_questions,
            inputs=[resume, cover_letter, job_description, company_info],
            outputs=[question_checkboxes]
        )

    return app

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()

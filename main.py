import gradio as gr
import numpy as np
import speech_recognition as sr

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

def generate_model_answer(question, user_answer):
    """Generate a model answer based on the question and user's response"""
    # This is a mock implementation - in a real app, you might use an LLM
    model_answers = {
        "Tell me about yourself": 
            "Hi, I'm [Name], a [profession] with [X] years of experience in [industry]. I've developed expertise in [key skills] through my work at [previous companies]. In my current role at [company], I [key achievement]. I'm particularly passionate about [relevant interest], which aligns well with this position.",
        "What's your greatest strength?":
            "One of my greatest strengths is [specific strength]. For example, in my previous role at [company], I [specific example that demonstrates the strength]. This resulted in [quantifiable outcome]. I believe this strength would be particularly valuable in this position because [reason].",
    }
    return model_answers.get(question, "Model answer not available for this question.")

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
            generate_btn = gr.Button("Generate Questions", variant="primary")

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
            with gr.Row():
                score_output = gr.Json(label="Scores")
                feedback = gr.Textbox(
                    label="Feedback",
                    lines=2
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

        # Event handlers
        def update_questions(job_desc, company_info, resume):
            questions = generate_sample_questions(job_desc, company_info, resume)
            return gr.Radio(choices=questions)

        def update_selected_question(question):
            if question:
                hint = question_hints.get(question, "No hint available for this question.")
                return question, hint
            return "", ""

        def process_answer(audio, answer):
            if audio is not None:
                answer = speech_to_text(audio)
            scores, feedback = analyze_answer(answer)
            return scores, feedback

        def get_model_answer(question, user_answer):
            return generate_model_answer(question, user_answer)

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
            inputs=[audio_input, answer_text],
            outputs=[score_output, feedback]
        )

        # Add new event binding for model answer
        generate_model_btn.click(
            get_model_answer,
            inputs=[selected_question, answer_text],
            outputs=[model_answer]
        )

    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch()

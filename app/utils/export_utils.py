"""
Export utilities - Functions for creating and exporting HTML reports
"""
import tempfile
from datetime import datetime


def save_to_html(
    job_desc, company_info, resume, company_name, position_title,
    company_values, tech_skills, soft_skills, job_duties,
    selected_question, answer_text, feedback, model_answer,
    follow_up_questions=None
):
    """Generate HTML content for download.
    
    Args:
        job_desc (str): Job description text
        company_info (str): Company information
        resume (str): User's resume
        company_name (str): Company name
        position_title (str): Position title
        company_values (str): Company values text
        tech_skills (str): Technical skills text
        soft_skills (str): Soft skills text
        job_duties (str): Job duties text
        selected_question (str): The interview question
        answer_text (str): User's answer
        feedback (str): Feedback on user's answer
        model_answer (str): Model answer
        follow_up_questions (list): List of follow-up questions
        
    Returns:
        str: Path to the generated HTML file
    """
    # Build HTML content
    html_content = f"""
    <html>
    <head>
        <title>Interview Preparation Summary</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;'
                + '400;500;600;700&display=swap');
            
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
                    <strong>Company:</strong> 
                    {company_name if company_name else 'Not specified'} |
                    <strong>Position:</strong> 
                    {position_title if position_title else 'Not specified'}
                </div>
                <div class="timestamp">
                    Generated on 
                    {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
                </div>
            </div>
        </header>
        <div class="container">
            <!-- Parsed Information -->
            <div class="info-section">
                <h2 class="section-title">Job Analysis</h2>
                <ul>
                    <li class="info-item">
                        <strong>Company Values:</strong> {company_values}
                    </li>
                    <li class="info-item">
                        <strong>Tech Skills:</strong> {tech_skills}
                    </li>
                    <li class="info-item">
                        <strong>Soft Skills:</strong> {soft_skills}
                    </li>
                    <li class="info-item">
                        <strong>Job Duties:</strong> {job_duties}
                    </li>
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
            {follow_up_html(follow_up_questions) if 
            follow_up_questions else ""}
            
            <!-- Feedback -->
            <div class="info-section">
                <h2 class="section-title">Performance Analysis</h2>
                <div class="score-section">
                    <div class="score-item">
                        <div class="score-title">Clarity</div>
                        <div class="stars">{"★" * 5 + "☆" * 5}</div>
                    </div>
                    <div class="score-item">
                        <div class="score-title">Relevance</div>
                        <div class="stars">{"★" * 5 + "☆" * 5}</div>
                    </div>
                    <div class="score-item">
                        <div class="score-title">Confidence</div>
                        <div class="stars">{"★" * 5 + "☆" * 5}</div>
                    </div>
                </div>
                
                <div class="feedback">
                    <h3 style="margin-top: 0; margin-bottom: 15px; color: #006241;">
                        Detailed Feedback
                    </h3>
                    {feedback}
                </div>
            </div>
            
            <footer>
                <p>Husky Interview Prep &copy; 2025</p>
                <p>
                    Generated on 
                    {datetime.now().strftime("%B %d, %Y at %I:%M %p")}
                </p>
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


def follow_up_html(follow_up_questions):
    """Generate HTML for follow-up questions section.
    
    Args:
        follow_up_questions (list): List of follow-up questions
        
    Returns:
        str: HTML for the follow-up questions section
    """
    section_html = """
    <div class='info-section'>
        <h2 class='section-title'>Potential Follow-up Questions</h2>
        <div class='follow-up-questions'>
    """
    
    for question in follow_up_questions:
        section_html += f"""
        <div class='follow-up-question'>
            <p>{question}</p>
        </div>
        """
    
    section_html += """
        </div>
    </div>
    """
    
    return section_html 
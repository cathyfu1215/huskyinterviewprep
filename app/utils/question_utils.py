"""
Question utilities - Functions for managing interview questions and hints
"""


def get_question_hints():
    """
    Return a dictionary of questions and their hints.
    
    Returns:
        dict: Mapping of question text to hint text
    """
    return {
        "Tell me about yourself": 
            "Focus on your professional background, key achievements, and why "
            "you're a good fit for this role.",
        
        "What's your greatest strength?": 
            "Choose a strength relevant to the job. Provide specific examples "
            "that demonstrate this strength.",
        
        "Why do you want this job?": 
            "Connect your skills and career goals to the role and company. "
            "Show you've done your research.",
        
        "Where do you see yourself in 5 years?": 
            "Discuss your career goals and how they align with the company's "
            "growth trajectory.",
        
        "Why do you want to work at our company?": 
            "Demonstrate your knowledge of the company's values, culture, "
            "and mission.",
        
        "Tell me about your most relevant experience for this role": 
            "Focus on experience that directly relates to the job requirements. "
            "Use the STAR method.",
        
        "Describe a time you led/motivated others. How were you able to?":
            "Describe a time when you led a team through a challenge. "
            "Explain how you tailored your approach to different team members.",
        
        "What's your greatest weakness?":
            "Choose a weakness that isn't critical to the job, and show "
            "how you're actively working to improve it.",
        
        "What's your biggest accomplishment?":
            "Focus on a significant achievement that showcases your skills "
            "and dedication.",
        
        "What's your biggest failure?":
            "Share a failure that taught you a valuable lesson, emphasizing "
            "what you learned and how you grew from it.",
        
        "How do you motivate team members?":
            "Focus on your ability to inspire and guide others, using "
            "your own experiences as examples.",
        
        "Tell me about a time you worked in a team. How did you contribute?":
            "Describe your specific role in the team. Highlight a successful "
            "outcome that resulted from your teamwork.",
        
        "Can you describe a time when you faced a conflict in a team setting? "
        "How did you handle the situation, and what was the outcome?":
            "Highlight empathy, communication, and positive resolution. "
            "Focus on lessons learned.",
        
        "Can you describe a situation where you had to work with a difficult "
        "colleague or client?":
            "Describe the specific challenges faced and positive outcomes. "
            "Emphasize professional handling of the situation.",
        
        "Describe a time you experienced a major change at work. How did you adapt?":
            "Pick an example where you adapted to significant change. "
            "Mention if you helped others adapt as well.",
        
        "Can you share an example of a time when you used creativity to solve "
        "a challenging problem? What approach did you take, and what was the result?":
            "Focus on analysis, creativity, and creating opportunities. "
            "Describe the outcome and impact.",
        
        "Can you share an example of a time when you took initiative? "
        "What was the situation, and what impact did your actions have?":
            "Highlight being proactive, making positive changes, and "
            "the measurable impact of your actions.",
        
        "How do you tackle challenges? Name a difficult challenge you faced "
        "while working on a project, how you overcame it, and what you learned.":
            "Demonstrate perseverance, resourcefulness, and "
            "problem-solving abilities.",
        
        "How do you handle ambiguity or uncertainty in your work?":
            "Emphasize strategies for approaching uncertain situations. "
            "Discuss taking ownership and being adaptable."
    }


def generate_sample_questions(job_desc, company_info, resume):
    """
    Generate categorized interview questions based on all inputs.
    
    Args:
        job_desc (str): The job description text
        company_info (str): Information about the company
        resume (str): The user's resume
        
    Returns:
        dict: Categorized interview questions
    """
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
            "Can you describe a time when you faced a conflict in a team setting? "
            "How did you handle the situation, and what was the outcome?",
            "Can you describe a situation where you had to work with a difficult "
            "colleague or client?"
        ],
        "Leadership & Initiative": [
            "Describe a time you led/motivated others. How were you able to?",
            "How do you motivate team members?",
            "Can you share an example of a time when you took initiative? "
            "What was the situation, and what impact did your actions have?"
        ],
        "Problem Solving & Adaptability": [
            "How do you tackle challenges? Name a difficult challenge you faced "
            "while working on a project, how you overcame it, and what you learned.",
            "Describe a time you experienced a major change at work. How did you adapt?",
            "Can you share an example of a time when you used creativity to solve "
            "a challenging problem? What approach did you take, and what was the result?",
            "How do you handle ambiguity or uncertainty in your work?"
        ]
    }
    
    # Add company-specific question if company info is provided
    if (company_info.strip() and 
            "Why do you want to work at our company?" not in categorized_questions["Career Goals"]):
        categorized_questions["Career Goals"].append("Why do you want to work at our company?")
    
    # Add experience-related question if resume is provided
    if (resume.strip() and
            "Tell me about your most relevant experience for this role" not in 
            categorized_questions["Introduction"]):
        categorized_questions["Introduction"].append(
            "Tell me about your most relevant experience for this role"
        )
    
    return categorized_questions 
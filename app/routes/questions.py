"""
Questions blueprint - Handles routes for interview question generation
"""
from flask import Blueprint, request, jsonify, session

# Create a blueprint for question-related routes
questions_bp = Blueprint('questions', __name__, url_prefix='/api/questions')


@questions_bp.route('/generate', methods=['POST'])
def generate_questions_endpoint():
    """
    Generate categorized interview questions based on job description.
    
    This endpoint accepts job description, company info, and resume as JSON data
    and returns categorized questions and hints.
    
    Returns:
        JSON response with questions and hints
    """
    data = request.get_json()
    job_desc = data.get('job_desc', '')
    company_info = data.get('company_info', '')
    resume = data.get('resume', '')
    
    # Store in session for later use
    session['job_desc'] = job_desc
    session['company_info'] = company_info
    session['resume'] = resume
    
    # Import utility functions
    from app.utils.question_utils import generate_sample_questions, get_question_hints
    
    categorized_questions = generate_sample_questions(job_desc, company_info, resume)
    question_hints = get_question_hints()
    
    return jsonify({
        'questions': categorized_questions,
        'hints': question_hints
    })


@questions_bp.route('/follow-up', methods=['POST'])
def generate_follow_up_questions_endpoint():
    """
    Generate follow-up questions based on the user's answer.
    
    This endpoint accepts question, answer, job description and resume as JSON data
    and returns potential follow-up questions.
    
    Returns:
        JSON response with follow-up questions
    """
    data = request.get_json()
    question = data.get('question', '')
    job_desc = data.get('job_desc', session.get('job_desc', ''))
    resume = data.get('resume', session.get('resume', ''))
    answer_text = data.get('answer_text', '')
    
    if not question or not answer_text:
        return jsonify({
            'follow_up_questions': [
                'Please provide both a question and your answer to generate follow-up questions.'
            ]
        })
    
    try:
        # Access the interview_manager from app context
        from flask import current_app
        interview_manager = current_app.interview_manager
        
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
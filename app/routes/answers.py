"""
Answers blueprint - Handles routes for answer evaluation and generation
"""
from flask import Blueprint, request, jsonify, session

# Create a blueprint for answer-related routes
answers_bp = Blueprint('answers', __name__, url_prefix='/api/answers')


@answers_bp.route('/analyze', methods=['POST'])
def analyze_answer_endpoint():
    """
    Analyze a user's interview answer and provide feedback.
    
    This endpoint accepts the user's answer and returns evaluation scores and feedback.
    
    Returns:
        JSON response with scores and feedback
    """
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
        # Access the interview manager from app context
        from flask import current_app
        interview_manager = current_app.interview_manager
        
        scores, feedback = interview_manager.evaluator.evaluate_answer(
            voice_answer, job_desc, company_values
        )
        
        # Ensure feedback is not empty
        if not feedback or len(feedback.strip()) < 10:
            feedback = """I couldn't properly evaluate your answer. Here are some general tips:
            
- Structure your response with a clear beginning, middle, and end
- Relate your experience directly to the job requirements
- Use specific examples from your past experience
- Show confidence in your tone and delivery
            
Try recording again with these tips in mind."""
            scores = {'clarity': 5, 'relevance': 5, 'confidence': 5}
        
        # Create formatted output with star ratings
        def stars(score):
            return "⭐" * score + "☆" * (10 - score)
            
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
        default_feedback = "I'm having trouble analyzing your answer right now. Please try again later."
        scores = {'clarity': 5, 'relevance': 5, 'confidence': 5}
        
        def stars(score):
            return "⭐" * score + "☆" * (10 - score)
            
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


@answers_bp.route('/model-answer', methods=['POST'])
def generate_model_answer_endpoint():
    """
    Generate a model answer for the given interview question.
    
    This endpoint accepts a question and related context and returns a model answer.
    
    Returns:
        JSON response with model answer
    """
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
        # Access the interview manager from app context
        from flask import current_app
        interview_manager = current_app.interview_manager
        
        model_answer = interview_manager.drafter.generate_answer(
            question, company_info, job_desc, resume, voice_answer
        )
        
        # Ensure model answer is not empty
        if not model_answer or len(model_answer.strip()) < 10:
            model_answer = f"""I couldn't generate a complete sample answer for "{question}"
            
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
        default_answer = f"""I'm having trouble generating a sample answer right now.

Here are some general tips for this type of question:
- Use the STAR method: Situation, Task, Action, Result
- Relate your answer to the job you're applying for
- Be specific and use concrete examples
- Keep your answer concise and to the point
- Practice your delivery to sound confident"""
        
        return jsonify({'model_answer': default_answer}) 
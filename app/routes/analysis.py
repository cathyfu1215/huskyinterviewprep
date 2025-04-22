"""
Analysis blueprint - Handles routes for job analysis functionality
"""
from flask import Blueprint, request, jsonify, session

# Create a blueprint for analysis-related routes
analysis_bp = Blueprint('analysis', __name__, url_prefix='/api/analysis')


@analysis_bp.route('/analyze-info', methods=['POST'])
def analyze_info_endpoint():
    """
    Extract insights from job description and company info.
    
    This endpoint accepts job description and company info as JSON data
    and returns the parsed information.
    
    Returns:
        JSON response with parsed job information
    """
    data = request.get_json()
    job_desc = data.get('job_desc', '')
    company_info = data.get('company_info', '')
    
    # Access the interview_manager from app context
    from flask import current_app
    interview_manager = current_app.interview_manager
    
    parsed_info = interview_manager.analyzer.parse_job_info(job_desc, company_info)
    
    # Store in session for later use
    session['job_desc'] = job_desc
    session['company_info'] = company_info
    session['parsed_info'] = parsed_info
    
    return jsonify(parsed_info) 
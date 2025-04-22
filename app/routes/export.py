"""
Export blueprint - Handles routes for exporting interview data to HTML
"""
from flask import Blueprint, request, jsonify, session, send_file
import uuid
from datetime import datetime

# Create a blueprint for export-related routes
export_bp = Blueprint('export', __name__, url_prefix='/api/export')


@export_bp.route('/save-html', methods=['POST'])
def save_to_html_endpoint():
    """
    Save interview data to an HTML file for download.
    
    This endpoint accepts interview data and generates an HTML file with a
    formatted summary of the interview preparation session.
    
    Returns:
        JSON response with file ID for later download
    """
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
    company_name = data.get(
        'company_name', parsed_info.get('company_name', '')
    )
    position_title = data.get(
        'position_title', parsed_info.get('position_title', '')
    )
    
    try:
        # Import the HTML export function
        from app.utils.export_utils import save_to_html
        
        html_file_path = save_to_html(
            job_desc, company_info, resume, company_name, position_title,
            company_values, tech_skills, soft_skills, job_duties,
            selected_question, answer_text, feedback, model_answer, 
            follow_up_questions
        )
        
        # Generate a unique ID for this file for the frontend to request it
        file_id = str(uuid.uuid4())
        session[f'html_file_{file_id}'] = html_file_path
        
        return jsonify({'file_id': file_id})
    except Exception as e:
        print(f"Error saving to HTML: {str(e)}")
        return jsonify({
            'error': 'An error occurred while generating the HTML file'
        }), 500


@export_bp.route('/download/<file_id>', methods=['GET'])
def download_html(file_id):
    """
    Download a previously generated HTML file.
    
    Args:
        file_id: Unique identifier for the HTML file
        
    Returns:
        HTML file for download or 404 if not found
    """
    file_path = session.get(f'html_file_{file_id}')
    if not file_path:
        return "File not found", 404
    
    # Set the name for the downloaded file
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    download_name = f"interview_summary_{current_time}.html"
    
    return send_file(
        file_path, 
        as_attachment=True, 
        download_name=download_name
    ) 
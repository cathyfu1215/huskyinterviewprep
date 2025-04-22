"""
Interview Manager Service - Coordinates the interview analysis process
"""
from app.services.analyzer import Analyzer
from app.services.drafter import Drafter
from app.services.evaluator import Evaluator
from app.services.follow_up import FollowUpQuestioner


class InterviewAgentManager:
    """
    Manages the full interview process by coordinating between services.
    """
    
    def __init__(self):
        """Initialize all the required services."""
        self.analyzer = Analyzer()
        self.drafter = Drafter()
        self.evaluator = Evaluator()
        self.follow_up_questioner = FollowUpQuestioner()
    
    def process_interview(
        self, job_description, company_values, question,
        company_info, resume, voice_answer
    ):
        """
        Manages the full process from analysis to evaluation.
        
        Args:
            job_description (str): The job description text
            company_values (str): The company values information
            question (str): The interview question
            company_info (str): Information about the company
            resume (str): The user's resume
            voice_answer (str): The user's recorded/transcribed answer
            
        Returns:
            dict: Combined results from all services
        """
        # Parse job information
        parsed_info = self.analyzer.parse_job_info(job_description, company_values)
        
        # Generate model answer
        model_answer = self.drafter.generate_answer(
            question, company_info, job_description, resume, voice_answer
        )
        
        # Evaluate user's answer
        scores, feedback = self.evaluator.evaluate_answer(
            voice_answer, job_description, company_values
        )
        
        # Return organized results
        return {
            "parsed_info": parsed_info,
            "model_answer": model_answer,
            "scores": scores,
            "feedback": feedback
        } 
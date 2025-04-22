"""
Evaluator Service - Evaluates user's interview answers
"""
import re
from app.services.llm import prompt_llm


class Evaluator:
    """
    Evaluator class for assessing the quality of interview answers.
    """
    
    def evaluate_answer(self, voice_answer, job_description, company_values):
        """
        Evaluates the user's voice answer based on clarity, relevance, and confidence.
        
        Args:
            voice_answer (str): The user's recorded/transcribed interview answer
            job_description (str): The job description text
            company_values (str): The company values information
            
        Returns:
            tuple: (scores dict, feedback text)
        """
        prompt = f"""
        SYSTEM: You are an experienced interviewer in the tech industry for over 
        30 years. Also you are an expert evaluator for interview responses. 
        Assess the answer based on the following criteria:
        
        INSTRUCTIONS:
        - Clarity: Is the response structured and easy to understand?
        - Relevance: Does it address the job's required skills and reflect company values?
        - Confidence: Does the tone convey certainty and professionalism?
        - Consider that the user could be nervous, so don't be too strict.
        - Consider that the user is not a native English speaker, so don't be too strict.
        - Provide constructive feedback and a score out of 10 for each category.
        - Always give some positive feedback at the beginning, then give suggestions for improvement.
        - Use a friendly and professional tone and encourage the user to do better.
        - Keep the feedback concise and to the point.
        - Keep the feedback in 150 words to 250 words.

        USER VOICE ANSWER: {voice_answer}
        JOB DESCRIPTION: {job_description}
        COMPANY VALUES: {company_values}
        """
        response = prompt_llm(prompt)
        
        # Extract scores using regex
        clarity_match = re.search(r"Clarity: (\d+)/10", response)
        relevance_match = re.search(r"Relevance: (\d+)/10", response)
        confidence_match = re.search(r"Confidence: (\d+)/10", response)
        
        # Convert to integers, defaulting to 0 if not found
        clarity_score = int(clarity_match.group(1)) if clarity_match else 0
        relevance_score = int(relevance_match.group(1)) if relevance_match else 0
        confidence_score = int(confidence_match.group(1)) if confidence_match else 0
        
        # Include the full response as feedback
        feedback = response
        
        # Organize scores in a dictionary
        scores = {
            "clarity": clarity_score,
            "relevance": relevance_score,
            "confidence": confidence_score
        }
        
        return scores, feedback 
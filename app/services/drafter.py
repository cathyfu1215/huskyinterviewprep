"""
Drafter Service - Generates model answers for interview questions
"""
from app.services.llm import prompt_llm


class Drafter:
    """
    Drafter class for generating model interview answers based on provided information.
    """
    
    def generate_answer(self, question, company_info, job_description, resume, voice_answer):
        """
        Drafts a model answer based on user inputs.
        
        Args:
            question (str): The interview question
            company_info (str): Information about the company
            job_description (str): The job description text
            resume (str): The user's resume or relevant experience
            voice_answer (str): The user's voice answer to build upon
            
        Returns:
            str: Model answer for the interview question
        """
        prompt = f"""
        SYSTEM: You are a professional interview coach and writer with over 30 years
        of experience in the tech industry. Draft a strong, structured answer to get 
        this user hired by a top tech company, based on the following inputs:
        
        INSTRUCTIONS:
        - Ensure clarity and logical flow.
        - Incorporate company values where relevant.
        - Highlight technical and soft skills from the job description.
        - Use Amazon Leadership Principles to guide the answer.
        - Use user's voice answer, experience and skills in the resume to answer the question.
        - Use the situation, task, action, and result (STAR) method to structure the answer.
        - Improve conciseness while maintaining completeness.
        - Maintain a confident and positive tone.
        - Keep the answer in 90 seconds to 2 minutes long.
        - If possible, use the same language as the user's voice answer.
        - If there is no information, just output "Not found".
        
        QUESTION: {question}
        COMPANY INFO: {company_info}
        JOB DESCRIPTION: {job_description}
        USER RESUME: {resume}
        USER VOICE ANSWER: {voice_answer}
        """
        return prompt_llm(prompt) 
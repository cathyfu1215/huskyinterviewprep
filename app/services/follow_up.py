"""
FollowUpQuestioner Service - Generates insightful follow-up interview questions
"""
from app.services.llm import prompt_llm


class FollowUpQuestioner:
    """
    Generates potential follow-up questions based on user's interview answers.
    """
    
    def generate_follow_up_questions(self, job_description, resume, question, answer):
        """
        Generates insightful follow-up questions based on the user's answer.
        
        Args:
            job_description (str): The job description text
            resume (str): The user's resume
            question (str): The original interview question
            answer (str): The user's answer to the question
            
        Returns:
            list: List of follow-up questions
        """
        prompt = f"""
        SYSTEM: You are an expert interviewer with 30 years of experience hiring 
        for top tech companies. Your task is to generate 2-3 thoughtful follow-up 
        questions based on a candidate's interview answer.

        INSTRUCTIONS:
        - Analyze the candidate's answer for areas that could be explored further
        - Look for opportunities to dive deeper into their experience or skills
        - Consider the job requirements and the candidate's resume
        - Generate questions that allow the candidate to elaborate on strengths
        - Include questions that address potential weaknesses constructively
        - Each question should be specific and related to the candidate's response
        - Keep questions concise and direct
        - Format the output with a numbered list (1., 2., 3.)

        JOB DESCRIPTION:
        {job_description}

        RESUME:
        {resume}

        INTERVIEW QUESTION:
        {question}

        CANDIDATE'S ANSWER:
        {answer}

        FOLLOW-UP QUESTIONS (generate exactly 2-3):
        """
        
        response = prompt_llm(prompt)
        
        # Extract the questions as a list
        questions = []
        for line in response.strip().split('\n'):
            line = line.strip()
            if line and (line.startswith('1.') or 
                         line.startswith('2.') or 
                         line.startswith('3.')):
                questions.append(line)
        
        # If no questions were extracted, return the full response
        if not questions:
            # Split by newlines and filter out empty lines
            questions = [line.strip() for line in response.strip().split('\n') 
                         if line.strip()]
            # Limit to 3 questions
            questions = questions[:3]
        
        return questions 
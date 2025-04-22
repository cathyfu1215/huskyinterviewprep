"""
Analyzer Service - Extracts insights from job descriptions and company information
"""
import re
from sentence_transformers import SentenceTransformer
from app.services.llm import prompt_llm


class Analyzer:
    """
    Analyzer class for parsing job information and extracting relevant insights.
    """
    def __init__(self):
        """Initialize the analyzer with a sentence transformer model."""
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")

    def parse_job_info(self, job_description, company_values):
        """
        Extract key insights and fills relevant fields from job description.
        
        Args:
            job_description (str): The job description text
            company_values (str): Additional company information
            
        Returns:
            dict: Dictionary with parsed information
        """
        prompt = f"""
        SYSTEM: You are an expert career coach and interviewer with over 30 years 
        of experience in the tech industry. Your task is to thoroughly analyze 
        the job description and company values to extract and classify all relevant 
        information.

        INSTRUCTIONS:
        1. First, identify the company name and position title.
        2. Then analyze the entire text for any skills, requirements, or values, 
           looking for both explicit and implicit mentions.
        3. Classify all found information into these categories:

        Company Name:
        - Extract the company name from the text
        - If not explicitly stated, write "Company name not specified"

        Position Title:
        - Extract the job title/position from the text
        - It contains no additional adjectives or adverbs, only the job title
        - If not explicitly stated, write "Position title not specified"
        - Example: From "NetNation is seeking Junior to Mid-Range UX/UI Software Developers"
          Extract only: "UX/UI Software Developer"
          (Remove level indicators like "Junior to Mid-Range")

        Company Values:
        - Look for mentions of culture, principles, mission, values
        - Include both explicit values and implied ones from the company's description
        - Examples: integrity, innovation, customer focus, diversity, sustainability

        Technical Skills:
        - All technical requirements, tools, languages, platforms
        - Both required and preferred technical qualifications
        - Include domain-specific technical knowledge
        - Examples: programming languages, frameworks, methodologies, systems

        Soft Skills:
        - All interpersonal and professional skills
        - Leadership and management capabilities
        - Personal qualities and attributes
        - Examples: communication, leadership, problem-solving, teamwork

        Job Duties:
        - Primary responsibilities and expectations
        - Day-to-day tasks and long-term objectives
        - Project responsibilities and deliverables
        - Team and organizational contributions

        FORMAT YOUR RESPONSE EXACTLY AS FOLLOWS:
        **Company Name:**
        [company name]

        **Position Title:**
        [position title]
        
        **Key Company Values:**
        - [value 1]
        - [value 2]
        
        **Essential Technical Skills:**
        - [skill 1]
        - [skill 2]
        
        **Necessary Soft Skills:**
        - [skill 1]
        - [skill 2]
        
        **Summary of Key Job Duties:**
        - [duty 1]
        - [duty 2]

        Keep each bullet point concise (under 10 words).

        JOB DESCRIPTION: {job_description}
        COMPANY VALUES: {company_values}
        """
        response = prompt_llm(prompt)
        
        # Extract information using regex patterns
        company_name_match = re.search(
            r"\*\*Company Name:\*\*(.*?)\*\*Position Title:\*\*", 
            response, 
            re.DOTALL
        )
        position_title_match = re.search(
            r"\*\*Position Title:\*\*(.*?)\*\*Key Company Values:\*\*", 
            response, 
            re.DOTALL
        )
        company_values_match = re.search(
            r"\*\*Key Company Values:\*\*(.*?)\*\*Essential Technical Skills:\*\*", 
            response, 
            re.DOTALL
        )
        tech_skills_match = re.search(
            r"\*\*Essential Technical Skills:\*\*(.*?)\*\*Necessary Soft Skills:\*\*", 
            response, 
            re.DOTALL
        )
        soft_skills_match = re.search(
            r"\*\*Necessary Soft Skills:\*\*(.*?)\*\*Summary of Key Job Duties:\*\*", 
            response, 
            re.DOTALL
        )
        job_duties_match = re.search(
            r"\*\*Summary of Key Job Duties:\*\*(.*)", 
            response, 
            re.DOTALL
        )

        # Extract matched content or provide defaults
        company_name = company_name_match.group(1).strip() if company_name_match else "Company name not specified"
        position_title = position_title_match.group(1).strip() if position_title_match else "Position title not specified"
        company_values = company_values_match.group(1).strip() if company_values_match else "Not found"
        tech_skills = tech_skills_match.group(1).strip() if tech_skills_match else "Not found"
        soft_skills = soft_skills_match.group(1).strip() if soft_skills_match else "Not found"
        job_duties = job_duties_match.group(1).strip() if job_duties_match else "Not found"

        # Organize results in a dictionary
        parsed_info = {
            "company_name": company_name,
            "position_title": position_title,
            "company_values": company_values,
            "tech_skills": tech_skills,
            "soft_skills": soft_skills,
            "job_duties": job_duties
        }
        return parsed_info 
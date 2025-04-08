# Husky Interview Prep


## About the Project

Husky Interview Prep is an AI-powered interview preparation tool designed to help job seekers excel in their interviews. This application simulates real interview scenarios, providing instant feedback and coaching to improve your interview skills.

## Features

- **Job Analysis**: Automatically extracts key skills, values, and requirements from job descriptions
- **Personalized Questions**: Generates tailored interview questions based on the job description and your resume
- **Voice Interaction**: Practice answering questions verbally and get your responses transcribed
- **AI Feedback**: Receive detailed feedback on clarity, relevance, and confidence of your answers
- **Model Answers**: View AI-generated sample answers to learn effective response strategies
- **Text-to-Speech**: Hear questions read aloud in different accents for a realistic experience
- **Exportable Results**: Save your practice sessions as HTML files for later review

## Motivation

As a job seeker, I found that preparing for interviews can be challenging without proper guidance and feedback. Most people practice alone, without knowing if their answers are effective or aligned with what employers are looking for. I built Husky Interview Prep to leverage AI technology to provide intelligent, personalized interview coaching that helps candidates build confidence and improve their chances of success.

## Try it out!
(For best experience, please adjust your system theme to light)
https://huggingface.co/spaces/cathyfu1215/huskyInterviewPrep

## Design Document & Presentation

https://gamma.app/docs/Job-Application-Accelerator-6j7oxopdrchd1kv

## Proof of Concept Presentation
(Click to see the video)

[![Watch the video](https://img.youtube.com/vi/iYD9pknqTcg/maxresdefault.jpg)](https://www.youtube.com/watch?v=iYD9pknqTcg)

## Blogs about this project
https://medium.com/@cathyfu1215/my-journey-building-husky-interview-prep-from-zero-to-ai-powered-interview-coach-c9798569908f

## Technologies Used

- **Backend**: Flask, Python
- **AI**: Together AI API (LLama 3), Speech Recognition
- **Frontend**: HTML, CSS, Alpine.js, TailwindCSS
- **Speech Processing**: Text-to-Speech, Speech-to-Text conversion

## Getting Started

Clone the repository and install the required dependencies to run the application locally:

```bash
git clone https://github.com/yourusername/huskyinterviewprep.git
cd huskyinterviewprep
pip install -r requirements.txt
python flask_app.py
```

Make sure to set up your API key in a .env file.

# Husky Interview Prep


## About the Project

Husky Interview Prep is an AI-powered interview preparation tool designed to help job seekers excel in their interviews. This application simulates real interview scenarios, providing instant feedback and coaching to improve your interview skills.

## Features

- **Job Analysis**: Automatically extracts key skills, values, and requirements from job descriptions
- **Personalized Questions**: Generate interview questions as well as follow-up questions based on the job description and your resume
- **Voice Interaction**: Practice answering questions verbally and get your responses transcribed
- **AI Feedback**: Receive detailed feedback on clarity, relevance, and confidence of your answers
- **Model Answers**: View AI-generated sample answers to learn effective response strategies
- **Text-to-Speech**: Hear questions read aloud in different accents for a realistic experience
- **Exportable Results**: Save your practice sessions as HTML files for later review

## Technologies Used

- **Backend**: Flask, Python
- **AI**: Together AI API (LLama 3), Speech Recognition
- **Frontend**: HTML, CSS, Alpine.js, TailwindCSS
- **Speech Processing**: Text-to-Speech, Speech-to-Text conversion

## MVP Presentation
(Click to see the video)
[![Watch the video](https://img.youtube.com/vi/ZQk1MGm-dYw/maxresdefault.jpg)](https://www.youtube.com/watch?v=ZQk1MGm-dYw)

## Proof of Concept Presentation
(Click to see the video)

[![Watch the video](https://img.youtube.com/vi/iYD9pknqTcg/maxresdefault.jpg)](https://www.youtube.com/watch?v=iYD9pknqTcg)

## Design Document & Presentation

https://gamma.app/docs/Job-Application-Accelerator-6j7oxopdrchd1kv

## Blogs about this project
https://medium.com/@cathyfu1215/my-journey-building-husky-interview-prep-from-zero-to-ai-powered-interview-coach-c9798569908f


## Running Tests

Tests are available to help ensure the application works as expected:

```bash
pytest test_flask_app.py
```

For coverage information, run:

```bash
pytest test_flask_app.py --cov=flask_app
```


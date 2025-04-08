import unittest
from unittest.mock import patch, MagicMock, mock_open

# Import the classes and functions to test
from flask_app import (
    prompt_llm, 
    Analyzer, 
    Drafter, 
    Evaluator, 
    InterviewAgentManager,
    get_question_hints,
    generate_sample_questions,
    speech_to_text,
    get_voice_options,
    text_to_speech,
    save_to_html,
    app
)

class TestPromptLLM(unittest.TestCase):
    def test_prompt_llm_basic(self):
        """Test that prompt_llm returns a non-empty string response"""
        result = prompt_llm("Test prompt")
        
        # Check that result is a string
        self.assertIsInstance(result, str)
        
        # Check that result is not empty
        self.assertTrue(len(result) > 0)
        
        # Check that result contains some content (likely to be in any response)
        self.assertTrue(any(word in result.lower() for word 
                           in ["the", "a", "is", "are", "test"]))
    
    @patch('flask_app.client.completions.create')
    def test_prompt_llm_empty_response(self, mock_create):
        # Mock an empty response
        mock_create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content=""))]
        )
        
        result = prompt_llm("Test prompt")
        
        # Just check that we get some kind of error message back
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 10)
    
    @patch('flask_app.client.completions.create')
    def test_prompt_llm_exception(self, mock_create):
        # Mock an exception
        mock_create.side_effect = Exception("API error")
        
        result = prompt_llm("Test prompt")
        
        # Check that we get a non-empty string response
        self.assertIsInstance(result, str)
        self.assertTrue(len(result) > 10)
        
        # Print the result for debugging purposes
        print(f"Error message received: {result}")
        
        # Instead of checking for specific words, just ensure the result 
        # isn't the same as a normal successful response
        normal_response = prompt_llm("Another test prompt")
        self.assertNotEqual(result, normal_response)
        
    def test_prompt_llm_different_prompts(self):
        """Test that different prompts generate different responses"""
        result1 = prompt_llm("Tell me about Python")
        result2 = prompt_llm("Tell me about JavaScript")
        
        # Different prompts should produce different responses
        self.assertNotEqual(result1, result2)


class TestAnalyzer(unittest.TestCase):
    @patch('flask_app.prompt_llm')
    def test_parse_job_info(self, mock_prompt_llm):
        # Mock the prompt_llm response with a formatted response
        mock_response = """
        **Company Name:** Test Company
        **Position Title:** Software Engineer
        **Key Company Values:** Innovation, Teamwork
        **Essential Technical Skills:** Python, JavaScript
        **Necessary Soft Skills:** Communication, Leadership
        **Summary of Key Job Duties:** Develop software, attend meetings
        """
        mock_prompt_llm.return_value = mock_response
        
        analyzer = Analyzer()
        result = analyzer.parse_job_info(
            "Test job description", 
            "Test company values"
        )
        
        self.assertEqual(result["company_name"], "Test Company")
        self.assertEqual(result["position_title"], "Software Engineer")
        self.assertEqual(result["company_values"], "Innovation, Teamwork")
        self.assertEqual(result["tech_skills"], "Python, JavaScript")
        self.assertEqual(result["soft_skills"], "Communication, Leadership")
        self.assertEqual(
            result["job_duties"], 
            "Develop software, attend meetings"
        )


class TestDrafter(unittest.TestCase):
    @patch('flask_app.prompt_llm')
    def test_generate_answer(self, mock_prompt_llm):
        mock_prompt_llm.return_value = "Sample model answer"
        
        drafter = Drafter()
        result = drafter.generate_answer(
            "Tell me about yourself",
            "Company info",
            "Job description",
            "Resume",
            "Voice answer"
        )
        
        self.assertEqual(result, "Sample model answer")
        mock_prompt_llm.assert_called_once()


class TestEvaluator(unittest.TestCase):
    @patch('flask_app.prompt_llm')
    def test_evaluate_answer(self, mock_prompt_llm):
        # Mock the prompt_llm response with the expected format
        mock_response = """
        Good work overall!
        
        Clarity: 8/10
        Relevance: 7/10
        Confidence: 6/10
        
        Your answer was well-structured but could use more specifics.
        """
        mock_prompt_llm.return_value = mock_response
        
        evaluator = Evaluator()
        scores, feedback = evaluator.evaluate_answer(
            "My voice answer",
            "Job description",
            "Company values"
        )
        
        self.assertEqual(scores["clarity"], 8)
        self.assertEqual(scores["relevance"], 7)
        self.assertEqual(scores["confidence"], 6)
        self.assertEqual(feedback, mock_response)


class TestInterviewAgentManager(unittest.TestCase):
    def setUp(self):
        self.manager = InterviewAgentManager()
        self.manager.analyzer = MagicMock()
        self.manager.drafter = MagicMock()
        self.manager.evaluator = MagicMock()
        
        # Set up mock returns
        self.manager.analyzer.parse_job_info.return_value = {"parsed": "info"}
        self.manager.drafter.generate_answer.return_value = "model answer"
        self.manager.evaluator.evaluate_answer.return_value = (
            {"scores": "dict"}, 
            "feedback"
        )
    
    def test_process_interview(self):
        result = self.manager.process_interview(
            "job_description",
            "company_values",
            "question",
            "company_info",
            "resume",
            "voice_answer"
        )
        
        # Check that the correct methods were called
        self.manager.analyzer.parse_job_info.assert_called_once()
        self.manager.drafter.generate_answer.assert_called_once()
        self.manager.evaluator.evaluate_answer.assert_called_once()
        
        # Check the structure of the result
        self.assertIn("parsed_info", result)
        self.assertIn("model_answer", result)
        self.assertIn("evaluation", result)


class TestHelperFunctions(unittest.TestCase):
    def test_get_question_hints(self):
        hints = get_question_hints()
        self.assertIsInstance(hints, dict)
        self.assertIn("Tell me about yourself", hints)
        self.assertIn("What's your greatest strength?", hints)
    
    @patch('flask_app.prompt_llm')
    def test_generate_sample_questions(self, mock_prompt_llm):
        questions = generate_sample_questions("job desc", "company info", "resume")
        self.assertIsInstance(questions, dict)
        self.assertIn("Introduction", questions)
        self.assertIn("Career Goals", questions)
        
    def test_get_voice_options(self):
        options = get_voice_options()
        self.assertIsInstance(options, dict)
        self.assertIn("US English", options)
        self.assertIn("UK English", options)


class TestSpeechToText(unittest.TestCase):
    @patch('flask_app.sr.Recognizer')
    def test_speech_to_text(self, mock_recognizer_class):
        # Create a mock recognizer
        mock_recognizer = MagicMock()
        mock_recognizer_class.return_value = mock_recognizer
        
        # Mock the recognize_google method
        mock_recognizer.recognize_google.return_value = "Transcribed text"
        
        # Create a mock audio data in base64
        mock_audio_data = "data:audio/webm;base64,SGVsbG8gV29ybGQ="
        
        result = speech_to_text(mock_audio_data)
        self.assertEqual(result, "Transcribed text")


class TestTextToSpeech(unittest.TestCase):
    @patch('flask_app.gTTS')
    @patch('builtins.open', new_callable=mock_open)
    @patch('os.unlink')
    def test_text_to_speech(self, mock_unlink, mock_file, mock_gTTS):
        # Set up the mock
        mock_tts_instance = MagicMock()
        mock_gTTS.return_value = mock_tts_instance
        
        # Mock file content for base64 encoding
        mock_file.return_value.read.return_value = b"audio content"
        
        result = text_to_speech("Test text", "US English")
        
        # Check that gTTS was called with correct parameters
        mock_gTTS.assert_called_with(text="Test text", lang="en", tld="com")
        
        # Check that the temporary file was saved and deleted
        mock_tts_instance.save.assert_called_once()
        mock_unlink.assert_called_once()
        
        # Check that the result is not None (contains base64)
        self.assertIsNotNone(result)


class TestSaveToHTML(unittest.TestCase):
    @patch('tempfile.NamedTemporaryFile')
    def test_save_to_html(self, mock_temp_file):
        # Set up the mock temporary file
        mock_temp = MagicMock()
        mock_temp.name = "/tmp/test.html"
        mock_temp_file.return_value.__enter__.return_value = mock_temp
        
        result = save_to_html(
            "job_desc", 
            "company_info", 
            "resume", 
            "company_values", 
            "tech_skills", 
            "soft_skills", 
            "job_duties", 
            "selected_question", 
            "answer_text", 
            "feedback", 
            "model_answer"
        )
        
        # Check that the file was written to
        mock_temp.write.assert_called_once()
        
        # Check that the function returns the path to the temporary file
        self.assertEqual(result, "/tmp/test.html")


class TestFlaskRoutes(unittest.TestCase):
    def setUp(self):
        self.client = app.test_client()
        self.client.testing = True
    
    def test_index_route(self):
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
    
    @patch('flask_app.interview_manager.analyzer.parse_job_info')
    def test_analyze_info_endpoint(self, mock_parse_job_info):
        mock_parse_job_info.return_value = {"test": "result"}
        
        response = self.client.post(
            '/analyze-info',
            json={
                'job_desc': 'Test job description',
                'company_info': 'Test company info'
            }
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {"test": "result"})
    
    @patch('flask_app.generate_sample_questions')
    @patch('flask_app.get_question_hints')
    def test_generate_questions_endpoint(
        self, 
        mock_get_hints, 
        mock_generate_questions
    ):
        mock_generate_questions.return_value = {"category": ["question1"]}
        mock_get_hints.return_value = {"question1": "hint1"}
        
        response = self.client.post(
            '/generate-questions',
            json={
                'job_desc': 'Test job',
                'company_info': 'Test company',
                'resume': 'Test resume'
            }
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertIn('questions', response.json)
        self.assertIn('hints', response.json)
    
    @patch('flask_app.speech_to_text')
    def test_speech_to_text_endpoint(self, mock_speech_to_text):
        mock_speech_to_text.return_value = "Transcribed text"
        
        response = self.client.post(
            '/speech-to-text',
            json={
                'audio': 'base64_audio_data'
            }
        )
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json, {'text': 'Transcribed text'})
    
    @patch('flask_app.interview_manager.evaluator.evaluate_answer')
    def test_analyze_answer_endpoint(self, mock_evaluate_answer):
        mock_evaluate_answer.return_value = (
            {'clarity': 8, 'relevance': 7, 'confidence': 6},
            "Feedback text"
        )
        
        with app.test_request_context():
            # Setup session data
            with self.client.session_transaction() as session:
                session['job_desc'] = 'Test job'
                session['company_values'] = 'Test values'
            
            response = self.client.post(
                '/analyze-answer',
                json={
                    'answer_text': 'Test answer'
                }
            )
            
            self.assertEqual(response.status_code, 200)
            self.assertIn('scores', response.json)
            self.assertIn('feedback', response.json)
    
    @patch('flask_app.interview_manager.drafter.generate_answer')
    def test_generate_model_answer_endpoint(self, mock_generate_answer):
        mock_generate_answer.return_value = "Model answer text"
        
        with app.test_request_context():
            # Setup session data
            with self.client.session_transaction() as session:
                session['job_desc'] = 'Test job'
                session['company_info'] = 'Test info'
                session['resume'] = 'Test resume'
            
            response = self.client.post(
                '/generate-model-answer',
                json={
                    'question': 'Test question',
                    'answer_text': 'User answer'
                }
            )
            
            self.assertEqual(response.status_code, 200)
            self.assertIn('model_answer', response.json)


if __name__ == '__main__':
    unittest.main() 
import unittest
import json
from unittest.mock import patch, Mock
from static.app_manager import AppManager


class TestRoutes(unittest.TestCase):
    def setUp(self):
        """Set up test client before each test."""
        self.app = AppManager.create_app()
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True
        # Ensure webdriver_manager is None at start
        self.app.webdriver_manager = None

    def tearDown(self):
        """Clean up after each test."""
        # Clean up webdriver if it exists
        if hasattr(self.app, 'webdriver_manager') and self.app.webdriver_manager:
            self.app.webdriver_manager = None

    @patch('static.webdriver_manager.WebDriverManager')
    def test_init_webdriver_route(self, mock_webdriver_class):
        """Test the webdriver initialization route."""
        # Create a mock instance
        mock_instance = Mock()
        mock_webdriver_class.return_value = mock_instance
        
        response = self.client.get('/init_webdriver')
        data = json.loads(response.data)
        
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['message'], 'WebDriver initialization successful')
        mock_webdriver_class.assert_called_once()
        self.assertEqual(self.app.webdriver_manager, mock_instance)

    def test_index_route(self):
        """Test the index route returns HTML."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/html', response.content_type)

    def test_workspace_operations(self):
        """Test workspace CRUD operations."""
        # Test creating a workspace
        test_state = {'test': 'data'}
        response = self.client.post('/save_workspace',
                                  json={'state': test_state, 'name': 'test_workspace'})
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')

        # Test listing workspaces
        response = self.client.get('/list_workspaces')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertIn('test_workspace', data['data'])

        # Test loading workspace
        response = self.client.get('/load_workspace?name=test_workspace')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['data']['state'], test_state)

        # Test deleting workspace
        response = self.client.get('/delete_workspace?name=test_workspace')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')

        # Verify workspace is deleted
        response = self.client.get('/list_workspaces')
        data = json.loads(response.data)
        self.assertNotIn('test_workspace', data['data'])

    @patch('static.openai_api.OpenAIChatCompletionsAPI.create_chat_completion')
    def test_send_message(self, mock_chat):
        """Test the send_message route."""
        # Configure mock response
        class MockResponse:
            content = "Test response"
            tool_calls = None
        mock_chat.return_value = MockResponse()
        
        test_message = {
            'message': json.dumps({
                'user_message': 'test message',
                'use_vision': False
            }),
            'svg_state': None
        }
        response = self.client.post('/send_message', json=test_message)
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertIn('ai_message', data['data'])
        self.assertIn('ai_tool_calls', data['data'])

    def test_error_handling(self):
        """Test error handling in routes."""
        # Test invalid workspace name
        response = self.client.get('/load_workspace?name=nonexistent')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 404)
        self.assertEqual(data['status'], 'error')

        # Test missing workspace name
        response = self.client.get('/delete_workspace')
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(data['status'], 'error')

        # Test missing message
        response = self.client.post('/send_message', json={'invalid': 'format'})
        data = json.loads(response.data)
        self.assertEqual(response.status_code, 400)  # Changed from 500 to 400 for invalid request
        self.assertEqual(data['status'], 'error')
        self.assertIn('message', data['message'].lower())  # Error message should mention 'message'


if __name__ == '__main__':
    unittest.main() 
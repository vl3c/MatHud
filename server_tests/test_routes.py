from __future__ import annotations

import json
import os
import unittest
from typing import Optional
from unittest.mock import Mock, patch

from static.app_manager import AppManager, MatHudFlask
from static.routes import CANVAS_SNAPSHOT_PATH, save_canvas_snapshot_from_data_url


class TestRoutes(unittest.TestCase):
    SAMPLE_PNG_BASE64 = (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8"
        "/w8AAwMB/aqVw0sAAAAASUVORK5CYII="
    )

    def setUp(self) -> None:
        """Set up test client before each test."""
        # Set test environment variables to disable authentication
        self.original_require_auth: Optional[str] = os.environ.get('REQUIRE_AUTH')
        os.environ['REQUIRE_AUTH'] = 'false'
        
        self.app: MatHudFlask = AppManager.create_app()
        self.client = self.app.test_client()
        self.app.config['TESTING'] = True
        # Ensure webdriver_manager is None at start
        self.app.webdriver_manager = None
        self._remove_canvas_snapshot()

    def tearDown(self) -> None:
        """Clean up after each test."""
        # Clean up webdriver if it exists
        if hasattr(self.app, 'webdriver_manager') and self.app.webdriver_manager:
            self.app.webdriver_manager = None
        self._remove_canvas_snapshot()
        
        # Restore original REQUIRE_AUTH environment variable
        if self.original_require_auth is not None:
            os.environ['REQUIRE_AUTH'] = self.original_require_auth
        else:
            os.environ.pop('REQUIRE_AUTH', None)

    def _remove_canvas_snapshot(self) -> None:
        if os.path.exists(CANVAS_SNAPSHOT_PATH):
            try:
                os.remove(CANVAS_SNAPSHOT_PATH)
            except OSError:
                pass

    @patch('static.webdriver_manager.WebDriverManager')
    def test_init_webdriver_route(self, mock_webdriver_class: Mock) -> None:
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

    def test_index_route(self) -> None:
        """Test the index route returns HTML."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn('text/html', response.content_type)

    def test_workspace_operations(self) -> None:
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
    def test_send_message(self, mock_chat: Mock) -> None:
        """Test the send_message route."""
        # Configure mock response
        class MockMessage:
            content = "Test response"
            tool_calls = None
            
        class MockResponse:
            message = MockMessage()
            finish_reason = "stop"
            
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

    def test_save_canvas_snapshot_helper(self) -> None:
        data_url = f"data:image/png;base64,{self.SAMPLE_PNG_BASE64}"
        saved = save_canvas_snapshot_from_data_url(data_url)
        self.assertTrue(saved)
        self.assertTrue(os.path.exists(CANVAS_SNAPSHOT_PATH))
        self.assertGreater(os.path.getsize(CANVAS_SNAPSHOT_PATH), 0)

    @patch('static.openai_api.OpenAIChatCompletionsAPI.create_chat_completion')
    def test_send_message_uses_canvas_snapshot(self, mock_chat: Mock) -> None:
        class MockMessage:
            content = "Test response with vision"
            tool_calls = None

        class MockResponse:
            message = MockMessage()
            finish_reason = "stop"

        mock_chat.return_value = MockResponse()

        canvas_data_url = f"data:image/png;base64,{self.SAMPLE_PNG_BASE64}"
        payload = {
            'message': json.dumps({
                'user_message': 'vision request',
                'use_vision': True
            }),
            'vision_snapshot': {
                'renderer_mode': 'canvas2d',
                'canvas_image': canvas_data_url
            }
        }

        response = self.client.post('/send_message', json=payload)
        data = json.loads(response.data)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertTrue(os.path.exists(CANVAS_SNAPSHOT_PATH))
        self.assertGreater(os.path.getsize(CANVAS_SNAPSHOT_PATH), 0)
        self.assertIsNone(self.app.webdriver_manager)

    def test_new_conversation_route(self) -> None:
        """Test the new_conversation route resets the AI conversation history."""
        # First, send a message to create some conversation history
        with patch.object(self.app.ai_api.client.chat.completions, 'create') as mock_create:
            class MockMessage:
                content = "Test response"
                tool_calls = None
            
            class MockChoice:
                message = MockMessage()
                finish_reason = "stop"

            class MockResponse:
                choices = [MockChoice()]

            mock_create.return_value = MockResponse()
            
            test_message = {
                'message': json.dumps({
                    'user_message': 'test message',
                    'use_vision': False
                }),
                'svg_state': None
            }
            self.client.post('/send_message', json=test_message)

        # Check that the conversation history has more than the initial developer message
        self.assertGreater(len(self.app.ai_api.messages), 1)

        # Now, call the new_conversation route
        response = self.client.post('/new_conversation')
        data = json.loads(response.data)

        # Check that the response is successful
        self.assertEqual(response.status_code, 200)
        self.assertEqual(data['status'], 'success')
        self.assertEqual(data['message'], 'New conversation started.')

        # Check that the conversation history has been reset
        self.assertEqual(len(self.app.ai_api.messages), 1)
        self.assertEqual(self.app.ai_api.messages[0]["role"], "developer")

    def test_error_handling(self) -> None:
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
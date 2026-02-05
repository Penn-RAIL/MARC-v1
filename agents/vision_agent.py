"""
Generic Vision Agent - Works with any vision API
No hardcoded providers - just reads config and makes HTTP requests
"""
import os
import base64
import time
import aiohttp
import json
from typing import Dict, Any, Optional
from PIL import Image
import io


class GenericVisionAgent:
    """Universal vision agent that works with any API via configuration"""
    
    def __init__(
        self,
        name: str,
        api_key_env: str,
        api_type: str,
        endpoint: str,
        model: str,
        prompt_template: str,
        max_tokens: int = 1500,
        timeout: int = 30
    ):
        """
        Initialize a generic vision agent
        
        Args:
            name: Display name
            api_key_env: Environment variable name for API key
            api_type: Request format ('openai', 'anthropic', 'google', or 'custom')
            endpoint: API endpoint URL
            model: Model identifier
            prompt_template: Prompt to send with image
            max_tokens: Maximum response tokens
            timeout: Request timeout in seconds
        """
        self.name = name
        self.api_key_env = api_key_env
        self.api_type = api_type.lower()
        self.endpoint = endpoint
        self.model = model
        self.prompt_template = prompt_template
        self.max_tokens = max_tokens
        self.timeout = timeout
        
        # Get API key from environment
        self.api_key = os.getenv(api_key_env)
        if not self.api_key:
            raise ValueError(f"API key not found: {api_key_env}")
    
    async def run(self, image_data: bytes) -> Dict[str, Any]:
        """
        Analyze image with configured API
        
        Args:
            image_data: Raw image bytes
            
        Returns:
            Dictionary with response and metadata
        """
        print(f"--- Running {self.name} ---")
        
        try:
            # Build request based on API type
            if self.api_type == "openai":
                return await self._run_openai_format(image_data)
            elif self.api_type == "anthropic":
                return await self._run_anthropic_format(image_data)
            elif self.api_type == "google":
                return await self._run_google_format(image_data)
            else:
                return {
                    'provider': self.name,
                    'model': self.model,
                    'response': None,
                    'error': f"Unknown API type: {self.api_type}",
                    'tokens': 0,
                    'latency': 0
                }
        except Exception as e:
            return {
                'provider': self.name,
                'model': self.model,
                'response': None,
                'error': str(e),
                'tokens': 0,
                'latency': 0
            }
    
    async def _run_openai_format(self, image_data: bytes) -> Dict[str, Any]:
        """Handle OpenAI-compatible API format"""
        start_time = time.time()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt_template},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            "max_tokens": self.max_tokens
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                response.raise_for_status()
                data = await response.json()
        
        latency = time.time() - start_time
        
        return {
            'provider': self.name,
            'model': self.model,
            'response': data['choices'][0]['message']['content'],
            'tokens': data.get('usage', {}).get('total_tokens', 0),
            'latency': latency,
            'error': None
        }
    
    async def _run_anthropic_format(self, image_data: bytes) -> Dict[str, Any]:
        """Handle Anthropic API format"""
        start_time = time.time()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Detect media type
        try:
            img = Image.open(io.BytesIO(image_data))
            media_type = f"image/{img.format.lower()}"
        except:
            media_type = "image/jpeg"
        
        headers = {
            "Content-Type": "application/json",
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": base64_image
                            }
                        },
                        {
                            "type": "text",
                            "text": self.prompt_template
                        }
                    ]
                }
            ]
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                self.endpoint,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                response.raise_for_status()
                data = await response.json()
        
        latency = time.time() - start_time
        
        return {
            'provider': self.name,
            'model': self.model,
            'response': data['content'][0]['text'],
            'tokens': data.get('usage', {}).get('input_tokens', 0) + data.get('usage', {}).get('output_tokens', 0),
            'latency': latency,
            'error': None
        }
    
    async def _run_google_format(self, image_data: bytes) -> Dict[str, Any]:
        """Handle Google Gemini API format"""
        start_time = time.time()
        base64_image = base64.b64encode(image_data).decode('utf-8')
        
        # Google endpoint format
        url = f"{self.endpoint}/{self.model}:generateContent?key={self.api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": self.prompt_template},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": base64_image
                            }
                        }
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": self.max_tokens
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                url,
                headers=headers,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as response:
                response.raise_for_status()
                data = await response.json()
        
        latency = time.time() - start_time
        
        # Extract response text
        response_text = data.get('candidates', [{}])[0].get('content', {}).get('parts', [{}])[0].get('text', '')
        tokens = data.get('usageMetadata', {}).get('totalTokenCount', 0)
        
        return {
            'provider': self.name,
            'model': self.model,
            'response': response_text,
            'tokens': tokens,
            'latency': latency,
            'error': None
        }
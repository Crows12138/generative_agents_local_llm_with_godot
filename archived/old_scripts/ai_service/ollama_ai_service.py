#!/usr/bin/env python
"""
Ollama AI Service Adapter
Provides interface to Ollama models with GPU acceleration
"""

import logging
import time
import json
import requests
from typing import Dict, Any, Optional, List
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OllamaModelType(Enum):
    """Ollama model types"""
    FAST = "qwen3:4b"   # Fast 4B model on CPU
    DEEP = "qwen3-14b-optimized"  # Optimized 14B model on GPU
    NO_THINK = "qwen3-no-thinking"  # Qwen3 with thinking disabled


class NoThinkingFilter:
    """Filter to remove <think> tags from responses"""
    
    def __init__(self):
        self.in_thinking = False
        self.buffer = ""
    
    def process(self, text: str) -> str:
        """Process text and remove thinking tags"""
        self.buffer += text
        result = ""
        
        while True:
            if '<think>' in self.buffer:
                idx = self.buffer.find('<think>')
                result += self.buffer[:idx]
                self.buffer = self.buffer[idx+7:]
                self.in_thinking = True
            elif '</think>' in self.buffer:
                idx = self.buffer.find('</think>')
                self.buffer = self.buffer[idx+8:]
                self.in_thinking = False
            else:
                break
        
        if not self.in_thinking:
            result += self.buffer
            self.buffer = ""
        
        return result


class OllamaService:
    """Service for interacting with Ollama models"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        """Initialize Ollama service
        
        Args:
            base_url: Ollama server URL
        """
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.no_thinking_filter = NoThinkingFilter()
        
        # Test connection
        if not self._test_connection():
            logger.warning("Ollama service not available, starting in offline mode")
        else:
            logger.info("Ollama service connected successfully")
            self._list_models()
    
    def _test_connection(self) -> bool:
        """Test if Ollama service is running
        
        Returns:
            True if service is available
        """
        try:
            response = requests.get(f"{self.base_url}/api/version", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def _list_models(self):
        """List available models"""
        try:
            response = requests.get(f"{self.api_url}/tags")
            if response.status_code == 200:
                models = response.json().get("models", [])
                logger.info(f"Available models: {[m['name'] for m in models]}")
                return models
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
        return []
    
    def generate(self, 
                prompt: str,
                model: OllamaModelType = OllamaModelType.FAST,
                temperature: float = 0.7,
                max_tokens: Optional[int] = None,
                stream: bool = False,
                no_thinking: bool = False) -> str:
        """Generate response using Ollama
        
        Args:
            prompt: Input prompt
            model: Model to use
            temperature: Temperature for sampling
            max_tokens: Maximum tokens to generate
            stream: Whether to stream response
            
        Returns:
            Generated text
        """
        start_time = time.time()
        
        # Add /no_think command if requested
        if no_thinking:
            prompt = f"/no_think\n{prompt}"
        
        # Prepare request
        data = {
            "model": model.value,
            "prompt": prompt,
            "temperature": temperature,
            "stream": stream
        }
        
        if max_tokens:
            data["num_predict"] = max_tokens
        
        try:
            # Send request
            response = requests.post(
                f"{self.api_url}/generate",
                json=data,
                timeout=120  # 2 minutes timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                
                # Filter thinking tags if present
                if "<think>" in generated_text and no_thinking:
                    generated_text = self._remove_thinking_tags(generated_text)
                
                elapsed = time.time() - start_time
                tokens = result.get("eval_count", 0)
                tokens_per_sec = tokens / elapsed if elapsed > 0 else 0
                
                logger.info(f"Generated with {model.value}: {elapsed:.2f}s, {tokens_per_sec:.1f} tokens/s")
                
                return generated_text
            else:
                logger.error(f"Generation failed: {response.status_code}")
                return f"Error: Failed to generate response (status: {response.status_code})"
                
        except requests.exceptions.Timeout:
            logger.error("Request timed out")
            return "Error: Request timed out"
        except Exception as e:
            logger.error(f"Generation error: {e}")
            return f"Error: {str(e)}"
    
    def _remove_thinking_tags(self, text: str) -> str:
        """Remove <think>...</think> content from text"""
        while '<think>' in text and '</think>' in text:
            start = text.find('<think>')
            end = text.find('</think>') + 8
            text = text[:start] + text[end:]
        return text.strip()
    
    def generate_stream(self,
                       prompt: str,
                       model: OllamaModelType = OllamaModelType.FAST,
                       temperature: float = 0.7,
                       no_thinking: bool = False) -> Any:
        """Generate streaming response
        
        Args:
            prompt: Input prompt
            model: Model to use
            temperature: Temperature
            
        Yields:
            Response chunks
        """
        # Add /no_think if requested
        if no_thinking:
            prompt = f"/no_think\n{prompt}"
        
        data = {
            "model": model.value,
            "prompt": prompt,
            "temperature": temperature,
            "stream": True
        }
        
        filter_obj = NoThinkingFilter() if no_thinking else None
        
        try:
            with requests.post(
                f"{self.api_url}/generate",
                json=data,
                stream=True
            ) as response:
                for line in response.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        if "response" in chunk:
                            text = chunk["response"]
                            if filter_obj:
                                text = filter_obj.process(text)
                            if text:
                                yield text
                        if chunk.get("done", False):
                            break
        except Exception as e:
            logger.error(f"Streaming error: {e}")
            yield f"Error: {str(e)}"
    
    def chat(self,
            messages: List[Dict[str, str]],
            model: OllamaModelType = OllamaModelType.FAST,
            temperature: float = 0.7,
            no_thinking: bool = False) -> str:
        """Chat completion with conversation history
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            model: Model to use
            temperature: Temperature
            
        Returns:
            Response text
        """
        # Add /no_think to system message if requested
        if no_thinking:
            # Insert system message with /no_think
            messages_copy = messages.copy()
            if messages_copy and messages_copy[0].get("role") == "system":
                messages_copy[0]["content"] += " /no_think"
            else:
                messages_copy.insert(0, {"role": "system", "content": "/no_think"})
            messages = messages_copy
        
        data = {
            "model": model.value,
            "messages": messages,
            "temperature": temperature,
            "stream": False
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/chat",
                json=data,
                timeout=120
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("message", {}).get("content", "")
                
                # Filter thinking tags if present
                if "<think>" in content and no_thinking:
                    content = self._remove_thinking_tags(content)
                
                return content
            else:
                return f"Error: Chat failed (status: {response.status_code})"
                
        except Exception as e:
            logger.error(f"Chat error: {e}")
            return f"Error: {str(e)}"
    
    def get_embeddings(self,
                      text: str,
                      model: OllamaModelType = OllamaModelType.FAST) -> Optional[List[float]]:
        """Get text embeddings
        
        Args:
            text: Input text
            model: Model to use
            
        Returns:
            Embedding vector or None
        """
        data = {
            "model": model.value,
            "prompt": text
        }
        
        try:
            response = requests.post(
                f"{self.api_url}/embeddings",
                json=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get("embedding")
            else:
                logger.error(f"Embedding failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None


# Singleton instance
_ollama_instance = None


def get_ollama_service() -> OllamaService:
    """Get or create singleton Ollama service instance"""
    global _ollama_instance
    if _ollama_instance is None:
        _ollama_instance = OllamaService()
    return _ollama_instance


def test_ollama_service():
    """Test Ollama service functionality"""
    print("\n" + "="*50)
    print("Testing Ollama Service")
    print("="*50)
    
    service = get_ollama_service()
    
    # Test 1: Fast model
    print("\n[Test 1: Fast Model (4B)]")
    start = time.time()
    response = service.generate(
        "Hello! How are you?",
        model=OllamaModelType.FAST
    )
    elapsed = time.time() - start
    print(f"Response: {response[:100]}...")
    print(f"Time: {elapsed:.2f}s")
    
    # Test 2: Deep model
    print("\n[Test 2: Deep Model (30B)]")
    start = time.time()
    response = service.generate(
        "What is the meaning of life?",
        model=OllamaModelType.DEEP,
        max_tokens=100
    )
    elapsed = time.time() - start
    print(f"Response: {response[:100]}...")
    print(f"Time: {elapsed:.2f}s")
    
    # Test 3: Streaming
    print("\n[Test 3: Streaming Response]")
    print("Streaming: ", end="")
    for chunk in service.generate_stream(
        "Count to 5",
        model=OllamaModelType.FAST
    ):
        print(chunk, end="", flush=True)
    print()
    
    # Test 4: Chat
    print("\n[Test 4: Chat Completion]")
    messages = [
        {"role": "user", "content": "My name is Bob"},
        {"role": "assistant", "content": "Nice to meet you, Bob!"},
        {"role": "user", "content": "What's my name?"}
    ]
    response = service.chat(messages, model=OllamaModelType.FAST)
    print(f"Chat response: {response}")
    
    print("\n" + "="*50)
    print("Tests completed!")
    print("="*50)


if __name__ == "__main__":
    test_ollama_service()
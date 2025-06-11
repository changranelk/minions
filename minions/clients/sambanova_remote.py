import logging
from typing import Any, Dict, List, Optional, Tuple
from minions.usage import Usage
from minions.clients.base import MinionsClient
import os
import openai


class SambanovaRemoteClient(MinionsClient):
    def __init__(
        self,
        model_name: str = "Meta-Llama-3.1-8B-Instruct",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: str = "https://api.sambanova.ai/v1",
        **kwargs
    ):
        """
        Initialize the SambaNova Remote client for supervisor tasks.
        Similar to OpenAI client logic - handles single conversation history.

        Args:
            model_name: The name of the model to use
            api_key: The API key for SambaNova
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            base_url: Base URL for the API
        """
        super().__init__(
            model_name=model_name,
            api_key=api_key,
            temperature=temperature,
            max_tokens=max_tokens,
            base_url=base_url,
            **kwargs
        )
        
        # Client-specific configuration
        self.api_key = api_key or os.getenv("SAMBANOVA_API_KEY")
        self.base_url = base_url
        
        if not self.api_key:
            raise ValueError("SambaNova API Key is required. Set SAMBANOVA_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize the OpenAI client with SambaNova base URL
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        # Model context limits based on SambaNova documentation
        self.model_context_limits = {
            "Meta-Llama-3.1-8B-Instruct": 16384,      # 16k
            "Meta-Llama-3.1-70B-Instruct": 131072,    # 128k
            "Meta-Llama-3.3-70B-Instruct": 131072,    # 128k
            "DeepSeek-R1": 32768,                      # 32k
        }

        logging.info(f"Initialized SambaNova Remote client with model: {self.model_name}")

    def chat(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage]:
        """
        Handle chat completions for remote client (supervisor).
        Similar to OpenAI client - returns (responses, usage).

        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to the chat.completions.create

        Returns:
            Tuple of (List[str], Usage) containing response strings and token usage
        """
        assert len(messages) > 0, "Messages cannot be empty."

        # 打印调试信息
                    # logging.info(f"=== SambaNova Remote API Call Debug ===")
        logging.info(f"Model: {self.model_name}")
        logging.info(f"Messages: {messages}")
        
        max_retries = 3
        for retry in range(max_retries):
            try:
                params = {
                    "model": self.model_name,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    **kwargs,
                }

                logging.info(f"API params: {params}")

                response = self.client.chat.completions.create(**params)
                
                # 打印完整response用于调试
                logging.info(f"=== SambaNova Remote API Response ===")
                logging.info(f"Full response: {response}")

                # 安全处理response，确保不会出现subscript错误
                if response and response.choices and len(response.choices) > 0:
                    responses = []
                    for choice in response.choices:
                        if choice and choice.message and choice.message.content:
                            content = choice.message.content
                        else:
                            # 如果content是空的，打印详细信息
                            logging.error(f"❌ Empty content returned in remote chat!")
                            logging.error(f"Choice: {choice}")
                            logging.error(f"Message: {choice.message if choice else 'No choice'}")
                            content = ""
                        responses.append(content)
                    
                    # 处理usage
                    if response.usage:
                        usage = Usage(
                            prompt_tokens=response.usage.prompt_tokens or 0,
                            completion_tokens=response.usage.completion_tokens or 0
                        )
                    else:
                        usage = Usage()
                    
                    return responses, usage
                else:
                    logging.error("Invalid response structure from SambaNova Remote API")
                    logging.error(f"Response: {response}")
                    return [""], Usage()

            except Exception as e:
                logging.error(f"Error during SambaNova Remote API call (attempt {retry + 1}): {e}")
                
                # 检查是否是rate limit错误
                if "rate" in str(e).lower() or "429" in str(e) or "too many requests" in str(e).lower():
                    if retry < max_retries - 1:
                        wait_time = 30
                        logging.warning(f"🕒 Rate limit detected in remote chat. Waiting {wait_time}s before retry...")
                        
                        # 尝试在Streamlit中显示等待信息
                        try:
                            import streamlit as st
                            if hasattr(st, 'session_state'):
                                st.warning(f"⏳ Rate limit reached. Waiting {wait_time} seconds before retry...")
                                with st.spinner(f"Waiting {wait_time}s due to rate limit..."):
                                    import time
                                    time.sleep(wait_time)
                        except:
                            # 如果不在Streamlit环境中，直接sleep
                            import time
                            time.sleep(wait_time)
                        
                        continue  # 重试
                
                # 最后一次重试失败
                if retry == max_retries - 1:
                    logging.error(f"Final error during SambaNova Remote API call: {e}")
                    raise

    def get_context_limit(self) -> int:
        """Get the context limit for the current model."""
        return self.model_context_limits.get(self.model_name, 16384)  # Default to 16k

    def __str__(self) -> str:
        return f"SambanovaRemoteClient(model={self.model_name})" 
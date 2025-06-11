from typing import Any, Dict, List, Optional, Tuple, Union
from minions.usage import Usage
from minions.clients.base import MinionsClient
import logging
import os
import openai
import asyncio
import concurrent.futures


class SambanovaClient(MinionsClient):
    def __init__(
        self,
        model_name: str = "Meta-Llama-3.1-8B-Instruct",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: str = "https://api.sambanova.ai/v1",
        use_async: bool = True,  # 默认启用async
        is_remote_client: bool = False,  # 是否作为remote客户端使用
        **kwargs
    ):
        """
        Initialize the SambaNova client.

        Args:
            model_name: The name of the model to use
            api_key: The API key for SambaNova
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            base_url: Base URL for the API
            use_async: Whether to use async processing for batch requests
        """
        self.model_name = model_name
        self.api_key = api_key or os.getenv("SAMBANOVA_API_KEY")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.use_async = use_async
        self.is_remote_client = is_remote_client
        
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
            "Meta-Llama-3.1-405B-Instruct": 131072,   # 128k
            "Meta-Llama-3.3-70B-Instruct": 131072,    # 128k
            "DeepSeek-R1": 32768,                      # 32k
            "DeepSeek-V3": 65536,                      # 64k
            "Qwen2.5-72B-Instruct": 131072,           # 128k
            "Qwen2.5-Coder-32B-Instruct": 131072,     # 128k
        }

        logging.info(f"Initialized SambaNova client with model: {self.model_name}")

    def chat(self, messages: Union[List[Dict[str, Any]], Dict[str, Any]], **kwargs):
        """
        Handle chat completions using the SambaNova API.
        
        智能区分两种使用模式：
        1. Minions协议 (use_async=True): 每个message是独立对话，并行处理
        2. Minion协议 (use_async=False): messages是完整对话历史，作为整体处理
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            **kwargs: Additional arguments to pass to the API call

        Returns:
            If is_remote_client=True: Tuple[List[str], Usage] (for compatibility with Minion protocol)
            Otherwise: Tuple[List[str], Usage, List[str]] (full return with done_reasons)
        """
        assert len(messages) > 0, "Messages cannot be empty."
        
        # 根据use_async决定处理模式
        if self.use_async:
            # Minions模式：每个message是独立对话，需要并行处理
            responses, usage, done_reasons = self._process_async_batch(messages, **kwargs)
        else:
            # Minion模式：messages是完整对话历史，作为整体处理
            responses, usage, done_reasons = self._process_single_conversation(messages, **kwargs)
        
        # 根据是否为remote_client决定返回值数量
        if self.is_remote_client:
            # Remote客户端模式：只返回2个值以兼容Minion协议
            return responses, usage
        else:
            # Local客户端模式：返回3个值以兼容Minions协议
            return responses, usage, done_reasons

    def _process_single_conversation(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage, List[str]]:
        """
        处理单个完整对话历史 (Minion协议)
        将整个messages列表作为一个对话发送给API
        """
        # Normalize input to list format
        if isinstance(messages, dict):
            messages = [messages]
            
        try:
            params = {
                "model": self.model_name,
                "messages": messages,  # 整个对话历史
                "max_tokens": self.max_tokens,
                "temperature": self.temperature,
                **kwargs,
            }
            
            logging.info(f"Processing single conversation with {len(messages)} messages")
            response = self.client.chat.completions.create(**params)
            
            # Extract response
            content = response.choices[0].message.content
            
            # Create usage object
            usage = Usage(
                prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                completion_tokens=response.usage.completion_tokens if response.usage else 0,
            )
            
            # Extract done reason
            done_reason = response.choices[0].finish_reason or "completed"
            
            return [content], usage, [done_reason]
            
        except Exception as e:
            logging.error(f"Error in SambaNova API call: {e}")
            return [f"Error: {str(e)}"], Usage(), ["error"]

    def _process_async_batch(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage, List[str]]:
        """
        处理多个独立对话 (Minions协议)
        每个message都是独立的单轮对话，需要并行处理
        """
        # 标准化输入格式
        if isinstance(messages, dict):
            messages = [messages]
            
        logging.info(f"Processing {len(messages)} independent conversations in async batch mode")
        
        # 运行异步处理
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            return loop.run_until_complete(self._async_process_independent_messages(messages, **kwargs))
        finally:
            loop.close()

    async def _async_process_independent_messages(self, messages: List[Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage, List[str]]:
        """异步处理多个独立的单消息对话"""
        
        async def process_single_message(message) -> Tuple[str, Usage, str]:
            """处理单个独立消息"""
            try:
                # 确保message是正确的格式（列表或单个对象）
                if isinstance(message, list):
                    # 如果已经是列表，直接使用
                    conversation = message
                else:
                    # 如果是单个消息对象，包装成列表
                    conversation = [message]
                
                # 使用线程池执行同步API调用
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(self._make_single_api_call, conversation, **kwargs)
                    response = await asyncio.wrap_future(future)
                
                content = response.choices[0].message.content
                usage = Usage(
                    prompt_tokens=response.usage.prompt_tokens if response.usage else 0,
                    completion_tokens=response.usage.completion_tokens if response.usage else 0,
                )
                done_reason = response.choices[0].finish_reason or "completed"
                
                return content, usage, done_reason
                
            except Exception as e:
                logging.error(f"Error in async processing: {e}")
                return f"Error: {str(e)}", Usage(), "error"

        # 并行处理所有独立消息
        results = await asyncio.gather(
            *[process_single_message(msg) for msg in messages],
            return_exceptions=True
        )
        
        # 收集结果
        responses = []
        total_usage = Usage()
        done_reasons = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logging.error(f"Message {i} failed: {result}")
                responses.append(f"Error: {str(result)}")
                done_reasons.append("error")
            else:
                content, usage, done_reason = result
                responses.append(content)
                total_usage += usage
                done_reasons.append(done_reason)
        
        logging.info(f"Completed async processing of {len(messages)} independent conversations")
        return responses, total_usage, done_reasons

    def _make_single_api_call(self, messages: List[Dict[str, Any]], **kwargs):
        """执行单个API调用（用于异步处理中的线程池）"""
        params = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            **kwargs,
        }
        
        return self.client.chat.completions.create(**params)

    def get_context_limit(self) -> int:
        """Get the context limit for the current model."""
        return self.model_context_limits.get(self.model_name, 16384)  # Default to 16k

    def __str__(self) -> str:
        return f"SambanovaClient(model={self.model_name}, async={self.use_async})"
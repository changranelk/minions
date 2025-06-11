import asyncio
import concurrent.futures
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from minions.usage import Usage
from minions.clients.base import MinionsClient
import os
import openai


class SambanovaLocalClient(MinionsClient):
    def __init__(
        self,
        model_name: str = "Meta-Llama-3.1-8B-Instruct",
        api_key: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        base_url: str = "https://api.sambanova.ai/v1",
        num_ctx: int = 4096,
        structured_output_schema: Optional[Any] = None,
        use_async: bool = True,
        **kwargs
    ):
        """
        Initialize the SambaNova Local client for worker tasks.
        Similar to Ollama client logic - handles batch processing of independent messages.
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
        self.num_ctx = num_ctx
        self.use_async = use_async
        self.structured_output_schema = structured_output_schema
        
        if not self.api_key:
            raise ValueError("SambaNova API Key is required. Set SAMBANOVA_API_KEY environment variable or pass api_key parameter.")
        
        # Initialize the OpenAI client with SambaNova base URL
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )

        logging.info(f"Initialized SambaNova Local client with model: {self.model_name}, async: {self.use_async}")

    def _prepare_chat_kwargs(self):
        """Prepare common chat parameters"""
        params = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        # Add structured output if specified
        if self.structured_output_schema:
            params["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "schema": self.structured_output_schema.model_json_schema(),
                    "strict": False  # SambaNova requires this to be false
                }
            }
        
        return params

    async def _achat_internal(
        self,
        messages: Union[List[Dict[str, Any]], Dict[str, Any]],
        **kwargs,
    ) -> Tuple[List[str], Usage, List[str]]:
        """
        Handle async chat with multiple messages in parallel.
        Similar to Ollama's _achat_internal method.
        """
        # If single dict, wrap in list
        if isinstance(messages, dict):
            messages = [messages]

        chat_kwargs = self._prepare_chat_kwargs()

        async def process_one(msg):
            """Process one message (either single message or conversation)"""
            conversation = [msg] if isinstance(msg, dict) else msg
            
            # Use thread pool for synchronous API call
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(
                    self.client.chat.completions.create,
                    messages=conversation,
                    **chat_kwargs,
                    **kwargs
                )
                response = await asyncio.wrap_future(future)
            
            return response

        # Run all in parallel
        results = await asyncio.gather(*(process_one(m) for m in messages))

        # Gather results
        responses = []
        usage_total = Usage()
        done_reasons = []
        
        for response in results:
            responses.append(response.choices[0].message.content)
            if response.usage:
                usage_total += Usage(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                )
            done_reasons.append(response.choices[0].finish_reason or "stop")

        return responses, usage_total, done_reasons

    def achat(self, messages: Union[List[Dict[str, Any]], Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage, List[str]]:
        """Async chat wrapper (similar to Ollama)"""
        if not self.use_async:
            raise RuntimeError("This client is not in async mode. Set `use_async=True`.")

        # For Streamlit and other web environments, always use synchronous mode
        # to avoid event loop conflicts
        try:
            # Check if we're in an async context
            loop = asyncio.get_running_loop()
            # If we have a running loop, use thread executor to avoid conflicts
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(self._run_sync_version, messages, **kwargs)
                return future.result()
        except RuntimeError:
            # No running loop, safe to use asyncio.run
            try:
                return asyncio.run(self._achat_internal(messages, **kwargs))
            except Exception as e:
                logging.warning(f"Async execution failed, falling back to sync: {e}")
                return self._run_sync_version(messages, **kwargs)

    def _run_sync_version(self, messages, **kwargs):
        """Synchronous version to avoid event loop issues"""
        # Use schat logic but handle multiple messages
        if isinstance(messages, dict):
            messages = [messages]

        chat_kwargs = self._prepare_chat_kwargs()
        
        responses = []
        usage_total = Usage()
        done_reasons = []
        
        for msg_group in messages:
            conversation = [msg_group] if isinstance(msg_group, dict) else msg_group
            
            # 打印调试信息
            # logging.info(f"=== SambaNova Local API Call Debug ===")
            logging.info(f"Model: {self.model_name}")
            logging.info(f"Conversation: {conversation}")
            logging.info(f"Chat kwargs: {chat_kwargs}")
            
            max_retries = 3
            for retry in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        messages=conversation,
                        **chat_kwargs,
                        **kwargs
                    )
                    
                    # 打印完整response用于调试
                    logging.info(f"=== SambaNova API Response ===")
                    logging.info(f"Full response: {response}")
                    
                    # 安全处理response
                    if response and response.choices and len(response.choices) > 0:
                        choice = response.choices[0]
                        content = choice.message.content if choice.message else ""
                        
                        # 如果content是空的，打印详细信息
                        if not content or content.strip() == "":
                            logging.error(f"❌ Empty content returned!")
                            logging.error(f"Choice: {choice}")
                            logging.error(f"Message: {choice.message if choice else 'No choice'}")
                            logging.error(f"Finish reason: {choice.finish_reason if choice else 'No finish reason'}")
                        
                        responses.append(content or "")
                        
                        if response.usage:
                            usage_total += Usage(
                                prompt_tokens=response.usage.prompt_tokens or 0,
                                completion_tokens=response.usage.completion_tokens or 0,
                            )
                        
                        finish_reason = choice.finish_reason if choice else "unknown"
                        done_reasons.append(finish_reason or "stop")
                        
                        # 成功，跳出重试循环
                        break
                    else:
                        logging.error("Invalid response structure from SambaNova API")
                        logging.error(f"Response: {response}")
                        responses.append("")
                        done_reasons.append("error")
                        break
                    
                except Exception as e:
                    logging.error(f"Error during SambaNova API call (attempt {retry + 1}): {e}")
                    
                    # 检查是否是rate limit错误
                    if "rate" in str(e).lower() or "429" in str(e) or "too many requests" in str(e).lower():
                        if retry < max_retries - 1:
                            wait_time = 30
                            logging.warning(f"🕒 Rate limit detected. Waiting {wait_time}s before retry...")
                            
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
                    
                    # 最后一次重试失败，或非rate limit错误
                    if retry == max_retries - 1:
                        responses.append("")
                        done_reasons.append("error")
                        break
        
        return responses, usage_total, done_reasons

    def schat(self, messages: Union[List[Dict[str, Any]], Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage, List[str]]:
        """Synchronous chat (similar to Ollama)"""
        # If single dict, wrap in list
        if isinstance(messages, dict):
            messages = [messages]

        chat_kwargs = self._prepare_chat_kwargs()

        # 打印调试信息
                    # logging.info(f"=== SambaNova Local API Call Debug (schat) ===")
        logging.info(f"Model: {self.model_name}")
        logging.info(f"Messages: {messages}")
        logging.info(f"Chat kwargs: {chat_kwargs}")

        max_retries = 3
        for retry in range(max_retries):
            try:
                # Single API call with entire conversation
                response = self.client.chat.completions.create(
                    messages=messages,
                    **chat_kwargs,
                    **kwargs
                )
                
                # 打印完整response用于调试
                logging.info(f"=== SambaNova API Response (schat) ===")
                logging.info(f"Full response: {response}")
                
                # 安全处理response
                if response and response.choices and len(response.choices) > 0:
                    choice = response.choices[0]
                    content = choice.message.content if choice.message else ""
                    
                    # 如果content是空的，打印详细信息
                    if not content or content.strip() == "":
                        logging.error(f"❌ Empty content returned in schat!")
                        logging.error(f"Choice: {choice}")
                        logging.error(f"Message: {choice.message if choice else 'No choice'}")
                        logging.error(f"Finish reason: {choice.finish_reason if choice else 'No finish reason'}")
                    
                    responses = [content or ""]
                    
                    usage = Usage(
                        prompt_tokens=response.usage.prompt_tokens if response.usage and response.usage.prompt_tokens else 0,
                        completion_tokens=response.usage.completion_tokens if response.usage and response.usage.completion_tokens else 0,
                    )
                    
                    finish_reason = choice.finish_reason if choice else "unknown"
                    done_reasons = [finish_reason or "stop"]
                else:
                    logging.error("Invalid response structure from SambaNova API")
                    logging.error(f"Response: {response}")
                    responses = [""]
                    usage = Usage()
                    done_reasons = ["error"]
                
                return responses, usage, done_reasons
                
            except Exception as e:
                logging.error(f"Error during SambaNova API call (schat, attempt {retry + 1}): {e}")
                
                # 检查是否是rate limit错误
                if "rate" in str(e).lower() or "429" in str(e) or "too many requests" in str(e).lower():
                    if retry < max_retries - 1:
                        wait_time = 30
                        logging.warning(f"🕒 Rate limit detected in schat. Waiting {wait_time}s before retry...")
                        
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
                    self.logger.error(f"Final error during SambaNova API call: {e}")
                    raise

    def chat(self, messages: Union[List[Dict[str, Any]], Dict[str, Any]], **kwargs) -> Tuple[List[str], Usage, List[str]]:
        """Main chat method (similar to Ollama)"""
        if self.use_async:
            return self.achat(messages, **kwargs)
        else:
            return self.schat(messages, **kwargs)

    def get_context_limit(self) -> int:
        """Get the context limit for the current model."""
        model_context_limits = {
            "Meta-Llama-3.1-8B-Instruct": 16384,
            "Meta-Llama-3.3-70B-Instruct": 131072,
            "DeepSeek-R1": 32768,
        }
        return model_context_limits.get(self.model_name, 16384)

    def __str__(self) -> str:
        return f"SambanovaLocalClient(model={self.model_name}, async={self.use_async})" 
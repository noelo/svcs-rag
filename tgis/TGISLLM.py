import os
from typing import Any, Callable, Dict, Optional, Sequence

import requests
from tqdm import tqdm

from llama_index.bridge.pydantic import Field, PrivateAttr
from llama_index.callbacks import CallbackManager
from llama_index.constants import (
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_NUM_OUTPUTS,
    DEFAULT_TEMPERATURE,
)
from llama_index.core.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
)
from llama_index.llms.base import llm_chat_callback, llm_completion_callback
from llama_index.llms.custom import CustomLLM
from llama_index.llms.generic_utils import (
    completion_response_to_chat_response,
    stream_completion_response_to_chat_response,
)
from llama_index.types import BaseOutputParser, PydanticProgramMode
from llama_index.utils import get_cache_dir
from tgis.tgis_grpc_client import TgisGrpcClient

DEFAULT_MODEL="flan-t5-small"

class TGISLLM(CustomLLM):
    
    model_name: Optional[str] = Field(
        default=DEFAULT_MODEL,
        description="The name of the model to use"
    )
    
    tgis_client: Optional[TgisGrpcClient] = Field(
        description="TGIS Client"
    )
    
    tgis_host: str = Field(
        default="localhost",
        description="Hostname of the TGIS server.",
    )
    
    tgis_port: str = Field(
        default="443",
        description="Port to connect to the TGIS instance.",
    )

    temperature: float = Field(
        default=DEFAULT_TEMPERATURE,
        description="The temperature to use for sampling.",
        gte=0.0,
        lte=1.0,
    )
    max_new_tokens: int = Field(
        default=DEFAULT_NUM_OUTPUTS,
        description="The maximum number of tokens to generate.",
        gt=0,
    )
    context_window: int = Field(
        default=DEFAULT_CONTEXT_WINDOW,
        description="The maximum number of context tokens for the model.",
        gt=0,
    )
    generate_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Kwargs used for generation."
    )
    model_kwargs: Dict[str, Any] = Field(
        default_factory=dict, description="Kwargs used for model initialization."
    )

    _model: Any = PrivateAttr()

    def __init__(
        self,
        tgis_host: str,
        tgis_port: int = 443,
        model_name: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        max_new_tokens: int = DEFAULT_NUM_OUTPUTS,
        context_window: int = DEFAULT_CONTEXT_WINDOW,
        callback_manager: Optional[CallbackManager] = None,
        generate_kwargs: Optional[Dict[str, Any]] = None,
        model_kwargs: Optional[Dict[str, Any]] = None,
        verbose: bool = True,
        system_prompt: Optional[str] = None,
        messages_to_prompt: Optional[Callable[[Sequence[ChatMessage]], str]] = None,
        completion_to_prompt: Optional[Callable[[str], str]] = None,
        pydantic_program_mode: PydanticProgramMode = PydanticProgramMode.DEFAULT,
        output_parser: Optional[BaseOutputParser] = None,
    ) -> None:

        model_kwargs = {
            **{"n_ctx": context_window, "verbose": verbose},
            **(model_kwargs or {}),  # Override defaults via model_kwargs
        }

        generate_kwargs = generate_kwargs or {}
        generate_kwargs.update(
            {"temperature": temperature, "max_tokens": max_new_tokens}
        )
        
        model_name= model_name
        
        tgis_client = TgisGrpcClient(host = tgis_host,
                                     port = tgis_port,
                                     verify=False
                                     )

        super().__init__(
            tgis_client=tgis_client,
            temperature=temperature,
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            callback_manager=callback_manager,
            generate_kwargs=generate_kwargs,
            model_kwargs=model_kwargs,
            verbose=verbose,
            system_prompt=system_prompt,
            messages_to_prompt=messages_to_prompt,
            completion_to_prompt=completion_to_prompt,
            pydantic_program_mode=pydantic_program_mode,
            output_parser=output_parser,
        )

    @classmethod
    def class_name(cls) -> str:
        return "TGISLLM"

    @property
    def metadata(self) -> LLMMetadata:
        """LLM metadata."""
        return LLMMetadata(
            context_window=512,
            num_output=300,
            model_name=self.model_name,
        )

    @llm_chat_callback()
    def chat(self, messages: Sequence[ChatMessage], **kwargs: Any) -> ChatResponse:
        prompt = self.messages_to_prompt(messages)

        response = self.tgis_client.make_batched_request(prompt, model_id=self.model_name)
        completion_response = CompletionResponse(text=response.responses[0].text)
        return completion_response_to_chat_response(completion_response)

    @llm_chat_callback()
    def stream_chat(
        self, messages: Sequence[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        prompt = self.messages_to_prompt(messages)
        
        completion_response = self.stream_complete(prompt, formatted=True, **kwargs)
        return stream_completion_response_to_chat_response(completion_response)

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        self.generate_kwargs.update({"stream": False})

        if not formatted:
            prompt = self.completion_to_prompt(prompt)
            
        response = self.tgis_client.make_batched_request(prompt, model_id=self.model_name)

        return CompletionResponse(text=response.responses[0].text)

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        self.generate_kwargs.update({"stream": True})

        if not formatted:
            prompt = self.completion_to_prompt(prompt)

        response_iter = self.tgis_client.make_stream_request(prompt, model_id=self.model_name)

        def gen() -> CompletionResponseGen:
            text = ""
            for response in response_iter:
                delta = response.text
                text += delta
                yield CompletionResponse(delta=delta, text=text)

        return gen()

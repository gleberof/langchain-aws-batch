import json
import logging
import threading
import time
import uuid
from tempfile import NamedTemporaryFile
from typing import Any, Dict, Iterator, List, Optional, Tuple

from langchain_aws import ChatBedrock
from langchain_aws.function_calling import _tools_in_params
from langchain_aws.llms.bedrock import LLMInputOutputAdapter
from langchain_aws.utils import enforce_stop_tokens
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import ToolCall
from langchain_core.outputs import ChatGenerationChunk
from pydantic import Field


class ChatBedrockBatch(ChatBedrock):
    batch_size: int = Field(default=None, ge=100, le=50000)
    """batch size min 100 max 50000"""
    bucket: str = Field(default=None)
    """bucket for batch inference. Permissions have to be properly setup:
    https://docs.aws.amazon.com/bedrock/latest/userguide/batch-inference-prereq.html
    """

    current_batch: List[Any] = Field(default=[])
    results: Dict[int, Any] = Field(default=[])
    events: Dict[int, Any] = Field(default={})

    lock: Any = Field(default=[])

    def __init__(self, *args, batch_size: int, bucket: str, **kwargs):
        super().__init__(*args, **kwargs)
        self.streaming = False
        self.batch_size = batch_size
        self.bucket = bucket
        self.current_batch = []
        self.events = {}
        self.results = {}
        self.lock = threading.Lock()

    def contribute_task(self, task: dict, thread_id: int, event: threading.Event):
        """
        Thread-safe method for agents to contribute tasks to the batch.
        """

        with self.lock:
            self.current_batch.append({"task": task, "thread_id": thread_id})
            self.events[thread_id] = event
            if len(self.current_batch) == self.batch_size:
                self._process_batch()

    def _process_batch(self):
        """
        Process the batch: upload to S3, submit to Bedrock, and distribute results.
        """
        batch_file = f"batch_{uuid.uuid1()}.jsonl"
        self._write_batch_to_s3(batch_file)

        job_id = self._submit_batch(batch_file)
        results = self._wait_for_results(job_id)

        with self.lock:
            # Distribute results to threads
            for result in results:
                thread_id = result["thread_id"]
                self.results[thread_id] = result["output"]
                self.events[thread_id].set()

            # Clear current batch
            self.current_batch = []

    def _write_batch_to_s3(self, batch_file: str):
        """
        Write the current batch to a tmp JSONL file and
        upload to S3 using a temporary file.
        """
        # Serialize batch data to JSONL format
        jsonl_data = "\n".join(
            json.dumps({"recordId": f"{idx:011d}", "modelInput": task})
            for idx, task in enumerate(self.current_batch)
        )

        # Use NamedTemporaryFile to create a temporary file
        with NamedTemporaryFile(mode="w", delete=True) as temp_file:
            # Write JSONL data to the temporary file
            temp_file.write(jsonl_data)
            temp_file.flush()  # Ensure all data is written to disk

            # Upload the temporary file to S3
            self.client.upload_file(temp_file.name, self.bucket, batch_file)

    def _submit_batch(self, batch_file: str) -> str:
        """
        Submit the batch job to Bedrock.
        """
        response = self.client.start_batch_job(
            model_id=self.model_id,
            input_data_s3_uri=f"s3://{self.bucket}/{batch_file}",
        )
        return response["JobId"]

    def _wait_for_results(self, job_id: str) -> list:
        """
        Wait for batch results and retrieve them.
        """
        while True:
            status = self.client.get_batch_job_status(job_id)
            if status["JobStatus"].lower() == "completed":
                results_uri = status["OutputDataS3Uri"]
                return self._download_results(results_uri)
            elif status["JobStatus"].lower() == "failed":
                raise RuntimeError(f"Batch job {job_id} failed.")
            time.sleep(60)

    def _parse_s3_uri(self, uri: str):
        """
        Parse S3 URI into bucket and key.
        """
        uri_parts = uri.replace("s3://", "").split("/", 1)
        return uri_parts[0], uri_parts[1]

    def _download_results(self, results_uri: str) -> list:
        """
        Download results from S3.
        """
        bucket, key = self._parse_s3_uri(results_uri)

        with NamedTemporaryFile(mode="r", delete=True) as temp_file:
            self.client.download_file(bucket, key, temp_file.name)
            results = [json.loads(line) for line in temp_file]

        return results

    def _stream(
        self,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        """
        Stub to overwrite existing _stream method - batch inference not supporting this
        """
        raise NotImplementedError

    def _prepare_input_and_invoke(
        self,
        prompt: Optional[str] = None,
        system: Optional[str] = None,
        messages: Optional[List[Dict]] = None,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Tuple[
        str,
        List[ToolCall],
        Dict[str, Any],
    ]:
        """Re implement the method for batch inference reference:
        https://github.com/langchain-ai/langchain-aws/blob/28f3718abcd05ad0e640ed234361404f51c40fd6/libs/aws/langchain_aws/llms/bedrock.py#L478
        """
        _model_kwargs = self.model_kwargs or {}

        provider = self._get_provider()
        params = {**_model_kwargs, **kwargs}
        if "claude-3" in self._get_model() and _tools_in_params(params):
            input_body = LLMInputOutputAdapter.prepare_input(
                provider=provider,
                model_kwargs=params,
                prompt=prompt,
                system=system,
                messages=messages,
                tools=params["tools"],
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        else:
            input_body = LLMInputOutputAdapter.prepare_input(
                provider=provider,
                model_kwargs=params,
                prompt=prompt,
                system=system,
                messages=messages,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
            )
        body = json.dumps(input_body)
        accept = "application/json"
        contentType = "application/json"

        request_options = {
            "body": body,
            "modelId": self.model_id,
            "accept": accept,
            "contentType": contentType,
        }

        if self._guardrails_enabled:
            request_options["guardrailIdentifier"] = self.guardrails.get(  # type: ignore[union-attr]
                "guardrailIdentifier", ""
            )
            request_options["guardrailVersion"] = self.guardrails.get(  # type: ignore[union-attr]
                "guardrailVersion", ""
            )
            if self.guardrails.get("trace"):  # type: ignore[union-attr]
                request_options["trace"] = "ENABLED"

        try:
            thread_id = threading.get_ident()
            event = threading.Event()
            # response = self.client.invoke_model(**request_options)
            self.contribute_task(
                task=request_options["task"], thread_id=thread_id, event=event
            )

            event.wait()

            response = self.get_response(thread_id)

            (
                text,
                tool_calls,
                body,
                usage_info,
                stop_reason,
            ) = LLMInputOutputAdapter.prepare_output(provider, response).values()

        except Exception as e:
            logging.error(f"Error raised by bedrock service: {e}")
            if run_manager is not None:
                run_manager.on_llm_error(e)
            raise e

        if stop is not None:
            text = enforce_stop_tokens(text, stop)

        llm_output = {"usage": usage_info, "stop_reason": stop_reason}

        return text, tool_calls, llm_output

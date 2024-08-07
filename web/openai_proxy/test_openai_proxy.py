import datetime
import itertools
from typing import Any, NamedTuple

import openai
import pytest


class ModelState(NamedTuple):
    model_name: str
    is_backend_ready: bool


example_messages: list[Any] = [
    {
        "role": "user",
        "content": "Please introduce yourself in one short sentence.",
    }
]


@pytest.mark.parametrize(
    "is_streaming,model_state",
    itertools.product(
        [True, False],
        [ModelState("Mistral-7B-Instruct-v0.2", is_backend_ready=True)],
    ),
)
def test_openai_request_ready(
    is_streaming: bool,
    model_state: ModelState,
) -> None:
    """Make a request to a running LLM instance and check response.

    With reference to platform.openai.com/docs/api-reference/streaming
    """
    client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:25765/v1")

    output = client.chat.completions.create(
        model=model_state.model_name,
        messages=example_messages,
        stream=is_streaming,
    )

    if is_streaming:
        assert isinstance(output, openai.Stream)
        streaming_output: list[str] = []
        time_deltas: list[datetime.timedelta] = []
        prev_time = datetime.datetime.now()

        for chunk in output:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content is not None:
                time_elapsed = datetime.datetime.now() - prev_time
                prev_time = datetime.datetime.now()

                time_deltas.append(time_elapsed)
                streaming_output.append(chunk_content)
                print((chunk_content, time_elapsed))

        assert len(streaming_output) > 1
        assert min(time_deltas) > datetime.timedelta(microseconds=100), time_deltas

        print("\n" + "".join(streaming_output))
        return

    assert not isinstance(output, openai.Stream)
    output = output.choices[0]
    print(output)


@pytest.mark.parametrize("is_streaming", [False])
def test_openai_request_not_ready(
    is_streaming: bool,
) -> None:
    """Make a request to an LLM instance that isn't running and check response."""
    model_state = ModelState("Meta-Llama-3-8B-Instruct", is_backend_ready=False)
    client = openai.OpenAI(api_key="EMPTY", base_url="http://localhost:25765/v1")

    with pytest.raises(openai.InternalServerError) as exception:
        client.chat.completions.create(
            model=model_state.model_name,
            messages=example_messages,
            stream=is_streaming,
        )
        print(exception)

# unifiedllm

A lightweight Python SDK that provides a unified interface for interacting with multiple Large Language Model (LLM) providers. `unifiedllm` simplifies working with Google Gemini, Anthropic, and OpenAI by exposing a single consistent `chat()` API, unified response objects, and structured error handling. Built with direct API integration, it has no dependencies on provider-specific SDKs.

`unifiedllm` makes it easy to experiment with different LLM providers without learning multiple SDKs. Google Gemini offers a free tier, making it an ideal starting point for learning and prototyping.

## Features

- **Unified API**: Single `chat()` method works across all providers
- **Provider-agnostic**: Switch between Gemini, Anthropic, and OpenAI with minimal code changes
- **Lightweight**: Direct API integration with zero provider SDK dependencies
- **Consistent responses**: Standardized `ChatResponse` object across all providers
- **Structured errors**: Clear error hierarchy for API, HTTP, and parsing issues
- **Simple configuration**: Set system prompts and parameters with intuitive methods

## Installation

```bash
pip install unifiedllm
```

**Requirements**: Python 3.10+

## Quick Start

Here's a minimal example using Google Gemini:

```python
from unifiedllm import LLM

# Initialize with Gemini (free tier available)
llm = LLM(provider="gemini", model="gemini-2.5-flash")

# Send a message
response = llm.chat(prompt="What is machine learning?")
print(response.text)
```

## API Keys & Authentication

Each provider requires an API key. You can provide keys in two ways:

### Environment Variables (Recommended)

Set the appropriate environment variable before running your code:

```bash
export GEMINI_API_KEY="your-google-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export OPENAI_API_KEY="your-openai-api-key"
```

Then initialize without passing the key explicitly:

```python
from unifiedllm import LLM

llm = LLM(provider="gemini", model="gemini-2.5-flash")
```

### Explicit API Key

Pass the API key directly when initializing:

```python
from unifiedllm import LLM

llm = LLM(
    provider="gemini",
    model="gemini-2.5-flash",
    api_key="your-google-api-key"
)
```

If no API key is provided and the environment variable is not set, a `MissingAPIKeyError` will be raised.

## Sending Messages

### Prompt-based Chat

The simplest way to send a message is with a text prompt:

```python
response = llm.chat(prompt="Explain photosynthesis in simple terms")
print(response.text)
```

### Message-based Chat

For multi-turn conversations, use the message format:

```python
messages = [
    {"role": "user", "content": "What is Python?"},
    {"role": "model", "content": "Python is a high-level programming language."},
    {"role": "user", "content": "What are its main features?"}
]

response = llm.chat(messages=messages)
print(response.text)
```

**Supported roles**: `"user"` and `"model"`. Using invalid roles will raise a `ValueError`.

## System Prompt & Configuration

### Setting a System Prompt

Define the behavior or persona of the assistant:

```python
llm.system_prompt("You are a helpful assistant specializing in biology.")
response = llm.chat(prompt="What is mitosis?")
```

### Configuring Parameters

Adjust model parameters like temperature and max tokens:

```python
llm.config(max_tokens=200, temperature=0.7)
response = llm.chat(prompt="Write a short poem about the ocean")
```

Unsupported configuration parameters will raise a `ValueError`.

## Response Object

All chat requests return a `ChatResponse` object with the following attributes:

- **`text`**: The generated response text
- **`usage`**: Token usage information (e.g., input tokens, output tokens)
- **`request_id`**: Unique identifier for the request
- **`raw`**: The raw response from the provider (for debugging)

Example:

```python
response = llm.chat(prompt="Hello, world!")

print(response.text)        # Generated text
print(response.usage)       # Token usage details
print(response.request_id)  # Request ID
```

## Error Handling

`unifiedllm` provides structured exceptions for common issues:

```python
from unifiedllm import LLM
from unifiedllm.errors import MissingAPIKeyError, ProviderAPIError

try:
    llm = LLM(provider="gemini", model="gemini-2.5-flash")
    response = llm.chat(prompt="Hello")
except MissingAPIKeyError as e:
    print(f"API key missing: {e}")
except ProviderAPIError as e:
    print(f"Provider error: {e}")
```

**Available exceptions**:
- `MissingAPIKeyError`: No API key provided
- `ProviderAPIError`: General provider-side error
- `ProviderHTTPError`: HTTP-related errors
- `ProviderParseError`: Response parsing errors

## Supported Providers

| Provider | Model Examples | Notes |
|----------|----------------|-------|
| **Google Gemini** | `gemini-2.5-flash`, `gemini-2.5-pro` | Free tier available; ideal for learning and prototyping |
| Anthropic | `claude-sonnet-4-20250514`, `claude-opus-4-1-20250805` | Requires API key |
| OpenAI | `gpt-4`, `gpt-4o-mini` | Requires API key |

## Examples

Example Jupyter notebooks are available in the `examples/` directory, with a focus on Google Gemini for students:

- **`gemini_basics.ipynb`**: Getting started with Gemini's free tier
- **`multi_turn_conversation.ipynb`**: Building conversational applications
- **`provider_comparison.ipynb`**: Comparing responses across providers

These examples are designed to help beginners learn LLM integration with minimal cost.

## Project Status

`unifiedllm` is currently in **pre-1.0 development** (version 0.1.0). The API is functional but may change as the library matures.

**Current limitations**:
- No streaming support
- No function/tool calling

These features may be added in future releases.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

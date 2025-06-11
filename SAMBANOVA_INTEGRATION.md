# SambaNova Integration for Minions

## Overview

This integration adds comprehensive SambaNova API support to the Minions framework, enabling both remote and local model usage through SambaNova's API.

## Features

- **Remote Models**: Access to SambaNova's cloud-hosted models like `DeepSeek-V3-0324`, `Meta-Llama-3.3-70B-Instruct`
- **Local Models**: Support for local model inference via SambaNova API
- **Rate Limiting**: Built-in rate limiting and error handling
- **Supervisor + Worker**: Full support for Minions protocol with SambaNova models as both supervisor and worker

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set API Key

```bash
export SAMBANOVA_API_KEY="your_api_key_here"
```

Or set in the Streamlit UI under the SambaNova provider section.

### 3. Available Models

**Remote Models:**
- `DeepSeek-V3-0324` (32k context)
- `DeepSeek-R1` (32k context)  
- `Meta-Llama-3.3-70B-Instruct` (128k context)
- `Meta-Llama-3.1-405B-Instruct` (16k context)
- `Meta-Llama-3.1-8B-Instruct` (16k context)
- `QwQ-32B` (16k context)
- `Meta-Llama-3.2-3B-Instruct` (4k context)
- `Meta-Llama-3.2-1B-Instruct` (16k context)

**Local Models:**
- `Meta-Llama-3.2-3B-Instruct` (via local API endpoint)

## Usage

### In Streamlit UI

1. Select "SambaNova" as your provider
2. Choose your remote model (e.g., `DeepSeek-V3-0324`)
3. Optionally configure local model for worker tasks
4. Set your API key if not in environment variables

### Programmatic Usage

```python
from minions.clients.sambanova_remote import SambaNovaRemoteClient
from minions.clients.sambanova_local import SambaNovaLocalClient
from minions.minions import Minions

# Remote client
remote_client = SambaNovaRemoteClient(
    model_name="DeepSeek-V3-0324",
    api_key="your_api_key",
    max_tokens=4096
)

# Local client (optional)
local_client = SambaNovaLocalClient(
    model_name="Meta-Llama-3.2-3B-Instruct",
    api_key="your_api_key",
    max_tokens=2048,
    num_ctx=4096
)

# Minions instance
minions = Minions(
    remote_client=remote_client,
    local_client=local_client
)

# Run task
result = minions(
    task="Analyze the provided document",
    context=["your document text here"],
    doc_metadata="Document description"
)
```

## Configuration

### Rate Limiting
- Default: 10 requests per minute
- Configurable via client initialization

### Context Windows
- **DeepSeek-V3-0324**: 32k tokens
- **Meta-Llama-3.3-70B**: 128k tokens  
- **Meta-Llama-3.2-3B**: 4k tokens

### Error Handling
- Automatic retry on rate limits
- Exponential backoff
- Graceful degradation on API errors

## Technical Details

### Files Added/Modified

1. **`minions/clients/sambanova_remote.py`** - Remote SambaNova API client
2. **`minions/clients/sambanova_local.py`** - Local SambaNova API client  
3. **`minions/clients/sambanova.py`** - Updated unified client
4. **`app.py`** - Streamlit UI integration
5. **`minions/minions.py`** - Core protocol bug fixes

### Key Bug Fixes

1. **Worker Prompt Template**: Fixed constructor logic in `minions.py` line 181
2. **JobOutput Model**: Enhanced to support `Union[str, List[str]]` for citations
3. **JSON Parsing**: Added control character cleaning
4. **Error Handling**: Improved API response handling

## Testing

The integration has been tested with:
- Document analysis tasks
- PDF processing 
- Multi-round conversations
- Error scenarios and rate limiting

## Troubleshooting

### Common Issues

1. **API Key Not Set**
   ```
   Error: SAMBANOVA_API_KEY not found
   ```
   Solution: Set the environment variable or enter key in UI

2. **Rate Limit Exceeded**
   ```
   Error: Rate limit exceeded
   ```
   Solution: Wait for rate limit reset (automatic retry included)

3. **Model Not Available**
   ```
   Error: Model not found
   ```
   Solution: Check model name spelling and availability

### Debug Mode

For debugging, you can enable logging:

```python
import logging
logging.basicConfig(level=logging.INFO)
```

## Contributing

When contributing to SambaNova integration:

1. Follow existing code patterns
2. Add appropriate error handling
3. Update model lists when new models are available
4. Test with both remote and local configurations

## License

Same as the main Minions project. 
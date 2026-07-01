import pytest
from unittest.mock import AsyncMock, MagicMock, patch

@pytest.mark.asyncio
async def test_call_openai_with_tool_use(monkeypatch):
    """Verify _call_openai implements the tool execution loop correctly."""
    from backend.analyzer.llm_client import _call_openai

    # Mock settings to return tools_enabled = True
    def fake_get_setting(db, key, default=""):
        return {
            "llm_enable_tools": "true",
            "searxng_url": "http://host.docker.internal:8043"
        }.get(key, default)
    
    # Mock database session query for llm_enable_tools
    mock_db = MagicMock()
    mock_setting = MagicMock()
    mock_setting.value = "true"
    mock_db.query.return_value.filter.return_value.first.return_value = mock_setting
    
    mock_SessionLocal = MagicMock(return_value=mock_db)
    monkeypatch.setattr("backend.models.db.SessionLocal", mock_SessionLocal)

    # We mock AsyncOpenAI client
    fake_client = MagicMock()
    
    # First response triggers a tool call
    fake_tc = MagicMock()
    fake_tc.id = "call_123"
    fake_tc.type = "function"
    fake_tc.function.name = "web_search"
    fake_tc.function.arguments = '{"query": "Senior Software Engineer profiles"}'
    
    first_choice = MagicMock()
    first_choice.message = MagicMock(
        content="Let me look up some profiles.",
        tool_calls=[fake_tc]
    )
    first_resp = MagicMock()
    first_resp.choices = [first_choice]
    first_resp.usage = MagicMock(prompt_tokens=100, completion_tokens=15)
    
    # Second response returns final text
    second_choice = MagicMock()
    second_choice.message = MagicMock(
        content="I have scored the profile based on the results.",
        tool_calls=None
    )
    second_resp = MagicMock()
    second_resp.choices = [second_choice]
    second_resp.usage = MagicMock(prompt_tokens=200, completion_tokens=50)
    
    # client.chat.completions.create is called twice
    fake_client.chat.completions.create = AsyncMock(side_effect=[first_resp, second_resp])
    
    def fake_ctor(**kwargs):
        return fake_client
        
    import openai
    monkeypatch.setattr(openai, "AsyncOpenAI", fake_ctor)

    # Mock web_search tool execution via TOOL_FUNCTIONS dict
    mock_search = AsyncMock(return_value="Search Results: Profile 1, Profile 2")
    import backend.analyzer.tools
    monkeypatch.setitem(backend.analyzer.tools.TOOL_FUNCTIONS, "web_search", mock_search)
    
    # Execute _call_openai
    result = await _call_openai("prompt", "system", "gpt-4o", "sk-test", 100)
    
    # Assertions
    assert result["text"] == "I have scored the profile based on the results."
    # Input tokens: 100 (first) + 200 (second) = 300
    assert result["usage"]["input_tokens"] == 300
    # Output tokens: 15 (first) + 50 (second) = 65
    assert result["usage"]["output_tokens"] == 65
    
    # Verify client.chat.completions.create was called twice
    assert fake_client.chat.completions.create.call_count == 2
    
    # Verify the tools parameter was passed
    first_call_args = fake_client.chat.completions.create.call_args_list[0]
    assert "tools" in first_call_args.kwargs
    assert first_call_args.kwargs["tools"][0]["function"]["name"] == "web_search"
    
    # Verify the tool result message was appended to the second call
    second_call_args = fake_client.chat.completions.create.call_args_list[1]
    messages = second_call_args.kwargs["messages"]
    
    # Messages should include:
    # 0: system
    # 1: user prompt
    # 2: assistant first message with tool calls
    # 3: tool result message
    assert len(messages) == 4
    assert messages[2]["role"] == "assistant"
    assert messages[3]["role"] == "tool"
    assert messages[3]["tool_call_id"] == "call_123"
    assert messages[3]["content"] == "Search Results: Profile 1, Profile 2"

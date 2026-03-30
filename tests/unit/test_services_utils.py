"""Unit tests for service utilities."""
import pytest

from almasim.services.utils import (
    log_message,
    as_progress_emitter,
    ProgressEmitterAdapter,
)


def test_log_message_with_logger():
    """Test log_message with a logger callback."""
    messages = []
    
    def logger(msg):
        messages.append(msg)
    
    log_message(logger, "Test message")
    assert len(messages) == 1
    assert messages[0] == "Test message"


def test_log_message_remote():
    """Test log_message in remote mode."""
    # In remote mode, it should print (we can't easily test print, but we can test it doesn't crash)
    log_message(None, "Test", remote=True)


def test_log_message_no_logger():
    """Test log_message without logger (should not crash)."""
    log_message(None, "Test message")


def test_progress_emitter_adapter():
    """Test ProgressEmitterAdapter."""
    values = []
    
    def callback(value):
        values.append(value)
    
    adapter = ProgressEmitterAdapter(callback)
    adapter.emit(50)
    adapter.emit(100)
    
    assert values == [50, 100]


def test_as_progress_emitter_with_emit():
    """Test as_progress_emitter with object that has emit method."""
    class Emitter:
        def emit(self, value):
            self.value = value
    
    emitter = Emitter()
    result = as_progress_emitter(emitter)
    assert result is emitter


def test_as_progress_emitter_with_callback():
    """Test as_progress_emitter with callback function."""
    def callback(value):
        pass
    
    result = as_progress_emitter(callback)
    assert isinstance(result, ProgressEmitterAdapter)


def test_as_progress_emitter_none():
    """Test as_progress_emitter with None."""
    result = as_progress_emitter(None)
    assert result is None



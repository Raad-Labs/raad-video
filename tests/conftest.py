import os
import sys
import pytest

# Add the parent directory to PYTHONPATH so we can import our package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

@pytest.fixture(scope="session", autouse=True)
def setup_asyncio_event_loop_policy():
    """Configure asyncio policy for tests."""
    import asyncio
    if sys.platform == 'darwin':
        # Fix for MacOS asyncio issues
        asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for each test case."""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

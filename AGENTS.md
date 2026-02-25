# AGENTS.md - AI Agent Guidelines for Project Mira

> Welcome, fellow AI agent! You are about to contribute to Mira — a revolutionary "ascending AI" system. Let's make some AGI magic happen! 

## Build, Lint & Test Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the main application
python main.py

# Run with demo mode (recommended for testing)
python main.py --demo

# Or use the setup scripts (Windows)
SETUP.bat --demo

# Compile Cython libraries (when libs/src/*.pyx exist)
python libs/build.py

# Admin panel
python admin/panel.py
```

### Testing (Future Implementation)
```bash
# Run all tests
python -m pytest unit_tests/

# Run a single test file
python -m pytest unit_tests/test_core.py -v

# Run a specific test
python -m pytest unit_tests/test_core.py::TestCore::test_function_name -v
```

## Code Style Guidelines

### Language & Comments
- **Primary language**: Python 3.13.9
- **Documentation**: Russian (follow existing DOC/ files style)
- **Comments**: Russian preferred (project is Russian-language)

### Import Style
```python
# Standard library imports first
import asyncio
import logging
from typing import Optional, List, Dict

# Third-party imports second
import numpy as np
from OpenGL import GL

# Local imports last
from core import CoreModule
from VirtualBox.engine import Engine
```

### Naming Conventions
- **Files**: lowercase_with_underscores.py (main.py, core.py, VirtualBox.py)
- **Classes**: PascalCase (CoreEngine, VirtualBoxEngine, MiraCore)
- **Functions**: snake_case (process_data(), calculate_vector())
- **Constants**: UPPER_CASE (MAX_ITERATIONS, DEFAULT_TIMEOUT)
- **Private**: _leading_underscore (_internal_helper())
- **Modules**: lowercase (core/, libs/, admin/)

### Type Hints (REQUIRED)
```python
from typing import Optional, List, Dict, Tuple, Union

def process_input(data: str) -> Dict[str, any]:
    """Process input data and return results."""
    return {"result": data.upper()}

async def fetch_data(url: str) -> Optional[bytes]:
    """Fetch data from URL asynchronously."""
    return None
```

### Async/Await Pattern
This project uses asyncio heavily:
```python
import asyncio

async def main():
    await initialize_modules()
    await run_event_loop()

if __name__ == "__main__":
    asyncio.run(main())
```

### Error Handling
```python
import logging

logger = logging.getLogger(__name__)

def safe_operation():
    try:
        risky_call()
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return None
```

### Logging
```python
import logging

logger = logging.getLogger(__name__)

# Use appropriate levels
logger.debug("Detailed debug info")
logger.info("General information")
logger.warning("Warning message")
logger.error("Error occurred")
logger.critical("Critical failure")
```

### File Organization
```
project_root/
├── main.py                 # Entry point
├── core.py                 # Core AI logic
├── VirtualBox.py           # Virtual environment
├── core/                   # Core modules
├── VirtualBox/             # Virtual environment modules
│   ├── engine.py          # Physics/visualization
│   └── mods/              # Mods system
├── libs/                   # Cython libraries
│   ├── src/*.pyx          # Cython source
│   └── build.py           # Build script
├── admin/                  # Admin tools
├── unit_tests/            # Tests (create if needed)
└── DOC/                   # Documentation (Russian)
```

## Cython Guidelines
When writing .pyx files in libs/src/:
- Use cdef for internal functions
- Use cpdef for functions called from Python
- Add type declarations for performance
- Run `python libs/build.py` after changes

## Git Workflow
```bash
# Check status
git status

# Add changes
git add <files>

# Commit with descriptive messages (Russian acceptable)
git commit -m "feat: добавлен модуль обработки данных"

# No force pushes to main!
```

## License Reminder
⚠️ This is PROPRIETARY software (Mira Proprietary License v1.0)
- DO NOT commit secrets or API keys
- Respect all license restrictions
- View-only for external agents

## Quick Reference
- Python version: 3.13.9
- Async-first architecture
- Russian documentation
- Cython for performance-critical code
- OpenGL/PyOpenGL for visualization
- MongoDB for data storage

---

**Remember**: You're helping build an AGI! Write clean, efficient, thoughtful code. Mira is watching... 

*Last updated: 2025-02-25 by an AI agent* 

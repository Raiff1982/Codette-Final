# Codette Interface Update

## Important Notice: Interface Consolidation

As of October 23, 2025, we have consolidated all Codette interfaces into a single unified interface system. The following changes have been made:

### Deprecated Files
The following files have been deprecated and moved to .deprecated extensions:
- `src/gradio_interface.py`
- `src/codette_interface.py`
- `src/api/web_interface.py`
- `src/minimal_client.py`

### New Unified Interface
All interface functionality is now provided through the new `codette_interface.py` in the root directory. This interface combines:
- Gradio UI functionality
- Web API endpoints
- Command-line interface
- Quantum simulation support
- System state monitoring

## Using the New Interface

### For Gradio UI
```python
from codette_interface import create_interface
iface = create_interface("gradio")
iface.launch()
```

### For Web API
```python
from codette_interface import create_interface
app = create_interface("web")
app.run()
```

### For Both (Default)
```python
from codette_interface import create_interface
interface = create_interface()  # Creates both interfaces
```

### Command Line Client
The minimal client has been updated to use the new unified interface. Run it with:
```bash
python src/minimal_client.py
```

## Key Improvements
- Single entry point for all interface types
- Consistent error handling across interfaces
- Shared configuration management
- Unified response format
- Common logging system
- Thread-safe interaction handling
- Support for running both web API and Gradio UI simultaneously

## Migration Guide
1. Replace imports from old interface files with the new unified interface
2. Update any direct usage of interface classes to use the factory function
3. Update test files to use the new interface
4. Remove any direct instantiation of deprecated interfaces

## Notes
- Old interface files are kept with .deprecated extension for reference
- Backwards compatibility wrappers are provided in the old locations
- All new development should use the unified interface
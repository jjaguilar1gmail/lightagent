# lightagent

This is currently just initial scaffolding for a tiny, lightweight agent built with Python and LangGraph. The goal is to keep things minimal — no unnecessary abstractions, just the essentials to get started.

## Getting Started

### 1. Create a virtual environment

```bash
python -m venv venv
```

### 2. Activate the virtual environment

- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```
- **Windows:**
  ```bash
  venv\Scripts\activate
  ```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Running Tests

This repo now includes a small stdlib `unittest` suite for the graph control-flow invariants.

```bash
python -m unittest discover -s tests -p "test_*.py"
```

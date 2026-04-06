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

## Running The Benchmark

The repo includes a benchmark harness that runs the main graph, then the evaluator, over a broad annotated question set.

```bash
python -m app.benchmark
```

Useful options:

```bash
python -m app.benchmark --limit 10
python -m app.benchmark --dataset benchmarks/questions.json --output-dir benchmark_results
python -m app.benchmark --split train
python -m app.benchmark --split holdout
```

Each run writes a JSON result file under `benchmark_results/` with per-question records plus an aggregate summary of:

- answer success rate
- evaluator best-next-support accuracy
- evaluator outcome accuracy
- helpful tool family match rate
- weak-point analysis by tag and expected support path

For iterative tuning, use the split manifest in `benchmarks/splits.json`:

- `train`: use while adjusting prompts, policies, and evaluator behavior
- `holdout`: use to check whether improvements generalize instead of overfitting the whole benchmark

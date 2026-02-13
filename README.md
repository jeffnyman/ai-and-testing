# TesterStories "AI and Testing" Code

This repository contains the Python scripts and data files associated with the the ["AI and Testing"](https://testerstories.com/category/ai/ai-and-testing/) series, as part of my TesterStories blog. Each script is designed to be standalone and corresponds to a specific article in the series.

### Configuration Files

**pyrightconfig.json** — Configures the [Pyright](https://github.com/microsoft/pyright) static type checker. It uses `basic` type checking mode with several `unknown` type warnings suppressed. This keeps type checking useful without being overly strict for the exploratory, script-based nature of this repo.

**.env** — Contains environment variables for [LangSmith](https://smith.langchain.com/), LangChain's tracing and observability platform. Placeholder values are provided for `LANGSMITH_API_KEY` and `LANGSMITH_PROJECT`. `LANGSMITH_TRACING` is set to `false` so that LangChain scripts do not automatically send traces to LangSmith unless you opt in by setting it to `true` and filling in your credentials. It also configures [DeepEval](https://docs.confident-ai.com/) with a local model (Ollama running `qwen2.5`) as the default evaluation model.

### Setup & Usage

To run these scripts locally, ensure you have **Python 3.x** installed, then follow these steps:

#### Clone the Repo

```bash
git clone https://github.com/jeffnyman/ai-and-testing
cd ai-and-testing
```

#### Option 1: The Recommended Way (Using uv)

If you have [uv](https://docs.astral.sh/uv/) installed, you don't need to manually install Python or manage virtual environments. Simply run:

```bash
uv run scripts/script_name.py
```

Replace script name with any of the scripts in the relevant directories.

#### Option 2: The Manual Way (Standard Python)

If you prefer standard tools, follow these steps:

```bash
python -m venv .venv

source .venv/bin/activate      # Linux/macOS
.venv\Scripts\activate         # Windows

python -m pip install -r requirements.txt
python scripts/script_name.py
```

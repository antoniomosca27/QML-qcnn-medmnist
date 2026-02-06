# Contributing

## Development Setup
1. Create and activate a virtual environment.
2. Install editable dependencies: `pip install -e ".[dev]"`.
3. Install hooks: `pre-commit install`.

## Quality Gates
- Format with `black .`.
- Sort imports with `isort .`.
- Run tests with `pytest`.

## Pull Requests
- Keep changes scoped to one objective.
- Include tests for behavioral changes.
- Document CLI or configuration updates in `README.md`.

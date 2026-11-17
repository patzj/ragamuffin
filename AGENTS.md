# Project Principles

- Build a lightweight, fully local RAG service.
- Prefer simple, maintainable Python and established libraries.
- Keep the application portable and Dockerized.

# Coding Guidelines

- **Keep comments useful.** Explain why, decisions, workarounds, or non-obvious behavior.
- **Prefer simple, boring code.** Favor clear control flow over cleverness or unnecessary indirection.
- **Don't over-engineer reuse.** Reuse obvious shared code, but prefer some duplication over premature abstraction.
- **Use Python pragmatically.** Prefer type hints without unnecessary abstractions.
- **Follow existing patterns.** Prefer consistency unless there is a clear reason to change.
- **Keep errors concise.** Use short, technical messages and let the error type provide context.

# Agent Instructions

- **Prefer safe local work.** Reading, editing, and running existing checks are allowed.
- **Protect dependencies.** Do not install, remove, or upgrade packages unless explicitly asked.
- **Protect external state.** Do not commit, push, publish, delete, or run remote operations unless explicitly asked.
- **Protect runtime data.** Do not delete or reset databases, documents, or persistent volumes without explicit permission.

Simplicity is the ultimate sophistication.

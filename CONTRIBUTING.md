# Contributing to ACOLYTE

Thank you for your interest in contributing to ACOLYTE!

## 🤝 How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported in [Issues](https://github.com/unmasSk/acolyte/issues)
2. Create a new issue with:
   - Clear title and description
   - Steps to reproduce
   - Expected vs actual behavior
   - System information (OS, Python version, etc.)

### Suggesting Features

1. Check existing [Issues](https://github.com/unmasSk/acolyte/issues) and [Discussions](https://github.com/unmasSk/acolyte/discussions)
2. Create a new discussion with:
   - Clear description of the feature
   - Use cases and benefits
   - Possible implementation approach

### Code Contributions

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`.scripts/dev/dev.sh test`)
5. Run linters (`.scripts/dev/dev.sh lint`)
6. Commit with conventional commits (`feat:`, `fix:`, etc.)
7. Push to your fork
8. Create a Pull Request

## 💻 Development Setup

```bash
# Clone your fork
git clone https://github.com/unmasSk/acolyte.git
cd acolyte

# Install dependencies
poetry install

# Run in development mode
./scripts/dev/dev.sh init
./scripts/dev/dev.sh start
```

## 📏 Code Standards

- **Python**: Follow PEP 8, use Black formatter
- **Line Length**: 100 characters
- **Imports**: Absolute imports from `acolyte`
- **Docstrings**: Google style
- **Type Hints**: Required for all functions
- **Tests**: Maintain >90% coverage

## 🧪 Testing

```bash
# Run all tests
./scripts/dev/dev.sh test

# Run specific test
poetry run pytest tests/test_specific.py

# Run with coverage
./scripts/dev/dev.sh coverage
```

## 📝 Commit Messages

Follow [Conventional Commits](https://www.conventionalcommits.org/):

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes
- `refactor:` Code refactoring
- `test:` Test changes
- `chore:` Build/tool changes

## 🏗️ Architecture Decisions

- **Local First**: No cloud dependencies
- **Privacy**: No telemetry or data collection
- **Simplicity**: Mono-user design
- **Performance**: Async everywhere
- **Quality**: High test coverage

## 📚 Documentation

- Update docstrings for API changes
- Update README for new features
- Add examples for complex features
- Keep PROMPT.md updated

## ❓ Questions?

- Check [Discussions](https://github.com/unmasSk/acolyte/discussions)
- Ask in [Issues](https://github.com/unmasSk/acolyte/issues)
- Read the [Documentation](docs/)

## 📜 License

By contributing, you agree that your contributions will be licensed under the BSL license.

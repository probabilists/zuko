# Contributing guidelines

First off, thank you for taking the time to contribute! 🎉

This document is a set of guidelines for contributing to the `zuko` package, which includes how to ask questions, report issues, suggest enhancements, contribute code, etc.

## I just have a question

Please **don't file an issue** to ask a question. We use [GitHub discussions](https://github.com/probabilists/zuko/discussions) as community forum for people to ask questions, share ideas or seek help. Before submitting your question, check whether it is addressed by the [documentation](https://zuko.readthedocs.io) or has already been asked in the discussions. If it has but the answer does not satisfy you, add a comment to the existing discussion instead of opening a new one.

## Submit an issue

Bugs and enhancements are tracked as [GitHub issues](https://github.com/probabilists/zuko/issues). For common issues, such as bug reports and feature requests, templates are provided. It is strongly recommended to use them as it helps understand and resolve issues faster. A clear and concise title (e.g. "RuntimeError with X when Y") also helps other users and developers to find relevant issues.

Before submitting any issue, please perform a thorough search to see if your problem or a similar one has already been reported. If it has and the issue is still open, add a comment to the existing issue instead of opening a new one. If you only find closed issues related to your problem, open a new one and include links to the closed issues in the description.

## Contribute code

If you like the project and wish to contribute, you can start by looking at issues labeled `good first issue` (should only require a few lines of code) or `help wanted` (more involved). If you found a bug and want to fix it, please create an issue reporting the bug before creating a pull request. Similarly, if you want to add a new feature, first create a feature request issue. This allows to separate the discussions related to the bug/feature, from the discussions related to the fix/implementation.

### Testing

We use [pytest](https://docs.pytest.org) to test our code base. If your contribution introduces new components, you should write new tests to make sure your code doesn't crash under normal circumstances. After installing `pytest`, add the tests to the [tests/](tests) directory and run them with

```
pytest tests
```

When you submit a pull request, tests are automatically (upon approval) executed for several versions of Python and PyTorch.

### Code conventions

We use [Ruff](https://github.com/astral-sh/ruff) to lint and format all Python code. After installing `ruff`, you can check if your code follows our conventions with

```
ruff check .
ruff format --check .
```

Additionally, please follow these rules:

* Use single quotes for strings (`'single-quoted'`) but double quotes (`"double-quoted"`) for text such as error messages.
* Use informative but concise variable names. Single-letter names are fine if the context is clear.
* Avoid explaining code with comments. If something is hard to understand, simplify or decompose it.

### Documentation

The package's [documentation](https://zuko.readthedocs.io) is automatically built by [Sphinx](https://www.sphinx-doc.org) using type hints and docstrings. All classes and functions accessible to the user should be documented with [Google style](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings) docstrings. You can build it locally with

```
cd docs
pip install -r requirements.txt
sphinx-build . html
```

### Commits

There are no rules for commits and commit messages, but we recommend to

* Avoid uninformative commit messages (e.g. "fix bug", "update", "typo").
* Use the present tense and imperative mood ("Add X" instead of "Added X" or "Adds X").
* Avoid small commits that revert/fix something introduced in the previous ones. Remember `git commit --amend` is your best friend.
* Consider [starting commit messages with an emoji](https://gitmoji.dev) to illustrate the intent of commits.
* Have fun!

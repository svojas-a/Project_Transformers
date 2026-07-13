# Project Transformers

## Overview

This repository contains the implementation for the Project Transformers research project.

The repository is organized into modular phases, allowing different team members to work independently while maintaining a consistent development environment.

---

## Repository Structure

```
configs/
data/
docs/
notebooks/
scripts/
src/
tests/
.github/workflows/
```

---

## Environment Setup

### Clone the repository

```bash
git clone <repository-url>
cd Project_Transformers
```

### Create the environment

If using Conda:

```bash
conda activate mlenv
```

or create one if needed:

```bash
conda create -n mlenv python=3.10
conda activate mlenv
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running Tests

```bash
pytest
```

---

## Formatting

```bash
black .
```

---

## Linting

```bash
ruff check .
```

---

## Branching Strategy

- Never push directly to `main`
- Create feature branches
- Open Pull Requests for review
- CI must pass before merging

Example branch:

```
feat/m0-repo-setup
```

---

## Contributors

Project Transformers Team
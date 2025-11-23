# Project Overview

This repository contains the codebase and notebooks for performing both technical stock analysis and financial text analysis.
All scripts and notebooks are organized into modular folders, each with its own README describing functionality in detail.

### Reproducing the Environment

1. Clone the project from GitHub (replace <your-username> with your actual GitHub username):
   
```
git clone https://github.com/<your-username>/solar-challenge-week0.git

cd news-sentiment-week1
```

2. Create and Activate the Virtual Environment
   
- Create the venv:
```
python -m venv .venv
```

- Activate the venv
```
.venv\Scripts\activate
```
3. Install Dependencies
```
pip install -r requirements.txt
```
4. CI Workflow

A GitHub Actions workflow is set up in .github/workflows/ci.yml to automatically run ```pip install -r requirements.txt``` and verify the Python version on every push and pull request.


### Components

scripts/ – Core utilities for data loading, preprocessing, EDA, technical indicator calculations, and financial text analytics.

notebooks/ – Interactive Jupyter notebooks demonstrating the analysis workflows end-to-end.

### Documentation

See the individual README files within each folder (scripts/ and notebooks/) for detailed descriptions.
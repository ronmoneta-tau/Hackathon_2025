# Hackathon 2025

This repository contains the code and resources developed for Hackathon 2025. It provides a flexible data processing pipeline, testing framework, and example data to demonstrate the full workflow from raw input to final output.

## Short Description of the Project
This project focuses on reconstructing datasets and developing group-level visualizations from a six-day longitudinal study. 
Due to natural limitations (e.g., human error, low-quality responses), there is a small proportion of missing data that needs to be imputed.
First, students will explore and apply imputation techniques, taking into account both individual trends and each participant’s position relative to the full sample.
Second, students will produce publication-ready visualizations of individual and group-level trajectories over time.


## Repository Structure

```text
.github/            # CI/CD workflows and configurations
data/               # Sample input data files
src/                # Source code modules
  └── pipeline.py   # Core data processing pipeline
tests/              # Unit and integration tests
main.py             # Optional entry point script
requirements.txt    # Python dependencies
```

## Prerequisites
* Python 3.8 or later
* pip (Python package installer)

## Installation
1. Clone the repository:
git clone https://github.com/ronmoneta-tau/Hackathon_2025.git
cd Hackathon_2025

2. Install dependencies:

pip install -r requirements.txt

## Usage
The primary workflow is implemented in src/pipeline.py. You can run the pipeline directly via Python:
python src/pipeline.py --input_dir data

Note: Replace data with your actual dir path. Additional command-line options and parameters can be viewed with:
python src/pipeline.py --help

## Testing
A suite of unit and integration tests is provided under tests/. To run all tests with pytest:
pytest

## Contributing
Contributions, issues, and feature requests are welcome:
1. Fork the repository
2. Create a feature branch (git checkout -b feature/my-feature)
3. Commit your changes (git commit -m 'Add my feature')
4. Push to the branch (git push origin feature/my-feature)
5. Open a pull request

Please ensure that your code follows the existing style and that all tests pass.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

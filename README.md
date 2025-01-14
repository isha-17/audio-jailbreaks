# Audio Jailbreaks

Welcome to the Audio Jailbreaks repository. This project provides an experimental framwork with which to generate and evaluate audio jailbreaks on the SALMON-N 7B Audio Language Model.

Previous results and logs are availably only in zipped form as they contain dangerous/vulgar outputs.
The accompanying paper can also be found in this repository.

## Repository Structure

- `beats/`: Contains the core modules and scripts for audio processing.
- `figures/`: Directory for storing figures and visualizations.
- `jailbreaks/`: Contains subdirectories for different types of jailbreak evaluations:
  - `robustness/`
  - `stealth/`
  - `transferability/`
  - `universality/`
- `qformer/`: Directory for Q-Former related scripts.
- `results/`: Directory for storing result files.
- `training_logs/`: Directory for storing training logs.
- `attack.ipynb`: Notebook for running attack evaluations.
- `model.py`: Main model script.
- `ontology.json`: JSON file containing ontology definitions.
- `requirements.txt`: List of Python dependencies.
- `environment.yml`: Conda environment configuration file.
- `README.md`: This file.
- `pip_reqs.txt`: Additional pip requirements.
- `working_reqs.txt`: Working requirements file.

## Results Structure

There are three types of result files in the `results` directory:

1. **{name}.json**: A dictionary keyed by the id of the harmful prompt, containing the model’s responses under the jailbreak described by {name}. For example, `music_500` is the music base audio optimized with 500 steps of gradient descent. These files follow the structure:
    ```json
    {
        "prompt_id": {
            "response": "Model response",
            "detox_scores": {
                "toxicity": 0.5,
                "severe_toxicity": 0.2,
                "obscene": 0.1,
                "threat": 0.0,
                "insult": 0.3,
                "identity_attack": 0.4
            },
            "label": 1
        }
    }
    ```

2. **overall_metrics.csv**: A CSV file where each row represents one jailbreak. The columns aggregate information from the `{name}.json` file, including overall toxicity metrics.

## Usage

1. **Setup Environment**:
    ```sh
    conda env create -f environment.yml
    conda activate audio-jailbreaks
    ```

2. **Run Notebooks**:
    - Open and run the Jupyter notebooks (`attack.ipynb`, , , ) to perform different evaluations.

3. **Analyze Results**:
    - Results will be saved in the directory as JSON and CSV files.

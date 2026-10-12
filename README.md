\# Self-Validating AI Agent for Automated Exploratory Data Analysis



<p align="center">

&#x20; <strong>Automating Data Analysis with Validation, Self-Correction, and Evidence-Grounded Insights</strong>

</p>



<p align="center">

&#x20; <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python 3.11">

&#x20; <img src="https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit" alt="Streamlit">

&#x20; <img src="https://img.shields.io/badge/Data%20Science-EDA-orange" alt="Data Science">

&#x20; <img src="https://img.shields.io/badge/Project-Under%20Development-yellow" alt="Under Development">

</p>



\## Overview



Exploratory Data Analysis (EDA) is an essential step in data science, but it often involves repetitive tasks such as examining datasets, planning analyses, validating results, interpreting findings, and preparing reports.



This project explores a \*\*self-validating AI agent\*\* that automates parts of this workflow through a modular Python pipeline. It combines data profiling, analysis planning, code execution, validation, insight generation, and evidence mapping to support more reliable and traceable data analysis.



The objective is to reduce repetitive analytical work while keeping generated findings connected to supporting evidence.



\## Key Features



| Feature | Description |

|---|---|

| Data Profiling | Examines dataset structure, columns, data types, and data quality. |

| EDA Planning | Organizes analytical tasks into a structured workflow. |

| Code Generation and Execution | Supports programmatic execution of analytical steps. |

| Result Validation | Checks analytical outputs using implemented validation logic. |

| Self-Correction | Attempts to recover from supported execution errors. |

| Insight Validation | Evaluates generated insights against available analytical evidence. |

| Evidence Mapping | Connects findings with supporting results. |

| Recommendations | Generates recommendations based on analysis findings. |

| Interactive Interface | Provides a Streamlit application for interacting with the workflow. |



\*Feature availability and behavior depend on the current implementation and configuration.\*



\## System Architecture



The intended analytical workflow is illustrated below.



```mermaid

flowchart TD

&#x20;   A\[Input Dataset] --> B\[Data Profiling]

&#x20;   B --> C\[Task Analysis]

&#x20;   C --> D\[EDA Planning]

&#x20;   D --> E\[Code Generation]

&#x20;   E --> F\[Code Execution]

&#x20;   F --> G\[Result Validation]

&#x20;   G -->|Valid| H\[Insight Generation]

&#x20;   G -->|Failure| I\[Error Correction]

&#x20;   I --> E

&#x20;   H --> J\[Insight Validation]

&#x20;   J --> K\[Evidence Mapping]

&#x20;   K --> L\[Recommendations]

&#x20;   L --> M\[Final Report]

```



\## Technology Stack



\- \*\*Programming Language:\*\* Python

\- \*\*Data Processing:\*\* Pandas, NumPy

\- \*\*Data Visualization:\*\* Matplotlib, Seaborn

\- \*\*Machine Learning Utilities:\*\* Scikit-learn

\- \*\*Application Framework:\*\* Streamlit

\- \*\*AI Integration:\*\* OpenAI API, where configured

\- \*\*Development Tools:\*\* Jupyter Notebook, VS Code, Git, GitHub



\## Repository Structure



```text

Self\_Validating\_Data\_Agent/

│

├── app.py

├── requirements.txt

├── commands.txt

├── .gitignore

│

├── data/

│   └── raw/

│       ├── sales.csv

│       └── test\_data.csv

│

├── src/

│   ├── data\_loader.py

│   ├── data\_profiler.py

│   ├── task\_analyzer.py

│   ├── eda\_planner.py

│   ├── code\_generator.py

│   ├── code\_executor.py

│   ├── code\_corrector.py

│   ├── result\_aggregator.py

│   ├── insight\_generator.py

│   ├── insight\_validator.py

│   ├── claim\_validator.py

│   ├── evidence\_mapping.py

│   ├── self\_correction.py

│   ├── recommendation\_generator.py

│   └── final\_report.py

│

└── tests/

&#x20;   └── test\_data\_loader.py

```



Additional modules and test scripts are included in the repository.



\## Getting Started



\### Prerequisites



\- Python 3.11

\- Conda or another Python environment manager

\- Git

\- An OpenAI API key if your selected workflow requires the OpenAI API



\### 1. Clone the Repository



```bash

git clone https://github.com/arkatvijay0/Self\_Validating\_Data\_Agent.git

cd Self\_Validating\_Data\_Agent

```



\### 2. Create the Environment



```bash

conda create -n data\_agent python=3.11

conda activate data\_agent

```



\### 3. Install Dependencies



```bash

python -m pip install -r requirements.txt

```



\### 4. Configure API Access (If Required)



Store your API key in an environment variable or a local `.env` file. Never commit API keys or other credentials to GitHub.



For example, in Windows Command Prompt:



```cmd

set OPENAI\_API\_KEY=your\_api\_key\_here

```



Replace the placeholder with your actual key locally. Do not publish the key or include it in source code.



\## Running the Application



Launch the Streamlit interface:



```bash

streamlit run app.py

```



Open the local URL displayed in your terminal.



To run the analysis pipeline directly, inspect the current entry point and its required arguments before using:



```bash

python src/main.py

```



The direct pipeline command may require a dataset path or an analysis question, depending on the implementation.



\## Testing



The repository includes test modules for data loading, result validation, insight validation, claim validation, and error-recovery workflows.



Run the data loader test with:



```bash

python -m pytest tests/test\_data\_loader.py -v

```



Run the broader test suite with:



```bash

python -m pytest

```



Some test scripts may require additional configuration or an API key. Review their prerequisites before running the full suite.



\## Potential Applications



\- Initial exploratory analysis of structured datasets

\- Data quality inspection

\- Exploratory sales and business analytics

\- Evidence-supported analytical reporting

\- Automation of repetitive data analysis tasks



\## Current Limitations



\- Analytical quality depends on dataset quality and the validation rules implemented.

\- Generated insights may require manual verification.

\- Some workflows may depend on external AI services and API configuration.

\- Runtime can vary with dataset size, model calls, and execution complexity.

\- The project is under development, and not all workflows may be fully automated.



\## Future Enhancements



\- Expand statistical and data-quality validation.

\- Improve execution speed and error recovery.

\- Add comprehensive tests and reproducible benchmarks.

\- Improve interactive visualizations and report generation.

\- Evaluate insight accuracy and evidence traceability across multiple datasets.



\## Author



\*\*Vijay Kumar AG\*\*  

M.Tech in Data Science | Presidency University, Bengaluru



\- GitHub: \[@arkatvijay0](https://github.com/arkatvijay0)

\- Project Repository: \[Self\_Validating\_Data\_Agent](https://github.com/arkatvijay0/Self\_Validating\_Data\_Agent)



\---



<p align="center">

&#x20; <i>Building toward reliable, automated, and evidence-grounded data analysis.</i>

</p>


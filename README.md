\# Self-Validating AI Agent for Automated Exploratory Data Analysis



<p align="center">

&#x20; <strong>An AI-powered framework for automated data analysis, result validation, and evidence-grounded insights.</strong>

</p>



<p align="center">

&#x20; <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python" alt="Python">

&#x20; <img src="https://img.shields.io/badge/Streamlit-Web\_App-FF4B4B?logo=streamlit" alt="Streamlit">

&#x20; <img src="https://img.shields.io/badge/Domain-Data\_Science-orange" alt="Data Science">

&#x20; <img src="https://img.shields.io/badge/Status-Under\_Development-yellow" alt="Under Development">

</p>



\---



\## \*\*📌 1. Project Overview\*\*



Exploratory Data Analysis (EDA) is an essential step in data science, but it often involves repetitive tasks such as data profiling, statistical analysis, result interpretation, and report generation.



This project aims to automate parts of the EDA workflow using a modular Python pipeline with \*\*result validation, self-correction, insight verification, and evidence mapping\*\*.



The goal is to improve analytical efficiency and traceability by connecting generated findings to supporting analytical results.



\## \*\*🎯 2. Project Objectives\*\*



\- Automate repetitive exploratory data analysis tasks.

\- Organize analytical tasks into a structured pipeline.

\- Validate analytical results and generated insights.

\- Support recovery from selected execution errors.

\- Map insights to supporting evidence.

\- Generate summaries and recommendations from analytical findings.



\## \*\*🚀 3. Key Features\*\*



| \*\*Feature\*\* | \*\*Description\*\* |

|:---|:---|

| \*\*Data Profiling\*\* | Examines dataset structure, columns, data types, and data quality. |

| \*\*EDA Planning\*\* | Organizes analytical tasks into a structured workflow. |

| \*\*Code Generation \& Execution\*\* | Supports programmatic execution of analytical steps. |

| \*\*Result Validation\*\* | Checks analytical outputs using implemented validation logic. |

| \*\*Self-Correction\*\* | Attempts to recover from supported execution errors. |

| \*\*Insight Validation\*\* | Evaluates generated insights against available analytical evidence. |

| \*\*Evidence Mapping\*\* | Connects findings with supporting analytical results. |

| \*\*Recommendation Generation\*\* | Generates recommendations based on analysis findings. |

| \*\*Interactive Interface\*\* | Provides a Streamlit interface for interacting with the workflow. |



\*Feature behavior depends on the current implementation and configuration.\*



\## \*\*🏗️ 4. System Architecture\*\*



The intended workflow is:



\*\*Dataset Input → Data Profiling → Task Analysis → EDA Planning → Code Generation → Execution → Result Validation → Insight Validation → Evidence Mapping → Recommendations → Final Report\*\*



```mermaid

flowchart TD

&#x20;   A\["Dataset Input"] --> B\["Data Profiling"]

&#x20;   B --> C\["Task Analysis"]

&#x20;   C --> D\["EDA Planning"]

&#x20;   D --> E\["Code Generation"]

&#x20;   E --> F\["Code Execution"]

&#x20;   F --> G{"Result Validation"}

&#x20;   G -->|Pass| H\["Insight Generation"]

&#x20;   G -->|Fail| I\["Error Correction"]

&#x20;   I --> E

&#x20;   H --> J\["Insight Validation"]

&#x20;   J --> K\["Evidence Mapping"]

&#x20;   K --> L\["Recommendations"]

&#x20;   L --> M\["Final Report"]

```



\## \*\*🛠️ 5. Technology Stack\*\*



| \*\*Category\*\* | \*\*Technologies\*\* |

|:---|:---|

| \*\*Programming Language\*\* | Python 3.11 |

| \*\*Data Processing\*\* | Pandas, NumPy |

| \*\*Visualization\*\* | Matplotlib, Seaborn |

| \*\*Machine Learning Utilities\*\* | Scikit-learn |

| \*\*Application Framework\*\* | Streamlit |

| \*\*AI Integration\*\* | OpenAI API, where configured |

| \*\*Development Tools\*\* | Jupyter Notebook, VS Code |

| \*\*Version Control\*\* | Git, GitHub |



\## \*\*📂 6. Repository Structure\*\*



| \*\*Path\*\* | \*\*Purpose\*\* |

|:---|:---|

| `app.py` | Streamlit application entry point |

| `src/data\_loader.py` | Dataset loading |

| `src/data\_profiler.py` | Dataset profiling |

| `src/task\_analyzer.py` | Analytical task analysis |

| `src/eda\_planner.py` | EDA planning |

| `src/code\_generator.py` | Analytical code generation |

| `src/code\_executor.py` | Code execution |

| `src/insight\_generator.py` | Insight generation |

| `src/insight\_validator.py` | Insight validation |

| `src/claim\_validator.py` | Claim validation |

| `src/evidence\_mapping.py` | Evidence mapping |

| `src/recommendation\_generator.py` | Recommendation generation |

| `src/final\_report.py` | Final report generation |

| `tests/` | Automated tests |

| `data/raw/` | Input datasets |

| `requirements.txt` | Python dependencies |



\## \*\*⚙️ 7. Installation and Setup\*\*



\### \*\*Prerequisites\*\*



\- Python 3.11

\- Conda

\- Git

\- An OpenAI API key if required by the selected workflow



\### \*\*Step 1: Clone the Repository\*\*



```bash

git clone https://github.com/arkatvijay0/Self\_Validating\_Data\_Agent.git

cd Self\_Validating\_Data\_Agent

```



\### \*\*Step 2: Create the Environment\*\*



```bash

conda create -n data\_agent python=3.11

conda activate data\_agent

```



\### \*\*Step 3: Install Dependencies\*\*



```bash

python -m pip install -r requirements.txt

```



\## \*\*▶️ 8. How to Run\*\*



\### \*\*Launch the Streamlit Application\*\*



```bash

streamlit run app.py

```



\### \*\*Run the Analysis Pipeline\*\*



```bash

python src/main.py

```



The direct pipeline command may require additional input arguments. Check the current implementation for the expected dataset path and analysis question.



\## \*\*🧪 9. Testing\*\*



Run the data loader test:



```bash

python -m pytest tests/test\_data\_loader.py -v

```



Run the test suite:



```bash

python -m pytest

```



Some tests may require additional configuration or API access.



\## \*\*💡 10. Potential Applications\*\*



\- Automated exploratory data analysis

\- Initial data quality assessment

\- Exploratory sales and business analytics

\- Evidence-supported insight generation

\- Repetitive data analysis workflow automation



\## \*\*⚠️ 11. Limitations\*\*



\- Analysis quality depends on input data and validation rules.

\- AI-generated findings may require manual verification.

\- Some workflows may require external API access.

\- Runtime depends on dataset size and execution complexity.

\- The project is under development.



\## \*\*🔮 12. Future Enhancements\*\*



\- Expand statistical and data-quality validation.

\- Improve execution speed and error recovery.

\- Add comprehensive automated tests.

\- Improve report generation and interactive visualizations.

\- Evaluate insight accuracy across multiple datasets.



\## \*\*👨‍💻 13. Author\*\*



\*\*Vijay Kumar AG\*\*



M.Tech in Data Science | Presidency University, Bengaluru



\- \*\*GitHub:\*\* \[@arkatvijay0](https://github.com/arkatvijay0)

\- \*\*Project Repository:\*\* \[Self\_Validating\_Data\_Agent](https://github.com/arkatvijay0/Self\_Validating\_Data\_Agent)



\---



<p align="center">

&#x20; <strong>Building toward reliable, automated, and evidence-grounded data analysis.</strong>

</p>


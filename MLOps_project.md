# MLOps Project

The aim of the project is to simulate the real world process of deploying machine learning models, using the concepts discussed during the classes.

---

## Project Deliverables

### Report (maximum 6 pages)

- Explain why you chose that data and what you try to achieve. Define **your success metrics**.
- Project planning: how you organized and scheduled the different steps (you can be inspired by sprints in the agile methodology).
- **Results and conclusions** from data exploration and data modelling (plots, feature importance, explainability).
- Since this is a proof of concept, discuss how this would be implemented in production and what are the **advantages of the technologies used, risks and possible mitigations** — e.g. "we are using only Pandas, so if the amount of data scales up, our pipeline will not be efficient. We propose more x weeks to build in Spark, as a mitigation solution."
- **List of the packages and versions** used for the project.

### Code for Generating Your Pipeline

- Preference to use **Kedro organization and modular code**, also for orchestration. You can keep your exploration notebooks in the appropriate folder. *(Classes Week 3 and Week 4)*
- Try to include the following components in your pipeline:
  1. **Unit data tests and feature store**: you can use one of the tools from class or your own solutions, but it is important to have several asserts for data quality. *(Class Week 1)*
  2. **MLflow** for **experimentation and model versioning**. *(Class Week 2)*
  3. Save model main metrics and explainability (SHAP) and include explanations about them in the report.
  4. Model serving and containers. *(Class Week 4/5)*
  5. **Data drift evaluation**: if you build a pipeline to test a sample of data from your strongest model, include this component as well. You can play with your sample to generate drift or see how the metrics would change if drift happened. *(Class Week 6)*
  6. Try to **build tests** for your relevant functions and pipelines. *(Classes Week 3 and Week 4)*

In the end, everyone should be able to run their pipeline and produce the same results. Projects will be graded based on the **quality of the report, code, and creativity shown** for using the technologies.

---

## Project Structure

A real use case organized and ready to be used in MLOps environments with separated pipeline orchestration.

```
conf/
data/
├── 01_raw/
├── 02_intermediate/
├── 03_primary/
├── 04_feature/
├── 05_model_input/
├── 06_models/
├── 07_model_output/
└── 08_reporting/
docs/
notebooks/
src/
└── mlops_house_pricing/
    └── pipelines/
        ├── data_cleaning/
        ├── data_drifts/
        ├── data_feat_engineering/
        ├── data_quality/
        ├── data_split/
        ├── model_predict/
        ├── model_selection/
        ├── model_train/
        ├── __init__.py
    ├── __init__.py
    ├── __main__.py
    ├── pipeline_registry.py
    └── settings.py
tests/
```

Each component is a pipeline that can run in full sequential sequence:

- e.g. `data_quality` → `data_cleaning` → `data_feat_engineering` → ... → `model_train` → `data_drift`
- Or you can choose to run a separated pipeline — e.g. after the model is in production, only run the `data_quality` or `data_drift` part.

---

## Submission

> **You should send the report, a zip of the code with a sample of data to run, or a Git link.**

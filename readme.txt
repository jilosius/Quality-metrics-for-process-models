BPMN Model Similarity Tool

This is a command-line tool for analyzing Business Process Model and Notation (BPMN) models, applying modifications, and computing similarity metrics between reference and altered models.

Project Structure:
------------------
tool/
├── csv/                                # Stores input/output CSV data for experiments
├── logs/                               # Stores logs runs, experiments, etc.
├── model_alteration/                   # Contains all functions for altering BPMN models
│   ├── add_flow.py                     # Adds a flow between elements
│   ├── add_flowNode.py                 # Adds a new flow node (activity/gateway)
│   ├── add_gateway.py                  # Specifically adds gateways
│   ├── change_label.py                 # Changes the label of a node
│   ├── model_alteration.py             # Main module coordinating model alteration
│   ├── remove_activity.py              # Removes activity nodes
│   ├── remove_flow.py                  # Removes flows
│   ├── remove_flowNode.py              # General flow node removal
│   └── remove_gateway.py               # Removes gateways
├── models/                             # Stores input BPMN models (XML format)
├── plots/                              # Saves generated visual plots (if any)
├── process/                            # Contains the BPMN model representation
│   ├── data_obj.py                     # Class for BPMN DataObject elements
│   ├── flow.py                         # Class for BPMN Flow elements
│   ├── flowNode.py                     # Class for BPMN flowNode elements
│   ├── lane.py                         # Class for BPMN Lane elements
│   ├── pool.py                         # Class for BPMN Pool elements
│   └── process.py                      # The main Process class structure
├── similarity_metric/                  # Houses different similarity metrics
│   ├── compliance_metric.py            # Compliance Degree, Compliance Maturity computation
│   ├── f1_score.py                     # F1-Score computation
│   ├── node_structural_metric.py       # Node, Structral, and behavioral similarity computation
│   └── similarity_metric.py            # Base class / interface for metrics
├── tables/                             # Contains output LaTeX tables
├── combination_alterations_experiment.py   # Experiment with combinations of alterations
├── io_handler.py                      # Handles reading/writing BPMN models
├── metric_profiles.ipynb              # Jupyter notebook analyzing metric behavior
├── rq1_experiment.py                  # Experiment for answering RQ1                   (25 iterations each alteration)
├── test_process.py                    # Test importing processes              
└── tool_controller.py                 # Main controller                                (1 iteration)

Installation:
-------------
Install dependencies using:

    pip install -r requirements.txt

This will install required packages like pm4py, networkx, transformers, etc.

Running the Tool:
-----------------
Basic commands:

    python tool_controller.py <file_path> <output_path> <alteration_type> <num_alterations>
    python rq1_experiment.py <file_path> <alteration_type> <max_alterations> <output_csv> <log_txt>

To view the usage instructions with all available flags, run:

    python tool_controller.py -h
    python rq1_experiment.py -h

Example Commands:
-----------------
Remove a random activity:

    python tool_controller.py models/task_reference5.bpmn altered_model.bpmn -remove_activity 1

Multiple alterations:

    python tool_controller.py models/task_reference5.bpmn altered_model.bpmn -add_activity 2 -remove_gateway 1 -change_label 3


Reference model used for experiments: models/task_reference5.bpmn
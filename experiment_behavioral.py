import argparse
import time
from io_handler import IOHandler
from process.process import Process
from model_alteration.model_alteration import ModelAlteration
from similarity_metric.similarity_metric import SimilarityMetric
from similarity_metric.compliance_metric import ComplianceMetric
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb


class Experiment:
    def __init__(self, alteration_type, max_alterations):
        """
        Initializes the Experiment class.

        :param alteration_type: Type of alteration to apply (e.g., change_label).
        :param max_alterations: Maximum number of alterations to perform.
        """
        self.alteration_type = alteration_type
        self.max_alterations = max_alterations
        self.results = []  # Placeholder for storing similarity scores

    def simulate_alterations(self, file_path, metric_names, output_csv):
        """
        Simulates the given alteration incrementally and saves the results to a CSV file.

        :param file_path: Path to the BPMN file.
        :param metric_names: List of metric names to calculate (e.g., NodeStructuralBehavioralMetric, F1Score).
        :param output_csv: Path to save the results CSV.
        """
        print("Reading the BPMN model...")
        reference_bpmn_tree = IOHandler.read_bpmn(file_path)

        print("Converting BPMN model to Process object...")
        reference_model = Process()
        reference_model.from_bpmn(reference_bpmn_tree.getroot())

        num_alterations = 1  # Start with 1 alteration

        while num_alterations <= self.max_alterations:
            print(f"Applying {num_alterations} {self.alteration_type}(s)...")

            for repetition in range(25):  # Perform similarity calculation 5 times for each alteration count
                print(f"Iteration number: {repetition}")
                # Perform alteration
                model_alteration = ModelAlteration(reference_model)
                model_alteration.apply_alteration(self.alteration_type, num_alterations)

                # Calculate the similarity metrics
                altered_model = model_alteration.altered_model
                node_structural_scores = SimilarityMetric.get_metric(metric_names[0], reference_model, altered_model).calculate()
                f1_scores = SimilarityMetric.get_metric(metric_names[1], reference_model, altered_model).calculate()

                # Store the results
                self.results.append( (
                    num_alterations,
                    repetition + 1,
                    self.alteration_type,
                    node_structural_scores.get("node_similarity", 0),
                    node_structural_scores.get("structural_similarity", 0),
                    node_structural_scores.get("behavioral_similarity", 0),
                    f1_scores.get("Precision", 0),
                    f1_scores.get("Recall", 0),
                    f1_scores.get("F1 Score", 0)
                ))

            # Increment the number of alterations
            num_alterations += 1

        # Save results to CSV
        self.save_results(output_csv)

    def save_results(self, output_file):
        """
        Saves the experiment results to a CSV file.

        :param output_file: Path to the output file.
        """
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write header
            writer.writerow([
                "Alteration Count", "Repetition", "Alteration Type", "Node Similarity",
                "Structural Similarity", "Behavioral Similarity", "Precision", "Recall", "F1-Score"
            ])
            # Write data
            for alteration_count, repetition, alteration_type, node_sim, struct_sim, behav_sim, precision, recall, f1_score in self.results:
                writer.writerow([
                    alteration_count, repetition, alteration_type, node_sim, struct_sim,
                    behav_sim, precision, recall, f1_score
                ])


    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run similarity metric experiments.")
    parser.add_argument("file_path", type=str, help="Path to the BPMN file.")
    parser.add_argument("alteration_type", type=str, help="Type of alteration to apply (e.g., change_label).")
    parser.add_argument("max_alterations", type=int, help="Maximum number of alterations to perform.")
    parser.add_argument("output_csv", type=str, help="File to save the experiment results.")
    
    args = parser.parse_args()

    start_time = time.time()
    experiment = Experiment(args.alteration_type, args.max_alterations)
    experiment.simulate_alterations(args.file_path, ["NodeStructuralBehavioralMetric", "F1Score"], args.output_csv)

    end_time = time.time()

    print(f"Script completed in {end_time - start_time:.2f} seconds.")
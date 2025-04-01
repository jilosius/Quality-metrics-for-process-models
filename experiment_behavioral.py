import argparse
import time
import sys
import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb
from io_handler import IOHandler
from process.process import Process
from model_alteration.model_alteration import ModelAlteration
from similarity_metric.similarity_metric import SimilarityMetric
from similarity_metric.compliance_metric import ComplianceMetric

class WriteLog:
    def __init__(self, file):
        self.file = file
        self.stdout = sys.stdout
        self.stderr = sys.stderr 

    def write(self, text):
        self.file.write(text)  
        self.stdout.write(text)  

    def flush(self):
        self.file.flush()
        self.stdout.flush()


class Experiment:
    def __init__(self, alteration_type, max_alterations):

        self.alteration_type = alteration_type
        self.max_alterations = max_alterations
        self.results = [] 

    def simulate_alterations(self, file_path, metric_names, output_csv):
        print("Reading the BPMN model...")
        reference_bpmn_tree = IOHandler.read_bpmn(file_path)

        print("Converting BPMN model to Process object...")
        reference_model = Process()
        reference_model.from_bpmn(reference_bpmn_tree.getroot())

        num_alterations = 1  

        tokenizer = None
        model = None
        if self.alteration_type == "change_label":
            print("Loading paraphrasing model...")
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
            model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")


        while num_alterations <= self.max_alterations:
            print(f"Applying {num_alterations} {self.alteration_type}(s)...")

            for repetition in range(25):  
                print(f"Iteration number: {repetition + 1}")

                model_alteration = ModelAlteration(reference_model)

                if self.alteration_type == "change_label":
                    model_alteration.apply_alteration(
                        self.alteration_type,
                        num_alterations,
                        tokenizer=tokenizer,
                        model=model
                    )
                else:
                    model_alteration.apply_alteration(
                        self.alteration_type,
                        num_alterations
                    )

                altered_model = model_alteration.altered_model


                try:
                    node_structural_scores = SimilarityMetric.get_metric(metric_names[0], reference_model, altered_model).calculate()
                except Exception as e:
                    print(f"Error calculating {metric_names[0]}: {e}")
                    node_structural_scores = {"node_similarity": 0, "structural_similarity": 0, "behavioral_similarity": 0}

                try:
                    f1_scores = SimilarityMetric.get_metric(metric_names[1], reference_model, altered_model).calculate()
                except Exception as e:
                    print(f"Error calculating {metric_names[1]}: {e}")
                    f1_scores = {"Precision": 0, "Recall": 0, "F1 Score": 0}

                try:
                    compliance_scores = SimilarityMetric.get_metric(metric_names[2], reference_model, altered_model, file_path, "output_test.bpmn").calculate()
                except Exception as e:
                    print(f"Error calculating {metric_names[2]}: {e}")
                    compliance_scores = {"compliance_degree": 0, "compliance_maturity": 0}

                # Store the results safely
                result_entry = (
                    num_alterations,
                    repetition + 1,
                    self.alteration_type,
                    node_structural_scores.get("node_similarity", 0),
                    node_structural_scores.get("structural_similarity", 0),
                    node_structural_scores.get("behavioral_similarity", 0),
                    f1_scores.get("Precision", 0),
                    f1_scores.get("Recall", 0),
                    f1_scores.get("F1 Score", 0),
                    compliance_scores.get("compliance_degree", 0),
                    compliance_scores.get("compliance_maturity", 0)
                )

                self.results.append(result_entry)

                # Print to log
                print(f"Results for alteration {num_alterations}, repetition {repetition + 1}:")
                print(f"Node Similarity: {result_entry[3]:.6f}, Structural Similarity: {result_entry[4]:.6f}, "
                    f"Behavioral Similarity: {result_entry[5]:.6f}")
                print(f"Precision: {result_entry[6]:.6f}, Recall: {result_entry[7]:.6f}, F1-Score: {result_entry[8]:.6f}")
                print(f"Compliance Degree: {result_entry[9]:.6f}, Compliance Maturity: {result_entry[10]:.6f}\n")

            num_alterations += 1

        self.save_results(output_csv)


    def save_results(self, output_file):
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            writer.writerow([
                "Alteration Count", "Repetition", "Alteration Type","Node Similarity", "Structural Similarity", "Behavioral Similarity",  "Precision", "Recall", "F1-Score", "Compliance Degree", "Compliance Maturity"
            ]) 
            for alteration_count, repetition, alteration_type, node_sim, struct_sim, behav_sim, precision, recall, f1_score, compliance_degree, compliance_maturity in self.results:
                writer.writerow([
                    alteration_count, repetition, alteration_type, node_sim, struct_sim,
                    behav_sim, precision, recall, f1_score, compliance_degree, compliance_maturity
                ])

            


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run similarity metric experiments.")
    parser.add_argument("file_path", type=str, help="Path to the BPMN file.")
    parser.add_argument("alteration_type", type=str, help="Type of alteration to apply (e.g., change_label).")
    parser.add_argument("max_alterations", type=int, help="Maximum number of alterations to perform.")
    parser.add_argument("output_csv", type=str, help="File to save the experiment results.")
    parser.add_argument("log_txt", type=str, help="File to save execution log.")

    args = parser.parse_args()

    start_time = time.time()
    experiment = Experiment(args.alteration_type, args.max_alterations)

    output_log = args.log_txt

    with open(output_log, "w") as log_file:
        sys.stdout = sys.stderr = WriteLog(log_file)  
        try:
            experiment.simulate_alterations(
                args.file_path,
                ["NodeStructuralBehavioralMetric", "F1Score", "ComplianceMetric"],
                args.output_csv
            )
        finally:
            sys.stdout = sys.__stdout__
            sys.stderr = sys.__stderr__
            print(f"Logs can be found in: {output_log}")

    end_time = time.time()
    print(f"Script completed in {end_time - start_time:.2f} seconds.")

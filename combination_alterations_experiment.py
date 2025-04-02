import random
import csv
from io_handler import IOHandler
from process.process import Process
from model_alteration.model_alteration import ModelAlteration
from similarity_metric.similarity_metric import SimilarityMetric


class Experiment3:
    def __init__(self, alteration_types, iterations=5, sequence_length=8):
        self.alteration_types = alteration_types
        self.iterations = iterations
        self.sequence_length = sequence_length
        self.results = []

    def run(self, file_path, output_csv):
        print("Reading and parsing the BPMN model...")
        reference_bpmn_tree = IOHandler.read_bpmn(file_path)
        reference_model = Process()
        reference_model.from_bpmn(reference_bpmn_tree.getroot())

        all_node_ids = [node.flowNode_id for node in reference_model.flowNodes]

        for run_id in range(self.iterations):
            print(f"\n=== Iteration {run_id + 1}/{self.iterations} ===")
            model_copy = reference_model.clone()
            model_alteration = ModelAlteration(model_copy)

            used_node_ids = set()
            sequence = random.choices(self.alteration_types, k=self.sequence_length)
            applied_sequence = []
            alteration_counts = {alt: 0 for alt in self.alteration_types}

            for alteration in sequence:
                target_node_id = None

                if alteration == "change_label":
                    unused_nodes = [nid for nid in all_node_ids if nid not in used_node_ids]
                    if not unused_nodes:
                        print("No unused nodes left for label change.")
                        continue
                    target_node_id = random.choice(unused_nodes)
                    used_node_ids.add(target_node_id)

                try:
                    model_alteration.apply_alteration(alteration, repetitions=1, node_id=target_node_id)
                    alteration_counts[alteration] += 1
                    applied_sequence.append((alteration, target_node_id))
                except Exception as e:
                    print(f"Failed to apply {alteration}: {e}")

            # compute similarity metrics
            altered_model = model_alteration.altered_model
            try:
                node_structural_scores = SimilarityMetric.get_metric("NodeStructuralBehavioralMetric", reference_model, altered_model).calculate()
                f1_scores = SimilarityMetric.get_metric("F1Score", reference_model, altered_model).calculate()
                compliance_scores = SimilarityMetric.get_metric("ComplianceMetric", reference_model, altered_model, file_path, "output_test.bpmn").calculate()
            except Exception as e:
                print(f"Similarity metric failed: {e}")
                node_structural_scores, f1_scores, compliance_scores = {}, {}, {}

            self.results.append({
                "iteration": run_id + 1,
                "sequence": applied_sequence,
                "alteration_counts": alteration_counts,
                "node_similarity": node_structural_scores.get("node_similarity", 0),
                "structural_similarity": node_structural_scores.get("structural_similarity", 0),
                "behavioral_similarity": node_structural_scores.get("behavioral_similarity", 0),
                "precision": f1_scores.get("Precision", 0),
                "recall": f1_scores.get("Recall", 0),
                "f1_score": f1_scores.get("F1 Score", 0),
                "compliance_degree": compliance_scores.get("compliance_degree", 0),
                "compliance_maturity": compliance_scores.get("compliance_maturity", 0)
            })

        self._save_results(output_csv)

    def _save_results(self, output_file):
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)

            header = [
                "Iteration", "Alteration Sequence", "Alteration Counts",
                "Node Similarity", "Structural Similarity", "Behavioral Similarity",
                "Precision", "Recall", "F1 Score", "Compliance Degree", "Compliance Maturity"
            ]
            writer.writerow(header)

            for row in self.results:
                writer.writerow([
                    row["iteration"],
                    row["sequence"],
                    row["alteration_counts"],
                    row["node_similarity"],
                    row["structural_similarity"],
                    row["behavioral_similarity"],
                    row["precision"],
                    row["recall"],
                    row["f1_score"],
                    row["compliance_degree"],
                    row["compliance_maturity"]
                ])
        print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    file_path = "task_reference5.bpmn"
    output_csv = "experiment3_results.csv"

    alteration_types = [
        "change_label",
        "add_activity",
        "add_flow",
        "remove_activity",
        "remove_flow",
        "remove_gateway"
    ]

    experiment = Experiment3(alteration_types=alteration_types, iterations=5, sequence_length=8)
    experiment.run(file_path, output_csv)

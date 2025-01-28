import argparse
from io_handler import IOHandler
from process.process import Process
from similarity_metric.similarity_metric import SimilarityMetric

class Experiment2:
    def __init__(self, graph1, graph2):
        self.reference_graph = graph1
        self.altered_graph = graph2
        self.results = []  # Placeholder for storing similarity scores

    
    def calculate_similarity(file1_path, file2_path):
        """
        Calculate similarity between two BPMN models.
        """
        # Load and process the first model
        print(f"Loading and processing the reference model: {file1_path}")
        reference_bpmn_tree = IOHandler.read_bpmn(file1_path)
        reference_model = Process()
        reference_model.from_bpmn(reference_bpmn_tree.getroot())

        # Load and process the second model
        print(f"Loading and processing the altered model: {file2_path}")
        altered_bpmn_tree = IOHandler.read_bpmn(file2_path)
        altered_model = Process()
        altered_model.from_bpmn(altered_bpmn_tree.getroot())

        # Calculate similarity metrics
        metrics = ["NodeStructuralBehavioralMetric", "F1Score", "ComplianceMetric"]  # Define available metrics
        results_summary = []

        for metric_name in metrics:
            metric = SimilarityMetric.get_metric(metric_name, reference_model, altered_model)
            print(f"\nCalculating metric: {metric_name}...")
            results = metric.calculate()
            results_summary.append((metric_name, results))

        # Print all results at the end in a nice format
        print("\nSimilarity Metrics Results:")
        print("===========================")
        for metric_name, results in results_summary:
            print(f"\n{metric_name} Results:")
            print("=" * (len(metric_name) + 9))
            for key, value in results.items():
                print(f"{key:<25}: {value:.4f}" if isinstance(value, float) else f"{key:<25}: {value}")
        return results_summary


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Experiment2: Similarity Calculation for BPMN Models")
    parser.add_argument("file1_path", type=str, help="Path to the first BPMN file (reference model).")
    parser.add_argument("file2_path", type=str, help="Path to the second BPMN file (altered model).")
    args = parser.parse_args()

    # Execute Experiment2
    results = Experiment2.calculate_similarity(args.file1_path, args.file2_path)

import argparse
import time
from io_handler import IOHandler
from process.process import Process
from similarity_metric.similarity_metric import SimilarityMetric


class SimilarityCalculator:
    def __init__(self):
        self.metrics = ["NodeStructuralBehavioralMetric", "F1Score", "ComplianceMetric"]
        self.reference_model = None
        self.comparison_model = None

    def execute(self, reference_path: str, comparison_path: str):
        # Read BPMN models
        print("Reading the reference BPMN model...")
        reference_bpmn_tree = IOHandler.read_bpmn(reference_path)
        
        print("Reading the comparison BPMN model...")
        comparison_bpmn_tree = IOHandler.read_bpmn(comparison_path)

        # Convert BPMN to Process objects
        print("Converting BPMN models to Process objects...")
        self.reference_model = Process()
        self.reference_model.from_bpmn(reference_bpmn_tree.getroot())

        self.comparison_model = Process()
        self.comparison_model.from_bpmn(comparison_bpmn_tree.getroot())

        # Print state
        print("Reference Model:")
        self.reference_model.print_process_state()
        print("Comparison Model:")
        self.comparison_model.print_process_state()

        # Calculate similarity metrics
        results_summary = []
        for metric_name in self.metrics:
            metric = SimilarityMetric.get_metric(metric_name, self.reference_model, self.comparison_model, reference_path)
            print("\n================================================================================")
            print(f"\nCalculating metric: {metric_name}...")
            results = metric.calculate()
            results_summary.append((metric_name, results))

        # Print results summary
        print("\nSimilarity Metrics Results:")
        print("===========================")
        for metric_name, results in results_summary:
            print(f"\n{metric_name} Results:")
            print("=" * (len(metric_name) + 9))
            for key, value in results.items():
                print(f"{key:<25}: {value:.4f}" if isinstance(value, float) else f"{key:<25}: {value}")
        print("")


if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Process Model Similarity Calculator")
    parser.add_argument("reference_model", type=str, help="Path to the reference BPMN file.")
    parser.add_argument("comparison_model", type=str, help="Path to the comparison BPMN file.")
    
    args = parser.parse_args()
    
    # Execute similarity calculation
    start_time = time.time()
    calculator = SimilarityCalculator()
    calculator.execute(args.reference_model, args.comparison_model)
    end_time = time.time()
    
    print(f"Similarity calculation completed in {end_time - start_time:.2f} seconds.")

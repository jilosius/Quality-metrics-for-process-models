import argparse
from io_handler import IOHandler
from process.process import Process
from model_alteration.model_alteration import ModelAlteration
from similarity_metric.similarity_metric import SimilarityMetric

class ToolController:
    
    def __init__(self):
        self.metrics = ["NodeStructuralBehavioralMetric",
                        "F1Score"]  # List of available metrics
        self.reference_model = None
        self.altered_model = None

    def execute(self, file_path: str, alterations: list, output_path: str):
        # Read BPMN model
        print("Reading the BPMN model...")
        bpmn_tree = IOHandler.read_bpmn(file_path)

        # Convert process object
        print("Converting BPMN model to Process object...")
        self.reference_model = Process()
        self.reference_model.from_bpmn(bpmn_tree.getroot())

        self.reference_model.print_process_state()

        # ModelAlteration initialization
        print("Performing alterations...")
        model_alteration = ModelAlteration(self.reference_model)

        # Applying alterations
        for alteration, repetitions in alterations:
            print(f"Applying alteration: {alteration} {repetitions} time(s)...")
            model_alteration.apply_alteration(alteration, repetitions)

        self.altered_model = model_alteration.altered_model

        # Calculating similarity metrics
        for metric_name in self.metrics:
            metric = SimilarityMetric.get_metric(metric_name, self.reference_model, self.altered_model)
            print("\nMetric: ", metric_name)
            results = metric.calculate()

            # Print results in a nicer format
            print(f"\n{metric_name} Results:")
            print("=" * (len(metric_name) + 9))
            for key, value in results.items():
                print(f"{key:<25}: {value:.4f}" if isinstance(value, float) else f"{key:<25}: {value}")
            print("")



if __name__ == "__main__":
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Similarity Tool for BPMN Models")
    parser.add_argument("file_path", type=str, help="Path to the BPMN file to load.")
    parser.add_argument("output_path", type=str, help="Path to save the altered BPMN file.")
    parser.add_argument("-add_activity", type=int, nargs="?", const=1, help="Add an activity to the process model.")
    parser.add_argument("-add_flow", type=int, nargs="?", const=1, help="Add a flow to the process model.")
    parser.add_argument("-add_gateway", type=int, nargs="?", const=1, help="Add a gateway to the process model.")
    parser.add_argument("-remove_activity", type=int, nargs="?", const=1, help="Remove an activity from the process model.")
    parser.add_argument("-remove_flow", type=int, nargs="?", const=1, help="Remove a flow from the process model.")
    parser.add_argument("-remove_gateway", type=int, nargs="?", const=1, help="Remove a gateway from the process model.")
    parser.add_argument("-change_label", type=int, nargs="?", const=1, help="Change the label of a node in the process model.")
    args = parser.parse_args()

    # Collect alterations and their repetitions
    alterations = []
    if args.add_activity:
        alterations.append(("add_activity", args.add_activity))
    if args.add_flow:
        alterations.append(("add_flow", args.add_flow))
    if args.add_gateway:
        alterations.append(("add_gateway", args.add_gateway))
    if args.remove_activity:
        alterations.append(("remove_activity", args.remove_activity))
    if args.remove_flow:
        alterations.append(("remove_flow", args.remove_flow))
    if args.remove_gateway:
        alterations.append(("remove_gateway", args.remove_gateway))
    if args.change_label:
        alterations.append(("change_label", args.change_label))

    # Execute the tool
    tool = ToolController()
    tool.execute(args.file_path, alterations, args.output_path)

import argparse
import time
from io_handler import IOHandler
from process.process import Process
from model_alteration.model_alteration import ModelAlteration
from similarity_metric.similarity_metric import SimilarityMetric


class ToolController:
    
    
    def __init__(self):
        self.metrics = ["NodeStructuralBehavioralMetric", "F1Score","ComplianceMetric"]  # List of available metrics  
        self.reference_model = None
        self.altered_model = None

    def execute(self, file_path: str, alterations: list, output_path: str):
        print("Reading the BPMN model...")
        reference_bpmn_tree = IOHandler.read_bpmn(file_path)

        # load tokenizer/model if change_label is used
        tokenizer = None
        model = None
        if any(alteration == "change_label" for alteration, _ in alterations):
            from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
            tokenizer = AutoTokenizer.from_pretrained("Vamsi/T5_Paraphrase_Paws")
            model = AutoModelForSeq2SeqLM.from_pretrained("Vamsi/T5_Paraphrase_Paws")

        # convert process object
        print("Converting BPMN model to Process object...")
        self.reference_model = Process()
        self.reference_model.from_bpmn(reference_bpmn_tree.getroot())
        self.reference_model.print_process_state()

        # perform alterations
        print("\nPerforming alterations...")
        model_alteration = ModelAlteration(self.reference_model)
        
        if not alterations:
            print("No alterations specified. Cloning reference model as altered model.")
            self.altered_model = self.reference_model.clone()
        else:
            for alteration, param in alterations:
                if alteration == "remove_flowNode":
                    print(f"Removing FlowNode: {param}...")
                    model_alteration.altered_model = model_alteration.apply_alteration(
                        alteration, 1, param
                    )
                elif alteration == "change_label":
                    print(f"Applying alteration: {alteration} {param} time(s)...")
                    model_alteration.altered_model = model_alteration.apply_alteration(
                        alteration, param, tokenizer=tokenizer, model=model
                    )
                else:
                    print(f"Applying alteration: {alteration} {param} time(s)...")
                    model_alteration.altered_model = model_alteration.apply_alteration(
                        alteration, param
                    )

            self.altered_model = model_alteration.altered_model





        results_summary = []
        for metric_name in self.metrics:
            
            metric = SimilarityMetric.get_metric(metric_name, self.reference_model, self.altered_model, file_path, output_path)
            print("\n================================================================================")
            print(f"\nCalculating metric: {metric_name}...")
            try:
                results = metric.calculate()
            except Exception as e:
                print(f"Error calculating {metric_name}: {e}")

            results_summary.append((metric_name, results))


        print("\nSimilarity Metrics Results:")
        print("===========================")
        for metric_name, results in results_summary:
            print(f"\n{metric_name} Results:")
            print("=" * (len(metric_name) + 9))
            for key, value in results.items():
                print(f"{key:<25}: {value:.4f}" if isinstance(value, float) else f"{key:<25}: {value}")
        print("")

    def print_petri_net_details(self, petri_net, initial_marking, final_marking):
        """
        Prints the details of a Petri net in a readable format.

        Args:
            petri_net: The Petri net object.
            initial_marking: The initial marking of the Petri net.
            final_marking: The final marking of the Petri net.
        """
        print("\nPlaces:")
        print("-------")
        for place in petri_net.places:
            print(f"- {place.name}")

        print("\nTransitions:")
        print("------------")
        for transition in petri_net.transitions:
            label = transition.label if transition.label else "None"
            print(f"- {transition.name} (label: {label})")

        print("\nArcs:")
        print("------")
        for arc in petri_net.arcs:
            source = arc.source.name
            target = arc.target.name
            print(f"- {source} -> {target}")

        print("\nInitial Marking:")
        print("-----------------")
        for place, tokens in initial_marking.items():
            print(f"- {place.name}: {tokens} token(s)")

        print("\nFinal Marking:")
        print("---------------")
        for place, tokens in final_marking.items():
            print(f"- {place.name}: {tokens} token(s)")

    def print_graph_and_markings(self, graph):
        """
        Prints the graph structure, initial marking, and final marking in a clear and structured format.
        
        Args:
            graph: NetworkX DiGraph representing the Petri net.
            initial_marking: Initial marking of the Petri net.
            final_marking: Final marking of the Petri net.
        """
        print("Graph Overview:")
        print("----------------")
        print(f"Nodes: {len(graph.nodes)}")
        print(f"Edges: {len(graph.edges)}\n")

        print("Nodes:")
        print("------")
        for node in graph.nodes:
            print(f"- {node}")

        print("\nEdges:")
        print("------")
        for edge in graph.edges:
            print(f"- {edge[0]} -> {edge[1]}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Similarity Tool for BPMN Models")
    parser.add_argument("file_path", type=str, metavar="N", help="Path to the BPMN file to load.")
    parser.add_argument("output_path", type=str, metavar="N", help="Path to save the altered BPMN file.")
    parser.add_argument("-add_activity", type=int, nargs="?", const=1, metavar="N", help="Add N activity nodes.")
    parser.add_argument("-add_flow", type=int, nargs="?", const=1, metavar="N", help="Add N flows.")
    parser.add_argument("-add_gateway", type=int, nargs="?", const=1, metavar="N", help="Add N gateways.")
    parser.add_argument("-remove_activity", type=int, nargs="?", const=1, metavar="N", help="Remove N activity nodes.")
    parser.add_argument("-remove_flow", type=int, nargs="?", const=1, metavar="N", help="Remove N flows.")
    parser.add_argument("-remove_gateway", type=int, nargs="?", const=1, metavar="N", help="Remove N gateways.")
    parser.add_argument("-change_label", type=int, nargs="?", const=1, metavar="N", help="Change the label of N nodes.")
    parser.add_argument("-remove_flowNode", type=str, nargs="+", metavar="ID", help="Remove one or more flowNodes by ID.")


    args = parser.parse_args()

    # collect alterations and their repetitions
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
    if args.remove_flowNode:
        alterations.append(("remove_activity", args.remove_flowNode))

    #execute
    start_time = time.time()
    tool = ToolController()
    tool.execute(args.file_path, alterations, args.output_path)
    end_time = time.time()

    print(f"Script completed in {end_time - start_time:.2f} seconds.")

import argparse
import time
from io_handler import IOHandler
from process.process import Process
from model_alteration.model_alteration import ModelAlteration
from similarity_metric.similarity_metric import SimilarityMetric
from similarity_metric.compliance_metric import ComplianceMetric



class ToolController:
    
    def __init__(self):
        self.metrics = ["NodeStructuralBehavioralMetric", "F1Score"]  # List of available metrics 
        self.reference_model = None
        self.altered_model = None

    def execute(self, file_path: str, alterations: list, output_path: str):
        # Read BPMN model
        print("Reading the BPMN model...")
        reference_bpmn_tree = IOHandler.read_bpmn(file_path)

        # Convert process object
        print("Converting BPMN model to Process object...")
        self.reference_model = Process()
        self.reference_model.from_bpmn(reference_bpmn_tree.getroot())
        self.reference_model.print_process_state()

        # Perform alterations
        print("\nPerforming alterations...")
        model_alteration = ModelAlteration(self.reference_model)

        # Applying alterations
        for alteration, repetitions in alterations:
            print(f"Applying alteration: {alteration} {repetitions} time(s)...")
            model_alteration.apply_alteration(alteration, repetitions)

        self.altered_model = model_alteration.altered_model

        # Generate granularity mapping using process objects
        # print("\nGenerating granularity mapping using process objects...")
        # compliance_metric = ComplianceMetric(self.reference_model, self.altered_model)
        # granularity_mapping = compliance_metric.match_nodes(
        #     self.reference_model.flowNodes,  # Process object nodes
        #     self.altered_model.flowNodes
        # )

        # Formatting the output for better readability
        # formatted_output = "\n".join(f"Reference: {pair[0]}\nAltered: {pair[1]}\n" for pair in granularity_mapping)   
        # print(formatted_output)

        # Convert the reference model to BPMN and save to a temporary file
        # print("\nConverting the reference BPMN model to Petri net...")

        # reference_petri_net, reference_initial_marking, reference_final_marking = compliance_metric.convert_to_petri_net(file_path)

        # compliance_metric.visualize_petri_net(reference_petri_net,reference_initial_marking,reference_final_marking)

        # print("\nReference Model Petri net:")
        # self.print_petri_net_details(reference_petri_net, reference_initial_marking, reference_final_marking)

        # Convert the altered model to BPMN and save to the output path
        # altered_bpmn_tree = self.altered_model.to_bpmn()
        # IOHandler.write_bpmn(altered_bpmn_tree, output_path)

        # altered_petri_net, altered_initial_marking, altered_final_marking = compliance_metric.convert_to_petri_net(output_path)

        # print("\nAltered Model Petri net:")
        # self.print_petri_net_details(altered_petri_net, altered_initial_marking, altered_final_marking)


        # print("Initial Marking:", reference_initial_marking)
        # print("Initial Marking Details:")
        # for place, tokens in reference_initial_marking.items():
        #     print(f"Place: {place}, Tokens: {tokens}")

        # print("Final Marking:", reference_final_marking)


        # firing_seq_ref = compliance_metric.generate_all_firing_sequences(reference_petri_net, reference_initial_marking, reference_final_marking)
        # firing_seq_alt = compliance_metric.generate_all_firing_sequences(altered_petri_net, altered_initial_marking, altered_final_marking)
        
        # print(firing_seq_ref)
        # print("\n------------\n")
        # print(firing_seq_alt)

        # for transition in reference_petri_net.transitions:
        #     print(f"Transition: {transition}")
        #     for arc in transition.in_arcs:
        #         print(f"  Input Arc: {arc.source} -> {arc.target}, Weight: {arc.weight}")
        #     for arc in transition.out_arcs:
        #         print(f"  Output Arc: {arc.source} -> {arc.target}, Weight: {arc.weight}")


        # print("\nMatches:")
        # for ref_transition, alt_transition in matches:
        #     ref_name = ref_transition.label if ref_transition.label else ref_transition.name
        #     alt_name = alt_transition.label if alt_transition.label else alt_transition.name
        #     print(f"Reference Transition: {ref_name} matched with Altered Transition: {alt_name}")


        # Calculating similarity metrics
        results_summary = []
        for metric_name in self.metrics:
            metric = SimilarityMetric.get_metric(metric_name, self.reference_model, self.altered_model)
            print("\n================================================================================")
            print(f"\nCalculating metric: {metric_name}...")
            results = metric.calculate()

            # Collect results for summary printing
            results_summary.append((metric_name, results))

        # Print all results at the end in a nice format
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
    start_time = time.time()
    tool = ToolController()
    tool.execute(args.file_path, alterations, args.output_path)
    end_time = time.time()

    print(f"Script completed in {end_time - start_time:.2f} seconds.")



import uuid
from enum import Enum
from pm4py.objects.petri_net.utils import reduction
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to
from pm4py.util import exec_utils
from similarity_metric.similarity_metric import SimilarityMetric
import pm4py

from pm4py.objects.petri_net.obj import PetriNet, Marking
from typing import List, Tuple






class Parameters(Enum):
    USE_ID = "use_id"



class ComplianceMetric(SimilarityMetric):
    def __init__(self, reference_model=None, altered_model=None, label_similarity_threshold=0.8):
        super().__init__(reference_model, altered_model)
        self.reference_model = reference_model
        self.altered_model = altered_model
        self.label_similarity_threshold = label_similarity_threshold

    def convert_to_petri_net(self, bpmn_path):
        """
        Converts a BPMN file to a Petri net and optionally visualizes it.
        
        Args:
            bpmn_path (str): Path to the BPMN file.
        
        Returns:
            tuple: Petri net, initial marking, and final marking.
        """
        # Load the BPMN model
        bpmn_graph = pm4py.read_bpmn(bpmn_path)
    
        # Convert BPMN to Petri net
        net, im, fm = self.apply(bpmn_graph)

         # Visualize the Petri net
        # try:
        #     from pm4py.visualization.petri_net import visualizer as pn_visualizer

        #     print("Visualizing the Petri net...")
        #     gviz = pn_visualizer.apply(net, im, fm)
        #     pn_visualizer.view(gviz)  # Opens in the default viewer
        # except ImportError:
        #     print("Petri net visualization skipped because the required visualization library is not available.")
        # except Exception as e:
        #     print(f"An error occurred during visualization: {e}")

        return net, im, fm
    
    
    def calculate_similarity(self, label1, label2, type1, type2):
        """Calculate syntactic/semantic similarity between labels."""
        # print(f"Calculating similarity between labels: '{label1}' and '{label2}'")

        type_similarity = self.calculate_type_similarity(type1, type2)
        # print(f"  Type similarity: {type_similarity}")

        if type_similarity == 1:
            syntactic_score = self.calculate_syntactic_similarity(label1, label2)
            semantic_score = self.calculate_semantic_similarity(label1, label2)
            # print(f"  Syntactic similarity: {syntactic_score}\n  Semantic similarity: {semantic_score}")
            
            return max(syntactic_score, semantic_score)
        
        elif type_similarity == 0:
            # print("Flow Nodes not of same type. Moving on..")
            # print("-----------")
            return 0.0

        else:
            return 0.0

    def match_nodes(self, reference_flowNodes, altered_flowNodes):
        """Match nodes from the altered model to the reference model based on label similarity."""
        matches = []
        
        for alt_node in altered_flowNodes:  # Iterate over altered model nodes
            best_match = None
            best_score = 0
            # print(f"\nAltered node: {alt_node.flowNode_id} ({alt_node.type})")
            for ref_node in reference_flowNodes:  # Compare against reference model nodes
                # print(f"  Reference node: {ref_node.flowNode_id} ({ref_node.type})")
                similarity = self.calculate_similarity(alt_node.label, ref_node.label, alt_node.type, ref_node.type)
                # print(f"Label Similarity: {similarity}")
                # print("-----------")
                if similarity > self.label_similarity_threshold and similarity > best_score:
                    best_match = ref_node
                    best_score = similarity
            if best_match:
                # print(f"  Best match for node '{alt_node.label}': '{best_match.label}' with score {best_score}")
                matches.append((alt_node, best_match))
            # else:
                # print(f"  No match with same type & similarity > {self.label_similarity_threshold} found for node '{alt_node.label}'")
        
        return matches


    def is_transition_enabled(self, transition: PetriNet.Transition, marking: Marking) -> bool:
        for arc in transition.in_arcs:
            # print(f"Checking arc from {arc.source} with weight {arc.weight} and tokens {marking[arc.source]}")
            if marking[arc.source] < arc.weight:
                # print("Transition not enabled due to insufficient tokens.")
                return False
        # print(f"Transition {transition} is enabled.")
        return True


    def fire_transition(self, transition: PetriNet.Transition, marking: Marking) -> Marking:
        new_marking = marking.copy()
        # print(f"Firing Transition: {transition}")
        for arc in transition.in_arcs:
            # print(f"Consuming {arc.weight} tokens from {arc.source}")
            new_marking[arc.source] -= arc.weight
        for arc in transition.out_arcs:
            # print(f"Producing {arc.weight} tokens at {arc.target}")
            new_marking[arc.target] += arc.weight
        # print("New Marking After Firing:", new_marking)
        return new_marking


    def generate_all_firing_sequences(self, net: PetriNet, initial_marking: Marking, final_marking: Marking) -> List[List[PetriNet.Transition]]:
        """
        Generates all firing sequences for a Petri net from the initial marking to the final marking.

        Args:
            net (PetriNet): The Petri net to explore.
            initial_marking (Marking): The initial marking of the net.
            final_marking (Marking): The final marking of the net.

        Returns:
            List[List[PetriNet.Transition]]: A list of all possible firing sequences.
        """
        all_sequences = []

        def dfs(marking, current_sequence):
            # Normalize markings for comparison
            normalized_marking = {k: v for k, v in marking.items() if v > 0}
            normalized_final_marking = {k: v for k, v in final_marking.items() if v > 0}

            # Check if the final marking is reached
            if normalized_marking == normalized_final_marking:
                all_sequences.append(current_sequence[:])
                return

            # Get all enabled transitions
            enabled_transitions = [t for t in net.transitions if self.is_transition_enabled(t, marking)]

            for transition in enabled_transitions:
                # Fire the transition
                new_marking = self.fire_transition(transition, marking)
                # Add the transition to the current sequence
                current_sequence.append(transition)
                # Continue searching
                dfs(new_marking, current_sequence)
                # Backtrack
                current_sequence.pop()


        dfs(initial_marking, [])
        return all_sequences





    def compute_metric(self, process_object1, process_object2):
            """
            Computes the compliance metric for two given process objects.
            
            :param process_object1: The reference process object.
            :param process_object2: The target process object to evaluate.
            :return: Compliance metric value.
            """
            # Convert both process objects to WF-nets
            wf_net1 = self.convert_to_wf_net(process_object1)
            wf_net2 = self.convert_to_wf_net(process_object2)

            # Extract firing sequences for both WF-nets
            firing_sequences1 = self.extract_firing_sequences(wf_net1)
            firing_sequences2 = self.extract_firing_sequences(wf_net2)

            # Placeholder: Perform compliance calculation (to be implemented)
            # For now, return the number of firing sequences in both models as a placeholder value
            return {
                "firing_sequences_model1": len(firing_sequences1),
                "firing_sequences_model2": len(firing_sequences2)
            }


    def calculate(self):
        """
        Placeholder implementation for the abstract calculate method.
        """
        return {"placeholder_result": 0}
    

    #from python docs
    def build_digraph_from_petri_net(self,net):
        """
        Builds a directed graph from a Petri net
            (for the purpose to add invisibles between inclusive gateways)

        Parameters
        -------------
        net
            Petri net

        Returns
        -------------
        digraph
            Digraph
        """
        import networkx as nx
        graph = nx.DiGraph()
        for place in net.places:
            graph.add_node(place.name)
        for trans in net.transitions:
            in_places = [x.source for x in list(trans.in_arcs)]
            out_places = [x.target for x in list(trans.out_arcs)]
            for pl1 in in_places:
                for pl2 in out_places:
                    graph.add_edge(pl1.name, pl2.name)
        return graph

    def apply(self, bpmn_graph, parameters=None):
        """
        Converts a BPMN graph to an accepting Petri net, excluding Pools.

        Args:
            bpmn_graph: BPMN graph.
            parameters: Parameters of the algorithm.

        Returns:
            tuple: Petri net, initial marking, and final marking.
        """
        if parameters is None:
            parameters = {}

        import networkx as nx
        from pm4py.objects.bpmn.obj import BPMN

        use_id = exec_utils.get_param_value(Parameters.USE_ID, parameters, False)

        net = PetriNet("")
        source_place = PetriNet.Place("source")
        net.places.add(source_place)
        sink_place = PetriNet.Place("sink")
        net.places.add(sink_place)
        im = Marking()
        fm = Marking()
        im[source_place] = 1
        fm[sink_place] = 1

        non_pool_nodes = [
            node for node in bpmn_graph.get_nodes()
            if not isinstance(node, (BPMN.Collaboration, BPMN.Participant))  # Exclude pools and collaborations
            and isinstance(node, (BPMN.Task, BPMN.Event, BPMN.ParallelGateway, BPMN.InclusiveGateway, BPMN.ExclusiveGateway, BPMN.IntermediateCatchEvent, BPMN.IntermediateThrowEvent))  # Explicit gateway types
        ]

        # Filter flows to exclude any connected to Pools
        non_pool_flows = [
            flow for flow in bpmn_graph.get_flows()
            if flow.get_source() in non_pool_nodes and flow.get_target() in non_pool_nodes
        ]


        # Map flow places and counts
        flow_place = {}
        source_count = {}
        target_count = {}
        for flow in non_pool_flows:
            source = flow.get_source()
            target = flow.get_target()
            place = PetriNet.Place(str(flow.get_id()))
            net.places.add(place)
            flow_place[flow] = place
            if source not in source_count:
                source_count[source] = 0
            if target not in target_count:
                target_count[target] = 0
            source_count[source] += 1
            target_count[target] += 1

        # Process non-pool nodes
        nodes_entering = {}
        nodes_exiting = {}
        for node in non_pool_nodes:
            entry_place = PetriNet.Place("ent_" + str(node.get_id()))
            net.places.add(entry_place)
            exiting_place = PetriNet.Place("exi_" + str(node.get_id()))
            net.places.add(exiting_place)
            if use_id:
                label = str(node.get_id())
            else:
                label = str(node.get_name()) if isinstance(node, BPMN.Task) else None
                if not label:
                    label = None
            transition = PetriNet.Transition(name=str(node.get_id()), label=label)
            net.transitions.add(transition)
            add_arc_from_to(entry_place, transition, net)
            add_arc_from_to(transition, exiting_place, net)

            # Handle gateways
            if isinstance(node, BPMN.ParallelGateway) or isinstance(node, BPMN.InclusiveGateway):
                if source_count[node] > 1:
                    exiting_object = PetriNet.Transition(str(uuid.uuid4()), None)
                    net.transitions.add(exiting_object)
                    add_arc_from_to(exiting_place, exiting_object, net)
                else:
                    exiting_object = exiting_place

                if target_count[node] > 1:
                    entering_object = PetriNet.Transition(str(uuid.uuid4()), None)
                    net.transitions.add(entering_object)
                    add_arc_from_to(entering_object, entry_place, net)
                else:
                    entering_object = entry_place
                nodes_entering[node] = entering_object
                nodes_exiting[node] = exiting_object
            else:
                nodes_entering[node] = entry_place
                nodes_exiting[node] = exiting_place




            # Handle Start and End events
            if isinstance(node, BPMN.StartEvent):
                start_transition = PetriNet.Transition(node.get_id(), node.get_name())
                net.transitions.add(start_transition)
                add_arc_from_to(source_place, start_transition, net)
                add_arc_from_to(start_transition, entry_place, net)
            elif isinstance(node, BPMN.EndEvent):
                end_transition = PetriNet.Transition(node.get_id(), node.get_name())
                net.transitions.add(end_transition)
                add_arc_from_to(exiting_place, end_transition, net)
                add_arc_from_to(end_transition, sink_place, net)

            # if isinstance(node, BPMN.ExclusiveGateway):
            #     # Exclusive Split
            #     exiting_object = exiting_place  # Use the existing place representing the gateway

            #     for flow in [f for f in non_pool_flows if f.get_source() == node]:
            #         # Add an intermediate transition for each outgoing flow
            #         intermediate_transition = PetriNet.Transition(f"trans_{str(uuid.uuid4())}", None)
            #         net.transitions.add(intermediate_transition)

            #         # Connect the gateway's existing place to the intermediate transition
            #         add_arc_from_to(exiting_object, intermediate_transition, net)

            #         # Connect the intermediate transition to the flow's place
            #         flow_place_obj = flow_place[flow]
            #         add_arc_from_to(intermediate_transition, flow_place_obj, net)


                
            #     # Exclusive Join
            #     entering_object = entry_place  # Use the existing place representing the gateway

            #     for flow in [f for f in non_pool_flows if f.get_target() == node]:
            #         # Add an intermediate transition for each incoming flow
            #         intermediate_transition = PetriNet.Transition(f"trans_{str(uuid.uuid4())}", None)
            #         net.transitions.add(intermediate_transition)

            #         # Connect the flow's place to the intermediate transition
            #         flow_place_obj = flow_place[flow]
            #         add_arc_from_to(flow_place_obj, intermediate_transition, net)

            #         # Connect the intermediate transition to the gateway's existing place
            #         add_arc_from_to(intermediate_transition, entering_object, net)




            

            

        # Process non-pool flows
        for flow in non_pool_flows:
            source_object = nodes_exiting[flow.get_source()]
            target_object = nodes_entering[flow.get_target()]

            if isinstance(source_object, PetriNet.Place):
                inv1 = PetriNet.Transition(str(uuid.uuid4()), None)
                net.transitions.add(inv1)
                add_arc_from_to(source_object, inv1, net)
                source_object = inv1

            if isinstance(target_object, PetriNet.Place):
                inv2 = PetriNet.Transition(str(uuid.uuid4()), None)
                net.transitions.add(inv2)
                add_arc_from_to(inv2, target_object, net)
                target_object = inv2

            add_arc_from_to(source_object, flow_place[flow], net)
            add_arc_from_to(flow_place[flow], target_object, net)

        # Apply reduction
        reduction.apply_simple_reduction(net)

        return net, im, fm

    # def visualize_petri_net(self,net, im, fm):
    #     """
    #     Visualizes the Petri net with its initial and final markings.

    #     Args:
    #         net: The Petri net object.
    #         im: Initial marking.
    #         fm: Final marking.
    #     """
    #     # Visualize the Petri net
    #     try:
    #         from pm4py.visualization.petri_net import visualizer as pn_visualizer

    #         print("Visualizing the Petri net...")
    #         gviz = pn_visualizer.apply(net, im, fm)
    #         pn_visualizer.view(gviz)  # Opens in the default viewer
    #     except ImportError:
    #         print("Petri net visualization skipped because the required visualization library is not available.")
    #     except Exception as e:
    #         print(f"An error occurred during visualization: {e}")

    #     return net, im, fm


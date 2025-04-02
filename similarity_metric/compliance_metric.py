import uuid
from enum import Enum
from pm4py.objects.petri_net.utils import reduction
from pm4py.objects.petri_net.obj import PetriNet, Marking
from pm4py.objects.petri_net.utils.petri_utils import add_arc_from_to
from pm4py.util import exec_utils
from similarity_metric.similarity_metric import SimilarityMetric
import pm4py
from io_handler import IOHandler
import numpy as np
from pm4py.objects.petri_net.obj import PetriNet, Marking
from typing import List
from tabulate import tabulate

class Parameters(Enum):
    USE_ID = "use_id"


class ComplianceMetric(SimilarityMetric):
    def __init__(self, reference_model=None, altered_model=None, file_path=None, output_path=None, label_similarity_threshold=0.2):
        super().__init__(reference_model, altered_model)
        self.reference_model = reference_model
        self.altered_model = altered_model
        self.label_similarity_threshold = label_similarity_threshold
        self.file_path = file_path
        self.output_path = output_path if output_path else "output_test.bpmn"

    def convert_to_petri_net(self, bpmn_path):    
        
        bpmn_graph = pm4py.read_bpmn(bpmn_path)
        # pm4py.view_bpmn(bpmn_graph)
    
        # Convert BPMN to Petri net
        net, im, fm = self.apply(bpmn_graph)

        # # Visualize the Petri net
        # try:
        #     from pm4py.visualization.petri_net import visualizer as pn_visualizer
        #     print("Visualizing the Petri net...")
        #     gviz = pn_visualizer.apply(net, im, fm)
        #     pn_visualizer.view(gviz)  # Opens in the default viewer
        # except ImportError:
        #     print("Petri net visualization skipped (missing pm4py visualization libraries).")
        # except Exception as e:
        #     print(f"An error occurred during visualization: {e}")

        return net, im, fm

    def calculate_similarity(self, label1, label2, type1, type2):
        type_similarity = self.calculate_type_similarity(type1, type2)

        if type_similarity == 1:
            syntactic_score = self.calculate_syntactic_similarity(label1, label2)
            semantic_score = self.calculate_semantic_similarity(label1, label2)
            return max(syntactic_score, semantic_score)
        elif type_similarity == 0:
            return 0.0
        else:
            return 0.0

    def match_nodes(self, reference_flowNodes, altered_flowNodes):
        matches = {}
        for alt_node in altered_flowNodes:
            matched_refs = []
            for ref_node in reference_flowNodes:
                similarity = self.calculate_similarity(
                    alt_node.label,
                    ref_node.label,
                    alt_node.type,
                    ref_node.type
                )
                if similarity > self.label_similarity_threshold:
                    matched_refs.append(ref_node)

            if matched_refs:
                matches[alt_node] = matched_refs

        return matches
    
    def is_transition_enabled(self, transition: PetriNet.Transition, marking: Marking) -> bool:
        for arc in transition.in_arcs:
            if marking[arc.source] < arc.weight:
                return False
        return True

    def fire_transition(self, transition: PetriNet.Transition, marking: Marking) -> Marking:
        new_marking = marking.copy()
        for arc in transition.in_arcs:
            new_marking[arc.source] -= arc.weight
        for arc in transition.out_arcs:
            new_marking[arc.target] += arc.weight
        return new_marking

    def is_petri_net_sound(self, net: PetriNet, initial_marking: Marking, final_marking: Marking) -> bool:
        visited = set()
        queue = [initial_marking]

        def normalize_marking(marking):
            return tuple(sorted((place, tokens) for place, tokens in marking.items() if tokens > 0))

        normalized_final_marking = normalize_marking(final_marking)

       

        while queue:
            current_marking = queue.pop(0)
            normalized_marking = normalize_marking(current_marking)
            
            if normalized_marking == normalized_final_marking:
                for place, tokens in current_marking.items():
                    if place not in final_marking and tokens > 0:
                        return False
                continue

            if normalized_marking in visited:
                # print(f"Marking already visited: {normalized_marking}")
                continue
            visited.add(normalized_marking)

            enabled_transitions = [
                t for t in net.transitions if self.is_transition_enabled(t, current_marking)
            ]

            # If no transitions are enabled and it's not the final marking, it's a deadlock
            if not enabled_transitions:
                # print(f"Deadlock detected at marking: {normalized_marking}")
                return False

            # Fire transitions and add new markings to the queue
            for transition in enabled_transitions:
                new_marking = self.fire_transition(transition, current_marking)
                # print(f"Firing transition {transition.label}: New marking -> {normalize_marking(new_marking)}")
                queue.append(new_marking)

        print("Petri net is sound.")
        return True

    def generate_all_firing_sequences(self, net: PetriNet, initial_marking: Marking, final_marking: Marking, unroll_factor=2) -> List[List[PetriNet.Transition]]:

            all_sequences = []


            start_transitions = [t for t in net.transitions if t.label and (t.label.lower() == "start" or "startevent" in t.label.lower())]
            end_transitions = [t for t in net.transitions if t.label and (t.label.lower() == "end" or "endevent" in t.label.lower())]

            def normalize_marking(marking):
                return tuple(sorted((getattr(place, 'name', str(place)), tokens) for place, tokens in marking.items() if tokens > 0))

            def dfs(marking, current_sequence, visit_count, start_visit_count, end_visit_count):
                normalized_marking = normalize_marking(marking)
                normalized_final_marking = normalize_marking(final_marking)

                # Stop recursion if we reached the final marking
                if normalized_marking == normalized_final_marking:
                    all_sequences.append(current_sequence[:])
                    return

                # Check visit count for this marking
                if visit_count.get(normalized_marking, 0) >= unroll_factor:
                    return
                visit_count[normalized_marking] = visit_count.get(normalized_marking, 0) + 1

                # Count visits to start and end transitions
                last_transition = current_sequence[-1] if current_sequence else None
                if last_transition in start_transitions:
                    start_visit_count += 1
                if last_transition in end_transitions:
                    end_visit_count += 1

                # Exclude sequences that revisit start or end transitions
                if start_visit_count > 1 or end_visit_count > 1:
                    return

                # Get all enabled transitions
                enabled_transitions = [
                    t for t in net.transitions if self.is_transition_enabled(t, marking)
                ]

                for transition in enabled_transitions:
                    # Fire the transition to get the new marking
                    new_marking = self.fire_transition(transition, marking)
                    # Add the transition to the current sequence and recurse
                    current_sequence.append(transition)
                    dfs(new_marking, current_sequence, visit_count.copy(), start_visit_count, end_visit_count)
                    # Backtrack
                    current_sequence.pop()

            dfs(initial_marking, [], {}, 0, 0)
            return all_sequences

    def calculate_lcs(self, seq1, seq2):
        seq1_labels = [item.label for item in seq1]
        seq2_labels = [item.label for item in seq2]

        # Initialize DP table
        dp = [[0] * (len(seq1_labels) + 1) for _ in range(len(seq2_labels) + 1)]

        # Fill DP table
        for i in range(1, len(seq2_labels) + 1):
            for j in range(1, len(seq1_labels) + 1):
                if seq1_labels[j - 1] == seq2_labels[i - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        # Return the LCS length
        return dp[-1][-1]

    def get_mapped_firing_sequences(self, alt_sequences, granularity_mapping):
        mapped_sequences = []  
        
        for seq in alt_sequences:  
            mapped_seq = []  
            
            for transition in seq:  
                if transition in granularity_mapping:  
                    mapped_seq.append(granularity_mapping[transition])  
                else:
                    mapped_seq.append(transition)  
            
            mapped_sequences.append(mapped_seq)  
        
        return mapped_sequences  

    def get_extended_firing_sequences(self, ref_sequences):
        return ref_sequences

    def calculate(self):

        granularity_mapping = self.match_nodes(self.reference_model.flowNodes, self.altered_model.flowNodes)
        
       

        net_ref, im_ref, fm_ref = self.convert_to_petri_net(self.file_path)
        self.print_petri_net_details(net_ref, im_ref, fm_ref)

        altered_bpmn_tree = self.altered_model.to_bpmn()
        IOHandler.write_bpmn(altered_bpmn_tree, self.output_path)

        net_alt, im_alt, fm_alt = self.convert_to_petri_net(self.output_path)
        self.print_petri_net_details(net_alt, im_alt, fm_alt)



        print("\nGranularity Mapping:\n")
        # print(granularity_mapping)
        for alt_node, ref_nodes in granularity_mapping.items():
            alt_id = alt_node.flowNode_id
            ref_ids = [rn.flowNode_id for rn in ref_nodes]

            print(f"{alt_id} : {ref_ids}")



        # Generate firing sequences
        ref_sequences = self.generate_all_firing_sequences(net_ref, im_ref, fm_ref)
        print("\nReference Model Firing Sequences:")
        for seq in ref_sequences:
            print(", ".join([t.label if t.label is not None else "InvisibleTransition" for t in seq]))
        print("\n------------\n")

        alt_sequences = self.generate_all_firing_sequences(net_alt, im_alt, fm_alt)

        print("Altered Model Firing Sequences:")
        for seq in alt_sequences:
            print(", ".join([t.label if t.label is not None else "InvisibleTransition" for t in seq]))
        print("\n------------\n")

        

        # Compute extended and mapped firing sequences
        ext_ref_sequences = self.get_extended_firing_sequences(ref_sequences)
        print("\nExtended Firing Sequences:")
        for seq in ext_ref_sequences:
            print(", ".join([t.label if t.label is not None else "None" for t in seq]))

        
        
        map_alt_sequences = self.get_mapped_firing_sequences(alt_sequences, granularity_mapping)
        print("\nMapped Altered Firing Sequences:")
        for seq in map_alt_sequences:
            print(", ".join([t.label if t.label is not None else "None" for t in seq]))

        fsc_values = []
        for alt_seq in map_alt_sequences:
            max_fsc = 0
            for ref_seq in ext_ref_sequences:
                max_fsc = max(max_fsc, self.calculate_lcs(ref_seq, alt_seq))
            fsc_values.append(max_fsc)

        fscd_values = [] 
        for fsc, seq in zip(fsc_values, map_alt_sequences):  
            if len(seq) > 0:  
                fscd = fsc / len(seq)
            else:
                fscd = 0  
            fscd_values.append(fscd) 


        fscm_values = [] 
        for fsc, seq in zip(fsc_values, ext_ref_sequences):  
            if len(seq) > 0: 
                fscm = fsc / len(seq)  # Calculate FSC Maturity (FSC divided by the sequence length)
            else:
                fscm = 0 
            fscm_values.append(fscm)  # Append the calculated FSCM value to the list

        print("\nFSC Values:")
        print(fsc_values)

        print("\nFSCD Values:")
        print(fscd_values)

        print("\nFSCM Values:")
        print(fscm_values)


        compliance_degree = np.mean(fscd_values) if fscd_values else 0
        compliance_maturity = np.mean(fscm_values) if fscm_values else 0
        
        print("\nCompliance_Degree:")
        print(compliance_degree)

        print("\nCompliance_Maturity:")
        print(compliance_maturity)


        return {
            "compliance_degree": compliance_degree,
            "compliance_maturity": compliance_maturity,
        }

    def build_digraph_from_petri_net(self, net):
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
            bpmn_graph: The BPMN graph (pm4py.objects.bpmn.obj.BPMN).
            parameters: Optional parameters dict.

        Returns:
            tuple: (PetriNet, initial_marking, final_marking)
        """
        if parameters is None:
            parameters = {}

        import networkx as nx
        from pm4py.objects.bpmn.obj import BPMN

        use_id = exec_utils.get_param_value(Parameters.USE_ID, parameters, False)

        # --------------------------------------------------------------------------
        # 1. Create a new Petri net and add a 'source' and 'sink' place
        # --------------------------------------------------------------------------
        net = PetriNet("")
        source_place = PetriNet.Place("source")
        net.places.add(source_place)
        sink_place = PetriNet.Place("sink")
        net.places.add(sink_place)
        im = Marking()
        fm = Marking()
        im[source_place] = 1
        fm[sink_place] = 1

        # --------------------------------------------------------------------------
        # 2. Identify "non-pool" BPMN nodes (Tasks, Events, Gateways, etc.).
        #    Collect both the objects and their IDs.
        # --------------------------------------------------------------------------
        valid_types = (
            BPMN.Task,
            BPMN.Event,
            BPMN.ParallelGateway,
            BPMN.InclusiveGateway,
            BPMN.ExclusiveGateway,
            BPMN.IntermediateCatchEvent,
            BPMN.IntermediateThrowEvent,
            BPMN.NormalIntermediateThrowEvent,
            BPMN.MessageIntermediateCatchEvent,
            BPMN.MessageIntermediateThrowEvent
        )

        non_pool_nodes = [
            node for node in bpmn_graph.get_nodes()
            if not isinstance(node, (BPMN.Collaboration, BPMN.Participant)) 
            and isinstance(node, valid_types)
        ]
        non_pool_node_ids = {node.get_id() for node in non_pool_nodes}

        # --------------------------------------------------------------------------
        # 3. Filter flows based on ID membership. We check flow.source_id and
        #    flow.target_id (or flow.get_source().get_id() / get_target().get_id()).
        # --------------------------------------------------------------------------
        non_pool_flows = []
        for flow in bpmn_graph.get_flows():
            src_id = flow.get_source().get_id()
            tgt_id = flow.get_target().get_id()
            if src_id in non_pool_node_ids and tgt_id in non_pool_node_ids:
                non_pool_flows.append(flow)

        # print(f"\n Non-pool flows: {non_pool_flows}\n")

        # --------------------------------------------------------------------------
        # 4. Create a Place object for each flow (flow_place),
        #    track how many incoming/outgoing flows each node has.
        # --------------------------------------------------------------------------
        flow_place = {}
        source_count = {}
        target_count = {}
        for flow in non_pool_flows:
            source_node = flow.get_source()
            target_node = flow.get_target()
            place = PetriNet.Place(str(flow.get_id()))
            net.places.add(place)
            flow_place[flow] = place

            source_count[source_node] = source_count.get(source_node, 0) + 1
            target_count[target_node] = target_count.get(target_node, 0) + 1

        # --------------------------------------------------------------------------
        # 5. Create Petri places/transitions for each non-pool node.
        #    - We separate "entry" (ent_...) and "exit" (exi_...) places for each node
        #    - Then add a single transition in between them
        # --------------------------------------------------------------------------
        nodes_entering = {}
        nodes_exiting = {}

        for node in non_pool_nodes:
            # print(f"Processing node: {node.get_id()}, type: {type(node)}, name: {node.get_name()}")

            entry_place = PetriNet.Place("ent_" + str(node.get_id()))
            net.places.add(entry_place)
            # print(f"\n{entry_place}")

            exiting_place = PetriNet.Place("exi_" + str(node.get_id()))
            net.places.add(exiting_place)

            # Use either ID or name as the transition label
            if use_id:
                label = str(node.get_id())
            else:
                # If this is a task, use node.get_name(); else None
                label = None
                if isinstance(node, 
                              (BPMN.Task,BPMN.IntermediateCatchEvent,
                               BPMN.IntermediateThrowEvent,
                               BPMN.NormalIntermediateThrowEvent,
                               BPMN.MessageIntermediateCatchEvent,
                               BPMN.MessageIntermediateThrowEvent)):
                    label = node.get_name() if node.get_name() else ""

            # The "main" transition that represents this BPMN node
            transition = PetriNet.Transition(
                name=str(node.get_id()), 
                label=label
            )
            net.transitions.add(transition)

            add_arc_from_to(entry_place, transition, net)
            add_arc_from_to(transition, exiting_place, net)

            # Handle the special Parallel/Inclusive gateway logic
            if isinstance(node, (BPMN.ParallelGateway, BPMN.InclusiveGateway)):
                # If more than one incoming flow, we need an extra "join" transition
                if source_count.get(node, 0) > 1:
                    join_trans = PetriNet.Transition(str(uuid.uuid4()), None)
                    net.transitions.add(join_trans)
                    add_arc_from_to(exiting_place, join_trans, net)
                    exiting_object = join_trans
                else:
                    exiting_object = exiting_place

                # If more than one outgoing flow, we need an extra "split" transition
                if target_count.get(node, 0) > 1:
                    split_trans = PetriNet.Transition(str(uuid.uuid4()), None)
                    net.transitions.add(split_trans)
                    add_arc_from_to(split_trans, entry_place, net)
                    entering_object = split_trans
                else:
                    entering_object = entry_place
                nodes_entering[node] = entering_object
                nodes_exiting[node] = exiting_object

            else:
                nodes_entering[node] = entry_place
                nodes_exiting[node] = exiting_place

            # Handle start events
            if isinstance(node, BPMN.StartEvent):
                start_trans = PetriNet.Transition(node.get_id(), node.get_name())
                net.transitions.add(start_trans)
                add_arc_from_to(source_place, start_trans, net)
                add_arc_from_to(start_trans, entry_place, net)

            # Handle end events
            elif isinstance(node, BPMN.EndEvent):
                end_trans = PetriNet.Transition(node.get_id(), node.get_name())
                net.transitions.add(end_trans)
                add_arc_from_to(exiting_place, end_trans, net)
                add_arc_from_to(end_trans, sink_place, net)

        # --------------------------------------------------------------------------
        # 6. Create arcs for the flows themselves:
        #    - link the node's "exiting" object to this flow's place,
        #      and the flow's place to the node's "entering" object
        # --------------------------------------------------------------------------
        for flow in non_pool_flows:
            source_node = flow.get_source()
            target_node = flow.get_target()

            source_object = nodes_exiting[source_node]
            target_object = nodes_entering[target_node]
            
            # If either source_object or target_object is a Place, 
            # we might insert an invisible transition to keep correct semantics.
            if isinstance(source_object, PetriNet.Place):
                inv_trans = PetriNet.Transition(str(uuid.uuid4()), None)
                net.transitions.add(inv_trans)
                add_arc_from_to(source_object, inv_trans, net)
                source_object = inv_trans

            if isinstance(target_object, PetriNet.Place):
                inv_trans = PetriNet.Transition(str(uuid.uuid4()), None)
                net.transitions.add(inv_trans)
                add_arc_from_to(inv_trans, target_object, net)
                target_object = inv_trans

            # Finally connect them via the flow place
            add_arc_from_to(source_object, flow_place[flow], net)
            add_arc_from_to(flow_place[flow], target_object, net)

        # --------------------------------------------------------------------------
        # 7. Optionally apply a basic Petri net reduction to simplify
        #    (comment out if the net becomes too "disconnected")
        # --------------------------------------------------------------------------
        reduction.apply_simple_reduction(net)

        return net, im, fm

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

    def print_annotated_dp_table(self, dp, seq1, seq2):
        """
        Prints the DP table with annotated headers for better debugging and visualization using tabulate.

        Args:
            dp (list of list): The DP table as a 2D list or numpy array.
            seq1 (list): Sequence 1 (e.g., reference sequence).
            seq2 (list): Sequence 2 (e.g., altered sequence).
        """
        # Add a placeholder for the empty sequence (∅)
        seq1 = ["∅"] + [str(elem) if elem is not None else "None" for elem in seq1]
        seq2 = ["∅"] + [str(elem) if elem is not None else "None" for elem in seq2]

        # Prepare the table data
        table_data = []
        for i, row in enumerate(dp):
            table_data.append([seq2[i]] + row)  # Add the row header and row data

        # Print the table using tabulate
        print(tabulate(table_data, headers=seq1, tablefmt="grid"))



    # def visualize_petri_net(self, net, im, fm):
    #     """
    #     Visualizes the Petri net with its initial and final markings.
    #     """
    #     try:
    #         from pm4py.visualization.petri_net import visualizer as pn_visualizer
    #         print("Visualizing the Petri net...")
    #         gviz = pn_visualizer.apply(net, im, fm)
    #         pn_visualizer.view(gviz)
    #     except ImportError:
    #         print("Petri net visualization skipped (missing pm4py visualization libraries).")
    #     except Exception as e:
    #         print(f"An error occurred during visualization: {e}")

    #     return net, im, fm



        # def compute_metric(self, process_object1, process_object2):
    #     wf_net1 = self.convert_to_wf_net(process_object1)
    #     wf_net2 = self.convert_to_wf_net(process_object2)

    #     firing_sequences1 = self.extract_firing_sequences(wf_net1)
    #     firing_sequences2 = self.extract_firing_sequences(wf_net2)

    #     return {
    #         "firing_sequences_model1": len(firing_sequences1),
    #         "firing_sequences_model2": len(firing_sequences2)
    #     }


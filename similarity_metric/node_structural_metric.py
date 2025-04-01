from Levenshtein import ratio
from similarity_metric.similarity_metric import SimilarityMetric
from process.process import Process
from difflib import SequenceMatcher
import spacy
import numpy as np
import networkx as nx
from tabulate import tabulate
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances

def get_all_paths(graph, source, target):

    try:
        return list(nx.all_simple_paths(graph, source, target))
    except nx.NetworkXNoPath:
        return []

class NodeStructuralBehavioralMetric(SimilarityMetric):

    

    def __init__(self, reference_model: Process, altered_model: Process):
        super().__init__(reference_model, altered_model)
        self.reference_graph = self.process_to_graph(self.reference_model)
        self.altered_graph = self.process_to_graph(self.altered_model)

        print("\nGraphs initialized. Reference and Altered models converted to NetworkX graphs.")
        self.print_graph_table(self.reference_graph, "Reference Graph")
        self.print_graph_table(self.altered_graph, "Altered Graph")

    def calculate(self):
        node_similarity_score = self.node_matching_similarity()
        structural_similarity_score = self.calculate_structural_similarity()
        behavioral_similarity_score = self.calculate_behavioral_similarity()

        return {
            "node_similarity": node_similarity_score,
            "structural_similarity": structural_similarity_score,
            "behavioral_similarity": behavioral_similarity_score,
        }

    def process_to_graph(self, process: Process) -> nx.DiGraph:
        diagram_graph = nx.DiGraph()
        for node in process.flowNodes:
            diagram_graph.add_node(
                node.flowNode_id,
                label=node.label,
                type=node.type,
                lane=node.lane_id,
            )
        for flow in process.flows:
            diagram_graph.add_edge(
                flow.source.flowNode_id,
                flow.target.flowNode_id,
                label=flow.label,
            )
        return diagram_graph

    def calculate_optimal_equivalence_mapping(self, neighbors1, neighbors2, graph1, graph2):
        matched_pairs = []
        for neighbor1 in neighbors1:
            print(f"\nProcessing Neighbor1: {neighbor1}, Label: {graph1.nodes[neighbor1].get('label', '')}, Type: {graph1.nodes[neighbor1].get('type', '')}")
            best_match = None
            best_similarity = 0
            for neighbor2 in neighbors2:
                print(f"  Comparing with Neighbor2: {neighbor2}, Label: {graph2.nodes[neighbor2].get('label', '')}, Type: {graph2.nodes[neighbor2].get('type', '')}")

                syntactic_similarity = self.calculate_syntactic_similarity(
                    graph1.nodes[neighbor1].get("label", ""),
                    graph2.nodes[neighbor2].get("label", "")
                )
                print(f"    Syntactic Similarity: {syntactic_similarity}")

                semantic_similarity = self.calculate_semantic_similarity(
                    graph1.nodes[neighbor1].get("label", ""),
                    graph2.nodes[neighbor2].get("label", "")
                )
                print(f"    Semantic Similarity: {semantic_similarity}")

                label_similarity = max(syntactic_similarity, semantic_similarity)
                print(f"    Label Similarity (Max of Syntactic and Semantic): {label_similarity}")

                type_similarity = self.calculate_type_similarity(
                    graph1.nodes[neighbor1].get("type", ""),
                    graph2.nodes[neighbor2].get("type", "")
                )
                print(f"    Type Similarity: {type_similarity}")

                similarity = (label_similarity + type_similarity) / 2
                print(f"    Combined Similarity: {similarity}")

                if similarity > best_similarity:
                    print(f"      New Best Match Found: {neighbor2} with Similarity: {similarity}")
                    best_similarity = similarity
                    best_match = neighbor2

            if best_match is not None and best_similarity > 0:
                print(f"  Best Match for {neighbor1}: {best_match} with Similarity: {best_similarity}")
                matched_pairs.append((neighbor1, best_match))
                neighbors2.remove(best_match)
                print(f"  Neighbor2 {best_match} removed from further consideration.")

        print(f"\nFinal Matched Pairs: {matched_pairs}")
        return matched_pairs

    def calculate_contextual_equivalence_mapping(self, neighbors1, neighbors2, graph1, graph2):
        matched_pairs = []
        for neighbor1 in neighbors1:
            best_match = None
            best_similarity = 0
            for neighbor2 in neighbors2:

                syntactic_similarity = self.calculate_syntactic_similarity(
                    graph1.nodes[neighbor1].get("label", ""),
                    graph2.nodes[neighbor2].get("label", "")
                )

                semantic_similarity = self.calculate_semantic_similarity(
                    graph1.nodes[neighbor1].get("label", ""),
                    graph2.nodes[neighbor2].get("label", "")
                )

                label_similarity = max(syntactic_similarity, semantic_similarity)

                type_similarity = self.calculate_type_similarity(
                    graph1.nodes[neighbor1].get("type", ""),
                    graph2.nodes[neighbor2].get("type", "")
                )

                similarity = (label_similarity + type_similarity) / 2

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = neighbor2

            if best_match is not None and best_similarity > 0:
                matched_pairs.append((neighbor1, best_match))
                neighbors2.remove(best_match)

        return matched_pairs
    
    def calculate_contextual_similarity(self, node1, graph1, node2, graph2):
        incoming_neighbors1 = set(nx.ancestors(graph1, node1))  
        outgoing_neighbors1 = set(nx.descendants(graph1, node1))  

        incoming_neighbors2 = set(nx.ancestors(graph2, node2)) 
        outgoing_neighbors2 = set(nx.descendants(graph2, node2))  


        # Debugging: Print neighbors
        # print(f"\nNode1: {node1}")
        # print(f"Incoming Neighbors of Node1: {incoming_neighbors1}")
        # print(f"Outgoing Neighbors of Node1: {outgoing_neighbors1}")
        # print("------")
        # print(f"Node2: {node2}")
        # print(f"Incoming Neighbors of Node2: {incoming_neighbors2}")
        # print(f"Outgoing Neighbors of Node2: {outgoing_neighbors2}\n")


        # Calculate optimal equivalence mappings for input and output contexts
        optimal_in_mapping = self.calculate_contextual_equivalence_mapping(incoming_neighbors1, incoming_neighbors2.copy(), graph1, graph2)
        optimal_out_mapping = self.calculate_contextual_equivalence_mapping(outgoing_neighbors1, outgoing_neighbors2.copy(), graph1, graph2)

        # print(f"\nOptimal Input Mapping Size: {len(optimal_in_mapping)}")
        # print(f"Optimal Output Mapping Size: {len(optimal_out_mapping)}\n")
        # print(f"len(incoming_neighbors1): {len(incoming_neighbors1)}\n")
        
        # print(f"len(incoming_neighbors2): {len(incoming_neighbors2)}\n")
        # print(f"len(outgoing_neighbors1): {len(outgoing_neighbors1)}\n")
        # print(f"len(outgoing_neighbors2): {len(outgoing_neighbors2)}\n")
        

        if len(incoming_neighbors1) > 0 and len(incoming_neighbors2) > 0:
            test1 = len(optimal_in_mapping) / (2 * (len(incoming_neighbors1) ** 0.5) * (len(incoming_neighbors2) ** 0.5))

        else:
            # print("Skipping input context calculation (empty neighbor sets)")
            test1 = 0 

        if len(outgoing_neighbors1) > 0 and len(outgoing_neighbors2) > 0:
            test2 = len(optimal_out_mapping) / (2 * (len(outgoing_neighbors1) ** 0.5) * (len(outgoing_neighbors2) ** 0.5))

        else:
            test2 = 0  
       

        # print(f"\nInput Context Similarity: {test1}")

        # print(f"Output Context Similarity: {test2}\n")
        

        contextual_similarity = test1 + test2

        
        
        return contextual_similarity

    def node_matching_similarity(self, threshold=0.2, ignore_types=None):
        if ignore_types is None:
            ignore_types = set()

        total_similarity = 0
        matched_pairs = []
        self.similarity_scores = {} 

        for node1, attr1 in self.reference_graph.nodes(data=True):
            if attr1.get("type", "") in ignore_types:
                continue

            best_match_score = 0
            best_match_node = None

            for node2, attr2 in self.altered_graph.nodes(data=True):
                if attr2.get("type", "") in ignore_types:
                    continue

                # Calculate similarities
                syntactic_similarity = self.calculate_syntactic_similarity(attr1.get("label", ""), attr2.get("label", ""))
                semantic_similarity = self.calculate_semantic_similarity(attr1.get("label", ""), attr2.get("label", ""))
                label_similarity = max(syntactic_similarity, semantic_similarity)
                type_similarity = self.calculate_type_similarity(attr1.get("type", ""), attr2.get("type", ""))

                contextual_similarity = self.calculate_contextual_similarity(node1, self.reference_graph, node2, self.altered_graph)

                combined_similarity = (label_similarity + type_similarity + contextual_similarity) / 3

                print(f"\nNode1: {node1}, Node2: {node2}")
                print(f"  Syntactic Similarity: {syntactic_similarity}")
                print(f"  Semantic Similarity: {semantic_similarity}")
                print(f"  Label Similarity: {label_similarity}")
                print(f"  Type Similarity: {type_similarity}")
                print(f"  Contextual Similarity: {contextual_similarity}")
                print(f"  Combined Similarity: {combined_similarity}")
                if combined_similarity > best_match_score:
                    best_match_score = combined_similarity
                    best_match_node = node2

            if best_match_score >= threshold:
                total_similarity += best_match_score
                print(f"total similarity till now = {total_similarity}")
                matched_pairs.append((node1, best_match_node))
                self.similarity_scores[(node1, best_match_node)] = best_match_score  

        valid_nodes_graph1 = [n for n, attr in self.reference_graph.nodes(data=True) if attr.get("type", "") not in ignore_types]
        valid_nodes_graph2 = [n for n, attr in self.altered_graph.nodes(data=True) if attr.get("type", "") not in ignore_types]

        total_valid_nodes = len(valid_nodes_graph1) + len(valid_nodes_graph2)
        print(f"total valid nodes = {total_valid_nodes}")
        
        print(f"total_similarity = {total_similarity}")
        print(f" * 2 = {total_similarity * 2}")
        similarity_score = (2 * total_similarity) / total_valid_nodes if total_valid_nodes > 0 else 0
        print(f"similarity score = {similarity_score}")
        
        
        self.matched_pairs = matched_pairs

        print(matched_pairs)

        return similarity_score

    def calculate_structural_similarity(self):
        reference_graph_copy = self.reference_graph.copy()
        altered_graph_copy = self.altered_graph.copy()

        def remove_gateways(graph):
            gateways = [n for n, attr in graph.nodes(data=True) if attr.get('type', '').endswith('Gateway')]
            for gateway in gateways:
                predecessors = list(graph.predecessors(gateway))
                successors = list(graph.successors(gateway))
                for pred in predecessors:
                    for succ in successors:
                        if pred != succ:
                            graph.add_edge(pred, succ)
                graph.remove_node(gateway)

        remove_gateways(reference_graph_copy)
        remove_gateways(altered_graph_copy)

        
        if len(reference_graph_copy.nodes()) == 0 or len(altered_graph_copy.nodes()) == 0:
            print("One of the graphs is empty after gateway removal. Setting structural similarity to 0.")
            return 0  
        
        matched_altered_nodes = set()


        matched_pairs = []
        unmatched_reference_nodes = set(reference_graph_copy.nodes())
        unmatched_altered_nodes = set(altered_graph_copy.nodes())
        total_substitution_cost = 0

        for node1, attr1 in reference_graph_copy.nodes(data=True):
            best_match = None
            best_similarity = 0
            for node2, attr2 in altered_graph_copy.nodes(data=True):
                if node2 in matched_altered_nodes:
                    continue
                
                syntactic_similarity = self.calculate_syntactic_similarity(attr1.get('label', ''), attr2.get('label', ''))
                semantic_similarity = self.calculate_semantic_similarity(attr1.get('label', ''), attr2.get('label', ''))
                label_similarity = max(syntactic_similarity, semantic_similarity)
                type_similarity = self.calculate_type_similarity(attr1.get('type', ''), attr2.get('type', ''))
                similarity = (label_similarity + type_similarity) / 2

                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match = node2

            if best_match is not None and best_similarity > 0:
                matched_pairs.append((node1, best_match))
                unmatched_reference_nodes.remove(node1)
                if best_match in unmatched_altered_nodes:
                    unmatched_altered_nodes.remove(best_match)
                matched_altered_nodes.add(best_match)
                total_substitution_cost += (1 - best_similarity)

        nodes_deleted_or_inserted = len(unmatched_reference_nodes) + len(unmatched_altered_nodes)
        unmatched_edges = set()

        node_mapping = {ref: alt for ref, alt in matched_pairs}
        reverse_mapping = {alt: ref for ref, alt in matched_pairs}

        for source, target in reference_graph_copy.edges():
            mapped_source = node_mapping.get(source)
            mapped_target = node_mapping.get(target)
            if mapped_source is None or mapped_target is None or not altered_graph_copy.has_edge(mapped_source, mapped_target):
                unmatched_edges.add((source, target))

        for source, target in altered_graph_copy.edges():
            mapped_source = reverse_mapping.get(source)
            mapped_target = reverse_mapping.get(target)
            if mapped_source is None or mapped_target is None or not reference_graph_copy.has_edge(mapped_source, mapped_target):
                unmatched_edges.add((mapped_source, mapped_target))

        edges_deleted_or_inserted = len(unmatched_edges)

        # Calculate structural similarity
        snv = nodes_deleted_or_inserted / max(1, len(reference_graph_copy.nodes()) + len(altered_graph_copy.nodes()))
        sev = edges_deleted_or_inserted / max(1, len(reference_graph_copy.edges()) + len(altered_graph_copy.edges()))
        sbv_denominator = len(reference_graph_copy.nodes()) + len(altered_graph_copy.nodes()) - nodes_deleted_or_inserted
        sbv = 2 * total_substitution_cost / sbv_denominator if sbv_denominator > 0 else 0
        structural_similarity = 1 - ((snv + sev + sbv) / 3)

        return structural_similarity

    def generate_causal_footprint(self, graph): 
        causal_footprint = {}
        
        nodes = list(graph.nodes())
        for node in nodes:
            look_back = set()
            look_ahead = set()

            # Find look-back relationships
            for predecessor in nodes:
                if predecessor != node:
                    paths = list(nx.all_simple_paths(graph, predecessor, node))
                    if paths:
                        look_back.add(predecessor)

            # Find look-ahead relationships
            for successor in nodes:
                if successor != node:
                    paths = list(nx.all_simple_paths(graph, node, successor))
                    if paths:
                        look_ahead.add(successor)

            causal_footprint[node] = {"look_back": look_back,
                                      "look_ahead": look_ahead}

        return causal_footprint
    
    def generate_index_terms(self, causal_footprint1, causal_footprint2):
        # print("-------")
        # print("\n Causal Footprint1: ")
        # self.print_causal_footprint(causal_footprint1)
        # print("-------")
        # print("\n Causal Footprint2: ")
        # self.print_causal_footprint(causal_footprint2)

        # Step 1: Calculate mapped activities using calculate_optimal_equivalence_mapping
        reference_nodes = list(self.reference_graph.nodes())
        altered_nodes = list(self.altered_graph.nodes())
        
        # Use calculate_optimal_equivalence_mapping to get mapped activities
        optimal_mapping = self.matched_pairs
        self.mapping = {a1: a2 for a1, a2 in optimal_mapping}  
        mapped_activities = set(optimal_mapping)  
        
        print("\n-------")
        print("\nmapped_activities: ", mapped_activities)
        print("\n-------")

        # Step 2: Identify unmapped activities
        mapped_reference_nodes = set(a1 for a1, _ in mapped_activities)
        mapped_altered_nodes = set(a2 for _, a2 in mapped_activities)
        
        unmapped_activities1 = set(reference_nodes) - mapped_reference_nodes
        print(f"Unmapped ref graph: {unmapped_activities1}")
        unmapped_activities2 = set(altered_nodes) - mapped_altered_nodes
        print(f"Unmapped alt graph: {unmapped_activities2}")

        # Step 3: Extract causal links
        look_back_links1 = {
            (target, source)
            for target, links in causal_footprint1.items()
            for source in links["look_back"]
        }
        

        print("------")
        print("\nlook_back_links1: ", look_back_links1)

        look_ahead_links1 = {
            (source, target)
            for source, links in causal_footprint1.items()
            for target in links["look_ahead"]
        }

        print("\nlook_ahead_links1: ", look_ahead_links1)


        look_back_links2 = {
            (target, source)
            for target, links in causal_footprint2.items()
            for source in links["look_back"]
        }

        print("\nlook_back_links2: ", look_back_links2)

        look_ahead_links2 = {
            (source, target)
            for source, links in causal_footprint2.items()
            for target in links["look_ahead"]
        }

        print("\nlook_ahead_links2: ", look_ahead_links2)
        print("------")

        print("\nIndex terms: ", mapped_activities.union(
            unmapped_activities1, unmapped_activities2, look_back_links1, look_ahead_links1, look_back_links2, look_ahead_links2
        ))
        print("------")

        # Step 4: Combine all index terms
        return mapped_activities.union(
            unmapped_activities1, unmapped_activities2, look_back_links1, look_ahead_links1, look_back_links2, look_ahead_links2
        )

    def compute_weight(self, term, causal_footprint, graph):
        if isinstance(term, tuple) and len(term) == 2:  # Mapped activity or causal link

            # Case 1: Mapped Activities
            if term in self.mapping.items():
                node1, node2 = term
                if node1 in graph.nodes and node2 in graph.nodes:
                    if (node1, node2) in self.similarity_scores:
                        return self.similarity_scores[(node1, node2)]
                    else:
                        return 0  

            # Case 2: Look-Back Causal Links
            for target, links in causal_footprint.items():
                if term in {(source, target) for source in links["look_back"]}:
                    source, target_node = term
                    if source in graph.nodes and target_node in graph.nodes:
                        attr_source = graph.nodes[source]
                        attr_target = graph.nodes[target_node]

                        syntactic_similarity = self.calculate_syntactic_similarity(attr_source.get("label", ""), attr_target.get("label", ""))
                        semantic_similarity = self.calculate_semantic_similarity(attr_source.get("label", ""), attr_target.get("label", ""))
                        label_similarity = max(syntactic_similarity, semantic_similarity)
                        type_similarity = self.calculate_type_similarity(attr_source.get("type", ""), attr_target.get("type", ""))

                        contextual_similarity = self.calculate_contextual_similarity(source, graph, target_node, graph)

                        length_lookback = len(links["look_back"])
                        print(f"length of look back: {length_lookback}")

                        combined_similarity = (label_similarity + type_similarity + contextual_similarity) / (2 ** length_lookback)

                        return combined_similarity

            # Case 3: Look-Ahead Causal Links
                if term in {(target, successor) for successor in links["look_ahead"]}:
                    target_node, successor = term
                    if target_node in graph.nodes and successor in graph.nodes:
                        attr_target = graph.nodes[target_node]
                        attr_successor = graph.nodes[successor]

                        syntactic_similarity = self.calculate_syntactic_similarity(attr_target.get("label", ""), attr_successor.get("label", ""))
                        semantic_similarity = self.calculate_semantic_similarity(attr_target.get("label", ""), attr_successor.get("label", ""))
                        label_similarity = max(syntactic_similarity, semantic_similarity)
                        type_similarity = self.calculate_type_similarity(attr_target.get("type", ""), attr_successor.get("type", ""))

                        contextual_similarity = self.calculate_contextual_similarity(target_node, graph, successor, graph)

                        length_lookahead = len(links["look_ahead"])
                        print(f"length of look ahead: {length_lookahead}")

                        combined_similarity = (label_similarity + type_similarity + contextual_similarity) / (2 ** length_lookahead)

                        return combined_similarity

        # Case 4: Default to 0 
        return 0

    def generate_index_vectors(self, index_terms, causal_footprint1, causal_footprint2):
        index_terms = list(index_terms)  
        vector1 = []
        vector2 = []

        for i, term in enumerate(index_terms):  
            weight1 = self.compute_weight(term, causal_footprint1, self.reference_graph)
            weight2 = self.compute_weight(term, causal_footprint2, self.altered_graph)

            vector1.append(weight1)
            vector2.append(weight2)

            print(f"Index: {i:<3} Term: {str(term):<40} Weight1: {weight1:<10} Weight2: {weight2:<10}")


        return np.array(vector1), np.array(vector2)

    def calculate_cosine_similarity(self, vector1, vector2):
        print("\nComparison of Vectors:")
        print(f"{'Index':<10} {'Vector1':<20} {'Vector2':<20}")
        print("-" * 50)
        for i, (v1, v2) in enumerate(zip(vector1, vector2)):
            print(f"{i:<10} {v1:<20} {v2:<20}")
        
        print("\n-------")

        # Reshape vectors into 2D since sklearn expects 2D arrays
        vector1 = np.array(vector1).reshape(1, -1)
        vector2 = np.array(vector2).reshape(1, -1)

        similarity = cosine_similarity(vector1, vector2)[0][0]  # Extract scalar value
        return similarity
    
    def calculate_behavioral_similarity(self):
        causal_footprint1 = self.generate_causal_footprint(self.reference_graph)
        causal_footprint2 = self.generate_causal_footprint(self.altered_graph)

        self.print_causal_footprint(causal_footprint1)
        print("----------------")
        self.print_causal_footprint(causal_footprint2)

        # generate index terms
        index_terms = self.generate_index_terms(causal_footprint1, causal_footprint2)

        # generate index vectors
        vector1, vector2 = self.generate_index_vectors(index_terms, causal_footprint1, causal_footprint2)

        # cosine similarity
        return self.calculate_cosine_similarity(vector1, vector2)
    
    def print_graph_table(self, graph, title="Graph"):
            print(f"======== {title} ========")

        
            node_table = [
                [node, data.get("label", ""), data.get("type", ""), data.get("lane", "")]
                for node, data in graph.nodes(data=True)
            ]
            print("\nNodes:")
            print(tabulate(node_table, headers=["Node ID", "Label", "Type", "Lane"], tablefmt="grid"))

 
            edge_table = [
                [source, target, data.get("label", "")]
                for source, target, data in graph.edges(data=True)
            ]
            print("\nEdges:")
            print(tabulate(edge_table, headers=["Source", "Target", "Label"], tablefmt="grid"))

    def print_causal_footprint(self, causal_footprint):
        for node, links in causal_footprint.items():
            print(f"{node}:")
            for link_type, neighbors in links.items():
                print(f"  {link_type}: {neighbors}")


        
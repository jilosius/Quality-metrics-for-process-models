from .model_alteration import ModelAlteration
from process.process import Process
from process.flow import Flow

import random

class RemoveActivity:

    def __init__(self, node_id=None):
        """
        Optionally specify the node_id to remove.
        If None, a random node will be selected.
        """
        self.node_id = node_id

    def apply(self, model: Process) -> Process:
        # Step 1: Find the target node to remove
        if self.node_id:
            target_node = None
            for node in model.flowNodes:
                if node.flowNode_id == self.node_id:
                    target_node = node
                    break
            if not target_node:
                print(f"FlowNode with ID {self.node_id} not found. No changes made.")
                return model
        else:
            # Randomly select any flow node
            target_node = random.choice(model.flowNodes) if model.flowNodes else None

        if not target_node:
            print("No valid flow nodes available to remove.")
            return model

        print(f"Flow node to remove: {target_node}")

        # Step 2: Identify preceding and succeeding nodes
        preceding_nodes = [flow.source for flow in model.flows if flow.target == target_node]
        print(f"Preceding nodes: {preceding_nodes}")
        succeeding_nodes = [flow.target for flow in model.flows if flow.source == target_node]
        print(f"Succeeding nodes: {succeeding_nodes}")

        # Step 3: Remove associated flows
        updated_flows = []
        for flow in model.flows:
            if flow.source != target_node and flow.target != target_node:
                updated_flows.append(flow)
        model.flows = updated_flows

        # Step 4: Create new flows between preceding and succeeding nodes
        for source in preceding_nodes:
            for target in succeeding_nodes:
                ModelAlteration.flow_count += 1
                new_flow = Flow(
                    flow_id=f"flow_{ModelAlteration.flow_count}",
                    label=f"Flow from {source.flowNode_id} to {target.flowNode_id}",
                    source=source,
                    target=target
                )
                model.flows.append(new_flow)
                print(f"Added flow: {new_flow.label} ({source.flowNode_id} -> {target.flowNode_id})")

        # Step 5: Remove the target node
        model.flowNodes.remove(target_node)

        print(f"Removed FlowNode: {target_node.flowNode_id} (ID: {target_node.flowNode_id})")
        return model

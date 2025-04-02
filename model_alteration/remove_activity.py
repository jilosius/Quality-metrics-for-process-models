from .model_alteration import ModelAlteration
from process.process import Process
from process.flow import Flow
import random

class RemoveActivity:

    def __init__(self, node_ids=None):
        self.node_ids = node_ids if node_ids else []

    def apply(self, model: Process) -> Process:
        if not model.flowNodes:
            print("No flow nodes in the model.")
            return model

        if not self.node_ids:
            # Remove one random node
            target_node = random.choice(model.flowNodes)
            return self._remove_node(model, target_node)

        # Remove each node by ID
        for node_id in self.node_ids:
            target_node = next((n for n in model.flowNodes if n.flowNode_id == node_id), None)
            if target_node:
                model = self._remove_node(model, target_node)
            else:
                print(f"FlowNode with ID {node_id} not found. Skipping.")

        return model

    def _remove_node(self, model: Process, target_node) -> Process:
        print(f"Removing FlowNode: {target_node.flowNode_id}")

        preceding_nodes = [flow.source for flow in model.flows if flow.target == target_node]
        succeeding_nodes = [flow.target for flow in model.flows if flow.source == target_node]

        model.flows = [
            flow for flow in model.flows
            if flow.source != target_node and flow.target != target_node
        ]

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

        model.flowNodes.remove(target_node)
        print(f"Successfully removed {target_node.flowNode_id}")
        return model

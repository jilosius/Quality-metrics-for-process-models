from .model_alteration import ModelAlteration
from process.process import Process
from process.flowNode import FlowNode
from process.flow import Flow

import random

GATEWAY_TYPE_MAPPING = {
    "XOR": "exclusiveGateway",
    "AND": "parallelGateway",
    "OR": "inclusiveGateway"
}

class AddFlowNode:
    def __init__(self, element_type="userTask"):

        self.element_type = element_type
        self.gateway_type = (
            random.choice(list(GATEWAY_TYPE_MAPPING.values()))
            if element_type == "gateway"
            else None
        )

    def apply(self, model: Process) -> Process:
        # Step 1: Assign a unique ID and label
        ModelAlteration.flowNode_count += 1
        element_id = f"{self.element_type}_{ModelAlteration.flowNode_count}"
        element_label = (
            f"{self.gateway_type} Gateway {ModelAlteration.flowNode_count}"
            if self.element_type == "gateway"
            else f"{self.element_type} {ModelAlteration.flowNode_count}"
        )
        element_type = (
            self.gateway_type if self.element_type == "gateway" else self.element_type
        )

        # Step 2: Randomly assign a lane ID
        lane_id = random.choice(model.lanes).lane_id if model.lanes else None

        # Step 3: Create the new node
        new_node = FlowNode(
            flowNode_id=element_id,
            label=element_label,
            flowNode_type=element_type,
            lane_id=lane_id,
        )
        model.flowNodes.append(new_node)

        # Step 4: Ensure sufficient nodes for connections
        if len(model.flowNodes) < 2:
            print(f"Not enough nodes to connect {element_label}.")
            return model

        # Step 5: Randomly select source and target nodes
        # End nodes cannot have outgoing flows
        potential_sources = [
            node for node in model.flowNodes
            if node != new_node and node.type.lower() != "endevent"
        ]
        if not potential_sources:
            print(f"No valid source nodes available for {element_label}.")
            return model
        source_node = random.choice(potential_sources)

        # Start nodes cannot have incoming flows
        potential_targets = [
            node for node in model.flowNodes
            if node != source_node and node != new_node and node.type.lower() != "startevent"
        ]
        if not potential_targets:
            print(f"No valid target nodes available for {element_label}.")
            return model
        target_node = random.choice(potential_targets)

        # Step 6: Create new flows
        ModelAlteration.flow_count += 1
        flow_to_node = Flow(
            flow_id=f"flow_{ModelAlteration.flow_count}",
            label=f"Flow to {element_label}",
            source=source_node,
            target=new_node,
        )
        ModelAlteration.flow_count += 1
        flow_from_node = Flow(
            flow_id=f"flow_{ModelAlteration.flow_count}",
            label=f"Flow from {element_label}",
            source=new_node,
            target=target_node,
        )
        model.flows.extend([flow_to_node, flow_from_node])

        # Step 7: Debugging info
        if lane_id:
            print(f"Added {element_label} ({element_type}) in lane {lane_id} between {source_node.flowNode_id} and {target_node.flowNode_id}")
        else:
            print(f"Added {element_label} ({element_type}) with no lane between {source_node.flowNode_id} and {target_node.flowNode_id}")

        print(f"New flows: {flow_to_node.label}, {flow_from_node.label}")

        return model

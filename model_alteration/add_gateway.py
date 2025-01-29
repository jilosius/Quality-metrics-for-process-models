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

class AddGateway:
    def __init__(self):
        
        pass  

    def apply(self, model: Process) -> Process:
        # Select a random gateway type at EACH application
        gateway_type = random.choice(list(GATEWAY_TYPE_MAPPING.values()))
        print(f"DEBUG: Selected gateway type -> {gateway_type}")  # Debugging print

        # Step 1: Randomly assign a lane ID
        lane_id = random.choice(model.lanes).lane_id if model.lanes else None

        # Step 2: Create a new gateway node
        ModelAlteration.flowNode_count += 1
        gateway_id = f"gateway_{ModelAlteration.flowNode_count}"
        gateway_label = f"{gateway_type} Gateway {ModelAlteration.flowNode_count}"
        gateway_node = FlowNode(
            flowNode_id=gateway_id,
            label=gateway_label,
            flowNode_type=gateway_type,
            lane_id=lane_id
        )
        model.flowNodes.append(gateway_node)

        # Step 3: Ensure sufficient nodes for connections
        if len(model.flowNodes) < 2:
            print("Not enough nodes to add a gateway.")
            return model

        # Step 4: Randomly select source and target nodes
        potential_sources = [
            node for node in model.flowNodes if node != gateway_node and node.type.lower() != "endevent"
        ]
        if not potential_sources:
            print(f"No valid source nodes available for {gateway_label}.")
            return model
        source_node = random.choice(potential_sources)

        potential_targets = [
            node for node in model.flowNodes if node != source_node and node != gateway_node and node.type.lower() != "startevent"
        ]
        if not potential_targets:
            print(f"No valid target nodes available for {gateway_label}.")
            return model
        target_node = random.choice(potential_targets)

        # Step 5: Create new flows
        ModelAlteration.flow_count += 1
        flow_to_gateway = Flow(
            flow_id=f"flow_{ModelAlteration.flow_count}",
            label=f"Flow to {gateway_label}",
            source=source_node,
            target=gateway_node
        )

        ModelAlteration.flow_count += 1
        flow_from_gateway = Flow(
            flow_id=f"flow_{ModelAlteration.flow_count}",
            label=f"Flow from {gateway_label}",
            source=gateway_node,
            target=target_node
        )

        # Add the new flows to the model
        model.flows.extend([flow_to_gateway, flow_from_gateway])

        # Step 6: Debugging info
        lane_info = f" in lane {lane_id}" if lane_id else " with no lane"
        print(f"Added {gateway_label} ({gateway_type}){lane_info} between {source_node.flowNode_id} and {target_node.flowNode_id}")
        print(f"New flows: {flow_to_gateway.label}, {flow_from_gateway.label}")

        return model

from .model_alteration import ModelAlteration
from process.process import Process
from process.flow import Flow

import random

class RemoveGateway:

    def __init__(self, gateway_id=None):
        """
        Optionally specify the gateway_id to remove.
        If None, a random gateway will be selected.
        """
        self.gateway_id = gateway_id

    def apply(self, model: Process) -> Process:
        
        if self.gateway_id:
            target_gateway = None
            for node in model.flowNodes:
                if node.flowNode_id == self.gateway_id and node.type in ["exclusiveGateway", "parallelGateway", "inclusiveGateway"]:
                    target_gateway = node
                    break
            if not target_gateway:
                print(f"Gateway with ID {self.gateway_id} not found or is not a valid gateway. No changes made.")
                return model
        else:
            # select random gateway if not specified
            gateways = [node for node in model.flowNodes if node.type in ["exclusiveGateway", "parallelGateway", "inclusiveGateway"]]
            target_gateway = random.choice(gateways) if gateways else None

        if not target_gateway:
            print("No gateways available to remove.")
            return model

        print(f"Removing Gateway: {target_gateway.label} (Type: {target_gateway.type})")


        incoming_flows = [flow for flow in model.flows if flow.target == target_gateway]
        outgoing_flows = [flow for flow in model.flows if flow.source == target_gateway]

        print(f"Incoming flows: {len(incoming_flows)}")
        print(f"Outgoing flows: {len(outgoing_flows)}")


        for incoming in incoming_flows:
            for outgoing in outgoing_flows:
                ModelAlteration.flow_count += 1
                new_flow = Flow(
                    flow_id=f"flow_{ModelAlteration.flow_count}",
                    label=f"Flow from {incoming.source.flowNode_id} to {outgoing.target.flowNode_id}",
                    source=incoming.source,
                    target=outgoing.target
                )
                model.flows.append(new_flow)
                print(f"Added flow: {new_flow.label} ({incoming.source.flowNode_id} -> {outgoing.target.flowNode_id})")


        model.flowNodes.remove(target_gateway)
        model.flows = [flow for flow in model.flows if flow.source != target_gateway and flow.target != target_gateway]

        print(f"Removed Gateway: {target_gateway.label} (ID: {target_gateway.flowNode_id})")
        return model

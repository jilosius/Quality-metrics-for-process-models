import random
from process.process import Process

class RemoveFlow:

    def __init__(self, flow_id=None):
        """
        Optionally specify the flow_id to remove.
        If None, a random flow will be selected.
        """
        self.flow_id = flow_id

    def apply(self, model: Process) -> Process:
        # Step 1: Find the target flow
        if self.flow_id:
            target_flow = None
            for flow in model.flows:
                if flow.flow_id == self.flow_id:
                    target_flow = flow
                    break
            if not target_flow:
                print(f"Flow with ID {self.flow_id} not found. No changes made.")
                return model
        else:
            target_flow = random.choice(model.flows) if model.flows else None

        if not target_flow:
            print("No flows available to remove.")
            return model

        # Step 2: Remove the flow
        model.flows.remove(target_flow)

        # Step 3: Log the removal
        print(f"Removed Flow: {target_flow.label} (ID: {target_flow.id})")

        # Step 4: Return the updated model
        return model
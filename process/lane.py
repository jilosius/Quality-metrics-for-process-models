from .flowNode import FlowNode
from .data_obj import DataObj

class Lane:
    def __init__(self, lane_id: str):
        self.lane_id = lane_id
        self.flowNodes = []  # List of Activity objects

    def add_flowNode(self, flowNode: FlowNode):
        self.flowNodes.append(flowNode)

    def __repr__(self):
        return f"Lane(id={self.lane_id}, flowNodes={len(self.flowNodes)})"

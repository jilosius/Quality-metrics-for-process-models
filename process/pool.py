from .lane import Lane
from .flowNode import FlowNode


class Pool:
    def __init__(self, pool_id: str, name: str):
        """
        Represents a pool in the process.
        
        :param pool_id: Unique ID of the pool.
        :param name: Name/description of the pool.
        """
        self.pool_id = pool_id
        self.name = name
        self.lanes = []  # List of Lane objects
        self.flowNodes = []  # List of FlowNode objects

    def add_lane(self, lane: "Lane"):
        """
        Adds a Lane object to the pool.
        
        :param lane: Lane object to be added.
        """
        self.lanes.append(lane)

    def add_flowNode(self, flowNode: "FlowNode"):
        """
        Adds a FlowNode object to the pool.
        
        :param flowNode: FlowNode object to be added.
        """
        self.flowNodes.append(flowNode)

    def __repr__(self):
        return (
            f"Pool(id={self.pool_id}, name={self.name}, "
            f"lanes={len(self.lanes)}, flowNodes={len(self.flowNodes)})"
        )

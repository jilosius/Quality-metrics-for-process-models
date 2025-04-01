from .lane import Lane


class Pool:
    def __init__(self, pool_id: str, name: str):

        self.pool_id = pool_id
        self.name = name
        self.lanes = []  
        self.flowNodes = [] 

    def add_lane(self, lane: Lane):

        self.lanes.append(lane)



    def __repr__(self):
        return (
            f"Pool(id={self.pool_id}, name={self.name}, "
            f"lanes={len(self.lanes)}, flowNodes={len(self.flowNodes)})"
        )

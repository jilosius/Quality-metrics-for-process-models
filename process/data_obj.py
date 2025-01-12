class DataObj:
    def __init__(self, label: str, data_type: str):
        self.label = label
        self.type = data_type

    def __repr__(self):
        return f"DataObj(label={self.label}, type={self.type})"

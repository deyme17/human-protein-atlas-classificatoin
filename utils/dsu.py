class DSU:
    def __init__(self, ids: list[str]):
        self.parent = {i: i for i in ids}
        
    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x
        
    def union(self, a: str, b: str) -> None:
        self.parent[self.find(a)] = self.find(b)
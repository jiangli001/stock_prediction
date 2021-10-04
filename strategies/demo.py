class Demo(object):
    def __init__(self, name):
        self.message = "this is a demo strategy"
        self.name = name
        
    def train(self, X, y):
        return
    
    def test(self, X, y):
        return
    
    def predict(self, X):
        return
    

if __name__ == "__main__":
    demo_strategy = Demo("demo strategy")
    print(demo_strategy.name)
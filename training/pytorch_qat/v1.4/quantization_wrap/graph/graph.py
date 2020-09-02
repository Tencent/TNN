# danerli add 2020.8.14

from quantization_wrap.graph.graph_optimizer import *

class Graph():
    """
        init graph
    """
    def __init__(self, model, image):

        super(Graph, self).__init__()

        self.model = model

        self.image = image

        self.graph = self.build()

    def build(self):

        with torch.onnx.set_training(self.model, False):
            try:
                trace = torch.jit.trace(self.model, (self.image,))
                graph = trace.graph
                torch._C._jit_pass_inline(graph)
                return graph
            except RuntimeError as e:
                print(e)
                print('Error occurs, No graph saved')
                raise e

    def optimizer(self, fuse_mode=2):

        go = GraphOptimizer(self.model, self.graph, fuse_mode)
        go.graph_optimizer()

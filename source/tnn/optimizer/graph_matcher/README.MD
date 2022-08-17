# Graph Matcher 

Generic DAG Matcher and Rewriter Infrastructure 

## TNN::Graph 

Represents a model in TNN. Including Nodes, Tensors, Edges.

### Constructing Methods

1. From NetStructure:
    `Status Graph::fromInterpreted(NetStructure * , NetResource *);`

2. From TextGraphParser
    ```
        std::vector<std::string> text_graph_pattern = {
            "Convolution@conv",
            "AnyType@act",
        };

        TNN_NS::TextGraphParser parser;
        std::shared_ptr<Graph> pattern = nullptr;
        if (parser.parseFromString(text_graph_pattern)) {
            pattern = parser.getGraph();
        }
    ```

3. From GraphParser

    ```
    std::string graph_str = R"(
        graph(%a):
            %a = Add(%5)
            %b,%c = Mul(%a)
            return (%b)
    )";
    
    TNN_NS::GraphParser graph_parser;
    if (parser.parseFromString(graph_str)) {
        pattern = parser.getGraph();
    }
    ```
    
# SubGraph Matchitng
``` 
        Graph * graph = ...;
        Graph * pattern = ...;
        auto gen = [](std::shared_ptr<TNN_NS::AnchorGraph> in) -> std::shared_ptr<TNN_NS::Graph> {
            return nullptr;
        };
        graph->rewrite(pattern, gen);
```

The `gen` function will be called with each matched subgraph, which is type of `AnchorGraph`. 
The AnchorGraph represents a subgraph of some other Graph.  Note that :
1. The AnchorGraph do not store matched placeholders, but create fake ones for the inEdges.
2. It does not modify the original Graph, Nodes and Edges.
3. The AnchorGraph do not use marked_outputs.

# Graph Rewriting
``` 
        Graph * graph = ...;
        Graph * pattern = ...;
        auto gen = [](std::shared_ptr<TNN_NS::AnchorGraph> in) -> std::shared_ptr<TNN_NS::Graph> {
            // a pattern might matched with subgraphs that an internal node is referenced 
            // by nodes outside the subgraph, which leds to an incresement of outputs()
            // so we check it here.
            if (in->inputs().size() != 1 || in->outputs().size() != 1 ){
                return nullptr;
            }
            
            // node matched with PatternNode that has a name will be marked with prefix@ in the AnchorGraph 
            auto n_of_interest = in->getNodeByTensorName(std::string("@xxx"));
            if (!n_of_interest) {
                printf("roi node not found\n");
                return nullptr;
            }

            // Create a new Graph and return it, it will replace the AnchorGraph
            // Notes that the number of inputs and number of outputs for the AnchorGraph and new Graph must match.
            auto g = std::make_shared<TNN_NS::Graph>();
            auto in_name = "input_1";
            auto in1 = g->getNodeOrCreatePlaceHolder(in_name);
            auto status = g->createNode(TNN_NS::LAYER_TANH, {in_name}, {"new_heir_node"});
            if (status != TNN_NS::TNN_OK) {
                return nullptr;
            }

            return g;
        };
        
        graph->rewrite(pattern, gen);
```

# TextGraph Syntax

### Graph

Every line may contain multiple Node.  every Node has an default input edge to the above Node that `has a same indention`. PlaceHolder will be created if default input nodes found.

### Node
A node is Reresented by the following snippet(the + symbol is not included): `LayerType+Inputs`

`LayerType` can be any TNN LayerType, e.g. Add, Mul.

`Inputs` is defined as : `'+'.join(offset)`, here `offset` is a sequence of `<` or `>` of any size. 
`<` means shifting left once, `<<`  Means an shifting left twice, etc.
`>` means shifting right once, `>>`  Means an shifting right twice, etc.

For example: 
```
Add     Sub
Mul     Mul
Div+>
```

equals to :
```
graph(%1, %2):
        %3 = Add(%1)
        %4 = Sub(%2)
        %5 = Mul(%3)
        %6 = Mul(%4)
        %7 = Div(%5, %6)
        return (%7)
```

an larger TextGraph :

```        
        "LayerNorm # comments are allowed",
        "        MatMul<",
        "        Add",
        "                                      Mul<",
        "                            Mul<+>",
        "                Mul<        Add",
        "                Mul+>",
        "                Tanh@act #node name is register by @ symbol ",
        "        Mul     Add",
        "        Mul+>",
        "        MatMul",
        "        Add+{act} # Add has two input edges, one is MatMul, the other one is Tanh",
        "Add+>",
        "Add",
        "Mul     ",
        "Mul",
```


## How to debug

1. Set logger level to DEBUG.
```
    TNN_NS::Logger::instance().set_verbose_level("D");
```

2. Dump the Graph to proto:
```
    std::ofstream f("text_graph.tnnproto");
    graph.dump(f);
```
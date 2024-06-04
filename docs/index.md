----------------------------------------
### What is graph Neural Network (GNN) ? 
----------------------------------------

Graph Neural Networks (GNNs) are a type of neural network designed specifically to operate on graph structures. They are powerful tools for dealing with data that represents entities and their relationships, such as social networks, molecular structures, or communication networks.

#### Key Features of Graph Neural Networks:

- **Graph-Based Input**: Unlike traditional neural networks that expect inputs in the form of vectors (e.g., images, text), GNNs work directly with graphs. These graphs are composed of nodes (vertices) and edges, each potentially having their own attributes.

- **Message Passing**: GNNs use a technique called message passing, where nodes aggregate information from their neighbors. This process involves updating the representation of a node by combining its own features with the features of its adjacent nodes, often using functions like sum, mean, or max. This mechanism allows GNNs to learn the structural information of the graph.

- **Local Connectivity**: In GNNs, the computation for a node’s representation is localized to its neighborhood. This is fundamentally different from architectures like CNNs where filters slide over the entire input space, or RNNs where sequences are processed either entirely or in large segments.

#### Differences from Traditional Neural Networks:

- **Data Structure**: Traditional neural networks usually require fixed-size input and do not natively support irregular data structures like graphs. GNNs, on the other hand, can naturally handle graphs of varying sizes and complexities.

- **Invariance to Permutations**: The output of a GNN is invariant to the permutation of node labels in the input graph. This means that reordering the nodes of the graph does not change the output of the network, which is a desirable property for many graph-based tasks.

- **Relational Reasoning**: GNNs excel at tasks that require reasoning about the relationships and interconnections between entities, making them suitable for tasks like social network analysis, chemical molecule analysis, and recommendation systems, where interactions or relationships are key.

- **Adaptability to Dynamic Data**: GNNs are particularly adept at handling dynamic graphs where the structure may change over time, whereas traditional neural networks would require retraining or significant adjustments.

Overall, GNNs extend the capability of traditional neural networks to a broad range of data types and structures, particularly excelling in domains where data can be naturally represented as graphs. This specialization allows them to capture both node-level and graph-level properties effectively.
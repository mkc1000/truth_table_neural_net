# truth_table_neural_net
This never succeeded in outperforming a basic neural net, and took significantly longer to train. A neural network with the capacity for "truth table layers" is implemented here. 

Truth table layers work as follows. If the number of nodes in the one layer is n, the number of nodes in the next layer must be n choose 2. Indeed each node in the successive layer only depends on two nodes in the former layer, with all possibilities included. To compute the activation of the resulting node from the two original nodes (A and B), four parameters are used (restricted in value to belong to [0,1]), effectively weighting the following probabilities: A and B, A and not B, not A and B, and not A and not B (as if A and B were independent). Therefore, any logical combination of the actiavtions of any two nodes in one layer can be propegated to the next layer. Since this is fit into a continuous regime, such a layer can fit inside any other neural network.

As mentioned above, however, this underperformed an ordinary neural network, so the code herein is functional, but quite unpolished.

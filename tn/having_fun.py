import numpy as np
import tensornetwork as tn

# Create the nodes
a = tn.Node(np.ones(10,))
b = tn.Node(np.ones((10,)))
edge = a[0] ^ b[0]
final_node = tn.contract(edge)
print(final_node.tensor)
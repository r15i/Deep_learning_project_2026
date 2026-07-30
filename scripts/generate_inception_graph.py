import graphviz

dot = graphviz.Digraph('Inception', format='png')
dot.attr(rankdir='TB', size='8,8')

dot.node('input', 'Input Feature Map')

# Stem
dot.node('stem', 'Conv 7x7, s=2\\nBatchNorm + ReLU\\nMaxPool 3x3, s=2', shape='box')
dot.edge('input', 'stem')

# Inception Block
dot.node('inc_start', 'Inception Block', shape='plaintext')
dot.edge('stem', 'inc_start')

# Branch 1
dot.node('b1', 'MaxPool 2x2\\nstride 2', shape='box')
# Branch 2
dot.node('b2', 'Conv 2x2, s=2\\nBatchNorm + ReLU', shape='box')
# Branch 3
dot.node('b3_1', 'Conv 1x1, s=1\\nBatchNorm + ReLU', shape='box')
dot.node('b3_2', 'Conv 2x2, s=1\\nBatchNorm + ReLU', shape='box')
dot.node('b3_3', 'Conv 4x4, s=2\\nBatchNorm + ReLU', shape='box')

dot.edge('inc_start', 'b1')
dot.edge('inc_start', 'b2')
dot.edge('inc_start', 'b3_1')
dot.edge('b3_1', 'b3_2')
dot.edge('b3_2', 'b3_3')

dot.node('concat', 'Concatenate\\nAlong Channels', shape='box', style='filled', color='lightgrey')
dot.edge('b1', 'concat')
dot.edge('b2', 'concat')
dot.edge('b3_3', 'concat')

dot.node('dim_red', 'Conv 1x1, s=1\\nBatchNorm + ReLU', shape='box')
dot.edge('concat', 'dim_red')

dot.node('avg', 'Adaptive AvgPool (1x1)', shape='box')
dot.edge('dim_red', 'avg')

dot.node('fc', 'Linear (num_classes)\\n+ Dropout', shape='box')
dot.edge('avg', 'fc')

dot.render('paper/figures/inception_architecture', cleanup=True)
print("Graph generated at paper/figures/inception_architecture.png")

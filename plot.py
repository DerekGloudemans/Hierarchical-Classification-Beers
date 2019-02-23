import Utilfn
from ete3 import Tree, faces, AttrFace, TreeStyle, NodeStyle

#def layout(node):
#    if True:
#        N = AttrFace("name", fsize=30)
#        faces.add_face_to_node(N, node, 0, position="aligned")
    
#newick = Utilfn.convert_to_newick(aug_cluster_list,len(aug_cluster_list)-1)
#t = Tree(newick,format = 1)
#test = test + ":0;"
t2 =  Tree(test,format = 1)
circular_style = TreeStyle()
circular_style.mode = "r" # draw tree in circular mode
circular_style.scale = 200
circular_style.arc_span = 45
#circular_style.layout_fn = layout

t2.show(tree_style=circular_style)
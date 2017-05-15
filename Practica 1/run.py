# Search methods

import search

ab = search.GPSProblem('R', 'V', search.romania)

print "===================================================================================="
print "Busqueda en anchura\n", search.breadth_first_graph_search(ab).path()
print "===================================================================================="
print "Busqueda en profundidad\n", search.depth_first_graph_search(ab).path()
print "===================================================================================="
print "Busqueda ramificacion y acotacion\n", search.ram_ord_graph_search(ab).path()
print "===================================================================================="
print "Busqueda ramificacion y acotacion con subestimacion\n", search.ram_ord_heu_graph_search(ab).path()
print "===================================================================================="
#print search.iterative_deepening_search(ab).path()
#print search.depth_limited_search(ab).path()

#print search.astar_search(ab).path()

# Result:
# [<Node B>, <Node P>, <Node R>, <Node S>, <Node A>] : 101 + 97 + 80 + 140 = 418
# [<Node B>, <Node F>, <Node S>, <Node A>] : 211 + 99 + 140 = 450

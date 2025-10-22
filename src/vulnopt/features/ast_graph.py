import ast, networkx as nx
def python_ast_graph(code:str):
    G=nx.DiGraph()
    try:
        tree=ast.parse(code)
    except:
        return G
    idx=0
    def add(node,parent=None):
        nonlocal idx
        nid=idx; idx+=1
        G.add_node(nid,type=type(node).__name__,lineno=getattr(node,'lineno',None))
        if parent is not None:
            G.add_edge(parent,nid)
        for ch in ast.iter_child_nodes(node):
            add(ch,nid)
    add(tree)
    return G

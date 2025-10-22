import ast
def extract_functions_python(code:str):
    out=[]
    try:
        tree=ast.parse(code)
    except:
        return [{'name':None,'start_line':1,'end_line':len(code.splitlines()),'code':code}]
    for node in ast.walk(tree):
        if isinstance(node,(ast.FunctionDef,ast.AsyncFunctionDef)):
            start=getattr(node,'lineno',1); end=getattr(node,'end_lineno',start)
            lines=code.splitlines()
            snippet='\n'.join(lines[start-1:end])
            out.append({'name':node.name,'start_line':start,'end_line':end,'code':snippet})
    if not out:
        out=[{'name':None,'start_line':1,'end_line':len(code.splitlines()),'code':code}]
    return out

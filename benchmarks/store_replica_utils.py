"""Serialize the public fields of Mooncake native replica descriptors."""
def serialize(obj, depth=0):
    if obj is None or isinstance(obj,(str,int,float,bool)): return obj
    if isinstance(obj,(list,tuple)): return [serialize(v,depth+1) for v in obj]
    if isinstance(obj,dict): return {str(k):serialize(v,depth+1) for k,v in obj.items()}
    if depth>5: return str(obj)
    result={}
    for name in dir(obj):
        if name.startswith('_'):continue
        try: value=getattr(obj,name)
        except Exception:continue
        if not callable(value):result[name]=serialize(value,depth+1)
    return result or str(obj)

def describe(descriptors):
    rows=[]
    for desc in descriptors:
        row={'status':str(desc.status),'memory_replica':desc.is_memory_replica()}
        if row['memory_replica']:row['memory']=serialize(desc.get_memory_descriptor())
        rows.append(row)
    return rows

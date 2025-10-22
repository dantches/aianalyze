def near_duplicates(rows,key,max_hamming=3):
    seen=set(); out=[]
    for r in rows:
        k=key(r)
        if k in seen: continue
        seen.add(k); out.append(r)
    return out

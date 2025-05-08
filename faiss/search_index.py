import faiss

model_path = ""
embbeding = [] 
top_n = 10  
model = faiss.read_index(model_path)
dis, label = model.search(x=embbeding, k=min(top_n, model.ntotal))
label, dis = label[0], dis[0]
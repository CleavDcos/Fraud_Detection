from services.vector_store import search_similar

query = "Your account is suspended, click here"

results = search_similar(query)

for r in results:
    print(r)
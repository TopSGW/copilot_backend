from pymilvus import connections, db


connections.connect(host="127.0.0.1", port=19530)


databases = db.list_database()
print("Databases:", databases)

from nebula3.Config import Config
from nebula3.gclient.net import ConnectionPool

def main():
    # Configure the connection pool
    config = Config()
    config.max_connection_pool_size = 10

    connection_pool = ConnectionPool()
    if not connection_pool.init([('127.0.0.1', 9779)], config):
        print("Failed to initialize the connection pool!")
        return

    # Create a session with the Nebula Graph server
    session = connection_pool.get_session('root', 'nebula')
    
    try:
        # Define your nGQL command
        query = 'CREATE SPACE IF NOT EXISTS llamaindex_nebula_property_graph(vid_type=FIXED_STRING(256));'
        # Execute the command
        result = session.execute(query)
        print("Query executed successfully!")
        print(result)
    except Exception as e:
        print("Error executing query:", e)
    finally:
        # Always release the session and close the connection pool
        session.release()
        connection_pool.close()

if __name__ == "__main__":
    main()

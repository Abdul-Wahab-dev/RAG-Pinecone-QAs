from django.db import connections

def execute_raw_query(query, database='secondary'):
    """
    Execute a raw SQL query on the specified database.

    Parameters:
    - query (str): The raw SQL query to execute.
    - database (str): The name of the database to use (default is 'default').

    Returns:
    - list: A list of tuples containing the query results.
    """
    with connections[database].cursor() as cursor:
        cursor.execute(query)
        columns = [col[0] for col in cursor.description]  # Get column names
        rows = cursor.fetchall()
        
        # Convert rows to a list of dictionaries
        results = [dict(zip(columns, row)) for row in rows]
    
    return results
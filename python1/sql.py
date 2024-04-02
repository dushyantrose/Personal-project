import mysql.connector
 
# Replace these values with your MySQL server information
host = "localhost"
user = "root"
password = "1234@"
 
# Connect to MySQL
connection = mysql.connector.connect(
    host=host,
    user=user,
    password=password
)
 
# Create a cursor object to interact with the database
cursor = connection.cursor()
 
# Create a database
cursor.execute("CREATE DATABASE IF NOT EXISTS olympics")
print("Database created successfully")
 
# Switch to the created database
cursor.execute("USE olympics")
 
# Create a table
cursor.execute("""
    CREATE TABLE IF NOT EXISTS shooting (
        id INT AUTO_INCREMENT PRIMARY KEY,
        name VARCHAR(255),
        age INT
    )
""")
print("Table created successfully")
 
# Close the cursor and connection
cursor.close()
connection.close()





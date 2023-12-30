"""
Every hour:

1. sync with icloud and s3
2. check for differences (new files, deleted files, modified files)
    - check for new files if they are not in the database
    - check for deleted files if there are files in the database that are not in the folder
    - check for modified files if the MD5 hash of the file is different (keep these hashes in the database)
    
    a. if there are new files, add them to the database and set that as not indexed and not reviewed
    b. if there are deleted files, remove them from the database and the index
    c. if there are modified files, update the database (just metadata most likely) and set that as not indexed and not reviewed
3. index the files that are not indexed
4. review the files that are not reviewed

"""

import psycopg2


class Database:
    def __init__(self):
        pass
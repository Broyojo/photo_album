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

import argparse
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import magic
import psycopg2
from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS
from pillow_heif import register_avif_opener, register_heif_opener

register_avif_opener()
register_heif_opener()

@dataclass
class Media:
    path: str
    hash: str
    mimetype: str
    longitude: Optional[float] = None
    latitude: Optional[float] = None
    timestamp: Optional[datetime] = None
        
    @classmethod
    def from_path(cls: "Media", path: str) -> "Media":
        mimetype = magic.from_file(path, mime=True)
        with open(path, "rb") as f:
            hash = hashlib.md5(f.read()).hexdigest()
        instance: "Media" = cls(path=path, hash=hash, mimetype=mimetype)
        with Image.open(path) as image:
            exif = image.getexif()
        tags = {value: key for key, value in TAGS.items()}
         
        if (dt := exif.get(tags["DateTime"])) is not None:
            try:
                instance.timestamp = datetime.strptime(dt, "%Y:%m:%d %H:%M:%S")
            except ValueError:
                pass

        gps_data = {
            GPSTAGS.get(key, key): value
            for key, value in exif.get_ifd(tags["GPSInfo"]).items()
        }
        
        if gps_data:
            if "GPSLatitude" in gps_data and "GPSLatitudeRef" in gps_data and "GPSLongitude" in gps_data and "GPSLongitudeRef" in gps_data:
                lat_values = gps_data["GPSLatitude"]
                lat_ref = gps_data["GPSLatitudeRef"]
                instance.latitude = (lat_values[0] + lat_values[1] / 60 + lat_values[2] / 3600) * (-1 if lat_ref in ["W", "S"] else 1)
                
                lon_values = gps_data["GPSLongitude"]
                lon_ref = gps_data["GPSLongitudeRef"]
                instance.longitude = (lon_values[0] + lon_values[1] / 60 + lon_values[2] / 3600) * (-1 if lon_ref in ["W", "S"] else 1)
        return instance

class Database:
    def __init__(self, table_name: str, reset=False):
        self.table_name = table_name
        self.conn = psycopg2.connect(dbname="postgres", user="postgres", host="localhost") # just connect to default postgres database
        self.cur = self.conn.cursor()
        
        if reset:
            self.drop()
        
        self.cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id SERIAL PRIMARY KEY,
            path TEXT NOT NULL UNIQUE,
            hash TEXT NOT NULL,
            mimetype TEXT,
            location GEOMETRY(Point, 4326),
            timestamp TIMESTAMP,
            indexed BOOLEAN DEFAULT FALSE,
            reviewed BOOLEAN DEFAULT FALSE
        )
        """)
        self.conn.commit()
    
    def drop(self):
        self.cur.execute(f"""
        DROP TABLE IF EXISTS {self.table_name}
        """)
        self.conn.commit()
    
    def execute(self, query: str) -> list[tuple]:
        self.cur.execute(query)
        return self.cur.fetchall()

    def commit(self):
        self.conn.commit()

    def __len__(self):
        self.cur.execute(f"""
        SELECT COUNT(*) FROM {self.table_name}
        """)
        return self.cur.fetchone()[0]

    def __contains__(self, media: Media) -> bool:
        self.cur.execute(f"""
        SELECT EXISTS(
            SELECT 1 FROM {self.table_name} 
            WHERE path = '{media.path}' OR hash = '{media.hash}')
        """)
        return self.cur.fetchone()[0]

    def add(self, media: Media):
        if media not in self:
            self.cur.execute(f"""
            INSERT INTO {self.table_name} (path, hash, mimetype, location, datetime)
            VALUES ('{media.path}', '{media.hash}', '{media.mimetype}', ST_SetSRID(ST_MakePoint('{media.longitude}', '{media.latitude}'), 4326), '{media.timestamp}')
            """)
            self.conn.commit()
    
    def remove(self, media: Media):
        if media in self:
            self.cur.execute(f"""
            DELETE FROM {self.table_name} WHERE path = '{media.path}'
            """)
            self.conn.commit()

    def close(self):
        self.cur.close()
        self.conn.close()

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--download_path", type=str, default="media", required=False, help="Directory to download photos to")
    parser.add_argument("--table_name", type=str, default="photos", required=False, help="Name of the database table")
    parser.add_argument("--s3_bucket", type=str, required=False, help="S3 bucket to sync with")
    parser.add_argument("--reset", action="store_true", required=False, help="Reset the database")
    parser.add_argument("--embedding_batch_size", type=int, default=128, required=False, help="Batch size for embedding")
    parser.add_argument("--index_path", type=str, default="index.usearch", required=False, help="Path to store index")
    args = parser.parse_args()
    
    db = Database(args.table_name, reset=args.reset)
    print(f"Database has {len(db)} entries")
    
    db.close()

if __name__ == "__main__":
    main()
    
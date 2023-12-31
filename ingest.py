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
import os
import time
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pool
from typing import Optional

import magic
import numpy as np
import psycopg2
import torch
from imagebind import data
from imagebind.models.imagebind_model import ModalityType, imagebind_huge
from PIL import Image, ImageFile
from PIL.ExifTags import GPSTAGS, TAGS
from pillow_heif import register_avif_opener, register_heif_opener
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from usearch.index import Index

register_avif_opener()
register_heif_opener()

ImageFile.LOAD_TRUNCATED_IMAGES = True

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
        
        if mimetype.startswith("image"):
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
                    instance.latitude = float((lat_values[0] + lat_values[1] / 60 + lat_values[2] / 3600) * (-1 if lat_ref in ["W", "S"] else 1))
                    
                    lon_values = gps_data["GPSLongitude"]
                    lon_ref = gps_data["GPSLongitudeRef"]
                    instance.longitude = float((lon_values[0] + lon_values[1] / 60 + lon_values[2] / 3600) * (-1 if lon_ref in ["W", "S"] else 1))
            return instance
        elif mimetype.startswith("video"):
            return instance

@dataclass
class Row:
    id: int
    media: Media
    indexed: bool
    reviewed: bool
    inappropriate: bool

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
            reviewed BOOLEAN DEFAULT FALSE,
            inappropriate BOOLEAN DEFAULT FALSE
        )
        """)
        self.conn.commit()
    
    def drop(self):
        self.cur.execute(f"""
        DROP TABLE IF EXISTS {self.table_name}
        """)
        self.conn.commit()
    
    def search(self, query: str, vars: Optional[tuple] = None) -> list[Row]:
        self.cur.execute(query, vars)
        rows = []
        for row in self.cur.fetchall():
            id, path, hash, mimetype, location, timestamp, indexed, reviewed, inappropriate = row

            if location is not None:
                self.cur.execute(f"SELECT ST_Y(location::geometry), ST_X(location::geometry) FROM {self.table_name} WHERE id = %s", (id,))
                lat, lon = self.cur.fetchone()
            else:
                lat, lon = None, None

            media = Media(path=path, hash=hash, mimetype=mimetype, longitude=lon, latitude=lat, timestamp=timestamp)

            rows.append(Row(id=id, media=media, indexed=indexed, reviewed=reviewed, inappropriate=inappropriate))
        return rows
    
    def execute_batched(self, query: str, vars: list[tuple] = None):
        self.cur.executemany(query, vars)
        self.conn.commit()
    
    def execute(self, query: str, vars: Optional[tuple] = None) -> list[tuple]:
        self.cur.execute(query, vars)
        self.conn.commit()
        try:
            return self.cur.fetchall()
        except psycopg2.ProgrammingError:
            return []

    def __len__(self):
        self.cur.execute(f"""
        SELECT COUNT(*) FROM {self.table_name}
        """)
        return self.cur.fetchone()[0]

    def __contains__(self, media: Media) -> bool:
        self.cur.execute(f"""
        SELECT EXISTS(
            SELECT 1 FROM {self.table_name} 
            WHERE path = %s OR hash = %s)
        """, (media.path, media.hash))
        return self.cur.fetchone()[0]

    def add(self, media: Media):
        if media not in self:
            self.cur.execute(f"""
            INSERT INTO {self.table_name} (path, hash, mimetype, location, timestamp)
            VALUES (%s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326), %s)
            """, (media.path, media.hash, media.mimetype, media.longitude, media.latitude, media.timestamp))
            self.conn.commit()
    
    def remove(self, media: Media):
        if media in self:
            self.cur.execute(f"""
            DELETE FROM {self.table_name} WHERE path = %s
            """, (media.path,))
            self.conn.commit()

    def close(self):
        self.cur.close()
        self.conn.close()
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

class MediaDataset(Dataset):
    def __init__(self, db: Database):
        self.rows = db.search(f"SELECT * FROM {db.table_name} WHERE indexed = FALSE")
    
    def __len__(self):
        return len(self.rows)
    
    def __getitem__(self, idx):
        row: Row = self.rows[idx]
        if row.media.mimetype.startswith("image"):
            input = {ModalityType.VISION: data.load_and_transform_vision_data([row.media.path], "cpu"), "ids": row.id}
            return input
        if row.media.mimetype.startswith("video"):
            input = {ModalityType.VISION: data.load_and_transform_video_data([row.media.path], "cpu"), "ids": row.id}
            return input
        return {}

def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--photos_dir", type=str, default="photos", required=False, help="Directory to store photos in")
    parser.add_argument("--table_name", type=str, default="photos", required=False, help="Name of the table in the database")
    parser.add_argument("--s3_bucket", type=str, required=False, help="S3 bucket to sync with")
    parser.add_argument("--reset", action="store_true", required=False, help="Reset the database")
    parser.add_argument("--embedding_batch_size", type=int, default=16, required=False, help="Batch size for embedding")
    parser.add_argument("--index_path", type=str, default="index.usearch", required=False, help="Path to store index")
    args = parser.parse_args()
        
    db = Database(args.table_name, reset=args.reset)
    old_length = len(db)
    print(f"Database has {len(db)} entries")
    
    index = Index(ndim=1024, metric="cos", dtype="f32")
    
    if os.path.exists(args.index_path):
        index.load(args.index_path)
        print(f"Loaded index with {len(index)} entries")
    
    print("Downloading new iCloud photos...")
    time.sleep(1)
    print("Backing up to S3...")
    time.sleep(1)

    modified = 0
    for root, dirs, files in os.walk(args.photos_dir):
        for file in tqdm(files):
            path = os.path.join(root, file)
            media: Media = Media.from_path(path)
            result = db.search(f"SELECT * FROM {db.table_name} WHERE path = %s", (media.path,))
            if len(result) == 0: # new file, so try to add it
                db.add(media)
                continue
            if result[0].media.hash != media.hash:
                modified += 1
                db.execute(f"UPDATE {db.table_name} SET hash = %s, mimetype = %s, location = ST_SetSRID(ST_MakePoint(%s, %s), 4326), timestamp = %s, indexed = False, reviewed = FALSE, inappropriate = FALSE WHERE id = %s", (media.hash, media.mimetype, media.longitude, media.latitude, media.timestamp, result[0].id))
                if len(index) > 0:
                    index.remove(result[0].id)
            
    print(f"Added {len(db) - old_length} new entries and modified {modified}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = imagebind_huge(pretrained=True).to(device).eval()
    print("Loaded ImageBind model")
        
    ds = MediaDataset(db)
    dl = DataLoader(ds, batch_size=args.embedding_batch_size, num_workers=os.cpu_count())
    for batch in tqdm(dl):
        inputs = {k: v.to(device).float() for k, v in batch.items() if k != "ids"}
        with torch.inference_mode():
            embeddings = torch.cat(list(model(inputs).values()), dim=0).cpu().numpy().astype(np.float32)
        keys = batch["ids"].cpu().numpy().astype(np.int32)
        index.add(keys, embeddings)
        index.save(args.index_path)
        db.execute_batched(f"UPDATE {db.table_name} SET indexed = TRUE WHERE id = %s", [(int(id),) for id in keys])
    db.close()

if __name__ == "__main__":
    main()
    
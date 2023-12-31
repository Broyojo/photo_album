import argparse
import asyncio
import io
from datetime import datetime
from typing import List, Optional

import geocoder
import torch
from fastapi import FastAPI, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from imagebind import data
from imagebind.models.imagebind_model import ModalityType, imagebind_huge
from PIL import Image
from pillow_heif import register_avif_opener, register_heif_opener
from pydantic import BaseModel, Field
from usearch.index import Index

from ingest import Database

register_heif_opener()
register_avif_opener()

# parser = argparse.ArgumentParser()
# parser.add_argument("--table_name", type=str, default="photos", required=False, help="Name of the table to use")
# parser.add_argument("--index_path", type=str, default="index.usearch", required=False, help="Path to store index")
# args = parser.parse_args()

TABLE_NAME = "photos"
INDEX_PATH = "index.usearch"

app = FastAPI(title="Photo Database")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

db = Database(TABLE_NAME)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = imagebind_huge(pretrained=True).to(device).eval()

index = Index(ndim=1024, metric="cos", dtype="f32")
index.load(INDEX_PATH)
print(f"Loaded index with {index.size} vectors")

def convert_heic_to_jpeg(path):
    with Image.open(path) as image:
        bytes = io.BytesIO()
        image.save(bytes, format="jpeg")
        bytes = bytes.getvalue()
    return bytes


@app.get("/api/media/{id}")
async def read_media(id: str) -> FileResponse:
    result = db.execute(f"SELECT path, mimetype FROM {TABLE_NAME} WHERE id = %s", (id,))
    print(result)
        
    if len(result) == 0:
        raise HTTPException(status_code=404, detail="Image not found")
    
    path = result[0][0]
    mimetype = result[0][1]
    
    try:
        if "heic" in mimetype:
            bytes = await asyncio.get_event_loop().run_in_executor(None, convert_heic_to_jpeg, path)
            return StreamingResponse(io.BytesIO(bytes), media_type="image/jpeg")
        return FileResponse(path)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/metadata/{id}")
async def read_metadata(id: str) -> JSONResponse:
    raise NotImplementedError()

class SearchRequest(BaseModel):
    query: Optional[str] = None
    files: List[UploadFile] = []
    count: int = Field(10, ge=1)
    address: Optional[str] = None
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    radius: Optional[float] = Field(None, ge=0)
    start_datetime: Optional[datetime] = None
    end_datetime: Optional[datetime] = None

@app.post("/api/search")
async def search(request: SearchRequest) -> JSONResponse:
    inputs = {}
    inputs[ModalityType.TEXT] = data.load_and_transform_text([request.query], device)

    with torch.inference_mode():
        embedding = model(inputs)[ModalityType.TEXT].detach().cpu().numpy().astype("float32")

    matches = index.search(embedding, count=int(request.count), exact=True)
    matched_ids = [int(match.key) for match in matches]
    print(matched_ids)
    match_id_to_score = {int(match.key): float(1 - match.distance) for match in matches}

    query = f"SELECT * FROM {TABLE_NAME} WHERE id = ANY(%s)"
    query_conditions = [matched_ids]

    if request.address:
        geocoding_result = geocoder.osm(request.address)
        if geocoding_result.ok:
            request.latitude = geocoding_result.lat
            request.longitude = geocoding_result.lng
    
    print(request.latitude, request.longitude, request.radius)
    
    if request.latitude and request.longitude and request.radius:
        query += " AND ST_DWithin(location, ST_MakePoint(%s, %s)::geography, %s)"
        query_conditions.extend([request.longitude, request.latitude, request.radius])

    if request.start_datetime or request.end_datetime:
        if request.start_datetime and request.end_datetime:
            query += " AND timestamp BETWEEN %s AND %s"
            query_conditions.extend([request.start_datetime, request.end_datetime])
        elif request.start_datetime:
            query += " AND timestamp >= %s"
            query_conditions.append(request.start_datetime)
        elif request.end_datetime:
            query += " AND timestamp <= %s"
            query_conditions.append(request.end_datetime)

    print(query.format(*query_conditions))

    results = db.search(query, tuple(query_conditions))

    return sorted([
        {
            "id": result.id,
            "type": result.media.mimetype.split("/")[0],
            "similarity": match_id_to_score[result.id]
        } for result in results
    ], key=lambda x: x["similarity"], reverse=True)
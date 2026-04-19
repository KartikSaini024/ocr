import medical_inference
import os
import shutil
import json
import uuid
import asyncio
from typing import Dict
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
import csv

app = FastAPI(title="Medical OCR PDL Extractor")

pdl_mapping = {}
pdl_value_mapping = {}

def parse_import_export_codes(codes_str):
    mapping = {}
    if not codes_str:
        return mapping
    # Split by comma or newline, but be careful with commas inside labels if any
    # Usually they are "val = label, val = label" or "val = label\nval = label"
    import re
    parts = re.split(r'[,\n]', codes_str)
    for part in parts:
        if "=" in part:
            try:
                val, label = part.split("=", 1)
                mapping[val.strip()] = label.strip()
            except:
                continue
    return mapping

def load_mapping():
    try:
        with open("PACE/Primary Data List.csv", "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ref = row.get("NEW REF NUMB", "").strip()
                field = row.get("DATA ELEMENT", "").strip()
                codes = row.get("IMPORT/EXPORT CODES", "").strip()
                
                if ref:
                    if field:
                        pdl_mapping[ref] = field
                    if codes:
                        pdl_value_mapping[ref] = parse_import_export_codes(codes)
    except Exception as e:
        print(f"Error loading mapping: {e}")

load_mapping()


# Storage for background jobs
class JobManager:
    def __init__(self):
        self.jobs: Dict[str, Dict] = {}

    def create_job(self):
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "status": "processing",
            "logs": asyncio.Queue(),
            "result": None,
            "error": None
        }
        return job_id

    async def add_log(self, job_id, message):
        if job_id in self.jobs:
            await self.jobs[job_id]["logs"].put(message)

    def set_result(self, job_id, result):
        if job_id in self.jobs:
            self.jobs[job_id]["result"] = result
            self.jobs[job_id]["status"] = "completed"
            # Signal end of logs
            import json
            asyncio.create_task(self.add_log(job_id, json.dumps({"status": "completed", "message": "Done"})))
            asyncio.create_task(self.add_log(job_id, "EOF"))

    def set_error(self, job_id, error):
        if job_id in self.jobs:
            self.jobs[job_id]["error"] = error
            self.jobs[job_id]["status"] = "error"
            # Signal end of logs
            import json
            asyncio.create_task(self.add_log(job_id, json.dumps({"status": "failed", "message": str(error)})))
            asyncio.create_task(self.add_log(job_id, "EOF"))

job_manager = JobManager()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

async def run_background_process(job_id: str, temp_path: str):
    try:
        loop = asyncio.get_running_loop()
        def sync_callback(msg):
            # Safe way to call async code from another thread
            import json
            asyncio.run_coroutine_threadsafe(
                job_manager.add_log(job_id, json.dumps({"status": "processing", "message": msg})),
                loop
            )

        # Run the long-running inference
        result = await asyncio.to_thread(
            medical_inference.process_document, 
            temp_path, 
            progress_callback=sync_callback
        )
        
        job_manager.set_result(job_id, result)
        
    except Exception as e:
        import json
        await job_manager.add_log(job_id, json.dumps({"status": "failed", "message": str(e)}))
        job_manager.set_error(job_id, str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/api/upload")
async def upload_pdf(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        return JSONResponse(status_code=400, content={"error": "Only PDFs are supported."})
    
    # Save the file temporarily
    temp_dir = "temp_uploads"
    os.makedirs(temp_dir, exist_ok=True)
    temp_path = os.path.join(temp_dir, f"{uuid.uuid4()}_{file.filename}")
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    job_id = job_manager.create_job()
    background_tasks.add_task(run_background_process, job_id, temp_path)
    
    return JSONResponse(content={"status": "accepted", "job_id": job_id})

@app.get("/api/progress/{job_id}")
async def get_progress(job_id: str):
    if job_id not in job_manager.jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})

    async def event_generator():
        queue = job_manager.jobs[job_id]["logs"]
        while True:
            msg = await queue.get()
            if msg == "EOF":
                break
            yield f"data: {msg}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/result/{job_id}")
async def get_result(job_id: str):
    if job_id not in job_manager.jobs:
        return JSONResponse(status_code=404, content={"error": "Job not found"})
    
    job = job_manager.jobs[job_id]
    if job["status"] == "processing":
        return JSONResponse(content={"status": "processing"})
    elif job["status"] == "error":
        return JSONResponse(status_code=500, content={"error": job["error"]})
    else:
        return JSONResponse(content={"status": "success", "data": job["result"]})

@app.get("/api/mapping")
async def get_mapping():
    return JSONResponse(content={"mapping": pdl_mapping, "values": pdl_value_mapping})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)

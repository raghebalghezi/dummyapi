import shutil
import uvicorn
from typing import Optional
from fastapi import FastAPI, UploadFile, File, Query

app = FastAPI(title="Digitala",
                description="In-progress API developed by Aalto-Speech  Group",
                version="0.0.1")

@app.post("/")
async def root(file: UploadFile = File(...), 
                prompt: Optional[str] = None, 
                lang: str = Query("fin", enum=["fin","sv"]),
                task: str = Query("freeform", enum=["freeform","readaloud"]),
                key: Optional[str] = None):

    if key != "aalto":
        return {"Error": "Authentication Error"}

    
    KEEP_FILE = False

    if KEEP_FILE:
        with open(f'{file.filename}', "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    if task == 'readaloud':
            return {"file_name": file.filename,
                    "Language": lang,
                    "Task": task,
                    "prompt":  prompt,
                    "GOP_score": 0.7}

    return {"file_name": file.filename,
            "Language": lang,
            "Task": task,
            "prompt":  prompt,
            "Transcript": "Pretend that this is a trancript",
            "Fluency": {"score":3, "speech_rate":3.12, "mean_f1": 0.19},
            "TaskAchievement": 0.4,
            "Accuracy": {"score":2, "lexical_profile":3.12, "nativeity": 0.4},
            "Holistic": 2.4
            }


if __name__ == '__main__':
    uvicorn.run("app:app", reload=True)
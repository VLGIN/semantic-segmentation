import uvicorn
from fastapi import FastAPI, File, UploadFile
from dotenv import dotenv_values
from fastapi.responses import FileResponse

CONFIG = dotenv_values("./.env")
app = FastAPI()

@app.post("/upload_video/")
def upload_file(file: UploadFile = File(...)):
    pass

@app.get("/get_file/")
def get_file():
    pass


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)
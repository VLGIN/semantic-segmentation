import uvicorn
from fastapi import FastAPI, File, UploadFile, Request, Response, Header
from fastapi.staticfiles import StaticFiles
from dotenv import dotenv_values
from fastapi.responses import FileResponse

CONFIG = dotenv_values("./.env")

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.post("/upload_file/")
def upload_file(file: UploadFile = File(...)):
    pass

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)
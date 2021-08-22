import sys
import os
from fastapi import FastAPI

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.dirname(__file__) + "../routers")  # TODO Is this needed?
from .routers import summa

app = FastAPI()


app.include_router(summa.router)


@app.get("/")
def read_root():
    return {"Hello": "World"}

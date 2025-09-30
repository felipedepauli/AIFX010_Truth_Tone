import os
from uvicorn import run

def main():
    host = os.getenv("APP_HOST","0.0.0.0")
    port = int(os.getenv("APP_PORT","8000"))
    run("dspfusion.service.api:app", host=host, port=port, reload=False)

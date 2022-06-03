import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

from pdds.grid_view_pdds import grid_view_pdds

app = FastAPI()


class GridViewSpace(BaseModel):
    space_id: str = "75195"
    date_str: str = "07-08-2021"
    label_map: dict[int, str] = {
        0: "pgf_daylight_healthy",
        1: "pgf_daylight_unhealthy",
        2: "empty",
        3: "purple",
    }
    input_preprocessing_enum: str = "CONVNEXT"
    model_name: str = "convnext_onnx"
    model_version: str = ""
    batch_size: int = 16
    server_url: str = "localhost:8000"
    run_clustering: bool = False
    cluster_eps: int = 1000
    cluster_min_samples: int = 5
    slice_width: int = 1024
    slice_height: int = 1024
    debug: bool = False
    debug_folder: str = "/temp/debug_images_output"


@app.post("/pdds/grid_view")
async def predict_unhealthy(space: GridViewSpace):
    data = space.dict()
    results = grid_view_pdds(**data)
    return results


if __name__ == "__main__":
    uvicorn.run(app, port=8080, host="0.0.0.0")

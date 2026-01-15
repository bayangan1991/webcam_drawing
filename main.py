from fastapi import FastAPI
from starlette.responses import StreamingResponse

from camera import VideoCamera
from canvas import Canvas

app = FastAPI()
camera = VideoCamera()
game = Canvas(camera)


@app.get("/")
def feed(flip: bool | None = None) -> StreamingResponse:
    if flip is not None:
        game.camera_flip = flip
    game.clear_canvas()
    return StreamingResponse(
        game.gen(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )

from pydantic import BaseModel

class PredictRequest(BaseModel):
    street: str
    postcode: str
    area: str
    type: str
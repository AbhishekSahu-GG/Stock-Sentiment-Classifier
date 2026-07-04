from pydantic import BaseModel

class NewsInput(BaseModel):
    headline: str

class BatchInput(BaseModel):
    headlines: list[str]

class SentimentOutput(BaseModel):
    headline: str
    sentiment: str
    confidence: float

class BatchOutput(BaseModel):
    results: list[SentimentOutput]

class HealthResponse(BaseModel):
    status: str
    model: str
    version: str
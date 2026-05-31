from pydantic import BaseModel, Field


class QuotedInt(BaseModel):
    value: int = Field(ge=0)
    quote: str = Field(min_length=1, max_length=200)


class QuotedFloat(BaseModel):
    value: float = Field(ge=0)
    quote: str = Field(min_length=1, max_length=200)

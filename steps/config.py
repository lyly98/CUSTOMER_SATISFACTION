from pydantic import BaseModel

class ModelNameConfig(BaseModel):
    """
    model configs
    """

    model_name: str = "LinearRegression"

from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    github_token: str | None = Field(default=None)
    nvd_api_key: str | None = Field(default=None)
    work_dir: str = Field(default='.')
settings = Settings()

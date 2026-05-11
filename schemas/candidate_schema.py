from pydantic import BaseModel, Field, ConfigDict
from schemas.user_schema import UserSchema






class ResumeSchema(BaseModel):
    id: str = Field(..., description="简历ID")
    file_path: str = Field(..., description="简历存储路径")
    uploader: UserSchema = Field(..., description="建立上传者")

    model_config = ConfigDict(from_attributes=True)

class ResumeUploadRespSchema(BaseModel):
    resume: ResumeSchema | None = Field(..., description="简历信息")

from datetime import timedelta

from pydantic_settings import BaseSettings
from pydantic import computed_field,Field
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class Settings(BaseSettings):
    # DB
    DB_USERNAME: str = "postgres"
    DB_PASSWORD: str = "root"
    DB_HOST: str = "127.0.0.1"
    DB_PORT: int = 5432
    DB_NAME: str = "hr_system"
    DB_AGENT_NAME: str = "hr_system_agent"


    JWT_SECRET_KEY:str="sfsdfsadfsdfjg"
    # access_token：一般是2个小时过期
    # refresh_token：30天过期
    JWT_ACCESS_TOKEN_EXPIRES:timedelta = timedelta(days=365)#JWTtoken过期时间
    JWT_REFRESH_TOKEN_EXPIRES: timedelta = timedelta(days=365)
    #redis配置
    REDIS_HOST: str = "127.0.0.1"
    REDIS_PORT: int = 6379

    #邀请码过期时间
    INVITE_CODE_EXPIRE: int = 60*60*24*2
    #邮箱相关配置
    MAIL_USERNAME: str = Field(..., validation_alias="MAIL_USERNAME")
    MAIL_PASSWORD: str = Field(..., validation_alias="MAIL_PASSWORD")
    MAIL_FROM: str = Field(..., validation_alias="MAIL_USERNAME")
    MAIL_PORT: int = 587
    MAIL_SERVER: str = "smtp.qq.com"
    MAIL_FROM_NAME: str = "智能招聘"
    MAIL_STARTTLS: bool = True
    MAIL_SSL_TLS: bool = False
    # 邮箱机器人配置
    EMAIL_BOT_IMAP_HOST: str = "imap.qq.com"
    EMAIL_BOT_SMTP_HOST: str = "smtp.qq.com"
    EMAIL_BOT_EMAIL: str = Field(..., validation_alias="MAIL_USERNAME")
    EMAIL_BOT_PASSWORD: str = Field(..., validation_alias="MAIL_PASSWORD")
    # 阿里云百炼平台的API_KEY
    DASHSCOPE_API_KEY: str = Field(..., validation_alias="DASHSCOPE_API_KEY")
    #钉钉相关配置
    DINGTALK_CLIENT_ID: str = Field(..., validation_alias="DINGTALK_APP_KEY")
    DINGTALK_CLIENT_SECRET: str = Field(..., validation_alias="DINGTALK_APP_SECRET")
    #前端和后端的域名
    BACKEND_BASE_URL: str = "https://cornmeal-front-decency.ngrok-free.dev"
    # 简历上传存储路径
    RESUME_DIR: str = os.path.join(BASE_DIR, "upload")
    # RAG 知识库文件目录（手动放入 markdown 文件）
    DATA_DIR: str = os.path.join(BASE_DIR, "data")
    # ChromaDB 向量数据库持久化路径
    CHROMA_DB_PATH: str = os.path.join(BASE_DIR, "chroma_db")
    # Paddle OCR Access Token
    PADDLE_OCR_ACCESS_TOKEN: str = Field(..., validation_alias="PADDLE_OCR_ACCESS_TOKEN")



    @computed_field
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+psycopg://{self.DB_USERNAME}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"

    @computed_field
    @property
    def DATABASE_AGENT_URL(self) -> str:
        return f"postgresql://{self.DB_USERNAME}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_AGENT_NAME}"


settings = Settings()
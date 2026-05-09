from fastapi import FastAPI

from routers.user_router import router as user_router
from fastapi.middleware.cors import CORSMiddleware
from redis import asyncio as aioredis
from settings import settings
from contextlib import asynccontextmanager
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

@asynccontextmanager
async def lifespan(_: FastAPI):
    #yield之前的代码是程序运行前执行的
    redis = aioredis.from_url(
        f"redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}",#redis连接地址
        encoding="utf-8",
        decode_responses=True,#解码
    )
    #初始化 FastAPI-Cache(缓存中间件，用于缓存 API 接口的响应结果)
    redis_backend = RedisBackend(redis)
    FastAPICache.init(redis_backend, prefix="fastapi-cache")
    yield# 这里应用开始运行
    #yield之后的代码是程序即将退出之前执行的
    await redis.close()

app = FastAPI(lifespan=lifespan)

#允许跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],


)
app.include_router(user_router)

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}



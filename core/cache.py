from pydantic import EmailStr,BaseModel

from core.single import SingletonMeta
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

from settings import settings


class InviteInfoSchema(BaseModel):
    email:EmailStr
    department_id:str
    invite_code:str



class HRCache(metaclass=SingletonMeta):
    invite_prefix = "invite:"

    def __init__(self):
        self.cache_backend: RedisBackend = FastAPICache.get_backend()

    async def set(self, key, value, ex: int):
        await self.cache_backend.set(key, value, expire=ex if ex else None)

    async def get(self, key):
        value = await self.cache_backend.get(key)
        return value

    async def delete(self, key):
        await self.cache_backend.clear(key)
    #将邀请信息以 JSON 格式存储到 Redis
    async def set_invite_info(self, invite_info: InviteInfoSchema):
        key = f"{self.invite_prefix}{invite_info.email}"
        await self.set(key, invite_info.model_dump_json(),ex= settings.INVITE_CODE_EXPIRE)
    #根据邮箱获取邀请信息，并自动转换为 InviteInfoSchema 对象
    async def get_invite_info(self, email:str)->InviteInfoSchema|None:
        key=f"{self.invite_prefix}{email}"
        invite_info_json = await self.get(key)
        if invite_info_json is not None:
            invite_info = InviteInfoSchema.model_validate_json(invite_info_json)#将json转化为InviteInfoSchema(pydantic模型)
            return invite_info
        return None

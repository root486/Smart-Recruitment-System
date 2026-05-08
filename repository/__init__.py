from sqlalchemy.ext.asyncio import AsyncSession

#Repo中是和数据库交互的代码
class BaseRepo:
    def __init__(self, session: AsyncSession):
        self.session = session
from . import BaseRepo
from models.user import UserModel,DingdingUserModel,DepartmentModel

from sqlalchemy import select,delete
from typing import Sequence
class UserRepo(BaseRepo):
    #创建用户
    async def create_user(self,user_data:dict)->UserModel:
        user =UserModel(**user_data)
        self.session.add(user)
        return user
    #通过id查找用户
    async def get_by_id(self,user_id:str)->UserModel|None:
        user=await self.session.scalar(
            select(UserModel).where(UserModel.id==user_id)
        )
        return user
    #通过email查找用户
    async def get_by_email(self,email:str)->UserModel|None:
        user=await self.session.scalar(
            select(UserModel).where(UserModel.email==email)
        )
        return user
    #获取用户列表
    async def get_user_list(self,page:int=1,size:int=10,department_id:str|None=None)->Sequence[UserModel]| None:
        stmt = select(UserModel)
        # 如果在获取用户时有输入部门id则添加部门id的过滤条件
        if department_id:
            stmt = stmt.where(UserModel.department_id == department_id)
        # 分页查询
        limit = size  # 每页显示多少条
        offset = (page - 1) * size  # 跳过前面的条数
        stmt = stmt.limit(limit).offset(offset).order_by(UserModel.created_at.desc())
        users = await self.session.scalars(stmt)
        return users.all()
    # 获取钉钉用户并存储到数据库当中
    async def set_dingding_user(self,user_id:str,dingding_user_data:dict)->DingdingUserModel:
        user = await self.get_by_id(user_id)
        if not user:
            raise ValueError("设置钉钉的用户不存在")
        dingding_user=await self.session.scalar(
            select(DingdingUserModel).where(DingdingUserModel.user_id==user_id)

        )
        #有则更新
        if dingding_user:
            for key,value in dingding_user_data.items():
                setattr(dingding_user,key,value)
        else:
            dingding_user = DingdingUserModel(**dingding_user_data,user_id=user_id)
            self.session.add(dingding_user)
        return dingding_user
class DepartmentRepo(BaseRepo):
    #创建部门
    async def create_department(self,department_data:dict)->DepartmentModel:
        department = DepartmentModel(**department_data)
        self.session.add(department)
        return department
    #根据id查找部门
    async def get_by_id(self,department_id:str)->DepartmentModel|None:
        department = await self.session.scalar(
            select(DepartmentModel).where(DepartmentModel.id==department_id)
        )
        return department
    #根据名字查找部门
    async def get_by_name(self,department_name:str)->DepartmentModel|None:
        department = await self.session.scalar(
            select(DepartmentModel).where(DepartmentModel.name==department_name)
        )
        return department
    #获取部门列表
    async def get_department_list(self):
        departments =await self.session.scalars(
            select(DepartmentModel)
        )
        return departments.all()
    #删除部门
    async def delete_department(self,department_id:str)->None:
        await self.session.execute(
            delete(DepartmentModel).where(DepartmentModel.id==department_id)
        )



from fastapi import Depends, APIRouter, HTTPException
from models.user import UserModel
from dependencies import get_current_user, get_session_instance
from models import AsyncSession
from repository.user_repo import UserRepo

from schemas.position_schema import PositionCreateSchema, PositionRespSchema,PositionListRespSchema
from repository.position_repo import PositionRepo
from schemas import ResponseSchema
from fastapi import status

router = APIRouter(prefix="/position", tags=["position"])

@router.post("/create", summary="创建职位", response_model=PositionRespSchema)
async def create_position(
    position_data: PositionCreateSchema,
    current_user: UserModel = Depends(get_current_user),
    session: AsyncSession = Depends(get_session_instance),
):
    async with session.begin():
        position_repo = PositionRepo(session=session)
        position_dict = position_data.model_dump()#先把pydantic对象转为dict
        position_dict['creator_id'] = current_user.id#添加创建者id
        position_dict['department_id'] = current_user.department.id#当前用户所属部门id
        position = await position_repo.create_position(position_dict)
        return {"position": position}


@router.get('/list', summary="职位列表", response_model=PositionListRespSchema)
async def get_position_list(
        page: int = 1,
        size: int = 10,
        current_user: UserModel = Depends(get_current_user),
        session: AsyncSession = Depends(get_session_instance),
):
    async with session.begin():
        position_repo = PositionRepo(session=session)
        user_repo = UserRepo(session=session)

        # 如果是HR用户，重新查询并预加载managed_departments
        if current_user.is_hr and not current_user.is_superuser:
            hr_user = await user_repo.get_by_id_with_departments(current_user.id)
            positions = await position_repo.get_possition_list(hr_user, page=page, size=size)
        # 如果是普通用户
        else:

            positions = await position_repo.get_possition_list(current_user, page=page, size=size)

        return {"positions": positions}



@router.delete("/delete/{position_id}", summary="删除职位")
async def delete_position(
    position_id: str,
    session: AsyncSession = Depends(get_session_instance),
    current_user: UserModel = Depends(get_current_user),#登录以后才可进行删除操作
):
    async with session.begin():
        position_repo = PositionRepo(session=session)
        position = await position_repo.get_by_id(position_id)
        if not position:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="该职位不存在！")
        # 如果是superuser，那么可以直接删除；否则就只能是所属部门的人才能删除
        if (not current_user.is_superuser) and (position.department_id != current_user.department.id):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="没有权限执行操作！")

        # 执行删除操作
        try:
            await position_repo.delete_position(position_id)
        except Exception as e:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="职位删除失败！")
        return ResponseSchema()



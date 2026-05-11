from fastapi import Depends, APIRouter, HTTPException
from models.user import UserModel
from dependencies import get_current_user, get_session_instance
from models import AsyncSession
from repository.position_repo import PositionRepo
from schemas.position_schema import PositionCreateSchema, PositionRespSchema#, PositionListRespSchema
#from repository.position_repo import PositionRepo
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



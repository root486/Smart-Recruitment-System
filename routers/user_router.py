from fastapi import APIRouter,Depends
from schemas.user_schema import UserLoginSchema,UserLoginRespSchema
from dependencies import get_session_instance, get_auth_handler,AuthHandler
from models import AsyncSession
from repository.user_repo import UserRepo
from models.user import UserModel
from fastapi.exceptions import HTTPException
from fastapi import status

router =APIRouter (prefix= "/user",tags=["user"])

#登录功能
@router.post("/login",description="登录",response_model=UserLoginRespSchema)
async def login(
    login_data:UserLoginSchema,
    session:AsyncSession = Depends(get_session_instance),
    auth_handler:AuthHandler = Depends(get_auth_handler)
):
    async with session.begin():
        #1.验证用户是否存在
        user_repo = UserRepo(session)
        user :UserModel=await user_repo.get_by_email(str(login_data.email))
        if not user:
            raise HTTPException(status.HTTP_400_BAD_REQUEST,detail="用户不存在")
        #2.验证密码是否正确
        if not user.check_password(login_data.password):
            raise HTTPException(status.HTTP_400_BAD_REQUEST,detail="密码错误")
        #3.确认用户存在，密码正确后，生成JWT token
        tokens=auth_handler.encode_login_token(user.id)
        return {
            "access_token":tokens["access_token"],
            "refresh_token":tokens["refresh_token"],
            "user":user
        }





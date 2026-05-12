import os.path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, status, BackgroundTasks, Query

from core.cache import HRCache
from dependencies import get_session_instance, get_current_user, get_cache_instance
from models import AsyncSession
from models.user import UserModel
from settings import settings
from uuid import uuid4
import aiofiles
from core.pdf import WordToPdfConverter
from loguru import logger
from repository.candidate_repo import ResumeRepo#,CandidateRepo
from schemas.candidate_schema import ResumeUploadRespSchema, ResumePaseSchema, ResumeParseTaskRespSchema, ResumeParseTaskInfoRespSchema#, CandidateCreateSchema, CandidateStatusUpdateSchema, CandidateAIScoreRespSchema
from core.ocr import PaddleOcr
from tasks import ocr_parse_resume_task
from schemas import ResponseSchema
from repository.position_repo import PositionRepo
from repository.user_repo import UserRepo
#from tasks import run_candidate_agent
#from schemas.candidate_schema import CandidateSchema, CandidateListSchema
from schemas.position_schema import PositionSchema
from schemas.user_schema import UserSchema
from models.candidate import CandidateStatusEnum
#from repository.interview_repo import InterviewRepo
from models.interview import InterviewResultEnum
from models.interview import InterviewModel
#from repository.candidate_repo import CandidateAIScoreRepo
from pathlib import Path


router = APIRouter(prefix="/candidate", tags=["candidate"])

# 上传简历
@router.post("/resume/upload", summary="上传简历", response_model=ResumeUploadRespSchema)
async def resume_upload(
    file: UploadFile = File(...),
    session: AsyncSession = Depends(get_session_instance),
    current_user: UserModel = Depends(get_current_user),
):
    # 1. 校验文件类型
    # 简历：图片、pdf、word
    allowed_mime_types = [
        "application/pdf",
        "application/msword",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "image/jpeg",
        "image/png",
        "image/jpg",
    ]
    if file.content_type not in allowed_mime_types:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="该文件不支持！")

    # 2. 保存文件
    #获取目录
    resume_dir = settings.RESUME_DIR

    #1.获取文件后缀名
    file_extension = os.path.splitext(file.filename)[-1]
    #2.重新生成文件名
    unique_filename = f"{uuid4()}{file_extension}"
    #3.得到文件路径
    file_path = os.path.join(resume_dir, unique_filename)

    # 异步保存文件
    try:
        async with aiofiles.open(file_path, mode="wb") as fp:
            content = await file.read(1024)
            #写入文件
            while content:
                await fp.write(content)
                content = await file.read(1024)
    finally:
        await fp.close()

    # 3. 如果是word文档，那么就转化成pdf
    if file_extension == ".doc" or file_extension == ".docx":
        pdf_path = file_path.replace(file_extension, ".pdf")
        converter = WordToPdfConverter(
            word_path=file_path,#word路径
            output_pdf_path=pdf_path,#输出的pdf路径
        )
        try:
            await converter.convert()#进行转换
            file_path = pdf_path
        except Exception as e:
            logger.error(f"Word转PDF失败：{e}")

    # 4. 将简历数据存储到数据库中
    async with session.begin():
        resume_repo = ResumeRepo(session=session)
        # 做一个更改，在数据库中只保存文件名
        file_name = Path(file_path).name
        resume = await resume_repo.create_resume(file_path=file_name, uploader_id=current_user.id)
    return {"resume": resume}

# 1. 发起了一个简历识别的请求，创建一个后台任务，把task_id返回给前端
# 2. 前端就可以通过task_id来获取这个任务的执行结果，当执行结果为success时，那么就返回解析后的数据
@router.post("/resume/parse", summary="简历解析",response_model=ResumeParseTaskRespSchema)
async def parse_resume(
    resume_data: ResumePaseSchema,
    background_tasks: BackgroundTasks,
    _: UserModel = Depends(get_current_user),
):
    # 创建一个识别简历的后台任务
    task_id = str(uuid4())
    background_tasks.add_task(ocr_parse_resume_task, resume_id=resume_data.resume_id, task_id=task_id)
    return {"task_id": task_id}

@router.get("/resume/parse/{task_id}", summary="获取任务状态", response_model=ResumeParseTaskInfoRespSchema)
async def get_task_status(
    task_id: str,
    cache: HRCache = Depends(get_cache_instance),
    _: UserModel = Depends(get_current_user)
):
    task_info = await cache.get_task_info(task_id)
    return task_info.model_dump()



@router.get("/resume/ocr/test")
async def resume_ocr_test(

):
    file_path = os.path.join(settings.RESUME_DIR, "8651671b-2e1b-4879-bf71-6dffc27fad80.pdf")
    paddle_ocr = PaddleOcr()
    job_id= await paddle_ocr.create_job(file_path)
    json_url = await paddle_ocr.poll_for_state(job_id)
    contents=await paddle_ocr.fetch_parsed_contents(json_url)
    logger.info(contents)
    return "success"
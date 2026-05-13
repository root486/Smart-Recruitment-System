from http.client import responses

from langchain.agents import create_agent
from schemas.candidate_schema import CandidateSchema
from schemas.position_schema import PositionSchema
from schemas.user_schema import UserSchema
from langgraph.graph.message import BaseMessage
from .llms import qwen_llm,deepseek_llm
from .prompts import CANDIDATE_PROCESS_SYSTEM_PROMPT,SCORE_FOR_CANDIDATE_SYSTEM_PROMPT,SCORE_FOR_CANDIDATE_USER_PROMPT
from langchain.agents.middleware import ModelFallbackMiddleware,SummarizationMiddleware
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from settings import settings
from pydantic import BaseModel
from typing import Annotated, List, TypeVar, Optional
from langgraph.graph.message import add_messages
from langchain.tools import tool,ToolRuntime
from schemas.agent_schema import AgentCandidateScoreSchema
from langchain_core.prompts import PromptTemplate
from models import AsyncSession,AsyncSessionFactory
from repository.candidate_repo import CandidateAIScoreRepo, CandidateRepo
from models.candidate import CandidateStatusEnum
from loguru import logger


T = TypeVar("T")
#数据处理方法：当有新值（right）时更新状态，否则保持原值（left）
def assign_state_property(left: T, right: Optional[T]) -> T:
    return right if right is not None else left

# checkpointer会自动保存state中的数据到数据库中
class CandidateAgentState(BaseModel):
    messages: Annotated[List[BaseMessage], add_messages]
    candidate: Annotated[CandidateSchema, assign_state_property]
    position: Annotated[PositionSchema, assign_state_property]
    interviewer: Annotated[UserSchema, assign_state_property]
@tool
async def score_for_candidate(
    runtime: ToolRuntime[CandidateAgentState]
):
    """
    根据职位信息，给职位上的候选人进行评分。
    评分结果会存入数据库中，并根据评分结果修改候选人状态。
    return：
    - str | None: 评分结果的JSON字符串，如果评分失败则返回None
    """
    candidate: CandidateSchema = runtime.state['candidate']
    position: PositionSchema = runtime.state['position']

    score_agent = create_agent(
        model=qwen_llm,
        system_prompt=SCORE_FOR_CANDIDATE_SYSTEM_PROMPT,
        middleware=[ModelFallbackMiddleware(first_model=deepseek_llm)],
        response_format=AgentCandidateScoreSchema
    )
    user_prompt_template = PromptTemplate.from_template(SCORE_FOR_CANDIDATE_USER_PROMPT)
    user_prompt = user_prompt_template.invoke({
        "candidate": candidate.model_dump_json(),
        "position": position.model_dump_json()
    })
    response = await score_agent.ainvoke({
        "messages": [{
            "role": "user",
            "content": user_prompt.text
        }]
    })
    candiate_score: AgentCandidateScoreSchema = response['structured_response']
    # 将得分情况存储到数据库中
    async with AsyncSessionFactory() as session:
        async with session.begin():
            try:
                score_repo = CandidateAIScoreRepo(session)
                candidate_repo = CandidateRepo(session)
                # 1. 将得分插入到数据库中
                await score_repo.create_candidate_score(
                    candidate_id=candidate.id,
                    candidate_score_dict=candiate_score.model_dump()
                )
                # 2. 判断得分情况，如果超过8分，那么修改状态为AI_PASS，否则就是AI_FAILED
                status = CandidateStatusEnum.AI_FILTER_FAILED
                if candiate_score.overall_score > 8:
                    status = CandidateStatusEnum.AI_FILTER_PASSED

                await candidate_repo.update_candidate_status(candidate_id=candidate.id, status=status)
            except Exception as e:
                logger.error(e)
                raise ValueError(e)
                return f"得分工具执行失败，错误信息为：{e}"
    return f"得分工具执行成功！该候选人得分为：{candiate_score.model_dump_json()}"




class CandidateProcessAgent:
    def __init__(self,
            candidate: CandidateSchema|None=None,
            position: PositionSchema|None=None,
            interviewer: UserSchema|None=None
):
        self.candidate = candidate
        self.position = position
        self.interviewer = interviewer
        self._checkpointer=None

    async def ainvoke(self, messages: list[BaseMessage], thread_id: str):
        assert self._checkpointer is not None
        agent = create_agent(
            model=deepseek_llm,
            system_prompt=CANDIDATE_PROCESS_SYSTEM_PROMPT,
            state_schema=CandidateAgentState,
            middleware=[
                ModelFallbackMiddleware(first_model=deepseek_llm),
                SummarizationMiddleware(
                    model=deepseek_llm,
                    trigger=("tokens", 50000),
                    keep=("tokens", 10000)
                )
            ],
            tools=[score_for_candidate

            ],
            checkpointer=self._checkpointer
        )
        response = await agent.ainvoke({
            "messages": messages,
            "candidate": self.candidate,
            "position": self.position,
            "interviewer": self.interviewer,
        }, {"thread_id": thread_id})
        return response

    #进入上下文时执行
    async def __aenter__(self):
        # langchain如果大模型选择了一个工具，那么这个工具消息后的消息必须是工具调用后的结果，否则会报错
        self._checkpointer_conn = AsyncPostgresSaver.from_conn_string(settings.DATABASE_AGENT_URL)
        self._checkpointer = await self._checkpointer_conn.__aenter__()#建立数据库链接
        await self._checkpointer.setup()#创建表
        return self
    #退出上下文时执行
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self._checkpointer_conn.__aexit__(exc_type, exc_val, exc_tb)

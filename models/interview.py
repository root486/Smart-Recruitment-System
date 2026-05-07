import enum
from datetime import datetime
from typing import Optional

from sqlalchemy import Text, DateTime, Enum, ForeignKey
from sqlalchemy.orm import Mapped, mapped_column, relationship

from . import BaseModel
from .user import UserModel
from .candidate import CandidateModel

#面试结果
class InterviewResultEnum(str, enum.Enum):
    PASSED = "PASSED"#面试成功
    FAILED = "FAILED"#面试失败

#面试相关信息模型
class InterviewModel(BaseModel):
    __tablename__ = "interviews"

    scheduled_time: Mapped[Optional[datetime]] = mapped_column(DateTime)
    feedback: Mapped[Optional[str]] = mapped_column(Text)#反馈
    result: Mapped[Optional[InterviewResultEnum]] = mapped_column(Enum(InterviewResultEnum))

    candidate_id: Mapped[str] = mapped_column(ForeignKey("candidates.id"), unique=True)
    interviewer_id: Mapped[str] = mapped_column(ForeignKey("users.id"))#面试官id

    candidate: Mapped["CandidateModel"] = relationship(back_populates="interviews")
    interviewer: Mapped["UserModel"] = relationship()

#通过候选人获取候选人面试信息
CandidateModel.interviews = relationship("InterviewModel", back_populates="candidate")

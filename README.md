# 智能招聘系统 - 后端（前端：https://github.com/root486/Smart-Recruitment-System-frontend）

基于 **FastAPI + LangChain + ChromaDB** 的 AI 智能招聘平台后端，实现从简历解析、RAG+Agent 评分、自动面试邮件沟通到钉钉日程创建的全流程自动化。

## 技术栈

| 层级 | 技术 |
|------|------|
| Web 框架 | FastAPI + Uvicorn |
| 数据库 | PostgreSQL（SQLAlchemy 异步 + Alembic 迁移） |
| 缓存 | Redis + fastapi-cache |
| AI 框架 | LangChain + LangGraph（Agent 工作流） |
| 向量数据库 | ChromaDB（RAG 知识库） |
| 嵌入 & 重排 | DashScope text-embedding-v4 / gte-rerank-v2 |
| 大模型 | DeepSeek + 千问（OCR 降级） |
| OCR | PaddleOCR → 千问 VL（自动降级） |
| 定时任务 | APScheduler（邮件轮询） |
| 第三方集成 | 钉钉 OAuth / 日程 / QQ 邮箱 |

## 功能模块

| 模块 | 说明 |
|------|------|
| AI 智能评分 | LangGraph Agent 自动对候选人简历进行多维评分（技术、项目、软技能等），结合 RAG 知识库上下文 |
| 邮件自动化 | 邮箱机器人自动发送面试邀请、接收回复、确认时间 |
| 钉钉日程 | 查询面试官闲忙时段，自动创建面试日程 |
| 简历解析 | OCR 识别 + LLM 结构化抽取候选人信息 |
| RAG 检索 | 混合检索（稠密向量 + BM25）+ RRF 融合 + DashScope 重排序 |
| 用户管理 | 邀请码注册、部门分配、JWT 认证 |
| 仪表盘 | 候选人数据统计 |


## 项目结构

```
├── agents/          # LangGraph Agent（候选人评分、面试流程）
├── alembic/         # 数据库迁移脚本
├── core/            # 核心模块（钉钉、邮件、OCR、PDF、认证）
├── data/            # RAG 知识库 Markdown 文件
├── models/          # SQLAlchemy ORM 模型
├── rag/             # RAG 检索系统（嵌入、分块、重排、存储）
├── repository/      # 数据访问层
├── routers/         # API 路由
├── scheduler/       # 定时任务
├── schemas/         # Pydantic 数据模型
├── settings/        # 全局配置
├── templates/       # HTML 模板
├── utils/           # 工具函数
├── main.py          # 应用入口
├── init_data.py     # 初始化种子数据
└── requirements.txt # Python 依赖
```

## 快速开始

### 环境要求

- Python 3.13+
- PostgreSQL（本地 127.0.0.1:5432，用户名 `postgres`，密码 `root`）
- Redis（本地 127.0.0.1:6379）

### 安装

```bash

虚拟环境的Python版本为3.13

pip install -r requirements.txt
```

### 配置环境变量

详情请看.env.example

### 初始化数据库

```bash
# 1. 手动创建两个 PostgreSQL 数据库
#   hr_system       — 主业务库
#   hr_system_agent — LangGraph checkpoint

# 2. 执行迁移
alembic upgrade head

# 3. 插入种子数据（部门和用户）
python init_data.py
```

### 启动

```bash
python -m uvicorn main:app --reload
```

访问 http://127.0.0.1:8000/docs 查看 Swagger API 文档。


## 默认账号

| 角色 | 邮箱 | 密码 |
|------|------|------|
| 管理员 | boss@qq.com | 111111 |
| HR | hr@qq.com | 111111 |

> 更多账号见 `init_data.py`


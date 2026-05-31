# 高级Python工程师 - 详细岗位说明书

## 技术栈要求

- Python 3.10+，精通 asyncio 异步编程，熟悉协程、事件循环机制
  - 能独立设计高并发异步任务调度方案，熟练使用 asyncio.gather、TaskGroup、信号量等原语
  - 理解 GIL 限制下的多进程 + 协程混合架构，有 ProcessPoolExecutor + asyncio 实战经验
- FastAPI 框架，3年以上实战经验，熟悉中间件、依赖注入、生命周期管理
  - 能设计可复用的 Depends 依赖链，理解 scope（request/singleton）选择策略
  - 有自定义中间件开发经验（如请求日志、限流、全局异常捕获）
  - 熟悉 lifespan 事件管理，能在 startup/shutdown 中合理编排资源初始化与释放
- PostgreSQL 关系型数据库，熟练编写复杂 SQL，有慢查询优化和索引设计经验
  - 能使用 EXPLAIN ANALYZE 分析执行计划，针对慢查询设计联合索引或部分索引
  - 熟悉窗口函数、CTE、横向子查询等高级 SQL 特性
  - 有数据库迁移工具（Alembic）的实战经验，能编写幂等的 migration 脚本
- Redis 缓存，熟悉常用数据结构（String、Hash、List、Set、Sorted Set）及应用场景
  - 能设计缓存策略（Cache-Aside / Write-Through / Write-Behind），合理设置 TTL
  - 有分布式锁、限流器、消息队列等 Redis 高级场景的落地经验
- Docker、K8s 有生产环境部署经验，能编写 Dockerfile 和 docker-compose
  - 掌握多阶段构建优化镜像体积，理解 layer 缓存机制
  - 能编写 K8s Deployment/Service/ConfigMap/Ingress 配置，有 Helm chart 经验者优先
- SQLAlchemy 2.0 异步 ORM，熟悉 relationship、lazy loading 策略
  - 能设计复杂模型关系（一对多、多对多、自引用），合理选择加载策略避免 N+1
  - 有 AsyncSession 生命周期管理经验，理解 sessionmaker 与上下文传播机制
- 加分项：LangChain / LangGraph 生态、钉钉 API 对接、AI Agent 开发经验

## 日常工作

- 负责后端核心模块的架构设计与开发实现
- 参与代码评审，把控代码质量，输出可维护的高质量代码
- 与 AI 团队协作，将大模型能力集成到招聘业务中
- 优化系统性能，解决高并发场景下的瓶颈问题
- 编写技术文档，推动团队技术规范落地

## 软技能期待

- 能独立负责复杂模块的技术方案设计与落地
- 良好的跨部门沟通能力，能与产品、前端、算法团队高效协作
- 有较强的自驱力，能主动发现系统问题并推动解决
- 习惯编写清晰的技术文档和代码注释

## 业务领域

- 智能招聘 / HR SaaS 系统
- 有钉钉或企业微信集成经验者优先
- 有 AI Agent（LLM 调用、工具编排、状态管理）开发经验者优先
- 有邮件系统对接经验者优先

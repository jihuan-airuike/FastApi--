from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime
from sqlalchemy.orm import declarative_base
from datetime import datetime

engine = create_engine(
    'mysql+pymysql://root:123456@localhost:3306/查询记录?charset=utf8mb4'
)
Base = declarative_base()

class QuestionLog(Base):
    __tablename__ = '查询'
    id          = Column(Integer, primary_key=True, index=True, autoincrement=True)
    question    = Column(String(255), nullable=False)
    result      = Column(Text, nullable=False)
    created_at  = Column(DateTime, default=datetime.now)

Base.metadata.create_all(bind=engine)

import os
from datetime import datetime
from typing import Optional
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
Base = declarative_base()
class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    model_name = Column(String(255), nullable=False, index=True)
    model_version = Column(String(50), nullable=True)
    input_features = Column(JSON, nullable=False)
    prediction = Column(Integer, nullable=False)
    confidence = Column(Float, nullable=False)
    request_id = Column(String(36), nullable=True)
    latency_ms = Column(Float, nullable=True)
    def __repr__(self):
        return f"<PredictionLog(id={self.id}, prediction={self.prediction}, confidence={self.confidence:.2f})>"
_engine = None
_SessionLocal = None
def get_engine():
    global _engine
    if _engine is None:
        database_url = os.getenv(
            "DATABASE_URL",
            "postgresql://mlflow_user:mlflow_password@postgres:5432/mlflow_db"
        )
        _engine = create_engine(database_url, pool_pre_ping=True)
        Base.metadata.create_all(_engine)
    return _engine
def get_session() -> Session:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=get_engine())
    return _SessionLocal()

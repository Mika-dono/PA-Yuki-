#!/usr/bin/env python3
"""
Gestion de Base de Données pour AURA
"""

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy import Column, String, Integer, Float, DateTime, JSON, Text
from datetime import datetime
from typing import Optional
import logging

logger = logging.getLogger("AURA_Database")

Base = declarative_base()

class PortfolioSnapshot(Base):
    """Modèle pour les snapshots de portefeuille"""
    __tablename__ = "portfolio_snapshots"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    total_value = Column(Float)
    risk_score = Column(Float)
    asset_allocation = Column(JSON)
    analysis_results = Column(JSON)
    blockchain_hash = Column(String)

class TransactionRecord(Base):
    """Modèle pour les enregistrements de transaction"""
    __tablename__ = "transaction_records"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, nullable=False, index=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    asset = Column(String)
    action = Column(String)  # BUY, SELL, HOLD
    amount = Column(Float)
    price = Column(Float)
    aura_score = Column(Float)
    blockchain_tx_hash = Column(String)
    market_context = Column(JSON)

class UserProfile(Base):
    """Modèle pour les profils utilisateurs"""
    __tablename__ = "user_profiles"
    
    user_id = Column(String, primary_key=True)
    risk_tolerance = Column(String)
    investment_horizon = Column(String)
    financial_goals = Column(JSON)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

class DatabaseManager:
    """Gestionnaire de base de données"""
    
    def __init__(self, database_url: str):
        self.engine = create_async_engine(database_url, echo=False)
        self.async_session = sessionmaker(
            self.engine, class_=AsyncSession, expire_on_commit=False
        )
    
    async def save_portfolio_snapshot(self, snapshot_data: Dict) -> str:
        """Sauvegarde un snapshot de portefeuille"""
        async with self.async_session() as session:
            snapshot = PortfolioSnapshot(**snapshot_data)
            session.add(snapshot)
            await session.commit()
            return snapshot.id
    
    async def get_user_portfolio_history(self, user_id: str, limit: int = 100) -> List[PortfolioSnapshot]:
        """Récupère l'historique des portefeuilles d'un utilisateur"""
        async with self.async_session() as session:
            result = await session.execute(
                select(PortfolioSnapshot)
                .where(PortfolioSnapshot.user_id == user_id)
                .order_by(PortfolioSnapshot.timestamp.desc())
                .limit(limit)
            )
            return result.scalars().all()
    
    async def save_transaction(self, transaction_data: Dict) -> str:
        """Sauvegarde une transaction"""
        async with self.async_session() as session:
            transaction = TransactionRecord(**transaction_data)
            session.add(transaction)
            await session.commit()
            return transaction.id
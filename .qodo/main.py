"""
API Principale AURA - Service Enterprise
"""

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import jwt
from pydantic import BaseModel, validator
import logging
import asyncio

from core.pa_engine.advanced_pa import AURAPAEngine, FinancialAsset, AssetType, MarketContext
from core.blockchain.transaction_manager import BlockchainManager
from core.security.threat_detection import ThreatDetector
from core.data.market_data import MarketDataService

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AURA_API")

app = FastAPI(
    title="AURA Financial Intelligence API",
    description="API Enterprise pour la gestion intelligente de patrimoine",
    version="2.1.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

security = HTTPBearer()

class AssetModel(BaseModel):
    symbol: str
    asset_type: str
    quantity: float
    current_price: float
    purchase_price: float
    volatility: float
    sector: str
    region: str
    market_cap: Optional[float] = None
    beta: Optional[float] = None
    
    @validator('quantity')
    def validate_quantity(cls, v):
        if v <= 0:
            raise ValueError('La quantité doit être positive')
        return v

class PortfolioAnalysisRequest(BaseModel):
    assets: List[AssetModel]
    user_profile: Dict
    market_context: Optional[Dict] = None

class TransactionRequest(BaseModel):
    asset: str
    action: str 
    amount: float
    price: float
    user_id: str

pa_engine = None
blockchain_manager = None
threat_detector = None
market_data_service = None

@app.on_event("startup")
async def startup_event():
    """Initialisation des services au démarrage"""
    global pa_engine, blockchain_manager, threat_detector, market_data_service
    
    logger.info("🚀 Démarrage des services AURA...")
    
    config = {
        'redis_url': 'redis://localhost:6379',
        'database_url': 'postgresql://user:pass@localhost/aura',
        'jwt_secret': 'votre-secret-super-securise',
        'blockchain_rpc': 'https://mainnet.infura.io/v3/votre-projet'
    }
    
    try:
        pa_engine = AURAPAEngine(config)
        await pa_engine.initialize()
        
        blockchain_manager = BlockchainManager(config)
        threat_detector = ThreatDetector(config)
        market_data_service = MarketDataService(config)
        
        logger.info("✅ Tous les services AURA sont opérationnels")
        
    except Exception as e:
        logger.error(f"❌ Erreur initialisation: {e}")
        raise

@app.on_event("shutdown")
async def shutdown_event():
    """Nettoyage à l'arrêt"""
    logger.info("🛑 Arrêt des services AURA...")
    if hasattr(pa_engine, 'http_session') and pa_engine.http_session:
        await pa_engine.http_session.close()

# Routes API
@app.get("/")
async def root():
    """Endpoint de santé"""
    return {
        "status": "operational",
        "service": "AURA Financial Intelligence",
        "version": "2.1.0",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.get("/health")
async def health_check():
    """Check de santé complet"""
    services_status = {
        "pa_engine": pa_engine is not None,
        "blockchain": blockchain_manager is not None,
        "security": threat_detector is not None,
        "market_data": market_data_service is not None
    }
    
    all_healthy = all(services_status.values())
    
    return {
        "status": "healthy" if all_healthy else "degraded",
        "services": services_status,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/api/v1/analyze-portfolio")
async def analyze_portfolio(
    request: PortfolioAnalysisRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Analyse complète du portefeuille"""
    try:
        user_id = await verify_token(credentials.credentials)
        
        assets = []
        for asset_model in request.assets:
            asset = FinancialAsset(
                symbol=asset_model.symbol,
                asset_type=AssetType(asset_model.asset_type),
                quantity=asset_model.quantity,
                current_price=asset_model.current_price,
                purchase_price=asset_model.purchase_price,
                volatility=asset_model.volatility,
                sector=asset_model.sector,
                region=asset_model.region,
                market_cap=asset_model.market_cap,
                beta=asset_model.beta
            )
            assets.append(asset)
        
        market_context = MarketContext(
            timestamp=datetime.now(),
            market_volatility=0.18,  
            interest_rates={"USD": 0.05},
            economic_indicators={},
            geopolitical_risk=0.3
        )
        
        analysis = await pa_engine.analyze_portfolio_comprehensive(
            assets, request.user_profile, market_context
        )
        
        tx_hash = await blockchain_manager.record_analysis(
            user_id, analysis, "portfolio_analysis"
        )
        
        return {
            "success": True,
            "analysis": analysis.__dict__,
            "blockchain_tx": tx_hash,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Erreur analyse portefeuille: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de l'analyse: {str(e)}"
        )

@app.post("/api/v1/execute-transaction")
async def execute_transaction(
    request: TransactionRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Exécute une transaction sécurisée"""
    try:
        user_id = await verify_token(credentials.credentials)
        
        threat_analysis = await threat_detector.analyze_transaction(
            request.dict(), user_id
        )
        
        if threat_analysis.threat_level.value >= 3: 
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Transaction bloquée pour raisons de sécurité: {threat_analysis.description}"
            )
        
        tx_result = await blockchain_manager.execute_transaction(
            user_id, request.asset, request.action, 
            request.amount, request.price
        )
        
        return {
            "success": True,
            "transaction_hash": tx_result.tx_hash,
            "block_number": tx_result.block_number,
            "gas_used": tx_result.gas_used,
            "threat_analysis": threat_analysis.__dict__
        }
        
    except Exception as e:
        logger.error(f"Erreur transaction: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur transaction: {str(e)}"
        )

@app.get("/api/v1/market-data/{symbol}")
async def get_market_data(symbol: str):
    """Récupère les données marché en temps réel"""
    try:
        data = await market_data_service.get_real_time_data(symbol)
        return {
            "success": True,
            "symbol": symbol,
            "data": data,
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Données non disponibles pour {symbol}: {str(e)}"
        )

@app.get("/api/v1/portfolio-history/{user_id}")
async def get_portfolio_history(user_id: str, days: int = 30):
    """Historique des analyses de portefeuille"""
    try:
        history = await blockchain_manager.get_user_history(user_id, days)
        return {
            "success": True,
            "user_id": user_id,
            "history": history,
            "period_days": days
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Historique non disponible: {str(e)}"
        )

async def verify_token(token: str) -> str:
    """Vérifie et décode le JWT token"""
    try:
        payload = jwt.decode(
            token, 
            app.config['jwt_secret'], 
            algorithms=["HS256"]
        )
        return payload.get("user_id")
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token expiré"
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token invalide"
        )

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Gestionnaire d'erreurs global"""
    logger.error(f"Erreur non gérée: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": "Erreur interne du serveur",
            "detail": str(exc)
        }
    )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=4
    )
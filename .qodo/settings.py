import os
from dataclasses import dataclass
from typing import Dict, Any

@dataclass
class DatabaseConfig:
    url: str = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/aura")
    pool_size: int = 20
    max_overflow: int = 30
    pool_timeout: int = 30

@dataclass
class RedisConfig:
    url: str = os.getenv("REDIS_URL", "redis://localhost:6379/0")
    password: str = os.getenv("REDIS_PASSWORD", "")
    decode_responses: bool = True

@dataclass
class BlockchainConfig:
    rpc_url: str = os.getenv("BLOCKCHAIN_RPC", "https://mainnet.infura.io/v3/your-project")
    chain_id: int = 1  
    gas_limit: int = 300000
    gas_price: str = "50"  

@dataclass
class SecurityConfig:
    jwt_secret: str = os.getenv("JWT_SECRET", "your-super-secret-key")
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440  
    encryption_key: str = os.getenv("ENCRYPTION_KEY", "")

@dataclass
class MarketDataConfig:
    alpha_vantage_key: str = os.getenv("ALPHA_VANTAGE_KEY", "")
    coinbase_api_key: str = os.getenv("COINBASE_API_KEY", "")
    polygon_api_key: str = os.getenv("POLYGON_API_KEY", "")
    update_interval: int = 60  

@dataclass
class ModelConfig:
    risk_model_path: str = "models/risk_model.pth"
    nlp_model_name: str = "microsoft/DialoGPT-medium"
    device: str = "cuda" if os.getenv("GPU_ENABLED", "false").lower() == "true" else "cpu"
    batch_size: int = 32

@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = int(os.getenv("WORKERS", "4"))
    reload: bool = os.getenv("ENVIRONMENT", "production") == "development"
    cors_origins: list = None
    
    def __post_init__(self):
        if self.cors_origins is None:
            self.cors_origins = [
                "https://votre-domaine.com",
                "https://app.votre-domaine.com"
            ]

@dataclass
class MonitoringConfig:
    prometheus_port: int = 9090
    grafana_port: int = 3001
    metrics_interval: int = 15 
    log_level: str = "INFO"

class AURAConfig:
    """Configuration globale AURA"""
    
    def __init__(self):
        self.database = DatabaseConfig()
        self.redis = RedisConfig()
        self.blockchain = BlockchainConfig()
        self.security = SecurityConfig()
        self.market_data = MarketDataConfig()
        self.models = ModelConfig()
        self.api = APIConfig()
        self.monitoring = MonitoringConfig()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convertit la configuration en dictionnaire"""
        return {
            'database': self.database.__dict__,
            'redis': self.redis.__dict__,
            'blockchain': self.blockchain.__dict__,
            'security': self.security.__dict__,
            'market_data': self.market_data.__dict__,
            'models': self.models.__dict__,
            'api': self.api.__dict__,
            'monitoring': self.monitoring.__dict__
        }

config = AURAConfig()
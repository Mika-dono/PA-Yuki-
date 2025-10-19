"""
Moteur PA AURA - Système complet d'intelligence artificielle financière
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer,
    pipeline,
    TrainingArguments,
    Trainer
)
import aiohttp
from redis import asyncio as aioredis
from sqlalchemy.ext.asyncio import AsyncSession
import websockets
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AURA_PA")

class RiskLevel(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"

class AssetType(Enum):
    STOCK = "stock"
    CRYPTO = "crypto"
    BOND = "bond"
    ETF = "etf"
    COMMODITY = "commodity"

@dataclass
class FinancialAsset:
    symbol: str
    asset_type: AssetType
    quantity: float
    current_price: float
    purchase_price: float
    volatility: float
    sector: str
    region: str
    market_cap: Optional[float] = None
    beta: Optional[float] = None

@dataclass
class PortfolioAnalysis:
    total_value: float
    total_invested: float
    total_profit_loss: float
    risk_score: float
    diversification_score: float
    liquidity_score: float
    recommended_actions: List[Dict]
    risk_breakdown: Dict[str, float]
    market_correlation: float
    sharpe_ratio: Optional[float]
    var_95: Optional[float]  

@dataclass
class MarketContext:
    timestamp: datetime
    market_volatility: float
    interest_rates: Dict[str, float]
    economic_indicators: Dict[str, float]
    geopolitical_risk: float

class AdvancedFinancialRiskModel(nn.Module):
    """Modèle de risque financier avancé avec architecture transformer"""
    
    def __init__(self, input_dim=128, hidden_dims=[512, 256, 128, 64], num_heads=8):
        super().__init__()
        
        self.asset_embedding = nn.Linear(input_dim, 256)
        
        self.attention_layer = nn.MultiheadAttention(256, num_heads, dropout=0.1)
        
        layers = []
        prev_dim = 256
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 5))  
        self.network = nn.Sequential(*layers)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
    
    def forward(self, x, attention_mask=None):
        embedded = self.asset_embedding(x)
        
        if attention_mask is not None:
            attn_output, _ = self.attention_layer(
                embedded, embedded, embedded, 
                key_padding_mask=attention_mask
            )
        else:
            attn_output, _ = self.attention_layer(embedded, embedded, embedded)
        
        pooled = torch.mean(attn_output, dim=1)
        
        output = self.network(pooled)
        return torch.sigmoid(output)

class AURAPAEngine:
    """Moteur principal de la Personnalité Artificielle AURA"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        logger.info(f"Initialisation AURA PA sur device: {self.device}")
        
        self.risk_model = self._load_risk_model()
        self.nlp_model, self.tokenizer = self._load_nlp_model()
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis",
            device=0 if torch.cuda.is_available() else -1
        )
        
        self.redis_client = None
        self.http_session = None
        
        self.market_cache = {}
        self.analysis_cache = {}
        
    async def initialize(self):
        """Initialisation asynchrone des services"""
        self.redis_client = await aioredis.from_url(
            self.config['redis_url'],
            encoding="utf-8",
            decode_responses=True
        )
        self.http_session = aiohttp.ClientSession()
        
        await self._load_initial_data()
        
    async def analyze_portfolio_comprehensive(
        self,
        portfolio: List[FinancialAsset],
        user_profile: Dict,
        market_context: MarketContext
    ) -> PortfolioAnalysis:
        """
        Analyse complète du portefeuille avec calculs avancés
        """
        try:
            self._validate_portfolio_data(portfolio)
            
            analysis_tasks = [
                self._calculate_advanced_risk_metrics(portfolio, market_context),
                self._assess_portfolio_diversification(portfolio),
                self._evaluate_liquidity_profile(portfolio),
                self._calculate_var_metrics(portfolio),
                self._analyze_market_correlation(portfolio, market_context),
                self._generate_ai_recommendations(portfolio, user_profile, market_context)
            ]
            
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)

            valid_results = []
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Erreur dans l'analyse: {result}")
                    continue
                valid_results.append(result)
            
            if len(valid_results) != len(analysis_tasks):
                raise Exception("Certaines analyses ont échoué")
            
            risk_score, risk_breakdown = valid_results[0]
            diversification_score = valid_results[1]
            liquidity_score = valid_results[2]
            var_95, sharpe_ratio = valid_results[3]
            market_correlation = valid_results[4]
            recommendations = valid_results[5]
            
            total_value = sum(asset.current_price * asset.quantity for asset in portfolio)
            total_invested = sum(asset.purchase_price * asset.quantity for asset in portfolio)
            total_profit_loss = total_value - total_invested
            
            return PortfolioAnalysis(
                total_value=total_value,
                total_invested=total_invested,
                total_profit_loss=total_profit_loss,
                risk_score=risk_score,
                diversification_score=diversification_score,
                liquidity_score=liquidity_score,
                recommended_actions=recommendations,
                risk_breakdown=risk_breakdown,
                market_correlation=market_correlation,
                sharpe_ratio=sharpe_ratio,
                var_95=var_95
            )
            
        except Exception as e:
            logger.error(f"Erreur dans l'analyse du portefeuille: {e}")
            raise
    
    async def _calculate_advanced_risk_metrics(
        self, 
        portfolio: List[FinancialAsset],
        market_context: MarketContext
    ) -> Tuple[float, Dict[str, float]]:
        """Calcule les métriques de risque avancées"""
        
        portfolio_tensor = self._portfolio_to_tensor(portfolio, market_context)
        
        with torch.no_grad():
            risk_predictions = self.risk_model(portfolio_tensor)
        
        risk_metrics = {
            'market_risk': risk_predictions[0][0].item(),
            'credit_risk': risk_predictions[0][1].item(),
            'liquidity_risk': risk_predictions[0][2].item(),
            'operational_risk': risk_predictions[0][3].item(),
            'systemic_risk': risk_predictions[0][4].item()
        }
        
        weights = [0.3, 0.2, 0.15, 0.15, 0.2]  
        overall_risk = sum(risk_metrics[metric] * weight 
                          for metric, weight in zip(risk_metrics.keys(), weights))
        
        return overall_risk, risk_metrics
    
    async def _assess_portfolio_diversification(
        self, 
        portfolio: List[FinancialAsset]
    ) -> float:
        """Évalue la diversification du portefeuille"""
        
        if len(portfolio) <= 1:
            return 0.0
        
        sector_allocation = {}
        region_allocation = {}
        asset_type_allocation = {}
        
        total_value = sum(asset.current_price * asset.quantity for asset in portfolio)
        
        for asset in portfolio:
            asset_value = asset.current_price * asset.quantity
            weight = asset_value / total_value
            
            sector_allocation[asset.sector] = sector_allocation.get(asset.sector, 0) + weight
            region_allocation[asset.region] = region_allocation.get(asset.region, 0) + weight

            asset_type_allocation[asset.asset_type.value] = asset_type_allocation.get(
                asset.asset_type.value, 0
            ) + weight
        
        def herfindahl_index(allocations):
            return sum(weight ** 2 for weight in allocations.values())
        
        sector_diversification = 1 - herfindahl_index(sector_allocation)
        region_diversification = 1 - herfindahl_index(region_allocation)
        asset_type_diversification = 1 - herfindahl_index(asset_type_allocation)
        
        diversification_score = (
            sector_diversification * 0.4 +
            region_diversification * 0.3 +
            asset_type_diversification * 0.3
        )
        
        return max(0.0, min(1.0, diversification_score))
    
    async def _calculate_var_metrics(
        self, 
        portfolio: List[FinancialAsset]
    ) -> Tuple[Optional[float], Optional[float]]:
        """Calcule la Value at Risk et le ratio de Sharpe"""
        
        try:
            historical_returns = await self._get_historical_returns(portfolio)
            
            if len(historical_returns) < 30:  
                return None, None
            
            var_95 = np.percentile(historical_returns, 5)
            
            mean_return = np.mean(historical_returns) * 252  
            std_return = np.std(historical_returns) * np.sqrt(252)  
            risk_free_rate = 0.02  
            
            sharpe_ratio = (mean_return - risk_free_rate) / std_return if std_return > 0 else 0
            
            return abs(var_95), sharpe_ratio
            
        except Exception as e:
            logger.warning(f"Impossible de calculer VaR/Sharpe: {e}")
            return None, None
    
    async def _generate_ai_recommendations(
        self,
        portfolio: List[FinancialAsset],
        user_profile: Dict,
        market_context: MarketContext
    ) -> List[Dict]:
        """Génère des recommandations intelligentes basées sur l'IA"""
        
        recommendations = []
        
        for asset in portfolio:
            asset_analysis = await self._analyze_single_asset(asset, market_context)
            
            if asset_analysis['recommendation'] != 'HOLD':
                recommendations.append({
                    'asset': asset.symbol,
                    'action': asset_analysis['recommendation'],
                    'confidence': asset_analysis['confidence'],
                    'reason': asset_analysis['reason'],
                    'expected_impact': asset_analysis['expected_impact']
                })
        
        diversification_recs = await self._generate_diversification_recommendations(
            portfolio, user_profile
        )
        recommendations.extend(diversification_recs)
        
        recommendations.sort(key=lambda x: x['confidence'], reverse=True)
        
        return recommendations
    
    def _portfolio_to_tensor(
        self, 
        portfolio: List[FinancialAsset],
        market_context: MarketContext
    ) -> torch.Tensor:
        """Convertit le portefeuille en tensor pour le modèle ML"""
        
        features = []
        
        for asset in portfolio:
            asset_features = [
                asset.current_price,
                asset.volatility,
                asset.beta or 1.0,
                asset.market_cap or 0,
                market_context.market_volatility,
                market_context.geopolitical_risk,
            ]
            features.append(asset_features)
        
        max_assets = 50  
        while len(features) < max_assets:
            features.append([0.0] * len(features[0]) if features else [0.0] * 6)
        
        return torch.tensor(features[:max_assets], dtype=torch.float32).unsqueeze(0)
    
    async def _get_historical_returns(self, portfolio: List[FinancialAsset]) -> List[float]:
        """Récupère les rendements historiques du portefeuille"""
        # Implémentation simplifiée - à remplacer par vrai appel API
        try:
            returns = []
            for asset in portfolio:
                simulated_return = np.random.normal(0, asset.volatility, 100)
                returns.extend(simulated_return * (asset.quantity * asset.current_price))
            
            return returns
        except:
            return []
    
    def _load_risk_model(self) -> AdvancedFinancialRiskModel:
        """Charge le modèle de risque entraîné"""
        model = AdvancedFinancialRiskModel()
        
        try:
            checkpoint = torch.load(
                'models/risk_model.pth', 
                map_location=self.device
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Modèle de risque chargé avec succès")
        except FileNotFoundError:
            logger.warning("Poids du modèle non trouvés, utilisation du modèle non entraîné")
        
        model.to(self.device)
        model.eval()
        return model
    
    def _load_nlp_model(self):
        """Charge le modèle de langage"""
        try:
            model_name = "microsoft/DialoGPT-medium"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
            model.to(self.device)
            return model, tokenizer
        except Exception as e:
            logger.error(f"Erreur chargement modèle NLP: {e}")
            return None, None
    
    def _validate_portfolio_data(self, portfolio: List[FinancialAsset]):
        """Valide les données du portefeuille"""
        if not portfolio:
            raise ValueError("Portefeuille vide")
        
        for asset in portfolio:
            if asset.quantity <= 0:
                raise ValueError(f"Quantité invalide pour {asset.symbol}")
            if asset.current_price < 0:
                raise ValueError(f"Prix invalide pour {asset.symbol}")

async def main():
    """Exemple d'utilisation complète"""
    
    config = {
        'redis_url': 'redis://localhost:6379',
        'database_url': 'postgresql://user:pass@localhost/aura'
    }
    
    pa_engine = AURAPAEngine(config)
    await pa_engine.initialize()
    
    portfolio = [
        FinancialAsset(
            symbol="AAPL",
            asset_type=AssetType.STOCK,
            quantity=10,
            current_price=150.0,
            purchase_price=140.0,
            volatility=0.25,
            sector="Technology",
            region="US",
            market_cap=2.5e12,
            beta=1.2
        ),
    ]
    
    market_context = MarketContext(
        timestamp=datetime.now(),
        market_volatility=0.18,
        interest_rates={"USD": 0.05, "EUR": 0.03},
        economic_indicators={"GDP_growth": 0.02, "inflation": 0.03},
        geopolitical_risk=0.3
    )
    
    user_profile = {
        "risk_tolerance": "medium",
        "investment_horizon": "long_term",
        "financial_goals": ["retirement", "wealth_growth"]
    }
    
    analysis = await pa_engine.analyze_portfolio_comprehensive(
        portfolio, user_profile, market_context
    )
    
    print(f"Score AURA: {analysis.risk_score:.2f}")
    print(f"Recommandations: {len(analysis.recommended_actions)}")

if __name__ == "__main__":
    asyncio.run(main())
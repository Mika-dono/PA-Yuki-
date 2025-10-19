#!/usr/bin/env python3
"""
Service de Données de Marché pour AURA
"""

import asyncio
import aiohttp
import pandas as pd
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import logging

logger = logging.getLogger("AURA_MarketData")

class MarketDataService:
    """Service de récupération et traitement des données de marché"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Configuration des APIs
        self.api_endpoints = {
            'alpha_vantage': 'https://www.alphavantage.co/query',
            'yahoo_finance': 'https://query1.finance.yahoo.com/v8/finance/chart/',
            'coinbase': 'https://api.coinbase.com/v2/prices/',
            'polygon': 'https://api.polygon.io/v2/'
        }
    
    async def get_real_time_data(self, symbols: List[str]) -> Dict:
        """Récupère les données en temps réel pour plusieurs symboles"""
        tasks = []
        
        for symbol in symbols:
            task = asyncio.create_task(self._fetch_symbol_data(symbol))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        market_data = {}
        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Erreur données {symbol}: {result}")
                market_data[symbol] = {'error': str(result)}
            else:
                market_data[symbol] = result
        
        return market_data
    
    async def get_historical_data(self, symbol: str, period: str = "1mo") -> pd.DataFrame:
        """Récupère les données historiques"""
        cache_key = f"{symbol}_{period}"
        
        # Vérification du cache
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if datetime.now() - timestamp < timedelta(seconds=self.cache_ttl):
                return cached_data
        
        try:
            if self._is_crypto_symbol(symbol):
                data = await self._fetch_crypto_historical(symbol, period)
            else:
                data = await self._fetch_stock_historical(symbol, period)
            
            # Mise en cache
            self.cache[cache_key] = (data, datetime.now())
            
            return data
            
        except Exception as e:
            logger.error(f"Erreur données historiques {symbol}: {e}")
            return pd.DataFrame()
    
    async def get_market_indicators(self, symbol: str) -> Dict:
        """Calcule les indicateurs techniques"""
        historical_data = await self.get_historical_data(symbol, "6mo")
        
        if historical_data.empty:
            return {}
        
        prices = historical_data['close']
        
        return {
            'sma_20': self._calculate_sma(prices, 20),
            'sma_50': self._calculate_sma(prices, 50),
            'rsi': self._calculate_rsi(prices),
            'macd': self._calculate_macd(prices),
            'bollinger_bands': self._calculate_bollinger_bands(prices),
            'volume_avg': historical_data['volume'].tail(20).mean()
        }
    
    async def _fetch_symbol_data(self, symbol: str) -> Dict:
        """Récupère les données d'un symbole"""
        try:
            if self._is_crypto_symbol(symbol):
                return await self._fetch_crypto_data(symbol)
            else:
                return await self._fetch_stock_data(symbol)
        except Exception as e:
            logger.error(f"Erreur récupération {symbol}: {e}")
            raise
    
    async def _fetch_stock_data(self, symbol: str) -> Dict:
        """Récupère les données actions"""
        async with aiohttp.ClientSession() as session:
            url = f"{self.api_endpoints['yahoo_finance']}{symbol}"
            params = {
                'range': '1d',
                'interval': '1m'
            }
            
            async with session.get(url, params=params) as response:
                data = await response.json()
                
                result = data['chart']['result'][0]
                meta = result['meta']
                quotes = result['indicators']['quote'][0]
                
                return {
                    'symbol': symbol,
                    'price': meta['regularMarketPrice'],
                    'change': meta['regularMarketPrice'] - meta['previousClose'],
                    'change_percent': ((meta['regularMarketPrice'] - meta['previousClose']) / meta['previousClose']) * 100,
                    'volume': quotes['volume'][-1] if quotes['volume'] else 0,
                    'timestamp': datetime.now().isoformat(),
                    'currency': meta['currency'],
                    'exchange': meta['exchangeName']
                }
    
    def _calculate_sma(self, prices: pd.Series, window: int) -> float:
        """Calcule la moyenne mobile simple"""
        return prices.tail(window).mean()
    
    def _calculate_rsi(self, prices: pd.Series, window: int = 14) -> float:
        """Calcule le RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi.iloc[-1] if not rsi.empty else 50
    
    def _is_crypto_symbol(self, symbol: str) -> bool:
        """Détermine si un symbole est une cryptomonnaie"""
        crypto_keywords = ['BTC', 'ETH', 'XRP', 'ADA', 'DOT', 'USDT', 'USDC']
        return any(crypto in symbol.upper() for crypto in crypto_keywords)
#!/usr/bin/env python3
"""
Flux de Données Temps Réel pour AURA
"""

import asyncio
import websockets
import json
from typing import Dict, List, Callable
import logging
from datetime import datetime

logger = logging.getLogger("AURA_RealTimeFeeds")

class RealTimeDataFeed:
    """Flux de données temps réel pour les marchés financiers"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.connections = {}
        self.subscribers = {}
        self.is_running = False
    
    async def start_feeds(self):
        """Démarre tous les flux de données"""
        self.is_running = True
        
        feeds = [
            self._start_crypto_feed,
            self._start_stock_feed,
            self._start_forex_feed
        ]
        
        tasks = [asyncio.create_task(feed()) for feed in feeds]
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def subscribe(self, symbol: str, callback: Callable):
        """Abonne un callback à un symbole"""
        if symbol not in self.subscribers:
            self.subscribers[symbol] = []
        
        self.subscribers[symbol].append(callback)
        logger.info(f"Nouvel abonnement pour {symbol}")
    
    async def _start_crypto_feed(self):
        """Démarre le flux cryptomonnaies"""
        uri = "wss://ws-feed.pro.coinbase.com"
        
        try:
            async with websockets.connect(uri) as websocket:
                # Subscription aux symboles populaires
                subscription_msg = {
                    "type": "subscribe",
                    "product_ids": ["BTC-USD", "ETH-USD", "ADA-USD", "DOT-USD"],
                    "channels": ["ticker"]
                }
                
                await websocket.send(json.dumps(subscription_msg))
                
                while self.is_running:
                    try:
                        message = await asyncio.wait_for(websocket.recv(), timeout=30)
                        data = json.loads(message)
                        
                        if data['type'] == 'ticker':
                            await self._process_crypto_ticker(data)
                    
                    except asyncio.TimeoutError:
                        # Reconnexion silencieuse
                        continue
                    except Exception as e:
                        logger.error(f"Erreur flux crypto: {e}")
                        await asyncio.sleep(1)
                        
        except Exception as e:
            logger.error(f"Erreur connexion crypto WebSocket: {e}")
    
    async def _process_crypto_ticker(self, data: Dict):
        """Traite les données ticker crypto"""
        symbol = data['product_id'].replace('-', '')
        price_data = {
            'symbol': symbol,
            'price': float(data['price']),
            'volume': float(data['volume_24h']) if 'volume_24h' in data else 0,
            'change_24h': float(data['open_24h']) if 'open_24h' in data else 0,
            'timestamp': datetime.now().isoformat(),
            'source': 'coinbase'
        }
        
        await self._notify_subscribers(symbol, price_data)
    
    async def _notify_subscribers(self, symbol: str, data: Dict):
        """Notifie tous les abonnés d'un symbole"""
        if symbol in self.subscribers:
            for callback in self.subscribers[symbol]:
                try:
                    await callback(data)
                except Exception as e:
                    logger.error(f"Erreur notification abonné {symbol}: {e}")
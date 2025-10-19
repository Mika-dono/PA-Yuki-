#!/usr/bin/env python3
"""
Middleware de limitation de débit pour AURA API
"""

from fastapi import Request, HTTPException
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.middleware import SlowAPIMiddleware
import time

limiter = Limiter(key_func=get_remote_address)

# Stockage en mémoire pour les limitations (à remplacer par Redis en production)
rate_limit_storage = {}

class RateLimitMiddleware:
    """Middleware de limitation de débit"""
    
    def __init__(self, requests_per_minute: int = 60):
        self.requests_per_minute = requests_per_minute
    
    async def __call__(self, request: Request, call_next):
        client_ip = get_remote_address(request)
        current_time = time.time()
        
        # Nettoyage des anciennes entrées
        self._clean_old_entries(client_ip, current_time)
        
        # Vérification du taux
        if not self._check_rate_limit(client_ip, current_time):
            raise HTTPException(
                status_code=429, 
                detail="Trop de requêtes. Veuillez réessayer dans 1 minute."
            )
        
        response = await call_next(request)
        return response
    
    def _clean_old_entries(self, client_ip: str, current_time: float):
        """Nettoie les anciennes entrées du stockage"""
        if client_ip in rate_limit_storage:
            rate_limit_storage[client_ip] = [
                timestamp for timestamp in rate_limit_storage[client_ip]
                if current_time - timestamp < 60  # Garde seulement les dernières 60 secondes
            ]
    
    def _check_rate_limit(self, client_ip: str, current_time: float) -> bool:
        """Vérifie si le client n'a pas dépassé la limite"""
        if client_ip not in rate_limit_storage:
            rate_limit_storage[client_ip] = []
        
        requests = rate_limit_storage[client_ip]
        
        if len(requests) >= self.requests_per_minute:
            return False
        
        requests.append(current_time)
        return True
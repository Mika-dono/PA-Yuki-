#!/usr/bin/env python3
"""
Middleware d'authentification pour AURA API
"""

from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from datetime import datetime, timedelta
from typing import Optional

class JWTBearer(HTTPBearer):
    """Middleware JWT pour l'authentification"""
    
    def __init__(self, auto_error: bool = True, secret_key: str = "your-secret-key"):
        super().__init__(auto_error=auto_error)
        self.secret_key = secret_key
        self.algorithm = "HS256"
    
    async def __call__(self, request: Request) -> Optional[HTTPAuthorizationCredentials]:
        credentials = await super().__call__(request)
        
        if credentials:
            if not self.verify_jwt(credentials.credentials):
                raise HTTPException(status_code=403, detail="Token invalide ou expiré")
            return credentials
        else:
            raise HTTPException(status_code=403, detail="Authorization header manquante")
    
    def verify_jwt(self, token: str) -> bool:
        """Vérifie la validité d'un token JWT"""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload.get("exp", 0) > datetime.utcnow().timestamp()
        except jwt.PyJWTError:
            return False
    
    def create_jwt(self, user_id: str, expires_delta: Optional[timedelta] = None) -> str:
        """Crée un nouveau token JWT"""
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=24)
        
        payload = {
            "user_id": user_id,
            "exp": expire,
            "iat": datetime.utcnow()
        }
        
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
#!/usr/bin/env python3
"""
Module de Cryptographie pour AURA - Chiffrement des données sensibles
"""

from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
from typing import Optional

class CryptographicManager:
    """Gestionnaire de cryptographie pour la sécurisation des données"""
    
    def __init__(self, master_key: Optional[str] = None):
        if master_key:
            self.master_key = master_key.encode()
        else:
            # En production, utilisez une clé sécurisée depuis les variables d'environnement
            self.master_key = os.getenv('AURA_MASTER_KEY', 'default-insecure-key').encode()
        
        self.fernet = self._initialize_fernet()
    
    def _initialize_fernet(self) -> Fernet:
        """Initialise le système de chiffrement Fernet"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=b'aura_finance_salt',  # En production, utilisez un salt unique
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(self.master_key))
        return Fernet(key)
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """Chiffre les données sensibles"""
        return self.fernet.encrypt(data.encode()).decode()
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Déchiffre les données sensibles"""
        return self.fernet.decrypt(encrypted_data.encode()).decode()
    
    def generate_secure_hash(self, data: str) -> str:
        """Génère un hash sécurisé pour l'intégrité des données"""
        import hashlib
        return hashlib.sha3_256(data.encode()).hexdigest()

class KeyManager:
    """Gestionnaire des clés cryptographiques"""
    
    def __init__(self, key_storage_path: str = "/secure/keys/"):
        self.key_storage_path = key_storage_path
        self.loaded_keys = {}
    
    def generate_key_pair(self, key_id: str) -> dict:
        """Génère une paire de clés pour un utilisateur"""
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa
        
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        
        public_key = private_key.public_key()
        
        # Stockage sécurisé des clés
        key_pair = {
            'private_key': private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption()
            ).decode(),
            'public_key': public_key.public_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PublicFormat.SubjectPublicKeyInfo
            ).decode()
        }
        
        self.loaded_keys[key_id] = key_pair
        return key_pair
    
    def sign_data(self, key_id: str, data: str) -> str:
        """Signe des données avec une clé privée"""
        import hashlib
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.asymmetric import padding
        
        if key_id not in self.loaded_keys:
            raise ValueError(f"Clé {key_id} non trouvée")
        
        private_key = serialization.load_pem_private_key(
            self.loaded_keys[key_id]['private_key'].encode(),
            password=None
        )
        
        signature = private_key.sign(
            data.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        return base64.b64encode(signature).decode()
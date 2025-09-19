"""
API Key Generator and Management System for MailSift
Provides API access with pay-as-you-use pricing model.
"""

import secrets
import hashlib
import time
import json
import sqlite3
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

@dataclass
class APIKey:
    """API Key data structure"""
    key_id: str
    api_key: str
    user_email: str
    tier: str  # free, pay_as_you_use, pro, enterprise
    credits_remaining: int
    total_credits_used: int
    rate_limit_per_hour: int
    created_at: datetime
    expires_at: Optional[datetime]
    is_active: bool
    usage_stats: Dict[str, Any]

@dataclass
class APIUsage:
    """API Usage tracking"""
    key_id: str
    endpoint: str
    credits_used: int
    timestamp: datetime
    response_time_ms: int
    status_code: int
    user_ip: str

class APIGenerator:
    """API Key Generator and Management System"""

    def __init__(self, db_path: str = "api_keys.db"):
        self.db_path = db_path
        self.init_database()

        # Pricing tiers (credits per operation)
        self.pricing = {
            "free": {
                "daily_limit": 100,
                "credits_per_extraction": 1,
                "credits_per_validation": 0.5,
                "credits_per_enrichment": 0.5,
                "rate_limit_per_hour": 20,
                "price_per_credit": 0
            },
            "pay_as_you_use": {
                "daily_limit": None,  # No daily limit
                "credits_per_extraction": 1,
                "credits_per_validation": 0.5,
                "credits_per_enrichment": 1,
                "rate_limit_per_hour": 100,
                "price_per_credit": 0.001  # $0.001 per credit
            },
            "pro": {
                "daily_limit": 10000,
                "credits_per_extraction": 1,
                "credits_per_validation": 0.5,
                "credits_per_enrichment": 0.5,
                "rate_limit_per_hour": 500,
                "price_per_credit": 0
            },
            "enterprise": {
                "daily_limit": None,  # No daily limit
                "credits_per_extraction": 1,
                "credits_per_validation": 0.5,
                "credits_per_enrichment": 0.5,
                "rate_limit_per_hour": 1000,
                "price_per_credit": 0
            }
        }

    def init_database(self):
        """Initialize the database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS api_keys (
                    key_id TEXT PRIMARY KEY,
                    api_key TEXT UNIQUE NOT NULL,
                    user_email TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    credits_remaining INTEGER DEFAULT 0,
                    total_credits_used INTEGER DEFAULT 0,
                    rate_limit_per_hour INTEGER NOT NULL,
                    created_at INTEGER NOT NULL,
                    expires_at INTEGER,
                    is_active BOOLEAN DEFAULT 1,
                    usage_stats TEXT DEFAULT '{}'
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS api_usage (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    credits_used INTEGER NOT NULL,
                    timestamp INTEGER NOT NULL,
                    response_time_ms INTEGER NOT NULL,
                    status_code INTEGER NOT NULL,
                    user_ip TEXT,
                    FOREIGN KEY (key_id) REFERENCES api_keys (key_id)
                )
            ''')

            conn.execute('''
                CREATE TABLE IF NOT EXISTS credit_purchases (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    key_id TEXT NOT NULL,
                    credits_purchased INTEGER NOT NULL,
                    amount_paid REAL NOT NULL,
                    payment_method TEXT NOT NULL,
                    transaction_id TEXT,
                    timestamp INTEGER NOT NULL,
                    FOREIGN KEY (key_id) REFERENCES api_keys (key_id)
                )
            ''')

    def generate_api_key(self, user_email: str, tier: str = "free") -> APIKey:
        """Generate a new API key"""
        if tier not in self.pricing:
            raise ValueError(f"Invalid tier: {tier}")

        # Generate unique API key
        key_id = secrets.token_hex(16)
        api_key = f"ms_{tier}_{secrets.token_urlsafe(32)}"

        # Calculate initial credits
        initial_credits = 100 if tier == "free" else 0

        # Create API key object
        api_key_obj = APIKey(
            key_id=key_id,
            api_key=api_key,
            user_email=user_email,
            tier=tier,
            credits_remaining=initial_credits,
            total_credits_used=0,
            rate_limit_per_hour=self.pricing[tier]["rate_limit_per_hour"],
            created_at=datetime.now(),
            expires_at=None if tier in ["pay_as_you_use"
                "pro", "enterprise"] else datetime.now() + timedelta(days=30),
            is_active=True,
            usage_stats={}
        )

        # Save to database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO api_keys (
                    key_id, api_key, user_email, tier, credits_remaining,
                    total_credits_used, rate_limit_per_hour, created_at,
                    expires_at, is_active, usage_stats
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                api_key_obj.key_id,
                api_key_obj.api_key,
                api_key_obj.user_email,
                api_key_obj.tier,
                api_key_obj.credits_remaining,
                api_key_obj.total_credits_used,
                api_key_obj.rate_limit_per_hour,
                int(api_key_obj.created_at.timestamp()),
                int(api_key_obj.expires_at.timestamp())
                    api_key_obj.expires_at else None,
                api_key_obj.is_active,
                json.dumps(api_key_obj.usage_stats)
            ))

        logger.info(f"Generated API key for {user_email} with tier {tier}")
        return api_key_obj

    def validate_api_key(self, api_key: str) -> Optional[APIKey]:
        """Validate an API key and return key info"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute('''
                SELECT * FROM api_keys WHERE api_key = ? AND is_active = 1
            ''', (api_key,))

            row = cursor.fetchone()
            if not row:
                return None

            # Check expiration
            if row[8] and datetime.now().timestamp() > row[8]:
                return None

            return APIKey(
                key_id=row[0],
                api_key=row[1],
                user_email=row[2],
                tier=row[3],
                credits_remaining=row[4],
                total_credits_used=row[5],
                rate_limit_per_hour=row[6],
                created_at=datetime.fromtimestamp(row[7]),
                expires_at=datetime.fromtimestamp(row[8]) if row[8] else None,
                is_active=bool(row[9]),
                usage_stats=json.loads(row[10] or '{}')
            )

    def record_usage(self, api_key: str, endpoint: str, credits_used: int,
                    response_time_ms: int
                        status_code: int, user_ip: str = None):
        """Record API usage"""
        key_info = self.validate_api_key(api_key)
        if not key_info:
            return False

        # Check if user has enough credits
        if key_info.tier != "free"
            key_info.credits_remaining < credits_used:
            return False

        # Record usage
        with sqlite3.connect(self.db_path) as conn:
            # Add usage record
            conn.execute('''
                INSERT INTO api_usage (
                    key_id, endpoint, credits_used, timestamp,
                    response_time_ms, status_code, user_ip
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                key_info.key_id,
                endpoint,
                credits_used,
                int(time.time()),
                response_time_ms,
                status_code,
                user_ip
            ))

            # Update credits
            new_credits = key_info.credits_remaining - credits_used
            new_total_used = key_info.total_credits_used + credits_used

            conn.execute('''
                UPDATE api_keys
                SET credits_remaining = ?, total_credits_used = ?
                WHERE key_id = ?
            ''', (new_credits, new_total_used, key_info.key_id))

        return True

    def purchase_credits(self, api_key: str, credits: int, amount_paid: float,
                        payment_method: str
                            transaction_id: str = None) -> bool:
        """Purchase credits for pay-as-you-use tier"""
        key_info = self.validate_api_key(api_key)
        if not key_info or key_info.tier != "pay_as_you_use":
            return False

        with sqlite3.connect(self.db_path) as conn:
            # Record purchase
            conn.execute('''
                INSERT INTO credit_purchases (
                    key_id, credits_purchased, amount_paid,
                    payment_method, transaction_id, timestamp
                ) VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                key_info.key_id,
                credits,
                amount_paid,
                payment_method,
                transaction_id,
                int(time.time())
            ))

            # Add credits to account
            new_credits = key_info.credits_remaining + credits
            conn.execute('''
                UPDATE api_keys
                SET credits_remaining = ?
                WHERE key_id = ?
            ''', (new_credits, key_info.key_id))

        logger.info(f"Purchased {credits} credits for {key_info.user_email}")
        return True

    def get_usage_stats(self, api_key: str) -> Dict[str, Any]:
        """Get usage statistics for an API key"""
        key_info = self.validate_api_key(api_key)
        if not key_info:
            return {}

        with sqlite3.connect(self.db_path) as conn:
            # Get recent usage
            cursor = conn.execute('''
                SELECT endpoint
                    COUNT(*) as calls, SUM(credits_used) as total_credits,
                       AVG(response_time_ms) as avg_response_time
                FROM api_usage
                WHERE key_id = ? AND timestamp > ?
                GROUP BY endpoint
            ''', (key_info.key_id, int((datetime.now() - timedelta(days=30)).timestamp())))

            recent_usage = []
            for row in cursor.fetchall():
                recent_usage.append({
                    'endpoint': row[0],
                    'calls': row[1],
                    'total_credits': row[2],
                    'avg_response_time_ms': round(row[3], 2)
                })

            # Get daily usage for last 7 days
            cursor = conn.execute('''
                SELECT DATE(datetime(timestamp, 'unixepoch')) as date,
                       COUNT(*) as calls, SUM(credits_used) as credits
                FROM api_usage
                WHERE key_id = ? AND timestamp > ?
                GROUP BY DATE(datetime(timestamp, 'unixepoch'))
                ORDER BY date DESC
                LIMIT 7
            ''', (key_info.key_id, int((datetime.now() - timedelta(days=7)).timestamp())))

            daily_usage = []
            for row in cursor.fetchall():
                daily_usage.append({
                    'date': row[0],
                    'calls': row[1],
                    'credits': row[2]
                })

        return {
            'key_info': asdict(key_info),
            'pricing_tier': self.pricing[key_info.tier],
            'recent_usage': recent_usage,
            'daily_usage': daily_usage,
            'total_spent': key_info.total_credits_used
                self.pricing[key_info.tier]['price_per_credit']
        }

    def get_pricing_info(self) -> Dict[str, Any]:
        """Get pricing information for all tiers"""
        return {
            'tiers': self.pricing,
            'credit_packages': {
                'small': {'credits': 1000, 'price': 1.00, 'bonus': 0},
                'medium': {'credits': 5000, 'price': 4.50, 'bonus': 500},
                'large': {'credits': 10000, 'price': 8.00, 'bonus': 2000},
                'enterprise': {'credits': 50000
                    'price': 35.00, 'bonus': 15000}
            }
        }

# Global API generator instance
api_generator = APIGenerator()

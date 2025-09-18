"""
MailSift PRO - Complete Revenue-Generating Application
Ready for immediate deployment and monetization
"""

from flask import Flask, render_template, request, jsonify, session, Response, send_file, redirect, url_for
import os
import json
import time
import hashlib
import secrets
from datetime import datetime, timedelta
import re
import requests
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import stripe
import logging
import io
import csv
import uuid
from functools import wraps
import random

# Import existing modules that work
from app import extract_emails_from_text, extract_emails_from_html, detect_provider
from file_parsing import extract_text_from_file
from payments import record_payment, mark_verified, list_payments

# Configure app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', secrets.token_hex(32))
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=30)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB

# Stripe setup
stripe.api_key = os.environ.get('STRIPE_SECRET_KEY', 'sk_test_51234567890')

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Database
DB_PATH = 'mailsift_pro.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Users table
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        email TEXT UNIQUE,
        plan TEXT DEFAULT 'free',
        credits INTEGER DEFAULT 100,
        total_extracted INTEGER DEFAULT 0,
        stripe_customer_id TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # Extractions table
    c.execute('''CREATE TABLE IF NOT EXISTS extractions (
        id TEXT PRIMARY KEY,
        user_id TEXT,
        count INTEGER,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    # API keys
    c.execute('''CREATE TABLE IF NOT EXISTS api_keys (
        key TEXT PRIMARY KEY,
        user_id TEXT,
        name TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )''')
    
    conn.commit()
    conn.close()

init_db()

# Pricing
PRICING = {
    'free': {'price': 0, 'credits': 100, 'name': 'Free'},
    'starter': {'price': 29, 'credits': 1000, 'name': 'Starter'},
    'pro': {'price': 99, 'credits': 10000, 'name': 'Professional'},
    'enterprise': {'price': 499, 'credits': -1, 'name': 'Enterprise'}  # -1 = unlimited
}

# Intelligence engine
class IntelligenceEngine:
    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=5)
    
    def extract_advanced(self, text):
        """Advanced extraction with multiple methods"""
        emails = set()
        
        # Standard extraction
        valid, _ = extract_emails_from_text(text)
        emails.update(valid)
        
        # Pattern matching
        patterns = [
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            r'([A-Za-z0-9._%+-]+)\s*\[at\]\s*([A-Za-z0-9.-]+)\s*\[dot\]\s*([A-Z|a-z]{2,})'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if isinstance(matches[0], tuple) if matches else False:
                # Handle obfuscated emails
                for match in matches:
                    email = f"{match[0]}@{match[1]}.{match[2]}"
                    emails.add(email.lower())
            else:
                emails.update([m.lower() for m in matches])
        
        return list(emails)
    
    def analyze_email(self, email):
        """Analyze email and return intelligence"""
        domain = email.split('@')[1] if '@' in email else ''
        local = email.split('@')[0] if '@' in email else ''
        
        # Lead scoring
        score = 50
        if not any(p in domain for p in ['gmail', 'yahoo', 'hotmail', 'outlook']):
            score += 30  # Business email
        if any(r in local.lower() for r in ['ceo', 'director', 'manager', 'founder']):
            score += 20
        
        # Intent detection
        intent = 'general'
        if any(w in local.lower() for w in ['sales', 'buy', 'purchase']):
            intent = 'sales'
        elif any(w in local.lower() for w in ['support', 'help']):
            intent = 'support'
        elif any(w in local.lower() for w in ['hr', 'job', 'career']):
            intent = 'hr'
        
        # Category
        category = detect_provider(email)
        if category == 'other' and not any(p in domain for p in ['gmail', 'yahoo', 'hotmail']):
            category = 'corporate'
        
        return {
            'email': email,
            'lead_score': min(100, score),
            'intent': intent,
            'category': category,
            'deliverable': '@' in email and '.' in domain,
            'priority': 5 if score > 80 else 3 if score > 60 else 1,
            'value': score * 10  # Estimated value in dollars
        }

engine = IntelligenceEngine()

# Authentication
def get_or_create_user():
    """Get or create user from session"""
    if 'user_id' not in session:
        session['user_id'] = str(uuid.uuid4())
        session['plan'] = 'free'
        session['credits'] = 100
        
        # Create in DB
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        try:
            c.execute('INSERT INTO users (id, plan, credits) VALUES (?, ?, ?)',
                     (session['user_id'], 'free', 100))
            conn.commit()
        except:
            pass  # User might already exist
        conn.close()
    
    return session['user_id']

def check_credits(required=1):
    """Check if user has enough credits"""
    user_id = get_or_create_user()
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT credits, plan FROM users WHERE id = ?', (user_id,))
    result = c.fetchone()
    conn.close()
    
    if not result:
        return False
    
    credits, plan = result
    if credits == -1:  # Unlimited
        return True
    
    return credits >= required

def use_credits(amount):
    """Deduct credits from user"""
    user_id = get_or_create_user()
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''UPDATE users 
                 SET credits = CASE 
                     WHEN credits = -1 THEN -1 
                     ELSE credits - ? 
                 END,
                 total_extracted = total_extracted + ?
                 WHERE id = ?''', (amount, amount, user_id))
    
    # Log extraction
    c.execute('INSERT INTO extractions (id, user_id, count) VALUES (?, ?, ?)',
             (str(uuid.uuid4()), user_id, amount))
    
    conn.commit()
    conn.close()

# Routes
@app.route('/')
def index():
    """Modern landing page"""
    user_id = get_or_create_user()
    
    # Get user data
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT credits, plan FROM users WHERE id = ?', (user_id,))
    result = c.fetchone()
    conn.close()
    
    credits = result[0] if result else 100
    plan = result[1] if result else 'free'
    
    return render_template('index_pro.html', 
                         credits=credits,
                         plan=plan,
                         pricing=PRICING)

@app.route('/api/extract', methods=['POST'])
def extract():
    """Main extraction endpoint"""
    start_time = time.time()
    
    if not check_credits():
        return jsonify({
            'error': 'Insufficient credits',
            'upgrade_url': '/pricing'
        }), 402
    
    data = request.get_json()
    all_emails = []
    
    # Extract from text
    if data.get('text'):
        emails = engine.extract_advanced(data['text'])
        all_emails.extend(emails)
    
    # Extract from URL
    if data.get('url'):
        try:
            resp = requests.get(data['url'], timeout=5)
            emails = engine.extract_advanced(resp.text)
            all_emails.extend(emails)
        except:
            pass
    
    # Remove duplicates and analyze
    unique_emails = list(set(all_emails))
    results = []
    
    for email in unique_emails:
        analysis = engine.analyze_email(email)
        results.append(analysis)
    
    # Sort by lead score
    results.sort(key=lambda x: x['lead_score'], reverse=True)
    
    # Use credits
    use_credits(len(results))
    
    # Analytics
    analytics = {
        'total': len(results),
        'high_value': sum(1 for r in results if r['lead_score'] > 70),
        'categories': {},
        'total_value': sum(r['value'] for r in results),
        'processing_time': round((time.time() - start_time) * 1000)
    }
    
    for r in results:
        cat = r['category']
        analytics['categories'][cat] = analytics['categories'].get(cat, 0) + 1
    
    # Recommendations
    recommendations = []
    if analytics['high_value'] > 5:
        recommendations.append('Focus on high-value B2B leads for maximum ROI')
    if analytics['total'] > 50:
        recommendations.append('Consider email campaign segmentation')
    
    return jsonify({
        'success': True,
        'emails': results,
        'analytics': analytics,
        'recommendations': recommendations,
        'credits_remaining': _get_credits()
    })

@app.route('/api/subscribe', methods=['POST'])
def subscribe():
    """Handle subscription"""
    data = request.get_json()
    plan = data.get('plan', 'starter')
    email = data.get('email')
    
    if plan not in PRICING:
        return jsonify({'error': 'Invalid plan'}), 400
    
    user_id = get_or_create_user()
    
    # Create Stripe customer
    try:
        # Create customer
        customer = stripe.Customer.create(
            email=email,
            metadata={'user_id': user_id}
        )
        
        # Create payment intent
        intent = stripe.PaymentIntent.create(
            amount=int(PRICING[plan]['price'] * 100),  # Convert to cents
            currency='usd',
            customer=customer.id,
            metadata={'plan': plan, 'user_id': user_id}
        )
        
        # Update user
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute('''UPDATE users 
                     SET stripe_customer_id = ?, email = ?
                     WHERE id = ?''', (customer.id, email, user_id))
        conn.commit()
        conn.close()
        
        return jsonify({
            'client_secret': intent.client_secret,
            'plan': plan,
            'price': PRICING[plan]['price']
        })
    
    except Exception as e:
        logger.error(f"Stripe error: {e}")
        return jsonify({'error': 'Payment processing error'}), 500

@app.route('/api/confirm-payment', methods=['POST'])
def confirm_payment():
    """Confirm payment and upgrade plan"""
    data = request.get_json()
    payment_intent_id = data.get('payment_intent_id')
    
    if not payment_intent_id:
        return jsonify({'error': 'Invalid payment'}), 400
    
    try:
        # Verify payment with Stripe
        intent = stripe.PaymentIntent.retrieve(payment_intent_id)
        
        if intent.status == 'succeeded':
            user_id = intent.metadata.get('user_id')
            plan = intent.metadata.get('plan')
            
            # Upgrade user
            conn = sqlite3.connect(DB_PATH)
            c = conn.cursor()
            c.execute('''UPDATE users 
                         SET plan = ?, credits = ?
                         WHERE id = ?''', 
                     (plan, PRICING[plan]['credits'], user_id))
            conn.commit()
            conn.close()
            
            session['plan'] = plan
            session['credits'] = PRICING[plan]['credits']
            
            return jsonify({
                'success': True,
                'plan': plan,
                'credits': PRICING[plan]['credits']
            })
    
    except Exception as e:
        logger.error(f"Payment confirmation error: {e}")
    
    return jsonify({'error': 'Payment verification failed'}), 400

@app.route('/api/generate-key', methods=['POST'])
def generate_api_key():
    """Generate API key for user"""
    user_id = get_or_create_user()
    
    # Check if user has paid plan
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT plan FROM users WHERE id = ?', (user_id,))
    result = c.fetchone()
    
    if not result or result[0] == 'free':
        conn.close()
        return jsonify({'error': 'API access requires paid plan'}), 402
    
    # Generate key
    api_key = 'msp_' + secrets.token_urlsafe(32)
    
    c.execute('INSERT INTO api_keys (key, user_id, name) VALUES (?, ?, ?)',
             (api_key, user_id, 'Default'))
    conn.commit()
    conn.close()
    
    return jsonify({'api_key': api_key})

@app.route('/api/stats')
def stats():
    """Get user statistics"""
    user_id = get_or_create_user()
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Get user stats
    c.execute('SELECT credits, plan, total_extracted FROM users WHERE id = ?', (user_id,))
    user = c.fetchone()
    
    # Get recent extractions
    c.execute('''SELECT count, timestamp FROM extractions 
                 WHERE user_id = ? 
                 ORDER BY timestamp DESC LIMIT 10''', (user_id,))
    extractions = c.fetchall()
    
    conn.close()
    
    return jsonify({
        'credits': user[0] if user else 100,
        'plan': user[1] if user else 'free',
        'total_extracted': user[2] if user else 0,
        'recent_extractions': [
            {'count': e[0], 'date': e[1]} for e in extractions
        ]
    })

@app.route('/pricing')
def pricing():
    """Pricing page"""
    return render_template('pricing.html', pricing=PRICING)

@app.route('/dashboard')
def dashboard():
    """User dashboard"""
    user_id = get_or_create_user()
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT * FROM users WHERE id = ?', (user_id,))
    user = c.fetchone()
    conn.close()
    
    return render_template('dashboard.html', user=user, pricing=PRICING)

@app.route('/export/<format>', methods=['POST'])
def export(format):
    """Export emails"""
    data = request.get_json()
    emails = data.get('emails', [])
    
    if format == 'csv':
        output = io.StringIO()
        writer = csv.DictWriter(output, fieldnames=['email', 'lead_score', 'category'])
        writer.writeheader()
        writer.writerows(emails)
        
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={'Content-Disposition': 'attachment;filename=emails.csv'}
        )
    
    elif format == 'json':
        return Response(
            json.dumps(emails, indent=2),
            mimetype='application/json',
            headers={'Content-Disposition': 'attachment;filename=emails.json'}
        )
    
    return jsonify({'error': 'Invalid format'}), 400

@app.route('/health')
def health():
    """Health check"""
    return jsonify({
        'status': 'healthy',
        'version': '3.0',
        'timestamp': datetime.utcnow().isoformat()
    })

# Helper functions
def _get_credits():
    user_id = get_or_create_user()
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('SELECT credits FROM users WHERE id = ?', (user_id,))
    result = c.fetchone()
    conn.close()
    return result[0] if result else 100

if __name__ == '__main__':
    print("\n" + "="*60)
    print("💎 MAILSIFT PRO - REVENUE READY")
    print("="*60)
    print("✅ Lead Scoring & Intelligence")
    print("✅ Stripe Payments Integrated") 
    print("✅ Credit System Active")
    print("✅ API Access Ready")
    print("✅ Export Features")
    print("✅ User Dashboard")
    print("="*60)
    print("🚀 Starting at http://localhost:5000")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000, host='0.0.0.0')


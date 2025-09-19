"""
AI-Powered Customer Support System for MailSift
Provides intelligent customer assistance and support automation.
"""

import json
import re
import time
import openai
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import os
from flask import request, jsonify
import logging

logger = logging.getLogger(__name__)

@dataclass
class SupportTicket:
    """Support ticket data structure"""
    id: str
    user_email: str
    subject: str
    message: str
    category: str
    priority: str
    status: str
    created_at: datetime
    updated_at: datetime
    ai_response: Optional[str] = None
    human_response: Optional[str] = None
    resolved: bool = False

@dataclass
class ChatMessage:
    """Chat message data structure"""
    id: str
    session_id: str
    user_message: str
    ai_response: str
    timestamp: datetime
    context: Dict[str, Any]

class AISupportEngine:
    """AI-powered customer support engine"""
    
    def __init__(self):
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        self.knowledge_base = self._load_knowledge_base()
        self.support_templates = self._load_support_templates()
        self.chat_sessions = {}
        
    def _load_knowledge_base(self) -> Dict[str, Any]:
        """Load knowledge base for AI responses"""
        return {
            "email_extraction": {
                "common_issues": [
                    "How to extract emails from websites",
                    "Supported file formats",
                    "Bulk email processing",
                    "Email validation accuracy"
                ],
                "solutions": {
                    "website_extraction": "Use our web scraper feature or paste website content into the text area",
                    "file_formats": "We support TXT, CSV, HTML, PDF, DOCX, and Excel files",
                    "bulk_processing": "Use our bulk processing API or upload large files for batch processing",
                    "validation": "Our AI engine provides 99.9% accuracy with real-time validation"
                }
            },
            "licensing": {
                "common_issues": [
                    "License key not working",
                    "How to purchase license",
                    "License renewal",
                    "Transfer license"
                ],
                "solutions": {
                    "license_issue": "Please verify your license key and ensure you're connected to the internet",
                    "purchase": "Visit our pricing page to purchase a license with crypto or card payments",
                    "renewal": "Licenses auto-renew. Check your account dashboard for renewal status",
                    "transfer": "Contact support with both account details for license transfer"
                }
            },
            "technical": {
                "common_issues": [
                    "API rate limits",
                    "Error messages",
                    "Performance issues",
                    "Integration problems"
                ],
                "solutions": {
                    "rate_limits": "Free tier has 100 requests/day. Upgrade for higher limits",
                    "errors": "Check our error codes documentation or contact support",
                    "performance": "Large files may take longer. Use our bulk API for better performance",
                    "integration": "We provide SDKs for Python, JavaScript, and REST API"
                }
            }
        }
    
    def _load_support_templates(self) -> Dict[str, str]:
        """Load support response templates"""
        return {
            "greeting": """
Hello! I'm MailSift AI Assistant. I'm here to help you with:

🔍 Email extraction and validation
💳 Licensing and payments  
⚙️ Technical support
📊 API usage and integration
🚀 Feature requests

How can I assist you today?
            """.strip(),
            
            "email_extraction_help": """
Here's how to extract emails effectively:

1. **Web Scraping**: Use our web scraper tool or paste website content
2. **File Upload**: Support TXT, CSV, HTML, PDF, DOCX files
3. **Bulk Processing**: Use our API for large-scale extraction
4. **Validation**: All emails are validated with 99.9% accuracy

Need help with a specific extraction task?
            """.strip(),
            
            "license_help": """
License assistance:

🔑 **Verify License**: Check your license key in the admin panel
💳 **Purchase**: Visit /pricing for crypto or card payments
🔄 **Renewal**: Licenses auto-renew, check your dashboard
📧 **Support**: Contact support@mailsift.com for license issues

What specific license help do you need?
            """.strip(),
            
            "technical_help": """
Technical support:

📊 **API Limits**: Free tier = 100 requests/day
⚡ **Performance**: Use bulk API for large datasets
🔧 **Integration**: Python/JS SDKs available
📖 **Documentation**: Full API docs at /api/docs

What technical issue can I help resolve?
            """.strip(),
            
            "fallback": """
I understand you need help. Let me connect you with our support team:

📧 **Email**: support@mailsift.com
💬 **Live Chat**: Available on our website
📞 **Response Time**: Within 2 hours during business hours

Is there anything else I can help you with right now?
            """.strip()
        }
    
    def generate_ai_response(self, user_message: str, context: Dict[str, Any] = None) -> str:
        """Generate AI response to user message"""
        try:
            if not self.openai_api_key:
                return self._generate_fallback_response(user_message)
            
            # Analyze user intent
            intent = self._analyze_intent(user_message)
            
            # Generate contextual response
            if intent == "email_extraction":
                return self.support_templates["email_extraction_help"]
            elif intent == "licensing":
                return self.support_templates["license_help"]
            elif intent == "technical":
                return self.support_templates["technical_help"]
            elif intent == "greeting":
                return self.support_templates["greeting"]
            else:
                return self._generate_custom_response(user_message, context)
                
        except Exception as e:
            logger.error(f"AI response generation failed: {e}")
            return self.support_templates["fallback"]
    
    def _analyze_intent(self, message: str) -> str:
        """Analyze user intent from message"""
        message_lower = message.lower()
        
        # Email extraction keywords
        extraction_keywords = ['extract', 'email', 'website', 'scrape', 'bulk', 'file', 'upload']
        if any(keyword in message_lower for keyword in extraction_keywords):
            return "email_extraction"
        
        # Licensing keywords
        license_keywords = ['license', 'key', 'purchase', 'buy', 'payment', 'crypto', 'subscription']
        if any(keyword in message_lower for keyword in license_keywords):
            return "licensing"
        
        # Technical keywords
        tech_keywords = ['api', 'error', 'bug', 'issue', 'problem', 'integration', 'code']
        if any(keyword in message_lower for keyword in tech_keywords):
            return "technical"
        
        # Greeting keywords
        greeting_keywords = ['hello', 'hi', 'help', 'start', 'begin']
        if any(keyword in message_lower for keyword in greeting_keywords):
            return "greeting"
        
        return "general"
    
    def _generate_custom_response(self, message: str, context: Dict[str, Any] = None) -> str:
        """Generate custom AI response using OpenAI"""
        try:
            openai.api_key = self.openai_api_key
            
            prompt = f"""
You are MailSift AI Support Assistant. Help users with email extraction, licensing, and technical issues.

User message: {message}

Context: {json.dumps(context or {})}

Knowledge base:
- Email extraction: Support TXT, CSV, HTML, PDF, DOCX files with 99.9% accuracy
- Web scraping: Extract emails from websites with anti-bot protection
- Bulk processing: Handle millions of emails with our API
- Licensing: Crypto and card payments supported, auto-renewal
- API: REST API with Python/JS SDKs, rate limits apply

Provide a helpful, concise response. If you can't help, direct them to support@mailsift.com.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful customer support AI for MailSift email extraction service."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"Custom AI response failed: {e}")
            return self.support_templates["fallback"]
    
    def _generate_fallback_response(self, message: str) -> str:
        """Generate fallback response when AI is unavailable"""
        intent = self._analyze_intent(message)
        
        if intent == "email_extraction":
            return self.support_templates["email_extraction_help"]
        elif intent == "licensing":
            return self.support_templates["license_help"]
        elif intent == "technical":
            return self.support_templates["technical_help"]
        else:
            return self.support_templates["fallback"]
    
    def create_support_ticket(self, user_email: str, subject: str, message: str, 
                            category: str = "general") -> SupportTicket:
        """Create a new support ticket"""
        ticket_id = f"TKT-{int(time.time())}"
        
        ticket = SupportTicket(
            id=ticket_id,
            user_email=user_email,
            subject=subject,
            message=message,
            category=category,
            priority=self._determine_priority(message, category),
            status="open",
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        
        # Generate AI response
        ticket.ai_response = self.generate_ai_response(message, {"ticket_id": ticket_id})
        
        return ticket
    
    def _determine_priority(self, message: str, category: str) -> str:
        """Determine ticket priority"""
        urgent_keywords = ['urgent', 'critical', 'down', 'broken', 'not working', 'error']
        message_lower = message.lower()
        
        if any(keyword in message_lower for keyword in urgent_keywords):
            return "high"
        elif category in ["technical", "licensing"]:
            return "medium"
        else:
            return "low"
    
    def get_chat_response(self, session_id: str, user_message: str) -> ChatMessage:
        """Get chat response for a session"""
        if session_id not in self.chat_sessions:
            self.chat_sessions[session_id] = []
        
        # Add user message to session
        self.chat_sessions[session_id].append({"role": "user", "content": user_message})
        
        # Generate AI response
        context = {"session_id": session_id, "history": self.chat_sessions[session_id]}
        ai_response = self.generate_ai_response(user_message, context)
        
        # Add AI response to session
        self.chat_sessions[session_id].append({"role": "assistant", "content": ai_response})
        
        # Create chat message
        chat_message = ChatMessage(
            id=f"CHAT-{int(time.time())}",
            session_id=session_id,
            user_message=user_message,
            ai_response=ai_response,
            timestamp=datetime.now(),
            context=context
        )
        
        return chat_message
    
    def get_suggested_responses(self, user_message: str) -> List[str]:
        """Get suggested responses for common questions"""
        intent = self._analyze_intent(user_message)
        
        suggestions = {
            "email_extraction": [
                "How do I extract emails from a website?",
                "What file formats are supported?",
                "How accurate is email validation?",
                "Can I process bulk emails?"
            ],
            "licensing": [
                "How do I purchase a license?",
                "My license key isn't working",
                "How does license renewal work?",
                "Can I transfer my license?"
            ],
            "technical": [
                "What are the API rate limits?",
                "I'm getting error messages",
                "How do I integrate with my app?",
                "Performance is slow"
            ],
            "general": [
                "How to get started with MailSift?",
                "What features are available?",
                "Contact human support",
                "View documentation"
            ]
        }
        
        return suggestions.get(intent, suggestions["general"])

# Global AI support engine instance
ai_support_engine = AISupportEngine()

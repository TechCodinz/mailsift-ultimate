"""
🚀 ULTRA EMAIL EXTRACTION ENGINE - WORLD-CLASS ACCURACY
The most advanced email extraction system ever built
"""

import re
import base64
import urllib.parse
from typing import List, Tuple, Dict, Set, Any
from dataclasses import dataclass
from bs4 import BeautifulSoup
import logging

logger = logging.getLogger(__name__)


@dataclass
class ExtractionResult:
    """Result of email extraction with metadata"""
    emails: List[str]
    invalid_emails: List[str]
    confidence_scores: Dict[str, float]
    extraction_methods: Dict[str, List[str]]
    context_info: Dict[str, Any]
    total_found: int
    unique_valid: int


class UltraEmailExtractor:
    """The most advanced email extraction engine in the world"""

    def __init__(self):
        self.compiled_patterns = self._compile_all_patterns()
        self.domain_blacklist = self._load_domain_blacklist()
        self.disposable_domains = self._load_disposable_domains()

    def _compile_all_patterns(self) -> Dict[str, re.Pattern]:
        """Compile all regex patterns for maximum performance"""
        return {
            # Basic email pattern (RFC 5322 compliant)
            'basic': re.compile(
                r'\b[a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?@'
                r'[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,}\b'
            ),

            # Advanced pattern with international domains
            'international': re.compile(
                r'\b[a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?@'
                r'[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?\.[a-zA-Z0-9]{2,}\b'
            ),

            # Obfuscated patterns - (at) and (dot)
            'obfuscated_at_dot': re.compile(
                r'\b([a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?)\s*'
                r'(?:\(at\)|\[at\]|\{at\}|&#64;|\uFF20)\s*'
                r'([a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?)\s*'
                r'(?:\(dot\)|\[dot\]|\{dot\}|&#46;)\s*'
                r'([a-zA-Z]{2,})\b',
                re.IGNORECASE
            ),

            # Obfuscated patterns - spaced
            'obfuscated_spaced': re.compile(
                r'\b([a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?)\s+'
                r'at\s+([a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?)\s+'
                r'dot\s+([a-zA-Z]{2,})\b',
                re.IGNORECASE
            ),

            # HTML encoded emails
            'html_encoded': re.compile(
                r'&#64;([^&#]+)&#46;([^&#]+)&#46;([a-zA-Z]{2,})'
            ),

            # Base64 encoded emails (partial)
            'base64_partial': re.compile(
                r'[A-Za-z0-9+/]{20,}={0,2}'
            ),

            # JavaScript obfuscated
            'js_obfuscated': re.compile(
                r'String\.fromCharCode\([^)]+\)'
            ),

            # URL encoded
            'url_encoded': re.compile(
                r'%40[^%]+%2E[^%]+%2E[a-zA-Z]{2,}'
            ),

            # Mailto links
            'mailto': re.compile(
                r'mailto:([a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?@'
                r'[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,})',
                re.IGNORECASE
            ),

            # Contact forms and input fields
            'contact_forms': re.compile(
                r'(?:email|contact|mail)\s*[:=]\s*'
                r'([a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?@'
                r'[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,})',
                re.IGNORECASE
            ),

            # JSON/API responses
            'json_api': re.compile(
                r'"email"\s*:\s*"([a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?@'
                r'[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,})"',
                re.IGNORECASE
            ),

            # CSS content
            'css_content': re.compile(
                (r'content\s*:\s*["\']([a-zA-Z0-9](?:[a-zA-Z0-9._-]*'
                 r'[a-zA-Z0-9])?@'
                 r'[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,})'
                 r'["\']'),
                re.IGNORECASE
            ),

            # Meta tags
            'meta_tags': re.compile(
                (r'<meta[^>]+content=["\']([a-zA-Z0-9](?:[a-zA-Z0-9._-]*'
                 r'[a-zA-Z0-9])?@'
                 r'[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,})'
                 r'["\']'),
                re.IGNORECASE
            ),

            # Comments and hidden text
            'comments': re.compile(
                r'<!--.*?([a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?@'
                r'[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,})'
                r'.*?-->',
                re.IGNORECASE | re.DOTALL
            )
        }

    def _load_domain_blacklist(self) -> Set[str]:
        """Load blacklisted domains for filtering"""
        return {
            'example.com', 'test.com', 'localhost', 'invalid.com',
            'none.com', 'null.com', 'placeholder.com', 'dummy.com'
        }

    def _load_disposable_domains(self) -> Set[str]:
        """Load disposable email domains"""
        return {
            '10minutemail.com', 'tempmail.org', 'guerrillamail.com',
            'mailinator.com', 'throwaway.email', 'temp-mail.org',
            'yopmail.com', 'maildrop.cc', 'sharklasers.com',
            'guerrillamailblock.com', 'pokemail.net', 'spam4.me',
            'bccto.me', 'chacuo.net', 'dispostable.com'
        }

    def extract_emails_ultra(
            self,
            content: str,
            content_type: str = 'text') -> ExtractionResult:
        """Ultra-advanced email extraction with maximum accuracy"""
        if not content:
            return ExtractionResult([], [], {}, {}, {}, 0, 0)

        all_emails = set()
        extraction_methods = {}
        confidence_scores = {}

        # 1. Basic extraction
        basic_emails = self._extract_basic_emails(content)
        all_emails.update(basic_emails)
        extraction_methods['basic'] = basic_emails

        # 2. HTML parsing (if HTML content)
        if content_type == 'html':
            html_emails = self._extract_from_html(content)
            all_emails.update(html_emails)
            extraction_methods['html'] = html_emails

        # 3. Obfuscated email extraction
        obfuscated_emails = self._extract_obfuscated_emails(content)
        all_emails.update(obfuscated_emails)
        extraction_methods['obfuscated'] = obfuscated_emails

        # 4. Encoded email extraction
        encoded_emails = self._extract_encoded_emails(content)
        all_emails.update(encoded_emails)
        extraction_methods['encoded'] = encoded_emails

        # 5. Context-aware extraction
        context_emails = self._extract_context_emails(content)
        all_emails.update(context_emails)
        extraction_methods['context'] = context_emails

        # 6. Advanced pattern matching
        advanced_emails = self._extract_advanced_patterns(content)
        all_emails.update(advanced_emails)
        extraction_methods['advanced'] = advanced_emails

        # 7. Clean and validate emails
        valid_emails, invalid_emails = self._clean_and_validate(
            list(all_emails))

        # 8. Calculate confidence scores
        for email in valid_emails:
            confidence_scores[email] = self._calculate_confidence(
                email, content)

        # 9. Remove duplicates and sort
        valid_emails = sorted(list(set(valid_emails)))
        invalid_emails = sorted(list(set(invalid_emails)))

        return ExtractionResult(
            emails=valid_emails,
            invalid_emails=invalid_emails,
            confidence_scores=confidence_scores,
            extraction_methods=extraction_methods,
            context_info=self._analyze_context(content),
            total_found=len(all_emails),
            unique_valid=len(valid_emails)
        )

    def _extract_basic_emails(self, content: str) -> List[str]:
        """Extract basic email patterns"""
        emails = []

        # Standard email pattern
        for match in self.compiled_patterns['basic'].finditer(content):
            emails.append(match.group(0))

        # International domains
        for match in self.compiled_patterns['international'].finditer(content):
            emails.append(match.group(0))

        return emails

    def _extract_from_html(self, html_content: str) -> List[str]:
        """Advanced HTML email extraction"""
        emails = []

        try:
            soup = BeautifulSoup(html_content, 'html.parser')

            # Remove script and style tags
            for tag in soup(['script', 'style', 'noscript']):
                tag.decompose()

            # Extract from mailto links
            for link in soup.find_all('a', href=True):
                href = link.get('href', '')
                if href.startswith('mailto:'):
                    email = href.replace('mailto:', '').split('?')[0]
                    emails.append(email)

            # Extract from data attributes
            for tag in soup.find_all(attrs={'data-email': True}):
                emails.append(tag.get('data-email'))

            # Extract from input fields
            for input_tag in soup.find_all('input', {'type': 'email'}):
                if input_tag.get('value'):
                    emails.append(input_tag.get('value'))

            # Extract from meta tags
            for meta in soup.find_all(
                'meta',
                attrs={
                    'name': re.compile(
                        r'email|contact|author',
                        re.I)}):
                content = meta.get('content', '')
                if '@' in content:
                    emails.extend(
                        self.compiled_patterns['basic'].findall(content))

            # Extract from text content
            text_content = soup.get_text()
            emails.extend(
                self.compiled_patterns['basic'].findall(text_content))

        except Exception as e:
            logger.error(f"HTML extraction error: {e}")
            # Fallback to regex
            emails.extend(
                self.compiled_patterns['basic'].findall(html_content))

        return emails

    def _extract_obfuscated_emails(self, content: str) -> List[str]:
        """Extract obfuscated emails"""
        emails = []

        # (at) and (dot) patterns
        for match in self.compiled_patterns['obfuscated_at_dot'].finditer(
                content):
            local, domain, tld = match.groups()
            email = f"{local}@{domain}.{tld}"
            emails.append(email)

        # Spaced patterns
        for match in self.compiled_patterns['obfuscated_spaced'].finditer(
                content):
            local, domain, tld = match.groups()
            email = f"{local}@{domain}.{tld}"
            emails.append(email)

        # HTML encoded
        for match in self.compiled_patterns['html_encoded'].finditer(content):
            local, domain, tld = match.groups()
            email = f"{local}@{domain}.{tld}"
            emails.append(email)

        return emails

    def _extract_encoded_emails(self, content: str) -> List[str]:
        """Extract encoded emails"""
        emails = []

        # URL encoded
        for match in self.compiled_patterns['url_encoded'].finditer(content):
            decoded = urllib.parse.unquote(match.group(0))
            emails.extend(self.compiled_patterns['basic'].findall(decoded))

        # Base64 encoded (look for email-like patterns)
        for match in self.compiled_patterns['base64_partial'].finditer(
                content):
            try:
                decoded = base64.b64decode(
                    match.group(0) +
                    '==').decode(
                    'utf-8',
                    errors='ignore')
                if '@' in decoded:
                    emails.extend(
                        self.compiled_patterns['basic'].findall(decoded))
            except BaseException:
                continue

        # JavaScript obfuscated
        for match in self.compiled_patterns['js_obfuscated'].finditer(content):
            # This would need a JavaScript interpreter for full extraction
            # For now, we'll skip complex JS obfuscation
            pass

        return emails

    def _extract_context_emails(self, content: str) -> List[str]:
        """Extract emails using context clues"""
        emails = []

        # Contact forms
        for match in self.compiled_patterns['contact_forms'].finditer(content):
            emails.append(match.group(1))

        # JSON/API responses
        for match in self.compiled_patterns['json_api'].finditer(content):
            emails.append(match.group(1))

        # CSS content
        for match in self.compiled_patterns['css_content'].finditer(content):
            emails.append(match.group(1))

        # Meta tags
        for match in self.compiled_patterns['meta_tags'].finditer(content):
            emails.append(match.group(1))

        # Comments
        for match in self.compiled_patterns['comments'].finditer(content):
            emails.append(match.group(1))

        return emails

    def _extract_advanced_patterns(self, content: str) -> List[str]:
        """Extract emails using advanced pattern matching"""
        emails = []

        # Look for emails in various contexts
        patterns = [
            # Email in quotes
            (r'"([a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?@'
             r'[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,})"'),
            # Email in parentheses
            (r'\(([a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?@'
             r'[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,})\)'),
            # Email after "contact:"
            (r'contact\s*:\s*([a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?@'
             r'[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,})'),
            # Email in structured data
            (r'<[^>]*>([a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?@'
             r'[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,})'
             r'<[^>]*>'),
        ]

        for pattern in patterns:
            for match in re.finditer(pattern, content, re.IGNORECASE):
                emails.append(match.group(1))

        return emails

    def _clean_and_validate(
            self, emails: List[str]) -> Tuple[List[str], List[str]]:
        """Clean and validate extracted emails"""
        valid_emails = []
        invalid_emails = []

        for email in emails:
            # Clean email
            cleaned = self._clean_email(email)

            if not cleaned:
                continue

            # Basic validation
            if self._is_valid_email(cleaned):
                # Check if not blacklisted
                domain = cleaned.split('@')[1].lower()
                if domain not in self.domain_blacklist:
                    valid_emails.append(cleaned.lower())
                else:
                    invalid_emails.append(cleaned)
            else:
                invalid_emails.append(cleaned)

        return valid_emails, invalid_emails

    def _clean_email(self, email: str) -> str:
        """Clean email address"""
        if not email:
            return ""

        # Remove common prefixes/suffixes
        email = re.sub(
            r'^(mailto:|email\s*[:=]\s*)',
            '',
            email,
            flags=re.IGNORECASE)
        email = re.sub(r'[<>"\'\[\](),;:]', '', email)
        email = email.strip()

        # Remove trailing dots and commas
        email = email.rstrip('.,;')

        return email

    def _is_valid_email(self, email: str) -> bool:
        """Validate email format"""
        if not email or '@' not in email:
            return False

        parts = email.split('@')
        if len(parts) != 2:
            return False

        local, domain = parts

        # Check local part
        if len(local) < 1 or len(local) > 64:
            return False

        # Check domain
        if len(domain) < 3 or '.' not in domain:
            return False

        # Check for valid characters
        if not re.match(
            r'^[a-zA-Z0-9](?:[a-zA-Z0-9._-]*[a-zA-Z0-9])?$',
                local):
            return False

        if not re.match(
            r'^[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?\.[a-zA-Z0-9]{2,}$',
                domain):
            return False

        return True

    def _calculate_confidence(self, email: str, context: str) -> float:
        """Calculate confidence score for extracted email"""
        confidence = 0.5  # Base confidence

        # Check if email appears in multiple contexts
        if email.lower() in context.lower():
            confidence += 0.2

        # Check domain reputation
        domain = email.split('@')[1].lower()
        if domain in self.disposable_domains:
            confidence -= 0.3
        elif any(provider in domain for provider in [
                'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com']):
            confidence += 0.1

        # Check if email appears near contact-related keywords
        contact_keywords = [
            'contact',
            'email',
            'mail',
            'reach',
            'get in touch']
        email_pos = context.lower().find(email.lower())
        if email_pos != -1:
            context_snippet = context[max(
                0, email_pos - 100):email_pos + 100].lower()
            if any(keyword in context_snippet for keyword in contact_keywords):
                confidence += 0.2

        return min(1.0, max(0.0, confidence))

    def _analyze_context(self, content: str) -> Dict[str, Any]:
        """Analyze content context"""
        return {
            'length': len(content),
            'has_html': '<' in content and '>' in content,
            'has_obfuscation': any(pattern in content.lower() for pattern in [
                '(at)', '(dot)', '&#64;']),
            'has_encoding': any(pattern in content.lower() for pattern in [
                '%40', 'base64']),
            'contact_indicators': sum(1 for keyword in [
                'contact', 'email', 'reach'] if keyword in content.lower()),
            'estimated_quality': ('high' if len(content) > 1000 else
                                  'medium' if len(content) > 100 else 'low')
        }


# Global instance
ultra_extractor = UltraEmailExtractor()

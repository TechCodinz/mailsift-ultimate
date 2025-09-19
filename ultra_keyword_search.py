"""
🚀 ULTRA KEYWORD SEARCH ENGINE - WORLD-CLASS SEARCH CAPABILITIES
The most advanced keyword search system ever built for email discovery
"""

import re
import json
import time
import hashlib
from typing import List, Dict, Optional, Any, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict, Counter
import difflib
import math

# ML imports for advanced search
try:
    import numpy as np
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.cluster import KMeans
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

logger = logging.getLogger(__name__)

@dataclass
class SearchResult:
    """Result of keyword search operation"""
    email: str
    relevance_score: float
    match_type: str  # exact, fuzzy, semantic, contextual
    matched_keywords: List[str]
    confidence: float
    metadata: Dict[str, Any]
    snippet: Optional[str] = None

@dataclass
class SearchQuery:
    """Search query with advanced options"""
    keywords: List[str]
    search_type: str  # exact, fuzzy, semantic, all
    industry: Optional[str] = None
    domain_pattern: Optional[str] = None
    email_type: Optional[str] = None  # personal, business, role-based
    confidence_threshold: float = 0.7
    max_results: int = 100

class UltraKeywordSearchEngine:
    """The most advanced keyword search engine for email discovery"""
    
    def __init__(self):
        self.search_index = {}
        self.industry_keywords = self._load_industry_keywords()
        self.semantic_similarities = {}
        self.search_analytics = defaultdict(int)
        self.search_cache = {}
        self._build_search_index()
        
    def _load_industry_keywords(self) -> Dict[str, List[str]]:
        """Load comprehensive industry keyword database"""
        return {
            'technology': [
                'software', 'tech', 'it', 'systems', 'development', 'programming',
                'cloud', 'saas', 'ai', 'artificial intelligence', 'machine learning',
                'data science', 'analytics', 'cybersecurity', 'blockchain', 'crypto',
                'mobile', 'web', 'app', 'platform', 'api', 'devops', 'startup',
                'innovation', 'digital', 'automation', 'iot', 'vr', 'ar'
            ],
            'finance': [
                'finance', 'financial', 'banking', 'investment', 'capital', 'fund',
                'equity', 'venture', 'private equity', 'hedge fund', 'trading',
                'fintech', 'cryptocurrency', 'bitcoin', 'ethereum', 'crypto',
                'wealth management', 'asset management', 'portfolio', 'risk',
                'compliance', 'audit', 'accounting', 'cpa', 'tax', 'insurance'
            ],
            'healthcare': [
                'health', 'healthcare', 'medical', 'medicine', 'clinic', 'hospital',
                'pharmaceutical', 'pharma', 'biotech', 'biotechnology', 'research',
                'clinical', 'therapy', 'treatment', 'diagnosis', 'patient care',
                'telemedicine', 'digital health', 'medtech', 'healthtech',
                'wellness', 'fitness', 'nutrition', 'mental health', 'dental'
            ],
            'education': [
                'education', 'educational', 'school', 'university', 'college',
                'academy', 'institute', 'learning', 'training', 'e-learning',
                'edtech', 'online learning', 'course', 'curriculum', 'student',
                'teacher', 'professor', 'research', 'academic', 'scholarship'
            ],
            'real_estate': [
                'real estate', 'property', 'realty', 'realtor', 'broker',
                'homes', 'houses', 'apartments', 'commercial', 'residential',
                'construction', 'development', 'property management', 'leasing',
                'mortgage', 'lending', 'investment property', 'land'
            ],
            'marketing': [
                'marketing', 'advertising', 'advertisement', 'brand', 'branding',
                'digital marketing', 'social media', 'content marketing', 'seo',
                'ppc', 'email marketing', 'influencer', 'pr', 'public relations',
                'creative', 'design', 'agency', 'media', 'communications'
            ],
            'manufacturing': [
                'manufacturing', 'production', 'factory', 'industrial', 'machinery',
                'automation', 'supply chain', 'logistics', 'distribution',
                'quality control', 'engineering', 'mechanical', 'electrical',
                'materials', 'components', 'assembly', 'packaging'
            ],
            'retail': [
                'retail', 'ecommerce', 'e-commerce', 'online store', 'shopping',
                'merchandise', 'inventory', 'sales', 'customer service',
                'fashion', 'clothing', 'accessories', 'electronics', 'home',
                'beauty', 'cosmetics', 'jewelry', 'gifts', 'toys'
            ],
            'consulting': [
                'consulting', 'consultant', 'advisory', 'strategy', 'management',
                'business consulting', 'management consulting', 'strategy consulting',
                'operations', 'transformation', 'change management', 'process',
                'optimization', 'efficiency', 'growth', 'scaling'
            ],
            'legal': [
                'legal', 'law', 'attorney', 'lawyer', 'law firm', 'litigation',
                'corporate law', 'criminal law', 'family law', 'real estate law',
                'intellectual property', 'patent', 'trademark', 'copyright',
                'compliance', 'regulatory', 'contract', 'agreement'
            ],
            'government': [
                'government', 'public sector', 'federal', 'state', 'local',
                'municipal', 'city', 'county', 'public administration',
                'policy', 'regulation', 'public service', 'civil service',
                'election', 'voting', 'census', 'public health'
            ],
            'nonprofit': [
                'nonprofit', 'non-profit', 'charity', 'foundation', 'ngo',
                'social impact', 'community', 'volunteer', 'donation',
                'fundraising', 'grant', 'philanthropy', 'humanitarian',
                'environmental', 'sustainability', 'social justice'
            ]
        }
    
    def _build_search_index(self):
        """Build comprehensive search index"""
        # This would typically load from a database
        # For now, we'll build it dynamically as emails are processed
        self.search_index = {
            'emails': {},
            'keywords': defaultdict(list),
            'industries': defaultdict(list),
            'domains': defaultdict(list),
            'semantic_vectors': {}
        }
    
    def _extract_keywords_from_email(self, email: str) -> List[str]:
        """Extract keywords from email address"""
        try:
            local, domain = email.split('@', 1)
            
            # Extract keywords from local part
            local_keywords = re.findall(r'[a-zA-Z]+', local.lower())
            
            # Extract keywords from domain
            domain_keywords = re.findall(r'[a-zA-Z]+', domain.lower())
            
            # Combine and deduplicate
            all_keywords = list(set(local_keywords + domain_keywords))
            
            # Filter out common words
            common_words = {'com', 'org', 'net', 'edu', 'gov', 'co', 'io', 'me', 'us'}
            filtered_keywords = [kw for kw in all_keywords if kw not in common_words and len(kw) > 2]
            
            return filtered_keywords
            
        except Exception as e:
            logger.error(f"Keyword extraction failed for {email}: {e}")
            return []
    
    def _calculate_relevance_score(self, email: str, query_keywords: List[str], search_type: str) -> float:
        """Calculate relevance score for email against query"""
        try:
            email_keywords = self._extract_keywords_from_email(email)
            
            if search_type == 'exact':
                # Exact keyword matching
                matches = set(query_keywords) & set(email_keywords)
                return len(matches) / len(query_keywords) if query_keywords else 0
            
            elif search_type == 'fuzzy':
                # Fuzzy matching with similarity
                total_score = 0
                for query_kw in query_keywords:
                    best_match = 0
                    for email_kw in email_keywords:
                        similarity = difflib.SequenceMatcher(None, query_kw, email_kw).ratio()
                        best_match = max(best_match, similarity)
                    total_score += best_match
                return total_score / len(query_keywords) if query_keywords else 0
            
            elif search_type == 'semantic':
                # Semantic similarity (requires ML)
                if ML_AVAILABLE and email in self.search_index['semantic_vectors']:
                    query_vector = self._get_query_vector(query_keywords)
                    email_vector = self.search_index['semantic_vectors'][email]
                    similarity = cosine_similarity([query_vector], [email_vector])[0][0]
                    return float(similarity)
                else:
                    # Fallback to fuzzy matching
                    return self._calculate_relevance_score(email, query_keywords, 'fuzzy')
            
            else:  # 'all'
                # Combine all methods
                exact_score = self._calculate_relevance_score(email, query_keywords, 'exact')
                fuzzy_score = self._calculate_relevance_score(email, query_keywords, 'fuzzy')
                semantic_score = self._calculate_relevance_score(email, query_keywords, 'semantic')
                
                # Weighted combination
                return (exact_score * 0.4 + fuzzy_score * 0.3 + semantic_score * 0.3)
                
        except Exception as e:
            logger.error(f"Relevance calculation failed for {email}: {e}")
            return 0.0
    
    def _get_query_vector(self, keywords: List[str]) -> np.ndarray:
        """Get vector representation of query keywords"""
        if not ML_AVAILABLE:
            return np.zeros(100)  # Fallback
        
        try:
            # Simple TF-IDF vectorization
            text = ' '.join(keywords)
            vectorizer = TfidfVectorizer(max_features=100)
            vector = vectorizer.fit_transform([text])
            return vector.toarray()[0]
        except:
            return np.zeros(100)
    
    def _determine_match_type(self, email: str, query_keywords: List[str]) -> str:
        """Determine the type of match for the email"""
        email_keywords = self._extract_keywords_from_email(email)
        
        # Check for exact matches
        exact_matches = set(query_keywords) & set(email_keywords)
        if exact_matches:
            return 'exact'
        
        # Check for fuzzy matches
        fuzzy_threshold = 0.8
        for query_kw in query_keywords:
            for email_kw in email_keywords:
                similarity = difflib.SequenceMatcher(None, query_kw, email_kw).ratio()
                if similarity >= fuzzy_threshold:
                    return 'fuzzy'
        
        # Check for semantic matches
        if ML_AVAILABLE:
            try:
                query_vector = self._get_query_vector(query_keywords)
                if email in self.search_index['semantic_vectors']:
                    email_vector = self.search_index['semantic_vectors'][email]
                    similarity = cosine_similarity([query_vector], [email_vector])[0][0]
                    if similarity >= 0.7:
                        return 'semantic'
            except:
                pass
        
        return 'contextual'
    
    def _get_matched_keywords(self, email: str, query_keywords: List[str]) -> List[str]:
        """Get list of keywords that matched"""
        email_keywords = self._extract_keywords_from_email(email)
        matched = []
        
        # Exact matches
        exact_matches = set(query_keywords) & set(email_keywords)
        matched.extend(list(exact_matches))
        
        # Fuzzy matches
        for query_kw in query_keywords:
            for email_kw in email_keywords:
                similarity = difflib.SequenceMatcher(None, query_kw, email_kw).ratio()
                if similarity >= 0.8 and email_kw not in matched:
                    matched.append(email_kw)
        
        return matched
    
    def _detect_industry(self, email: str) -> str:
        """Detect industry from email"""
        email_keywords = self._extract_keywords_from_email(email)
        
        industry_scores = {}
        for industry, keywords in self.industry_keywords.items():
            score = 0
            for keyword in keywords:
                for email_kw in email_keywords:
                    if keyword.lower() in email_kw.lower() or email_kw.lower() in keyword.lower():
                        score += 1
            industry_scores[industry] = score
        
        # Return industry with highest score
        if industry_scores:
            return max(industry_scores, key=industry_scores.get)
        return 'other'
    
    def _generate_snippet(self, email: str, matched_keywords: List[str]) -> str:
        """Generate a snippet showing why the email matched"""
        local, domain = email.split('@', 1)
        
        snippet_parts = []
        for keyword in matched_keywords:
            if keyword.lower() in local.lower():
                snippet_parts.append(f"Local part contains '{keyword}'")
            elif keyword.lower() in domain.lower():
                snippet_parts.append(f"Domain contains '{keyword}'")
        
        return '; '.join(snippet_parts) if snippet_parts else f"Email: {email}"
    
    def index_email(self, email: str, metadata: Dict[str, Any] = None):
        """Add email to search index"""
        try:
            keywords = self._extract_keywords_from_email(email)
            industry = self._detect_industry(email)
            
            # Add to index
            self.search_index['emails'][email] = {
                'keywords': keywords,
                'industry': industry,
                'metadata': metadata or {},
                'indexed_at': datetime.now().isoformat()
            }
            
            # Update keyword index
            for keyword in keywords:
                self.search_index['keywords'][keyword].append(email)
            
            # Update industry index
            self.search_index['industries'][industry].append(email)
            
            # Update domain index
            domain = email.split('@')[1]
            self.search_index['domains'][domain].append(email)
            
            # Generate semantic vector if ML is available
            if ML_AVAILABLE:
                try:
                    vector = self._get_query_vector(keywords)
                    self.search_index['semantic_vectors'][email] = vector
                except:
                    pass
                    
        except Exception as e:
            logger.error(f"Failed to index email {email}: {e}")
    
    def search_emails(self, query: SearchQuery) -> List[SearchResult]:
        """Search emails with advanced keyword matching"""
        start_time = time.time()
        
        try:
            # Check cache first
            cache_key = hashlib.md5(json.dumps({
                'keywords': query.keywords,
                'search_type': query.search_type,
                'industry': query.industry,
                'domain_pattern': query.domain_pattern,
                'email_type': query.email_type
            }, sort_keys=True).encode()).hexdigest()
            
            if cache_key in self.search_cache:
                cached_result = self.search_cache[cache_key]
                if time.time() - cached_result['timestamp'] < 300:  # 5 minute cache
                    return cached_result['results']
            
            # Get candidate emails
            candidate_emails = self._get_candidate_emails(query)
            
            # Score and rank emails
            results = []
            for email in candidate_emails:
                relevance_score = self._calculate_relevance_score(email, query.keywords, query.search_type)
                
                if relevance_score >= query.confidence_threshold:
                    match_type = self._determine_match_type(email, query.keywords)
                    matched_keywords = self._get_matched_keywords(email, query.keywords)
                    snippet = self._generate_snippet(email, matched_keywords)
                    
                    result = SearchResult(
                        email=email,
                        relevance_score=relevance_score,
                        match_type=match_type,
                        matched_keywords=matched_keywords,
                        confidence=relevance_score,
                        metadata=self.search_index['emails'].get(email, {}),
                        snippet=snippet
                    )
                    results.append(result)
            
            # Sort by relevance score
            results.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Limit results
            results = results[:query.max_results]
            
            # Cache results
            self.search_cache[cache_key] = {
                'results': results,
                'timestamp': time.time()
            }
            
            # Update analytics
            self.search_analytics['total_searches'] += 1
            self.search_analytics[f'search_type_{query.search_type}'] += 1
            self.search_analytics['results_returned'] += len(results)
            
            logger.info(f"Search completed in {time.time() - start_time:.2f}s, found {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    def _get_candidate_emails(self, query: SearchQuery) -> Set[str]:
        """Get candidate emails based on query filters"""
        candidates = set()
        
        # Start with all indexed emails
        if not query.keywords and not query.industry and not query.domain_pattern:
            candidates = set(self.search_index['emails'].keys())
        else:
            # Filter by keywords
            if query.keywords:
                keyword_candidates = set()
                for keyword in query.keywords:
                    if keyword in self.search_index['keywords']:
                        keyword_candidates.update(self.search_index['keywords'][keyword])
                candidates = keyword_candidates if candidates else keyword_candidates
            
            # Filter by industry
            if query.industry:
                industry_candidates = set()
                if query.industry in self.search_index['industries']:
                    industry_candidates = set(self.search_index['industries'][query.industry])
                candidates = candidates & industry_candidates if candidates else industry_candidates
            
            # Filter by domain pattern
            if query.domain_pattern:
                domain_candidates = set()
                for domain, emails in self.search_index['domains'].items():
                    if re.search(query.domain_pattern, domain, re.IGNORECASE):
                        domain_candidates.update(emails)
                candidates = candidates & domain_candidates if candidates else domain_candidates
        
        return candidates
    
    def get_search_suggestions(self, partial_query: str, limit: int = 10) -> List[str]:
        """Get search suggestions based on partial query"""
        try:
            suggestions = []
            partial_lower = partial_query.lower()
            
            # Get suggestions from keywords
            for keyword in self.search_index['keywords']:
                if partial_lower in keyword.lower():
                    suggestions.append(keyword)
            
            # Get suggestions from industries
            for industry in self.industry_keywords:
                if partial_lower in industry.lower():
                    suggestions.append(industry)
            
            # Get suggestions from industry keywords
            for industry, keywords in self.industry_keywords.items():
                for keyword in keywords:
                    if partial_lower in keyword.lower() and keyword not in suggestions:
                        suggestions.append(keyword)
            
            # Sort by relevance and limit
            suggestions.sort(key=lambda x: len(x))
            return suggestions[:limit]
            
        except Exception as e:
            logger.error(f"Search suggestions failed: {e}")
            return []
    
    def get_search_analytics(self) -> Dict[str, Any]:
        """Get search analytics and performance metrics"""
        return {
            'total_searches': self.search_analytics['total_searches'],
            'search_types': {
                'exact': self.search_analytics.get('search_type_exact', 0),
                'fuzzy': self.search_analytics.get('search_type_fuzzy', 0),
                'semantic': self.search_analytics.get('search_type_semantic', 0),
                'all': self.search_analytics.get('search_type_all', 0)
            },
            'total_results_returned': self.search_analytics['results_returned'],
            'average_results_per_search': (
                self.search_analytics['results_returned'] / 
                max(self.search_analytics['total_searches'], 1)
            ),
            'indexed_emails': len(self.search_index['emails']),
            'indexed_keywords': len(self.search_index['keywords']),
            'industries_covered': len(self.search_index['industries']),
            'cache_hit_rate': len(self.search_cache) / max(self.search_analytics['total_searches'], 1)
        }
    
    def bulk_index_emails(self, emails: List[str], metadata_list: List[Dict[str, Any]] = None):
        """Bulk index multiple emails"""
        try:
            for i, email in enumerate(emails):
                metadata = metadata_list[i] if metadata_list and i < len(metadata_list) else None
                self.index_email(email, metadata)
            
            logger.info(f"Bulk indexed {len(emails)} emails")
            
        except Exception as e:
            logger.error(f"Bulk indexing failed: {e}")


# Global instance
ultra_search_engine = UltraKeywordSearchEngine()

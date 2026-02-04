"""
Katherine Orchestrator - Reddit Client
Fetches top posts from subreddits with submission statement detection.

Submission statements are explanatory comments required by some subreddits
(like r/geopolitics) where the OP must provide context for link posts.
"""
import asyncio
import time
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from datetime import datetime, timezone

import httpx
from loguru import logger

from config import settings


@dataclass
class RedditPost:
    """Represents a Reddit post with its submission statement."""
    title: str
    url: str
    author: str
    score: int
    created_utc: float
    permalink: str
    post_id: str
    selftext: Optional[str] = None
    submission_statement: Optional[str] = None
    num_comments: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for prompt injection."""
        created_dt = datetime.fromtimestamp(self.created_utc, tz=timezone.utc)
        return {
            "title": self.title,
            "url": self.url,
            "author": self.author,
            "score": self.score,
            "created": created_dt.strftime("%Y-%m-%d %H:%M UTC"),
            "submission_statement": self.submission_statement,
            "selftext": self.selftext,
            "permalink": self.permalink,
            "num_comments": self.num_comments
        }


@dataclass
class CachedPosts:
    """Cache container for Reddit posts."""
    posts: List[RedditPost] = field(default_factory=list)
    fetched_at: float = 0.0
    subreddit: str = ""
    
    def is_valid(self, ttl_seconds: int) -> bool:
        """Check if cache is still valid."""
        if not self.posts:
            return False
        return (time.time() - self.fetched_at) < ttl_seconds


class RedditClient:
    """
    Async client for fetching Reddit posts via the public JSON API.
    
    Features:
    - Automatic submission statement detection
    - Response caching to avoid rate limits
    - Graceful error handling
    
    Reddit's public JSON API doesn't require authentication for read-only access.
    """
    
    def __init__(self):
        self._client: Optional[httpx.AsyncClient] = None
        self._cache: Dict[str, CachedPosts] = {}
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize the HTTP client."""
        self._client = httpx.AsyncClient(
            base_url="https://www.reddit.com",
            timeout=httpx.Timeout(30.0, connect=10.0),
            headers={
                # Reddit requires a descriptive User-Agent
                "User-Agent": "Katherine-AI-Companion/1.0 (Memory-augmented chatbot)"
            },
            follow_redirects=True
        )
        logger.info("Reddit client initialized")
    
    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None
            logger.info("Reddit client closed")
    
    async def get_top_posts(
        self,
        subreddit: str = None,
        limit: int = None,
        time_filter: str = None,
        use_cache: bool = True
    ) -> List[RedditPost]:
        """
        Fetch top posts from a subreddit with submission statements.
        
        Args:
            subreddit: Subreddit name without r/ prefix (default from config)
            limit: Number of posts to fetch (default from config)
            time_filter: Time period - hour, day, week, month, year, all (default from config)
            use_cache: Whether to use cached results if available
        
        Returns:
            List of RedditPost objects with submission statements populated
        """
        if not self._client:
            logger.warning("Reddit client not initialized")
            return []
        
        # Use config defaults
        subreddit = subreddit or settings.reddit_subreddit
        limit = limit or settings.reddit_posts_limit
        time_filter = time_filter or settings.reddit_time_filter
        
        cache_key = f"{subreddit}:{time_filter}:{limit}"
        
        # Check cache first
        if use_cache and cache_key in self._cache:
            cached = self._cache[cache_key]
            if cached.is_valid(settings.reddit_cache_ttl_seconds):
                logger.debug(f"Using cached Reddit posts for r/{subreddit}")
                return cached.posts
        
        # Fetch fresh data with lock to prevent concurrent requests
        async with self._lock:
            # Double-check cache after acquiring lock
            if use_cache and cache_key in self._cache:
                cached = self._cache[cache_key]
                if cached.is_valid(settings.reddit_cache_ttl_seconds):
                    return cached.posts
            
            posts = await self._fetch_posts(subreddit, limit, time_filter)
            
            # Update cache
            self._cache[cache_key] = CachedPosts(
                posts=posts,
                fetched_at=time.time(),
                subreddit=subreddit
            )
            
            return posts
    
    async def _fetch_posts(
        self,
        subreddit: str,
        limit: int,
        time_filter: str
    ) -> List[RedditPost]:
        """Fetch posts from Reddit API."""
        try:
            # Fetch top posts
            url = f"/r/{subreddit}/top.json"
            params = {
                "limit": limit,
                "t": time_filter,
                "raw_json": 1  # Get unescaped JSON
            }
            
            response = await self._client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            posts = []
            children = data.get("data", {}).get("children", [])
            
            for child in children:
                if child.get("kind") != "t3":  # t3 = link/post
                    continue
                
                post_data = child["data"]
                post = self._parse_post(post_data)
                posts.append(post)
            
            # Fetch submission statements for posts that need them
            await self._populate_submission_statements(posts)
            
            logger.info(
                f"Fetched {len(posts)} posts from r/{subreddit} "
                f"({sum(1 for p in posts if p.submission_statement)} with submission statements)"
            )
            return posts
            
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                logger.warning("Reddit rate limit hit, using cached data if available")
            else:
                logger.error(f"Reddit API error: {e.response.status_code}")
            return []
        except Exception as e:
            logger.error(f"Failed to fetch Reddit posts: {e}")
            return []
    
    def _parse_post(self, post_data: Dict) -> RedditPost:
        """Parse raw post data into RedditPost object."""
        selftext = post_data.get("selftext", "")
        
        # Clean up selftext
        if selftext in ["[removed]", "[deleted]", ""]:
            selftext = None
        
        return RedditPost(
            title=post_data.get("title", ""),
            url=post_data.get("url", ""),
            author=post_data.get("author", "[deleted]"),
            score=post_data.get("score", 0),
            created_utc=post_data.get("created_utc", 0),
            permalink=f"https://www.reddit.com{post_data.get('permalink', '')}",
            post_id=post_data.get("id", ""),
            selftext=selftext,
            num_comments=post_data.get("num_comments", 0)
        )
    
    async def _populate_submission_statements(self, posts: List[RedditPost]) -> None:
        """
        Populate submission statements for all posts.
        
        Detection strategy:
        1. For self posts: selftext IS the submission statement
        2. For link posts: Find OP's first substantial comment
        3. Look for pinned/stickied OP comments first
        """
        for post in posts:
            # Self posts use selftext as submission statement
            if post.selftext and len(post.selftext.strip()) >= settings.reddit_min_ss_length:
                post.submission_statement = post.selftext
                continue
            
            # For link posts, fetch OP's comment
            if post.author != "[deleted]":
                ss = await self._find_submission_statement(post.post_id, post.author)
                if ss:
                    post.submission_statement = ss
    
    async def _find_submission_statement(
        self,
        post_id: str,
        op_username: str
    ) -> Optional[str]:
        """
        Find the submission statement in post comments.
        
        Looks for:
        1. Stickied comment from OP (highest priority)
        2. First substantial comment from OP
        3. OP comment containing keywords like "submission statement" or "SS:"
        """
        try:
            # Fetch post comments
            url = f"/comments/{post_id}.json"
            params = {
                "limit": 20,  # Check first 20 top-level comments
                "depth": 1,   # Only top-level
                "raw_json": 1
            }
            
            response = await self._client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if len(data) < 2:
                return None
            
            comments = data[1].get("data", {}).get("children", [])
            
            # Track candidates
            stickied_comment = None
            keyword_comment = None
            first_substantial_comment = None
            
            for item in comments:
                if item.get("kind") != "t1":  # t1 = comment
                    continue
                
                comment = item.get("data", {})
                author = comment.get("author", "")
                body = comment.get("body", "")
                is_stickied = comment.get("stickied", False)
                
                # Skip if not from OP or empty/deleted
                if author != op_username:
                    continue
                if not body or body in ["[removed]", "[deleted]"]:
                    continue
                
                body_stripped = body.strip()
                body_lower = body_stripped.lower()
                
                # Check if stickied (highest priority)
                if is_stickied and len(body_stripped) >= settings.reddit_min_ss_length:
                    stickied_comment = body_stripped
                    break  # Stickied OP comment is definitely the SS
                
                # Check for SS keywords
                ss_keywords = [
                    "submission statement",
                    "ss:",
                    "[ss]",
                    "rule 1",
                    "rule one"
                ]
                if any(kw in body_lower for kw in ss_keywords):
                    if len(body_stripped) >= settings.reddit_min_ss_length:
                        keyword_comment = body_stripped
                
                # Track first substantial comment as fallback
                if first_substantial_comment is None:
                    if len(body_stripped) >= settings.reddit_min_ss_length:
                        first_substantial_comment = body_stripped
            
            # Return in priority order
            if stickied_comment:
                logger.debug(f"Found stickied SS for post {post_id}")
                return stickied_comment
            if keyword_comment:
                logger.debug(f"Found keyword SS for post {post_id}")
                return keyword_comment
            if first_substantial_comment:
                logger.debug(f"Found fallback SS for post {post_id}")
                return first_substantial_comment
            
            return None
            
        except Exception as e:
            logger.debug(f"Could not fetch comments for post {post_id}: {e}")
            return None
    
    def clear_cache(self) -> None:
        """Clear the post cache."""
        self._cache.clear()
        logger.info("Reddit cache cleared")


# Singleton instance
reddit_client = RedditClient()

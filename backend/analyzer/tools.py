import asyncio
import logging
import re
import urllib.parse
import httpx
from bs4 import BeautifulSoup

from backend.models.db import SessionLocal, Setting

logger = logging.getLogger("jobnavigator.tools")

def _get_setting(key: str, default: str = "") -> str:
    db = SessionLocal()
    try:
        row = db.query(Setting).filter(Setting.key == key).first()
        return row.value if row and row.value else default
    finally:
        db.close()

async def web_search(query: str) -> str:
    """Perform a web search using Searxng (if configured) with DuckDuckGo fallback."""
    searxng_url = _get_setting("searxng_url", "http://host.docker.internal:8043")
    
    # Try Searxng first
    results = []
    if searxng_url:
        try:
            results = await _searxng_search_raw(query, searxng_url)
        except Exception as e:
            logger.warning(f"Searxng search failed: {e}. Falling back to DuckDuckGo.")
            
    # Fallback to DuckDuckGo HTML search if Searxng failed or returned no results
    if not results:
        try:
            results = await _ddg_fallback_search_raw(query)
        except Exception as e:
            logger.error(f"DuckDuckGo fallback search failed: {e}")
            return f"Error executing web search: {e}"
            
    if not results:
        return "No search results found."
        
    formatted = []
    for i, r in enumerate(results[:5], 1):
        formatted.append(f"{i}. Title: {r['title']}\n   URL: {r['url']}\n   Snippet: {r['content']}\n")
    return "\n".join(formatted)

async def _searxng_search_raw(query: str, searxng_url: str) -> list:
    url = searxng_url.strip()
    if not url.startswith("http"):
        url = "http://" + url
    url = url.rstrip("/") + "/search"
    
    params = {
        "q": query,
        "format": "json",
    }
    
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        results = []
        for item in data.get("results", []):
            results.append({
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "content": item.get("content", item.get("snippet", ""))
            })
        return results

async def _ddg_fallback_search_raw(query: str) -> list:
    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    }
    params = {"q": query}
    
    async with httpx.AsyncClient(timeout=10, headers=headers) as client:
        response = await client.get(url, params=params)
        if response.status_code != 200:
            raise RuntimeError(f"DuckDuckGo status {response.status_code}")
            
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for result in soup.select(".result"):
            title_el = result.select_one(".result__title a")
            snippet_el = result.select_one(".result__snippet")
            if title_el:
                title = title_el.get_text(strip=True)
                href = title_el.get("href", "")
                
                match = re.search(r"uddg=([^&]+)", href)
                if match:
                    href = urllib.parse.unquote(match.group(1))
                    
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                results.append({
                    "title": title,
                    "url": href,
                    "content": snippet
                })
        return results

# Helper map to run tool by name
TOOL_FUNCTIONS = {
    "web_search": web_search
}

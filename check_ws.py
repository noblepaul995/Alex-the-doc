from config.settings import get_settings
from utils.web_search import is_web_search_configured

s = get_settings()
print("tavily:", bool(s.tavily_api_key))
print("endpoint:", s.open_websearch_endpoint)
print("autostart:", s.open_websearch_autostart)
print("configured:", is_web_search_configured(s))
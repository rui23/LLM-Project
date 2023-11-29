from duckduckgo_search import ddg
from duckduckgo_search.utils import SESSION


SESSION.proxies = {
    "http": f"socks5h://localhost:7890",
    "https": f"socks5h://localhost:7890"
}
r = ddg("香港科技大学（广州）")
print(r[:2])
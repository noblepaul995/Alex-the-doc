import asyncio
import httpx

async def main():
    async with httpx.AsyncClient(base_url="http://localhost:11434") as client:
        r = await client.post(
            "/api/chat",
            json={
                "model": "qwen3:8b",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "stream": False,
            },
        )

        print(r.status_code)
        print(r.text)

asyncio.run(main())
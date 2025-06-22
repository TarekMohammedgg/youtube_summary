from api import app

if __name__ == "__main__":
    import nest_asyncio
    import uvicorn
    from pyngrok import ngrok

    nest_asyncio.apply()

    NGROK_AUTH_TOKEN = "your_ngrok_token_here"
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    for tunnel in ngrok.get_tunnels():
        ngrok.disconnect(tunnel.public_url)
    public_url = ngrok.connect(8001, bind_tls=True)
    print("🚀 Your API is live at:", public_url)

    uvicorn.run(app, host="0.0.0.0", port=8001)

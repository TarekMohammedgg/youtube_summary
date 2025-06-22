from api import app

if __name__ == "__main__":
    import nest_asyncio
    import uvicorn
    from pyngrok import ngrok

    nest_asyncio.apply()

    NGROK_AUTH_TOKEN = "2vhh5wt4kpowpHg2zxmvDJ7NNFO_3r3E2DC5oHZ8C3HjcJZRR"
    ngrok.set_auth_token(NGROK_AUTH_TOKEN)
    for tunnel in ngrok.get_tunnels():
        ngrok.disconnect(tunnel.public_url)
    public_url = ngrok.connect(8001, bind_tls=True)
    print("🚀 Your API is live at:", public_url)

    uvicorn.run(app, host="0.0.0.0", port=8001)

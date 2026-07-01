import chainlit as cl
from rag import chat


@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="🪐 Planetary Significations",
            message="What are the significations of Saturn?",
        ),
        cl.Starter(
            label="🏠 Houses",
            message="What does the 7th house signify?",
        ),
        cl.Starter(
            label="✨ Yogas",
            message="Explain Gaja Kesari Yoga.",
        ),
        cl.Starter(
            label="💼 Career",
            message="Which planets indicate career success?",
        ),
    ]


@cl.on_message
async def main(message: cl.Message):

    msg = cl.Message(content="🔮 Thinking...")
    await msg.send()

    try:
        response = chat(message.content)
        msg.content = response
    except Exception as e:
        msg.content = f"❌ {e}"

    await msg.update()
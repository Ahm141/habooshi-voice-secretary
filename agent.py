import os
from dotenv import load_dotenv
load_dotenv()

from livekit.agents import Agent, AgentSession, JobContext, RunContext, WorkerOptions, cli, function_tool
from livekit.agents.llm import ChatContext
from livekit.plugins import google
import core

SESSION = "voice"


class Secretary(Agent):
    def __init__(self, chat_ctx: ChatContext):
        super().__init__(instructions=core.system_prompt(), chat_ctx=chat_ctx)

    @function_tool
    async def add_appointment(self, context: RunContext, title: str, date: str, time: str, notes: str = "") -> str:
        """سجّل موعد جديد. date بصيغة YYYY-MM-DD و time بصيغة 24 ساعة HH:MM."""
        return core.add_appointment.invoke({"title": title, "date": date, "time": time, "notes": notes})

    @function_tool
    async def list_appointments(self, context: RunContext, date: str = "") -> str:
        """اعرض المواعيد الفعالة. اترك date فارغ لعرض كل المواعيد القادمة."""
        return core.list_appointments.invoke({"date": date})

    @function_tool
    async def cancel_appointment(self, context: RunContext, appointment_id: int) -> str:
        """الغِ موعد حسب رقمه."""
        return core.cancel_appointment.invoke({"appointment_id": appointment_id})


async def entrypoint(ctx: JobContext):
    await ctx.connect()

    model = os.getenv("GEMINI_LIVE_MODEL") or "gemini-3.8-live"
    session = AgentSession(llm=google.realtime.RealtimeModel(
        model=model,
        api_version="v1beta",
        voice=os.getenv("GEMINI_VOICE", "Puck"),   # صوت ولد
        temperature=0.8,
    ))

    @session.on("conversation_item_added")
    def _saved(ev):
        try:
            it = ev.item
            role = getattr(it, "role", None)
            text = getattr(it, "text_content", None)
            if role in ("user", "assistant") and text:
                core.save(SESSION, role, text)
        except Exception as e:
            print("save error:", e)

    hist = ChatContext()
    for role, content in core.history(SESSION):
        hist.add_message(role=role, content=content)

    await session.start(agent=Secretary(hist), room=ctx.room)
    await session.generate_reply(instructions="رحب بأحمد بجملة قصيرة وعفوية وسأله شنو يريد.")


if __name__ == "__main__":
    cli.run_app(WorkerOptions(entrypoint_fnc=entrypoint))

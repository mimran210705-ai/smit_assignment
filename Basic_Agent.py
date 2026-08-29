import asyncio
import os

# ==========================================
# DISABLE OPENAI AGENTS TRACING
# ==========================================

os.environ["OPENAI_AGENTS_DISABLE_TRACING"] = "1"

import re
from datetime import datetime
from zoneinfo import ZoneInfo
import requests
from pathlib import Path
from dotenv import load_dotenv
from openai import AsyncOpenAI

from agents import (
    Agent,
    Runner,
    OpenAIChatCompletionsModel,
    function_tool,
)


# ==========================================
# LOAD ENVIRONMENT VARIABLES
# ==========================================

load_dotenv(Path(__file__).resolve().parent / ".env", override=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEMINI_MODEL = os.getenv("GEMINI_MODEL")

if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY was not found in .env")

if not GEMINI_MODEL:
    raise ValueError("GEMINI_MODEL was not found in .env")


# ==========================================
# GEMINI CONNECTION
# ==========================================

client = AsyncOpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

gemini_model = OpenAIChatCompletionsModel(
    model=GEMINI_MODEL,
    openai_client=client,
)


# ==========================================
# TIME FUNCTION LOGIC
# ==========================================

def get_time_logic(city: str) -> str:

    timezones = {
        "faisalabad": "Asia/Karachi",
        "lahore": "Asia/Karachi",
        "karachi": "Asia/Karachi",
        "islamabad": "Asia/Karachi",

        "london": "Europe/London",

        "dubai": "Asia/Dubai",

        "tokyo": "Asia/Tokyo",

        "new york": "America/New_York",

        "los angeles": "America/Los_Angeles",

        "paris": "Europe/Paris",
    }

    city_name = city.lower().strip()

    if city_name not in timezones:
        return f"Sorry, I don't know the timezone for {city}."

    timezone = ZoneInfo(timezones[city_name])

    current_time = datetime.now(timezone)

    formatted_time = current_time.strftime("%I:%M:%S %p")

    formatted_date = current_time.strftime("%A, %d %B %Y")

    return (
        f"The current time in {city.title()} is {formatted_time}.\n"
        f"Date: {formatted_date}"
    )


# ==========================================
# TIME TOOL
# ==========================================

@function_tool
def get_time(city: str) -> str:
    """
    Get the current time and date for a city.
    """

    print(f"\n🔧 TIME TOOL CALLED: {city}")

    result = get_time_logic(city)

    print(f"🔧 TIME TOOL RESULT: {result}\n")

    return result


# ==========================================
# WIKIPEDIA TOOL
# ==========================================

@function_tool
def search_wikipedia(topic: str) -> str:
    """
    Search Wikipedia for information about a topic.
    """

    print(f"\n🔧 WIKIPEDIA TOOL CALLED: {topic}")

    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "list": "search",
        "srsearch": topic,
        "format": "json",
        "utf8": 1,
        "srlimit": 3,
    }

    try:

        response = requests.get(
            url,
            params=params,
            timeout=10
        )

        response.raise_for_status()

        data = response.json()

        results = data.get("query", {}).get("search", [])

        if not results:
            return f"No Wikipedia results found for {topic}."

        output = f"Wikipedia results for '{topic}':\n\n"

        for result in results:

            title = result["title"]

            snippet = result["snippet"]

            # Remove HTML tags
            snippet = re.sub("<.*?>", "", snippet)

            output += f"Title: {title}\n"
            output += f"Information: {snippet}\n\n"

        print("🔧 WIKIPEDIA TOOL FINISHED\n")

        return output

    except Exception as e:

        return f"Could not search Wikipedia: {str(e)}"


# ==========================================
# AGENT INSTRUCTIONS
# ==========================================

instruction = """
You are a helpful AI assistant.

You have two tools.

================================
TIME TOOL
================================

Use the get_time tool whenever the user asks about:

- current time
- time in a city
- current date
- date and time
- timezone
- what time is it

You MUST use get_time.

Never guess the current time.

Examples:

User:
What is the time in London?

Action:
Call get_time with city="London"


User:
What time is it in Karachi?

Action:
Call get_time with city="Karachi"


================================
WIKIPEDIA TOOL
================================

Use search_wikipedia whenever the user asks you to search Wikipedia.

Examples:

User:
Search Wikipedia for Albert Einstein.

Action:
Call search_wikipedia with topic="Albert Einstein"


User:
Search Wikipedia for Pakistan.

Action:
Call search_wikipedia with topic="Pakistan"


================================
GENERAL QUESTIONS
================================

For normal questions that do not require a tool, answer normally.

Always use the appropriate tool when it is available.
"""


# ==========================================
# CREATE AGENT
# ==========================================

agent = Agent(
    name="Gemini Assistant",

    instructions=instruction,

    model=gemini_model,

    tools=[
        get_time,
        search_wikipedia,
    ],
)


# ==========================================
# MAIN PROGRAM
# ==========================================

async def main():

    print("================================")
    print("      Gemini AI Assistant")
    print("================================")

    print(f"Model: {GEMINI_MODEL}")

    print("\nAvailable tools:")
    print("1. get_time")
    print("2. search_wikipedia")

    print("\nType 'exit' to quit.\n")


    while True:

        try:

            user_question = input("You: ").strip()

            if not user_question:
                continue

            if user_question.lower() == "exit":

                print("\nGoodbye!")
                break


            result = await Runner.run(
                agent,
                input=user_question
            )


            print("\nGemini:")
            print(result.final_output)
            print()


        except KeyboardInterrupt:

            print("\n\nGoodbye!")
            break


        except Exception as e:

            print("\n❌ ERROR:")

            error_text = str(e)

            if "429" in error_text or "quota" in error_text.lower():

                print(
                    "Gemini API quota has been exceeded.\n"
                    "Please wait for the quota to reset and try again."
                )

            else:

                print(error_text)

            print()


# ==========================================
# PROGRAM START
# ==========================================

if __name__ == "__main__":

    asyncio.run(main())